#!/usr/bin/env python3
"""Run the complete restartable Irodori-TTS data preparation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from dataset._io_utils import (
    atomic_write_json as _atomic_write_json,
)
from dataset._io_utils import (
    replace_with_retry,
)
from dataset.speech_pipeline import discover_audio_sources

PIPELINE_SCHEMA_VERSION = 1
STAGES = (
    "speech",
    "nonverbal",
    "transcribe",
    "context_correction",
    "latents",
    "publish",
)
# Stages that need CUDA; preflight only requires a GPU when one of these is in
# the selected --start-at/--stop-after range. The nonverbal stage is pure CPU.
GPU_STAGES = frozenset({"speech", "transcribe", "latents"})
# Everything that determines latent content. Part of the _latent_one cache key
# and the latents stage fingerprint so changed encode settings invalidate
# previously generated latents instead of silently reusing them.
LATENT_ENCODE_PARAMS: dict[str, Any] = {
    "codec_repo": "Aratako/Semantic-DACVAE-Japanese-32dim",
    "normalize_db": -16.0,
    "deterministic_encode": True,
    "deterministic_decode": True,
    "text_normalize": False,
}
PUBLISH_BACKUP_KEEP = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(payload: Any) -> str:
    value = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _path_record(path: Path) -> dict[str, Any]:
    """Content-addressed fingerprint record for a pipeline artifact.

    ``mtime_ns`` is only used when the file is too large to hash: including it
    for hashed files would change stage fingerprints whenever an earlier stage
    rewrites a byte-identical output, forcing needless re-runs forever.
    """
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {"path": resolved.as_posix(), "missing": True}
    stat = resolved.stat()
    record: dict[str, Any] = {
        "path": resolved.as_posix(),
        "size_bytes": stat.st_size,
    }
    if resolved.is_file() and stat.st_size <= 128 * 1024 * 1024:
        record["sha256"] = _sha256_file(resolved)
    else:
        record["mtime_ns"] = stat.st_mtime_ns
    return record


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def _backup_corrupt_state(path: Path, reason: str) -> None:
    backup: Path | None = path.with_name(f"{path.name}.bak")
    try:
        shutil.copy2(path, backup)
    except OSError:
        backup = None
    detail = f"previous file saved to {backup}" if backup else "previous file could not be saved"
    print(f"warning: pipeline state {path} was reset ({reason}); {detail}", flush=True)


def _load_state(path: Path) -> dict[str, Any]:
    fresh: dict[str, Any] = {"schema_version": PIPELINE_SCHEMA_VERSION, "stages": {}}
    if not path.is_file():
        return fresh
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _backup_corrupt_state(path, f"unreadable: {exc}")
        return fresh
    if not isinstance(payload, dict) or payload.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        _backup_corrupt_state(path, "schema mismatch")
        return fresh
    if not isinstance(payload.get("stages"), dict):
        payload["stages"] = {}
    return payload


class PipelineRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path.cwd().resolve()
        self.data_root = args.data_root.expanduser().resolve()
        self.work_dir = args.work_dir.expanduser().resolve()
        self.manifest_dir = args.manifest_dir.expanduser().resolve()
        self.latent_root = args.latent_root.expanduser().resolve()
        self.state_path = self.work_dir / "full_pipeline" / "state.json"
        self.log_dir = self.work_dir / "full_pipeline" / "logs"
        self.state = _load_state(self.state_path)
        self.force_stages = set(args.force_stage or ())
        self._gpu_indices_cache: tuple[int, ...] | None = None

    @property
    def gpu_indices(self) -> tuple[int, ...]:
        """Resolve GPUs lazily so CPU-only hosts can run GPU-free stage ranges."""
        if self._gpu_indices_cache is None:
            self._gpu_indices_cache = self._resolve_gpu_indices()
        return self._gpu_indices_cache

    def _resolve_gpu_indices(self) -> tuple[int, ...]:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The selected pipeline stages require CUDA, "
                "but torch.cuda.is_available() is false"
            )
        available = torch.cuda.device_count()
        if self.args.gpus == "all":
            indices = tuple(range(available))
        else:
            try:
                indices = tuple(int(value.strip()) for value in self.args.gpus.split(","))
            except ValueError as exc:
                raise ValueError("--gpus must be 'all' or comma-separated integer indices") from exc
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("--gpus must select at least one unique GPU")
        invalid = [index for index in indices if not 0 <= index < available]
        if invalid:
            raise ValueError(f"Invalid GPU indices {invalid}; available={available}")
        return indices

    @property
    def primary_device(self) -> str:
        return f"cuda:{self.gpu_indices[0]}"

    def save_state(self) -> None:
        self.state["schema_version"] = PIPELINE_SCHEMA_VERSION
        self.state["updated_at"] = _now()
        _atomic_write_json(self.state_path, self.state)

    def run_command(self, stage: str, command: Sequence[str]) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{stage}.log"
        display = subprocess.list2cmdline(list(command))
        print(f"[{stage}] command={display}", flush=True)
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUNBUFFERED"] = "1"
        with log_path.open("a", encoding="utf-8", newline="\n") as log:
            log.write(f"\n[{_now()}] command={display}\n")
            process = subprocess.Popen(
                list(command),
                cwd=self.root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"stage {stage!r} command failed with exit code {return_code}; see {log_path}"
            )

    def run_stage(
        self,
        name: str,
        *,
        fingerprint: str,
        outputs: Sequence[Path],
        action: Any,
    ) -> None:
        stage_state = self.state["stages"].get(name, {})
        complete = (
            name not in self.force_stages
            and stage_state.get("status") == "complete"
            and stage_state.get("fingerprint") == fingerprint
            and all(path.exists() for path in outputs)
        )
        if complete:
            print(f"[{name}] cached complete", flush=True)
            return
        self.state["stages"][name] = {
            "status": "running",
            "fingerprint": fingerprint,
            "started_at": _now(),
            "outputs": [path.as_posix() for path in outputs],
        }
        self.save_state()
        try:
            action()
            missing = [path for path in outputs if not path.exists()]
            if missing:
                raise RuntimeError(f"stage {name} did not produce: {missing}")
        except BaseException as exc:
            self.state["stages"][name].update(
                {
                    "status": "failed",
                    "failed_at": _now(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            self.save_state()
            raise
        self.state["stages"][name].update(
            {"status": "complete", "completed_at": _now(), "error": None}
        )
        self.save_state()
        print(f"[{name}] complete", flush=True)

    def _stage_range(self) -> tuple[int, int]:
        start = STAGES.index(self.args.start_at) if self.args.start_at else 0
        stop = STAGES.index(self.args.stop_after) if self.args.stop_after else len(STAGES) - 1
        if start > stop:
            raise ValueError("--start-at must not come after --stop-after")
        return start, stop

    def _selected_stages(self) -> tuple[str, ...]:
        start, stop = self._stage_range()
        return STAGES[start : stop + 1]

    def _preflight_problem(self, message: str) -> None:
        """Raise for real runs; downgrade to a warning during --dry-run."""
        if self.args.dry_run:
            print(f"preflight warning: {message}", flush=True)
            return
        raise RuntimeError(message)

    def selected_sources(self) -> list[Any]:
        # Every pipeline output directory that may live under data_root must be
        # excluded, or generated FLAC clips would be rediscovered as sources.
        excluded = (
            self.work_dir,
            self.manifest_dir,
            self.latent_root,
        )
        sources = discover_audio_sources(
            self.data_root, output_dir=self.work_dir, excluded_dirs=excluded
        )
        sources = [source for source in sources if source.duration > self.args.min_source_seconds]
        if self.args.max_files is not None:
            sources = sources[: self.args.max_files]
        return sources

    def preflight(self) -> None:
        selected_stages = set(self._selected_stages())
        sources = self.selected_sources()
        if not sources:
            raise RuntimeError(f"No supported audio files found under {self.data_root}")
        cached = sum(
            1
            for source in sources
            if (self.work_dir / "vad_responses" / f"{source.source_id}.json").is_file()
        )
        total_hours = sum(source.duration for source in sources) / 3600.0
        if (
            "context_correction" in selected_stages
            and shutil.which("codex") is None
            and shutil.which("claude") is None
        ):
            self._preflight_problem(
                "Neither codex nor claude CLI is installed for transcript correction"
            )
        gpu_display = "unused"
        if selected_stages & GPU_STAGES:
            try:
                gpu_display = str(self.gpu_indices)
            except (RuntimeError, ValueError) as exc:
                self._preflight_problem(str(exc))
                gpu_display = "unavailable"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        disk = shutil.disk_usage(self.data_root)
        print(
            f"preflight sources={len(sources)} cached_vad={cached} "
            f"duration={total_hours:.3f}h gpus={gpu_display} "
            f"source_duration=>{self.args.min_source_seconds:.3f}s "
            f"free_disk={disk.free / 1024**3:.1f}GiB",
            flush=True,
        )

    def speech(self) -> None:
        sources = self.selected_sources()
        # Execution environment (GPU list) is deliberately excluded: it does
        # not change output content and would invalidate the cache needlessly.
        fingerprint = _hash(
            {
                "sources": [source.metadata() for source in sources],
                "vad": {"speech_pad_ms": self.args.audio_padding_ms},
                "clip_padding": self.args.audio_padding_seconds,
                "segmentation": "vad-5-20s",
                "min_source_seconds": self.args.min_source_seconds,
                "max_source_errors": self.args.max_source_errors,
            }
        )
        command = [
            sys.executable,
            "-m",
            "dataset.cli.prepare_speech",
            "--input-dir",
            str(self.data_root),
            "--output-dir",
            str(self.work_dir),
            "--project-root",
            str(self.root),
            "--min-source-seconds",
            str(self.args.min_source_seconds),
            "--vad-speech-pad-ms",
            str(self.args.audio_padding_ms),
            "--padding-seconds",
            str(self.args.audio_padding_seconds),
            "--max-source-errors",
            str(self.args.max_source_errors),
        ]
        for directory in (self.work_dir, self.manifest_dir, self.latent_root):
            command.extend(["--exclude-dir", str(directory)])
        if self.args.rebuild_clips:
            # Clips are content-addressed (id embeds start/end ms) and written
            # atomically, so re-encoding them is opt-in instead of the default.
            command.append("--rebuild-clips")
        if self.args.max_files is not None:
            command.extend(["--max-files", str(self.args.max_files)])

        # The speech stage is CPU-bound (decode/resample/FLAC). A dynamic
        # in-process pool lets every worker pull the next source when free, so
        # uneven source durations cannot leave a single-worker tail.
        command.extend(
            [
                "--workers",
                str(max(1, int(self.args.speech_shards))),
                "--vad-devices",
                ",".join(f"cuda:{index}" for index in self.gpu_indices),
            ]
        )
        self.run_stage(
            "speech",
            fingerprint=fingerprint,
            outputs=(self.work_dir / "all.jsonl", self.work_dir / "train.jsonl"),
            action=lambda: self.run_command("speech", command),
        )

    def nonverbal(self) -> None:
        """Cut VAD-complement event clips; ASR + LLM assign text and category."""
        raw_paths = sorted((self.work_dir / "vad_responses").glob("*.json"))
        fingerprint = _hash(
            {
                "vad": [_path_record(path) for path in raw_paths],
                "segmentation": "acoustic-vad-complement-5-20s",
                "min_source_seconds": self.args.min_source_seconds,
            }
        )
        command = [
            sys.executable,
            "-m",
            "dataset.cli.prepare_nonverbal",
            "--input-dir",
            str(self.data_root),
            "--output-dir",
            str(self.work_dir),
            "--project-root",
            str(self.root),
            "--min-source-seconds",
            str(self.args.min_source_seconds),
            "--workers",
            str(max(1, int(self.args.speech_shards))),
            "--vad-speech-pad-ms",
            str(self.args.audio_padding_ms),
            "--max-source-errors",
            str(self.args.max_source_errors),
        ]
        for directory in (self.work_dir, self.manifest_dir, self.latent_root):
            command.extend(["--exclude-dir", str(directory)])
        if self.args.rebuild_clips:
            command.append("--rebuild-clips")
        if self.args.max_files is not None:
            command.extend(["--max-files", str(self.args.max_files)])
        self.run_stage(
            "nonverbal",
            fingerprint=fingerprint,
            outputs=(self.work_dir / "nonverbal_events.jsonl",),
            action=lambda: self.run_command("nonverbal", command),
        )

    def _merge_transcription_shards(
        self, input_manifest: Path, shard_paths: Sequence[Path], output: Path
    ) -> None:
        """Reassemble sharded ASR outputs in the input manifest's row order."""
        order: list[str] = []
        with input_manifest.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    order.append(str(json.loads(line)["id"]))
        by_id: dict[str, dict[str, Any]] = {}
        for shard in shard_paths:
            with shard.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        by_id[str(row["id"])] = row
        missing = [row_id for row_id in order if row_id not in by_id]
        if missing:
            raise RuntimeError(
                f"transcription shards are missing {len(missing)} row(s), "
                f"first: {missing[:5]}"
            )
        temporary = output.with_name(f".{output.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row_id in order:
                handle.write(json.dumps(by_id[row_id], ensure_ascii=False) + "\n")
        os.replace(temporary, output)

    def _transcribe_parallel(
        self,
        label: str,
        *,
        input_manifest: Path,
        output_manifest: Path,
        audio_root: Path,
        cache_dir: Path,
        replace_text: bool,
    ) -> None:
        """Run one ASR process per GPU over row shards, then merge in order."""
        base = [
            sys.executable,
            "-m",
            "dataset.cli.prepare_transcribe",
            "--input-manifest",
            str(input_manifest),
            "--audio-root",
            str(audio_root),
            "--cache-dir",
            str(cache_dir),
            "--batch-size",
            str(self.args.transcribe_batch_size),
        ]
        if replace_text:
            base.append("--replace-text")
        gpus = self.gpu_indices
        worker_count = len(gpus) * max(1, int(self.args.asr_workers_per_gpu))
        if worker_count == 1:
            command = [*base, "--device", f"cuda:{gpus[0]}", "--output-manifest", str(output_manifest)]
            self.run_command(f"transcribe_{label}", command)
            return
        shard_paths: list[Path] = []
        jobs: list[tuple[str, list[str]]] = []
        for shard_index in range(worker_count):
            gpu = gpus[shard_index % len(gpus)]
            shard_path = output_manifest.with_name(f"{output_manifest.name}.shard{shard_index}")
            shard_paths.append(shard_path)
            jobs.append(
                (
                    f"transcribe_{label}_gpu{gpu}",
                    [
                        *base,
                        "--device",
                        f"cuda:{gpu}",
                        "--output-manifest",
                        str(shard_path),
                        "--shard-index",
                        str(shard_index),
                        "--shard-count",
                        str(worker_count),
                    ],
                )
            )
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [executor.submit(self.run_command, stage, cmd) for stage, cmd in jobs]
            for future in futures:
                future.result()
        self._merge_transcription_shards(input_manifest, shard_paths, output_manifest)
        for shard in shard_paths:
            shard.unlink(missing_ok=True)

    def transcribe(self) -> None:
        speech_source = self.work_dir / "all.jsonl"
        nonverbal_source = self.work_dir / "nonverbal_events.jsonl"
        combined = self.work_dir / "all_clips.jsonl"
        output = self.work_dir / "all_aw.jsonl"
        fingerprint = _hash(
            {
                "speech": _path_record(speech_source),
                "nonverbal": _path_record(nonverbal_source),
                "backend": "faster-whisper",
                "model": "TransWithAI/whisper-ja-1.5B-ct2",
                "replace_text": True,
            }
        )

        def action() -> None:
            # One unified clip stream: speech and nonverbal rows share the same
            # shape, ASR backend, and (later) LLM classification.
            temporary = combined.with_name(f".{combined.name}.tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                for source in (speech_source, nonverbal_source):
                    with source.open("r", encoding="utf-8-sig") as reader:
                        for line in reader:
                            if line.strip():
                                handle.write(line.rstrip("\n") + "\n")
            os.replace(temporary, combined)
            self._transcribe_parallel(
                "clips",
                input_manifest=combined,
                output_manifest=output,
                audio_root=self.root,
                cache_dir=self.work_dir / "asr_cache",
                replace_text=True,
            )

        self.run_stage(
            "transcribe",
            fingerprint=fingerprint,
            outputs=(combined, output),
            action=action,
        )

    def context_correction(self) -> None:
        # The whole clip stream (speech + nonverbal) is corrected and
        # classified as one timeline; train.jsonl falls out of row status.
        speech_all = self.work_dir / "all_aw.jsonl"
        fingerprint = _hash(
            {
                "speech": _path_record(speech_all),
                "agents": self.args.correction_agents,
                "grok_model": self.args.correction_grok_model,
                "batch": self.args.correction_batch_size,
                "context": self.args.correction_context_segments,
            }
        )
        command = [
            sys.executable,
            "-m",
            "dataset.cli.correct_transcripts",
            "--speech-all",
            str(speech_all),
            "--speech-output-dir",
            str(self.work_dir),
            "--cache-dir",
            str(self.work_dir / "text_correction"),
            "--agent-priority",
            self.args.correction_agents,
            "--grok-model",
            self.args.correction_grok_model,
            "--target-batch-size",
            str(self.args.correction_batch_size),
            "--context-segments",
            str(self.args.correction_context_segments),
            "--workers",
            str(self.args.correction_workers),
            "--timeout-seconds",
            str(self.args.correction_timeout_seconds),
        ]
        self.run_stage(
            "context_correction",
            fingerprint=fingerprint,
            outputs=(self.work_dir / "train.jsonl",),
            action=lambda: self.run_command("context_correction", command),
        )

    def _latent_one(self, label: str, input_manifest: Path) -> Path:
        input_hash = _sha256_file(input_manifest)
        # Key the cache on input content AND encode parameters so changed
        # codec settings never silently reuse stale latents.
        key = _hash({"input_sha256": input_hash, "encode": LATENT_ENCODE_PARAMS})[:16]
        output_dir = self.manifest_dir / ".full_pipeline"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{label}_{key}.jsonl"
        expected = _line_count(input_manifest)
        if output.is_file() and _line_count(output) == expected:
            return output
        partial = output.with_suffix(".partial.jsonl")
        partial.unlink(missing_ok=True)
        partial_skips = partial.with_name(f"{partial.name}.skipped.jsonl")
        partial_skips.unlink(missing_ok=True)
        latent_dir = self.latent_root / f"{label}_{key}"
        command = [
            sys.executable,
            "-m",
            "dataset.cli.prepare_manifest",
            "--dataset",
            "json",
            "--data-files",
            f"train={input_manifest.as_posix()}",
            "--audio-column",
            "audio",
            "--text-column",
            "text",
            "--no-text-normalize",
            "--speaker-column",
            "speaker_id",
            "--speaker-id-prefix",
            "full-data-pipeline",
            "--codec-repo",
            str(LATENT_ENCODE_PARAMS["codec_repo"]),
            "--normalize-db",
            str(LATENT_ENCODE_PARAMS["normalize_db"]),
            "--output-manifest",
            str(partial),
            "--latent-dir",
            str(latent_dir),
            "--device",
            self.primary_device,
            "--prefetch",
            "16",
            "--prefetch-workers",
            "8",
            "--flush-every",
            "100",
        ]
        if len(self.gpu_indices) > 1:
            if self.gpu_indices != tuple(range(len(self.gpu_indices))):
                raise RuntimeError(
                    "multi-GPU latent generation currently requires contiguous GPU indices from 0"
                )
            command.extend(["--num-gpus", str(len(self.gpu_indices)), "--merge-output"])
        self.run_command(f"latents_{label}", command)
        actual = _line_count(partial)
        if actual != expected:
            skipped_hint = (
                f"; skipped-row details: {partial_skips}" if partial_skips.is_file() else ""
            )
            raise RuntimeError(
                f"{label} latent conversion wrote {actual}/{expected} rows; "
                f"refusing publication{skipped_hint}"
            )
        os.replace(partial, output)
        if partial_skips.is_file():
            os.replace(partial_skips, output.with_name(f"{output.name}.skipped.jsonl"))
        return output

    def _latent_stage_outputs(self, selected: Path) -> tuple[Path, ...]:
        """The selection pointer plus the manifests it references (when known).

        Including the referenced manifests makes the cached-complete check fail
        when they were deleted, instead of publishing from a stale pointer.
        """
        outputs: list[Path] = [selected]
        if selected.is_file():
            try:
                payload = json.loads(selected.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return tuple(outputs)
            if isinstance(payload, dict):
                value = payload.get("clips")
                if isinstance(value, str) and value:
                    outputs.append(Path(value))
        return tuple(outputs)

    def latents(self) -> None:
        clips = self.work_dir / "train.jsonl"
        fingerprint = _hash(
            {
                "clips": _path_record(clips),
                "encode": LATENT_ENCODE_PARAMS,
            }
        )
        selected = self.manifest_dir / ".full_pipeline" / "selected_latents.json"

        def action() -> None:
            clips_output = self._latent_one("clips", clips)
            _atomic_write_json(selected, {"clips": clips_output.as_posix()})

        self.run_stage(
            "latents",
            fingerprint=fingerprint,
            outputs=self._latent_stage_outputs(selected),
            action=action,
        )

    def _prune_publish_backups(self) -> None:
        backups = sorted(self.manifest_dir.glob("train_before_full_*.jsonl"))
        for stale in backups[:-PUBLISH_BACKUP_KEEP]:
            stale.unlink(missing_ok=True)

    def publish(self) -> None:
        selected_path = self.manifest_dir / ".full_pipeline" / "selected_latents.json"
        if not selected_path.is_file():
            raise RuntimeError(
                f"latents output is missing: {selected_path}; "
                "run the latents stage first (for example with --start-at latents)"
            )
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        clips = Path(selected["clips"]).resolve()
        if not clips.is_file():
            raise RuntimeError(
                f"latent manifest referenced by {selected_path} is missing: "
                f"{clips}; re-run the latents stage (--force-stage latents)"
            )
        fingerprint = _hash({"clips": _path_record(clips)})
        final = self.manifest_dir / "train.jsonl"
        receipt = self.work_dir / "full_pipeline" / "last_success.json"

        def action() -> None:
            self.manifest_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.manifest_dir / ".train.next.jsonl"
            command = [
                sys.executable,
                "-m",
                "dataset.cli.merge_latent_manifests",
                "--input",
                str(clips),
                "--output",
                str(temporary),
                # Speaker identity is per RJ work (one work = one CV), while raw
                # rows carry per-source-file ids. Normalizing at publish keeps
                # every upstream stage cache valid; doing it at the source would
                # change intermediate manifest hashes and force a full latent
                # re-encode.
                "--normalize-speaker-rj",
            ]
            self.run_command("publish", command)
            expected = _line_count(clips)
            actual = _line_count(temporary)
            if actual != expected:
                raise RuntimeError(f"merged manifest has {actual}/{expected} rows")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = None
            if final.is_file():
                backup = self.manifest_dir / f"train_before_full_{stamp}.jsonl"
                shutil.copy2(final, backup)
                self._prune_publish_backups()
            index = final.with_name(f"{final.name}.irodori_index.pt")
            if index.is_file():
                index_backup = self.manifest_dir / f"{index.name}.before_full_{stamp}"
                replace_with_retry(index, index_backup)
            try:
                replace_with_retry(temporary, final)
            except PermissionError as exc:
                raise RuntimeError(
                    f"could not replace {final}; close any process holding it open "
                    f"(for example a running training job) and re-run publish: {exc}"
                ) from exc
            _atomic_write_json(
                receipt,
                {
                    "completed_at": _now(),
                    "manifest": final.as_posix(),
                    "rows": actual,
                    "latent_manifest": clips.as_posix(),
                    "previous_manifest_backup": backup.as_posix() if backup else None,
                    "fingerprint": fingerprint,
                },
            )

        self.run_stage(
            "publish",
            fingerprint=fingerprint,
            outputs=(final, receipt),
            action=action,
        )

    def run(self) -> None:
        start, stop = self._stage_range()
        self.preflight()
        if self.args.dry_run:
            print("dry-run complete; no pipeline stages were run")
            return
        for index, name in enumerate(STAGES):
            if index < start or index > stop:
                continue
            getattr(self, name)()
        if stop == len(STAGES) - 1:
            print(f"FULL PIPELINE COMPLETE manifest={self.manifest_dir / 'train.jsonl'}")
        else:
            print(f"pipeline stopped after requested stage={STAGES[stop]}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fully automate local VAD segmentation, nonverbal GPU inference, "
            "whisper-ja transcription, contextual LLM correction, DACVAE "
            "latents, and atomic publication. No cloud STT is involved."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("dataset/data"))
    parser.add_argument("--work-dir", type=Path, default=Path("dataset/data/pipeline"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("dataset/data/manifests"))
    parser.add_argument("--latent-root", type=Path, default=Path("dataset/data/latents/full_pipeline"))
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--min-source-seconds",
        type=float,
        default=300.0,
        help="Exclude source audio at or below this duration from every pipeline stage.",
    )
    parser.add_argument(
        "--speech-shards",
        type=int,
        default=24,
        help=(
            "Parallel prepare_speech worker processes (CPU-bound stage); the "
            "light VAD load is spread round-robin over the selected GPUs."
        ),
    )
    parser.add_argument(
        "--asr-workers-per-gpu",
        type=int,
        default=2,
        help=(
            "faster-whisper worker processes per GPU during transcription. "
            "Each worker batches clips through CTranslate2 (~5GB VRAM each)."
        ),
    )
    parser.add_argument(
        "--transcribe-batch-size",
        "--anime-batch-size",
        dest="transcribe_batch_size",
        type=int,
        default=16,
        help="Cross-clip ASR batch size for the transcribe stage (alias: --anime-batch-size).",
    )
    parser.add_argument("--audio-padding-seconds", type=float, default=0.35)
    parser.add_argument("--audio-padding-ms", type=int, default=350)
    parser.add_argument("--correction-agents", default="grok")
    parser.add_argument(
        "--correction-grok-model",
        default="grok-4.5",
        help="Grok model pinned for transcript correction/classification.",
    )
    parser.add_argument("--correction-batch-size", type=int, default=16)
    parser.add_argument("--correction-context-segments", type=int, default=10)
    # 256 workers overwhelmed the grok CLI (mass 300 s timeouts); 128 with a
    # longer timeout clears the queue without failed batches.
    parser.add_argument("--correction-workers", type=int, default=128)
    parser.add_argument("--correction-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--rebuild-clips",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Force FLAC clip re-encoding in the speech stage. Clips are "
            "content-addressed and written atomically, so the default reuses them."
        ),
    )
    parser.add_argument(
        "--max-source-errors",
        type=int,
        default=0,
        help=(
            "Fail the speech stage when dataset building records more than this many "
            "source errors (clip extraction/decode failures). Default: 0."
        ),
    )
    parser.add_argument("--force-stage", action="append", choices=STAGES)
    parser.add_argument("--start-at", choices=STAGES, default=None)
    parser.add_argument("--stop-after", choices=STAGES, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "transcribe_batch_size",
        "correction_batch_size",
        "correction_workers",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive")
    if args.max_source_errors < 0:
        raise ValueError("--max-source-errors must be non-negative")
    if args.min_source_seconds < 0:
        raise ValueError("--min-source-seconds must be non-negative")
    if args.audio_padding_seconds < 0 or args.audio_padding_ms < 0:
        raise ValueError("audio padding cannot be negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = _parse_args(argv)
    PipelineRunner(args).run()


if __name__ == "__main__":
    main()
