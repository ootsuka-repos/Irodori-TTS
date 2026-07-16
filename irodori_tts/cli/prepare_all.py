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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from irodori_tts.data_prep.grok_stt import (
    GrokSubscriptionAuth,
    discover_audio_sources,
    load_cached_response,
)

PIPELINE_SCHEMA_VERSION = 1
STAGES = (
    "speech",
    "nonverbal_features",
    "nonverbal_classification",
    "anime_whisper",
    "context_correction",
    "training_manifests",
    "latents",
    "publish",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


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
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {"path": resolved.as_posix(), "missing": True}
    stat = resolved.stat()
    record: dict[str, Any] = {
        "path": resolved.as_posix(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if resolved.is_file() and stat.st_size <= 128 * 1024 * 1024:
        record["sha256"] = _sha256_file(resolved)
    return record


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": PIPELINE_SCHEMA_VERSION, "stages": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": PIPELINE_SCHEMA_VERSION, "stages": {}}
    if not isinstance(payload, dict) or payload.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        return {"schema_version": PIPELINE_SCHEMA_VERSION, "stages": {}}
    if not isinstance(payload.get("stages"), dict):
        payload["stages"] = {}
    return payload


class PipelineRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path.cwd().resolve()
        self.data_root = args.data_root.expanduser().resolve()
        self.work_dir = args.work_dir.expanduser().resolve()
        self.nonverbal_dir = args.nonverbal_dir.expanduser().resolve()
        self.nonverbal_training_dir = args.nonverbal_training_dir.expanduser().resolve()
        self.manifest_dir = args.manifest_dir.expanduser().resolve()
        self.latent_root = args.latent_root.expanduser().resolve()
        self.state_path = self.work_dir / "full_pipeline" / "state.json"
        self.log_dir = self.work_dir / "full_pipeline" / "logs"
        self.state = _load_state(self.state_path)
        self.force_stages = set(args.force_stage or ())
        self.gpu_indices = self._gpu_indices()

    def _gpu_indices(self) -> tuple[int, ...]:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The full pipeline requires CUDA, but torch.cuda.is_available() is false"
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

    def selected_sources(self) -> list[Any]:
        sources = discover_audio_sources(self.data_root, output_dir=self.work_dir)
        sources = [source for source in sources if source.duration > self.args.min_source_seconds]
        if self.args.max_files is not None:
            sources = sources[: self.args.max_files]
        return sources

    def preflight(self) -> None:
        sources = self.selected_sources()
        if not sources:
            raise RuntimeError(f"No supported audio files found under {self.data_root}")
        cached = sum(1 for source in sources if load_cached_response(self.work_dir, source))
        missing = len(sources) - cached
        total_hours = sum(source.duration for source in sources) / 3600.0
        speech_will_run = self.args.start_at in (None, "speech")
        if missing and self.args.stt_auth_mode == "subscription":
            if speech_will_run:
                auth = GrokSubscriptionAuth(
                    auth_file=self.args.grok_auth_file,
                    cli_command=self.args.grok_cli,
                )
                metadata = auth.validate()
                print(
                    "preflight grok_subscription_login=valid "
                    f"expires_at={metadata.get('expires_at') or 'unspecified'}",
                    flush=True,
                )
            else:
                print(
                    "preflight warning: uncached speech exists, but the selected stage range "
                    "does not run speech STT",
                    flush=True,
                )
        elif missing and not os.environ.get(self.args.xai_api_key_env, "").strip():
            message = (
                f"{missing}/{len(sources)} selected sources need Grok STT, but "
                f"{self.args.xai_api_key_env} is not set for --stt-auth-mode api-key"
            )
            if speech_will_run and not self.args.dry_run:
                raise RuntimeError(message)
            print(f"preflight warning: {message}", flush=True)
        if shutil.which("codex") is None and shutil.which("claude") is None:
            raise RuntimeError(
                "Neither codex nor claude CLI is installed for transcript correction"
            )
        beats_code = self.work_dir / "_models" / "beats" / "code"
        beats_checkpoint = self.work_dir / "_models" / "beats" / "BEATs_iter3_plus_AS2M.pt"
        if not beats_code.is_dir() or not beats_checkpoint.is_file():
            raise RuntimeError(
                f"Pinned BEATs files are missing: code={beats_code}, checkpoint={beats_checkpoint}"
            )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        disk = shutil.disk_usage(self.data_root)
        print(
            f"preflight sources={len(sources)} cached_stt={cached} missing_stt={missing} "
            f"duration={total_hours:.3f}h gpus={self.gpu_indices} "
            f"source_duration=>{self.args.min_source_seconds:.3f}s "
            f"free_disk={disk.free / 1024**3:.1f}GiB",
            flush=True,
        )

    def speech(self) -> None:
        sources = self.selected_sources()
        fingerprint = _hash(
            {
                "sources": [source.metadata() for source in sources],
                "vad": {
                    "devices": [f"cuda:{index}" for index in self.gpu_indices],
                    "speech_pad_ms": self.args.audio_padding_ms,
                },
                "clip_padding": self.args.audio_padding_seconds,
                "stt_auth_mode": self.args.stt_auth_mode,
                "min_source_seconds": self.args.min_source_seconds,
            }
        )
        command = [
            sys.executable,
            "-m",
            "irodori_tts.cli.prepare_grok_stt",
            "--input-dir",
            str(self.data_root),
            "--output-dir",
            str(self.work_dir),
            "--min-source-seconds",
            str(self.args.min_source_seconds),
            "--auth-mode",
            self.args.stt_auth_mode,
            "--api-key-env",
            self.args.xai_api_key_env,
            "--grok-cli",
            self.args.grok_cli,
            "--vad-devices",
            ",".join(f"cuda:{index}" for index in self.gpu_indices),
            "--vad-speech-pad-ms",
            str(self.args.audio_padding_ms),
            "--padding-seconds",
            str(self.args.audio_padding_seconds),
            "--workers",
            str(self.args.stt_workers),
            "--vad-prefetch-sources",
            str(self.args.vad_prefetch_sources),
            "--rebuild-clips",
        ]
        if self.args.grok_auth_file is not None:
            command.extend(["--grok-auth-file", str(self.args.grok_auth_file)])
        if self.args.max_files is not None:
            command.extend(["--max-files", str(self.args.max_files)])
        self.run_stage(
            "speech",
            fingerprint=fingerprint,
            outputs=(self.work_dir / "all.jsonl", self.work_dir / "train.jsonl"),
            action=lambda: self.run_command("speech", command),
        )

    def nonverbal_features(self) -> None:
        raw_paths = sorted((self.work_dir / "raw_responses").glob("*.json"))
        fingerprint = _hash(
            {
                "raw": [_path_record(path) for path in raw_paths],
                "device": self.primary_device,
                "segmentation": "acoustic",
                "min_source_seconds": self.args.min_source_seconds,
            }
        )
        command = [
            sys.executable,
            "-m",
            "irodori_tts.cli.cluster_nonverbal",
            "--data-root",
            str(self.data_root),
            "--raw-response-dir",
            str(self.work_dir / "raw_responses"),
            "--output-dir",
            str(self.nonverbal_dir),
            "--beats-code-dir",
            str(self.work_dir / "_models" / "beats" / "code"),
            "--beats-checkpoint",
            str(self.work_dir / "_models" / "beats" / "BEATs_iter3_plus_AS2M.pt"),
            "--device",
            self.primary_device,
            "--batch-size",
            str(self.args.beats_batch_size),
            "--min-source-seconds",
            str(self.args.min_source_seconds),
            "--segmentation-mode",
            "acoustic",
        ]
        if self.args.max_files is not None:
            command.extend(["--max-sources", str(self.args.max_files)])
        self.run_stage(
            "nonverbal_features",
            fingerprint=fingerprint,
            outputs=(self.nonverbal_dir / "candidates.jsonl", self.nonverbal_dir / "embeddings"),
            action=lambda: self.run_command("nonverbal_features", command),
        )

    def nonverbal_classification(self) -> None:
        candidates = self.nonverbal_dir / "candidates.jsonl"
        fingerprint = _hash(
            {
                "candidates": _path_record(candidates),
                "padding": self.args.audio_padding_seconds,
                "classifier_batch": self.args.classifier_batch_size,
            }
        )
        command = [
            sys.executable,
            "-m",
            "irodori_tts.cli.prepare_nonverbal_events",
            "--feature-dir",
            str(self.nonverbal_dir),
            "--output-dir",
            str(self.nonverbal_dir),
            "--data-root",
            str(self.data_root),
            "--device",
            self.primary_device,
            "--hf-cache-dir",
            str(self.work_dir / "_models" / "hf"),
            "--classifier-batch-size",
            str(self.args.classifier_batch_size),
            "--final-clip-padding-seconds",
            str(self.args.audio_padding_seconds),
        ]
        if self.args.allow_model_download:
            command.append("--allow-model-download")
        events = self.nonverbal_dir / "dataset" / "manifests" / "events.jsonl"
        self.run_stage(
            "nonverbal_classification",
            fingerprint=fingerprint,
            outputs=(events, self.nonverbal_dir / "dataset" / "summary.json"),
            action=lambda: self.run_command("nonverbal_classification", command),
        )

    def anime_whisper(self) -> None:
        dataset = self.nonverbal_dir / "dataset"
        source = dataset / "manifests" / "events.jsonl"
        output = dataset / "manifests" / "events_transcribed.jsonl"
        fingerprint = _hash(
            {
                "events": _path_record(source),
                "model": "litagin/anime-whisper@22e2008a8182b357da3922a6308d095008f72973",
                "device": self.primary_device,
            }
        )
        command = [
            sys.executable,
            "-m",
            "irodori_tts.cli.prepare_anime_whisper",
            "--input-manifest",
            str(source),
            "--output-manifest",
            str(output),
            "--audio-root",
            str(dataset),
            "--cache-dir",
            str(self.nonverbal_dir / "anime_whisper_cache"),
            "--device",
            self.primary_device,
            "--batch-size",
            str(self.args.anime_batch_size),
        ]
        self.run_stage(
            "anime_whisper",
            fingerprint=fingerprint,
            outputs=(output,),
            action=lambda: self.run_command("anime_whisper", command),
        )

    def context_correction(self) -> None:
        dataset = self.nonverbal_dir / "dataset"
        nonverbal_input = dataset / "manifests" / "events_transcribed.jsonl"
        nonverbal_output = dataset / "manifests" / "events_corrected.jsonl"
        speech_all = self.work_dir / "all.jsonl"
        fingerprint = _hash(
            {
                "speech": _path_record(speech_all),
                "nonverbal": _path_record(nonverbal_input),
                "agents": self.args.correction_agents,
                "batch": self.args.correction_batch_size,
                "context": self.args.correction_context_segments,
            }
        )
        command = [
            sys.executable,
            "-m",
            "irodori_tts.cli.correct_transcripts",
            "--speech-all",
            str(speech_all),
            "--speech-output-dir",
            str(self.work_dir),
            "--nonverbal-input",
            str(nonverbal_input),
            "--nonverbal-output",
            str(nonverbal_output),
            "--cache-dir",
            str(self.work_dir / "text_correction"),
            "--agent-priority",
            self.args.correction_agents,
            "--target-batch-size",
            str(self.args.correction_batch_size),
            "--context-segments",
            str(self.args.correction_context_segments),
            "--workers",
            str(self.args.correction_workers),
        ]
        self.run_stage(
            "context_correction",
            fingerprint=fingerprint,
            outputs=(nonverbal_output, self.work_dir / "train.jsonl"),
            action=lambda: self.run_command("context_correction", command),
        )

    def training_manifests(self) -> None:
        events = self.nonverbal_dir / "dataset" / "manifests" / "events_corrected.jsonl"
        speech = self.work_dir / "train.jsonl"
        fingerprint = _hash(
            {"events": _path_record(events), "speech": _path_record(speech), "schema": 3}
        )
        command = [
            sys.executable,
            "-m",
            "irodori_tts.cli.prepare_nonverbal_training",
            "--events",
            str(events),
            "--output-dir",
            str(self.nonverbal_training_dir),
            "--speech-manifest",
            str(speech),
            "--project-root",
            str(self.root),
            "--audio-base-dir",
            str(self.nonverbal_dir / "dataset"),
        ]
        self.run_stage(
            "training_manifests",
            fingerprint=fingerprint,
            outputs=(
                self.nonverbal_training_dir / "train_nonverbal.jsonl",
                self.nonverbal_training_dir / "train_combined.jsonl",
            ),
            action=lambda: self.run_command("training_manifests", command),
        )

    def _latent_one(self, label: str, input_manifest: Path) -> Path:
        input_hash = _sha256_file(input_manifest)
        key = input_hash[:16]
        output_dir = self.manifest_dir / ".full_pipeline"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{label}_{key}.jsonl"
        expected = _line_count(input_manifest)
        if output.is_file() and _line_count(output) == expected:
            return output
        partial = output.with_suffix(".partial.jsonl")
        partial.unlink(missing_ok=True)
        latent_dir = self.latent_root / f"{label}_{key}"
        command = [
            sys.executable,
            "-m",
            "irodori_tts.cli.prepare_manifest",
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
            "--output-manifest",
            str(partial),
            "--latent-dir",
            str(latent_dir),
            "--device",
            self.primary_device,
            "--prefetch",
            "8",
            "--prefetch-workers",
            "4",
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
            raise RuntimeError(
                f"{label} latent conversion wrote {actual}/{expected} rows; refusing publication"
            )
        os.replace(partial, output)
        return output

    def latents(self) -> None:
        speech = self.work_dir / "train.jsonl"
        nonverbal = self.nonverbal_training_dir / "train_nonverbal.jsonl"
        fingerprint = _hash(
            {
                "speech": _path_record(speech),
                "nonverbal": _path_record(nonverbal),
                "gpus": self.gpu_indices,
                "codec": "Aratako/Semantic-DACVAE-Japanese-32dim",
            }
        )
        outputs_holder: list[Path] = []

        def action() -> None:
            outputs_holder.extend(
                [self._latent_one("speech", speech), self._latent_one("nonverbal", nonverbal)]
            )
            _atomic_write_json(
                self.manifest_dir / ".full_pipeline" / "selected_latents.json",
                {"speech": outputs_holder[0].as_posix(), "nonverbal": outputs_holder[1].as_posix()},
            )

        selected = self.manifest_dir / ".full_pipeline" / "selected_latents.json"
        self.run_stage(
            "latents",
            fingerprint=fingerprint,
            outputs=(selected,),
            action=action,
        )

    def publish(self) -> None:
        selected_path = self.manifest_dir / ".full_pipeline" / "selected_latents.json"
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        speech = Path(selected["speech"]).resolve()
        nonverbal = Path(selected["nonverbal"]).resolve()
        fingerprint = _hash({"speech": _path_record(speech), "nonverbal": _path_record(nonverbal)})
        final = self.manifest_dir / "train.jsonl"
        receipt = self.work_dir / "full_pipeline" / "last_success.json"

        def action() -> None:
            self.manifest_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.manifest_dir / ".train.next.jsonl"
            command = [
                sys.executable,
                "-m",
                "irodori_tts.cli.merge_latent_manifests",
                "--input",
                str(speech),
                "--input",
                str(nonverbal),
                "--output",
                str(temporary),
            ]
            self.run_command("publish", command)
            expected = _line_count(speech) + _line_count(nonverbal)
            actual = _line_count(temporary)
            if actual != expected:
                raise RuntimeError(f"merged manifest has {actual}/{expected} rows")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = None
            if final.is_file():
                backup = self.manifest_dir / f"train_before_full_{stamp}.jsonl"
                shutil.copy2(final, backup)
            index = final.with_name(f"{final.name}.irodori_index.pt")
            if index.is_file():
                index_backup = self.manifest_dir / f"{index.name}.before_full_{stamp}"
                os.replace(index, index_backup)
            os.replace(temporary, final)
            _atomic_write_json(
                receipt,
                {
                    "completed_at": _now(),
                    "manifest": final.as_posix(),
                    "rows": actual,
                    "speech_rows": _line_count(speech),
                    "nonverbal_rows": _line_count(nonverbal),
                    "speech_latent_manifest": speech.as_posix(),
                    "nonverbal_latent_manifest": nonverbal.as_posix(),
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
        self.preflight()
        if self.args.dry_run:
            print("dry-run complete; no pipeline stages were run")
            return
        start = STAGES.index(self.args.start_at) if self.args.start_at else 0
        stop = STAGES.index(self.args.stop_after) if self.args.stop_after else len(STAGES) - 1
        if start > stop:
            raise ValueError("--start-at must not come after --stop-after")
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
            "Fully automate Grok STT, local VAD/nonverbal GPU inference, anime-whisper, "
            "contextual Codex/Claude correction, DACVAE latents, and atomic publication."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/grok_stt"))
    parser.add_argument(
        "--nonverbal-dir", type=Path, default=Path("data/grok_stt/nonverbal_events_new")
    )
    parser.add_argument(
        "--nonverbal-training-dir",
        type=Path,
        default=Path("data/grok_stt/nonverbal_training_all"),
    )
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--latent-root", type=Path, default=Path("data/latents/full_pipeline"))
    parser.add_argument(
        "--stt-auth-mode",
        choices=("subscription", "api-key"),
        default="subscription",
    )
    parser.add_argument("--xai-api-key-env", default="XAI_API_KEY")
    parser.add_argument("--grok-cli", default="grok")
    parser.add_argument("--grok-auth-file", type=Path, default=None)
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--min-source-seconds",
        type=float,
        default=300.0,
        help="Exclude source audio at or below this duration from every pipeline stage.",
    )
    parser.add_argument("--stt-workers", type=int, default=4)
    parser.add_argument("--vad-prefetch-sources", type=int, default=8)
    parser.add_argument("--beats-batch-size", type=int, default=32)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--anime-batch-size", type=int, default=16)
    parser.add_argument("--audio-padding-seconds", type=float, default=0.35)
    parser.add_argument("--audio-padding-ms", type=int, default=350)
    parser.add_argument("--correction-agents", default="codex,claude,grok")
    parser.add_argument("--correction-batch-size", type=int, default=16)
    parser.add_argument("--correction-context-segments", type=int, default=10)
    parser.add_argument("--correction-workers", type=int, default=12)
    parser.add_argument(
        "--allow-model-download", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--force-stage", action="append", choices=STAGES)
    parser.add_argument("--start-at", choices=STAGES, default=None)
    parser.add_argument("--stop-after", choices=STAGES, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "stt_workers",
        "vad_prefetch_sources",
        "beats_batch_size",
        "classifier_batch_size",
        "anime_batch_size",
        "correction_batch_size",
        "correction_workers",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive")
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
