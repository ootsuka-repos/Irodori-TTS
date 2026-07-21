"""Restartable local CUDA transcription for VAD-excluded audio events."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import soundfile as sf
import torch

from dataset._io_utils import atomic_write_json
from dataset._textnorm import meaningful_text, normalize_transcript

ANIME_WHISPER_MODEL = "litagin/anime-whisper"
ANIME_WHISPER_REVISION = "22e2008a8182b357da3922a6308d095008f72973"
FASTER_WHISPER_MODEL = "TransWithAI/whisper-ja-1.5B-ct2"
LOCAL_ASR_CACHE_SCHEMA_VERSION = 1

# Transcript-density plausibility gate shared by every ASR stream.
MIN_CHARS_PER_SECOND = 0.20
MAX_CHARS_PER_SECOND = 20.0


class BatchASR(Protocol):
    """Minimal interface used by the cache-aware manifest transcriber."""

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def transcribe(self, audio_paths: Sequence[Path]) -> list[str]: ...


@dataclass(frozen=True)
class AnimeWhisperConfig:
    """Pinned local inference settings for ``litagin/anime-whisper``."""

    model: str = ANIME_WHISPER_MODEL
    revision: str = ANIME_WHISPER_REVISION
    device: str = "cuda:0"
    dtype: str = "float16"
    batch_size: int = 16
    chunk_length_seconds: float = 30.0
    language: str = "Japanese"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not math.isfinite(self.chunk_length_seconds) or self.chunk_length_seconds <= 0:
            raise ValueError("chunk_length_seconds must be positive and finite")
        if self.dtype not in {"float16", "bfloat16"}:
            raise ValueError("dtype must be float16 or bfloat16")


class AnimeWhisperTranscriber:
    """Lazy Transformers pipeline that is required to execute on CUDA."""

    def __init__(self, config: AnimeWhisperConfig | None = None) -> None:
        self.config = config or AnimeWhisperConfig()
        self._pipeline: Any | None = None

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "provider": "local-transformers",
            "model": self.config.model,
            "revision": self.config.revision,
            "device": self.config.device,
            "dtype": self.config.dtype,
            "chunk_length_seconds": self.config.chunk_length_seconds,
            "language": self.config.language,
            "audio_uploaded": False,
        }

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        device = torch.device(self.config.device)
        if device.type != "cuda":
            raise RuntimeError(
                "anime-whisper is configured for local GPU inference; "
                f"a CUDA device is required, got {self.config.device!r}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for anime-whisper but is unavailable")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {index} does not exist; available={torch.cuda.device_count()}"
            )
        dtype = getattr(torch, self.config.dtype)
        from transformers import pipeline

        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.config.model,
            revision=self.config.revision,
            device=index,
            dtype=dtype,
            chunk_length_s=self.config.chunk_length_seconds,
            batch_size=self.config.batch_size,
        )
        model_device = torch.device(self._pipeline.model.device)
        if model_device.type != "cuda":
            raise RuntimeError(f"anime-whisper loaded on {model_device}, not CUDA")
        return self._pipeline

    def transcribe(self, audio_paths: Sequence[Path]) -> list[str]:
        if not audio_paths:
            return []
        pipe = self._load()
        outputs = pipe(
            [str(path) for path in audio_paths],
            batch_size=min(self.config.batch_size, len(audio_paths)),
            generate_kwargs={
                "language": self.config.language,
                "no_repeat_ngram_size": 0,
                "repetition_penalty": 1.0,
            },
        )
        if isinstance(outputs, Mapping):
            outputs = [outputs]
        values = list(outputs)
        if len(values) != len(audio_paths):
            raise RuntimeError(
                f"anime-whisper returned {len(values)} results for {len(audio_paths)} files"
            )
        result: list[str] = []
        for output in values:
            if not isinstance(output, Mapping):
                raise RuntimeError("anime-whisper returned a non-object result")
            result.append(normalize_transcript(str(output.get("text", ""))))
        return result


@dataclass(frozen=True)
class FasterWhisperConfig:
    """CTranslate2 inference settings for the whisper-ja-1.5B conversion.

    ``repetition_penalty`` plus the compression-ratio fallback keeps the known
    repetition loops of the base finetune in check without banning the
    legitimate onomatopoeia repeats this corpus is full of.
    """

    model: str = FASTER_WHISPER_MODEL
    device: str = "cuda"
    device_index: int = 0
    compute_type: str = "float16"
    language: str = "ja"
    beam_size: int = 5
    repetition_penalty: float = 1.1
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = False
    # Cross-clip batching: clips are at most ~20.5 s, i.e. exactly one Whisper
    # 30 s window each, so several clips can share one batched generate() call.
    batch_size: int = 8

    def __post_init__(self) -> None:
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


class FasterWhisperTranscriber:
    """Lazy faster-whisper (CTranslate2) model required to execute on CUDA."""

    def __init__(self, config: FasterWhisperConfig | None = None) -> None:
        self.config = config or FasterWhisperConfig()
        self._model: Any | None = None

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "provider": "local-faster-whisper",
            "model": self.config.model,
            "device": self.config.device,
            "compute_type": self.config.compute_type,
            "language": self.config.language,
            "beam_size": self.config.beam_size,
            "repetition_penalty": self.config.repetition_penalty,
            "condition_on_previous_text": self.config.condition_on_previous_text,
            "batch_size": self.config.batch_size,
            "audio_uploaded": False,
        }

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for faster-whisper but is unavailable")
        # ctranslate2 resolves cuDNN/cuBLAS DLLs from PATH; torch ships them.
        import os

        torch_lib = str(Path(torch.__file__).parent / "lib")
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.config.model,
            device=self.config.device,
            device_index=self.config.device_index,
            compute_type=self.config.compute_type,
        )
        return self._model

    def transcribe(self, audio_paths: Sequence[Path]) -> list[str]:
        if not audio_paths:
            return []
        if self.config.batch_size > 1:
            return self._transcribe_batched(audio_paths)
        model = self._load()
        results: list[str] = []
        for path in audio_paths:
            segments, _info = model.transcribe(
                str(path),
                language=self.config.language,
                beam_size=self.config.beam_size,
                repetition_penalty=self.config.repetition_penalty,
                compression_ratio_threshold=self.config.compression_ratio_threshold,
                log_prob_threshold=self.config.log_prob_threshold,
                no_speech_threshold=self.config.no_speech_threshold,
                condition_on_previous_text=self.config.condition_on_previous_text,
                vad_filter=False,
            )
            text = "".join(segment.text for segment in segments)
            results.append(normalize_transcript(text))
        return results

    def _transcribe_batched(self, audio_paths: Sequence[Path]) -> list[str]:
        """True cross-clip batching through the CTranslate2 Whisper API.

        Every training clip fits one 30 s Whisper window, so each clip is one
        batch element; the per-file fallback loops (temperature/compression
        retries) are traded for throughput — repetition_penalty plus beam
        search handles the repetition failure mode this corpus actually has.
        """
        import numpy as np
        from faster_whisper.audio import decode_audio
        from faster_whisper.tokenizer import Tokenizer

        model = self._load()
        extractor = model.feature_extractor
        tokenizer = Tokenizer(
            model.hf_tokenizer,
            True,
            task="transcribe",
            language=self.config.language,
        )
        prompt = model.get_prompt(tokenizer, [], without_timestamps=True)

        results: list[str] = []
        batch_size = self.config.batch_size
        for offset in range(0, len(audio_paths), batch_size):
            batch = audio_paths[offset : offset + batch_size]
            features = []
            for path in batch:
                audio = decode_audio(str(path), sampling_rate=extractor.sampling_rate)
                if audio.shape[0] < extractor.n_samples:
                    audio = np.pad(audio, (0, extractor.n_samples - audio.shape[0]))
                else:
                    audio = audio[: extractor.n_samples]
                features.append(extractor(audio, padding=0)[:, : extractor.nb_max_frames])
            encoder_output = model.encode(np.stack(features).astype(np.single))
            generation = model.model.generate(
                encoder_output,
                [prompt] * len(batch),
                beam_size=self.config.beam_size,
                repetition_penalty=self.config.repetition_penalty,
                no_repeat_ngram_size=0,
                max_length=448,
                return_scores=False,
                suppress_blank=True,
            )
            for result in generation:
                text = tokenizer.decode(result.sequences_ids[0])
                results.append(normalize_transcript(text))
        return results


def transcript_review_reasons(
    text: str,
    *,
    duration_seconds: float | None = None,
) -> list[str]:
    """Flag empty, repetitive, or duration-implausible local-ASR output."""
    meaningful = meaningful_text(text)
    reasons: list[str] = []
    if not meaningful:
        reasons.append("empty_transcript")
    compact = re.sub(r"\s+", "", text)
    if re.search(r"(.{1,8})\1{5,}", compact):
        reasons.append("repeated_transcript")
    if len(meaningful) > 400:
        reasons.append("implausibly_long_transcript")
    if duration_seconds is not None and duration_seconds > 0:
        characters_per_second = len(meaningful) / duration_seconds
        if meaningful and characters_per_second < MIN_CHARS_PER_SECOND:
            reasons.append("text_too_sparse")
        if characters_per_second > MAX_CHARS_PER_SECOND:
            reasons.append("text_too_dense")
    return reasons


def _cache_path(cache_dir: Path, row_id: str) -> Path:
    safe = re.sub(r"[^\w.-]+", "_", unicodedata.normalize("NFKC", row_id)).strip("_.")
    if not safe:
        safe = "event"
    digest = hashlib.sha1(row_id.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{safe[:80]}_{digest}.json"


def _audio_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.resolve().as_posix(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_cached(
    path: Path,
    *,
    row_id: str,
    audio: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    if (
        metadata.get("schema_version") != LOCAL_ASR_CACHE_SCHEMA_VERSION
        or metadata.get("row_id") != row_id
        or metadata.get("audio") != dict(audio)
        or metadata.get("model") != dict(model)
    ):
        return None
    result = payload.get("result")
    if not isinstance(result, Mapping):
        return None
    # Cached errors are diagnostics, not results: transient failures (OOM,
    # I/O hiccups) must be retried on the next run instead of permanently
    # pinning empty text to the row.
    if str(result.get("status", "")) == "error":
        return None
    return dict(result)


def _resolve_audio(row: Mapping[str, Any], audio_root: Path) -> Path:
    raw = str(row.get("audio", "") or "").strip()
    if not raw:
        raise ValueError("row is missing audio")
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (audio_root / path).resolve()


def transcribe_manifest_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    audio_root: str | Path,
    cache_dir: str | Path,
    transcriber: BatchASR,
    batch_size: int,
    force: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach cached raw transcripts to rows while isolating individual failures."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    root = Path(audio_root).expanduser().resolve()
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    model_metadata = dict(transcriber.metadata)
    output = [dict(row) for row in rows]
    seen: set[str] = set()
    cached_results: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, str, Path, dict[str, Any], Path]] = []
    cache_hits = 0
    audio_errors = 0

    for index, row in enumerate(output):
        row_id = str(row.get("id", "") or "").strip()
        if not row_id:
            raise ValueError(f"manifest row {index} has no non-empty id")
        if row_id in seen:
            raise ValueError(f"duplicate manifest id: {row_id}")
        seen.add(row_id)
        try:
            audio_path = _resolve_audio(row, root)
            audio_metadata = _audio_fingerprint(audio_path)
        except (OSError, ValueError) as exc:
            audio_errors += 1
            cached_results[index] = {
                "text": "",
                "status": "error",
                "review_reasons": ["audio_unavailable"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        path = _cache_path(cache_root, row_id)
        cached = None
        if not force:
            cached = _load_cached(
                path,
                row_id=row_id,
                audio=audio_metadata,
                model=model_metadata,
            )
        if cached is not None:
            cache_hits += 1
            cached_results[index] = cached
        else:
            pending.append((index, row_id, audio_path, audio_metadata, path))

    completed = 0

    def process(entries: list[tuple[int, str, Path, dict[str, Any], Path]]) -> None:
        nonlocal completed
        if not entries:
            return
        try:
            texts = transcriber.transcribe([entry[2] for entry in entries])
            if len(texts) != len(entries):
                raise RuntimeError("local ASR result count mismatch")
        except Exception as exc:
            if len(entries) > 1:
                midpoint = len(entries) // 2
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                process(entries[:midpoint])
                process(entries[midpoint:])
                return
            index, row_id, _audio_path, audio_metadata, path = entries[0]
            result = {
                "text": "",
                "status": "error",
                "review_reasons": ["transcription_error"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            cached_results[index] = result
            # Persisted for observability only; _load_cached skips error
            # results so the row is retried on the next run.
            atomic_write_json(
                path,
                {
                    "metadata": {
                        "schema_version": LOCAL_ASR_CACHE_SCHEMA_VERSION,
                        "row_id": row_id,
                        "audio": audio_metadata,
                        "model": model_metadata,
                        "transcribed_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "result": result,
                },
            )
            completed += 1
            return

        for entry, text in zip(entries, texts, strict=True):
            index, row_id, audio_path, audio_metadata, path = entry
            normalized = normalize_transcript(text)
            try:
                duration_seconds = float(sf.info(audio_path).duration)
            except (OSError, RuntimeError):
                duration_seconds = None
            reasons = transcript_review_reasons(normalized, duration_seconds=duration_seconds)
            result = {
                "text": normalized,
                "status": "review" if reasons else "ok",
                "review_reasons": reasons,
                "error": None,
            }
            cached_results[index] = result
            atomic_write_json(
                path,
                {
                    "metadata": {
                        "schema_version": LOCAL_ASR_CACHE_SCHEMA_VERSION,
                        "row_id": row_id,
                        "audio": audio_metadata,
                        "model": model_metadata,
                        "transcribed_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "result": result,
                },
            )
            completed += 1
        print(
            f"transcribe rows={completed}/{len(pending)} "
            f"cached={cache_hits} audio_errors={audio_errors}",
            flush=True,
        )

    for offset in range(0, len(pending), batch_size):
        process(pending[offset : offset + batch_size])

    for index, row in enumerate(output):
        result = cached_results[index]
        row["transcript_text_raw"] = str(result.get("text", ""))
        row["transcript_text"] = str(result.get("text", ""))
        row["transcript_backend"] = str(model_metadata.get("provider", "local-asr"))
        row["transcript_model"] = model_metadata.get("model")
        row["transcript_model_revision"] = model_metadata.get("revision")
        row["transcript_status"] = result.get("status", "error")
        row["transcript_review_reasons"] = list(result.get("review_reasons", []))
        if result.get("error"):
            row["transcript_error"] = str(result["error"])

    summary = {
        "rows": len(output),
        "cached": cache_hits,
        "audio_errors": audio_errors,
        "transcribed": len(pending),
        "ok": sum(1 for row in output if row["transcript_status"] == "ok"),
        "review": sum(1 for row in output if row["transcript_status"] == "review"),
        "errors": sum(1 for row in output if row["transcript_status"] == "error"),
        "model": model_metadata,
    }
    return output, summary
