"""Grok Speech-to-Text helpers for building timestamp-aligned TTS datasets."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import mimetypes
import os
import re
import subprocess
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from urllib.parse import urlencode

import aiohttp
import numpy as np
import requests
import soundfile as sf
import torch
import torchaudio

XAI_STT_ENDPOINT = "https://api.x.ai/v1/stt"
XAI_STT_WEBSOCKET_ENDPOINT = "wss://api.x.ai/v1/stt"
XAI_BATCH_PRICE_USD_PER_HOUR = 0.10
XAI_MAX_FILE_BYTES = 500_000_000
STT_CACHE_STRATEGY = "silero-vad-v1"
SILERO_VAD_REPO = "snakers4/silero-vad:v6.2.1"

AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mka",
    ".mkv",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
}

_MIME_TYPES = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mka": "audio/x-matroska",
    ".mkv": "video/x-matroska",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
}

_SENTENCE_END = frozenset("。！？!?…")
_OPEN_PUNCTUATION = frozenset("([{（［｛〈《「『【〔〘〚“‘")
_CLOSE_PUNCTUATION = frozenset(")]}）］｝〉》」』】〕〙〛、。，．：；！？!?…％%”’")


@dataclass(frozen=True)
class AudioSource:
    """A local source audio file and the metadata needed for safe resume."""

    path: Path
    relative_path: str
    source_id: str
    speaker_id: str
    size_bytes: int
    mtime_ns: int
    duration: float
    sample_rate: int
    channels: int
    frames: int

    def metadata(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "source_id": self.source_id,
            "speaker_id": self.speaker_id,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "duration": round(self.duration, 6),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
        }


@dataclass(frozen=True)
class Word:
    """One timestamped word returned by the STT endpoint."""

    text: str
    start: float
    end: float
    speaker: str | int | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class Segment:
    """A contiguous range of timestamped words."""

    word_start: int
    word_end: int
    start: float
    end: float
    text: str
    speaker: str | int | None = None
    confidence_mean: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class SegmentationConfig:
    """Rules used to turn word timestamps into training clips."""

    min_seconds: float = 1.0
    target_seconds: float = 15.0
    max_seconds: float = 29.5
    # ASMR dialogue often contains deliberate pauses inside one sentence.
    # A one-second boundary produced fragments such as "じゃ" + "あ…".
    hard_gap_seconds: float = 2.0
    soft_gap_seconds: float = 0.20
    # Preserve breath attacks and low-energy decay that word timestamps and VAD
    # commonly underestimate.
    padding_seconds: float = 0.35
    min_chars_per_second: float = 0.20
    max_chars_per_second: float = 20.0

    def validate(self) -> None:
        if self.min_seconds <= 0:
            raise ValueError("min_seconds must be greater than zero")
        if not self.min_seconds <= self.target_seconds <= self.max_seconds:
            raise ValueError("Expected min_seconds <= target_seconds <= max_seconds")
        if self.hard_gap_seconds <= 0 or self.soft_gap_seconds < 0:
            raise ValueError("Gap thresholds must be non-negative")
        if self.padding_seconds < 0:
            raise ValueError("padding_seconds must be non-negative")


@dataclass(frozen=True)
class TranscriptionOptions:
    """Supported options for xAI's dedicated REST STT endpoint."""

    language: str = "ja"
    format_text: bool = False
    filler_words: bool = True
    diarize: bool = False
    multichannel: bool = False
    keyterms: tuple[str, ...] = ()

    def request_fields(self) -> list[tuple[str, str]]:
        fields = [
            ("language", self.language),
            ("format", str(self.format_text).lower()),
            ("filler_words", str(self.filler_words).lower()),
            ("diarize", str(self.diarize).lower()),
            ("multichannel", str(self.multichannel).lower()),
        ]
        fields.extend(("keyterm", term) for term in self.keyterms)
        return fields

    def public_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["keyterms"] = list(self.keyterms)
        return payload


@dataclass(frozen=True)
class ChunkingConfig:
    """Upload chunking used to keep long-form STT responses complete."""

    seconds: float = 60.0
    overlap_seconds: float = 1.0

    def validate(self) -> None:
        if self.seconds <= 0:
            raise ValueError("chunk seconds must be greater than zero")
        if self.overlap_seconds < 0:
            raise ValueError("chunk overlap must be non-negative")
        if self.overlap_seconds >= self.seconds / 2:
            raise ValueError("chunk overlap must be less than half the chunk duration")


@dataclass(frozen=True)
class SileroVADConfig:
    """ASMR-oriented Silero VAD settings."""

    repo: str = SILERO_VAD_REPO
    threshold: float = 0.35
    neg_threshold: float | None = 0.20
    min_speech_duration_ms: int = 180
    min_silence_duration_ms: int = 450
    speech_pad_ms: int = 120
    max_speech_duration_s: float = 29.0
    max_join_gap_s: float = 1.2
    max_upload_duration_s: float = 29.5

    def validate(self) -> None:
        if not 0 < self.threshold < 1:
            raise ValueError("VAD threshold must be between zero and one")
        if self.neg_threshold is not None and not 0 <= self.neg_threshold < self.threshold:
            raise ValueError("VAD neg_threshold must be below threshold")
        if self.min_speech_duration_ms <= 0 or self.min_silence_duration_ms < 0:
            raise ValueError("VAD duration settings are invalid")
        if self.speech_pad_ms < 0 or self.max_speech_duration_s <= 0:
            raise ValueError("VAD padding/max speech settings are invalid")
        if self.max_join_gap_s < 0:
            raise ValueError("VAD max join gap must be non-negative")
        if self.max_upload_duration_s <= 0:
            raise ValueError("VAD max upload duration must be positive")


@dataclass(frozen=True)
class VADPlan:
    """Cached GPU-VAD output consumed independently by STT workers."""

    regions: tuple[tuple[float, float], ...]
    upload_ranges: tuple[tuple[float, float], ...]
    cached: bool = False


@dataclass(frozen=True)
class AudioStats:
    """Lightweight diagnostics gathered while writing a clip."""

    peak: float
    rms: float
    clipping_ratio: float


def _sanitize_component(value: str, *, fallback: str, max_length: int) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "-", normalized)
    normalized = re.sub(r"[^\w.\-]+", "-", normalized, flags=re.UNICODE)
    normalized = re.sub(r"[-_]{2,}", "_", normalized).strip("-_.")
    if not normalized:
        normalized = fallback
    if len(normalized) > max_length:
        normalized = normalized[:max_length].rstrip("-_.")
    return normalized or fallback


def _source_id(relative_path: str) -> str:
    relative = Path(relative_path)
    top = _sanitize_component(relative.parts[0], fallback="source", max_length=16)
    stem = _sanitize_component(relative.stem, fallback="audio", max_length=24)
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:10]
    return f"{top}_{stem}_{digest}"


def discover_audio_sources(input_dir: Path, *, output_dir: Path | None = None) -> list[AudioSource]:
    """Find supported audio recursively while excluding a nested output directory."""
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve() if output_dir is not None else None
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    sources: list[AudioSource] = []
    for path in sorted(input_dir.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        resolved = path.resolve()
        if output_dir is not None and resolved.is_relative_to(output_dir):
            continue

        relative_path = resolved.relative_to(input_dir).as_posix()
        stat = resolved.stat()
        info = sf.info(resolved)
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"Invalid audio metadata: {resolved}")
        source_id = _source_id(relative_path)
        sources.append(
            AudioSource(
                path=resolved,
                relative_path=relative_path,
                source_id=source_id,
                # The loader samples a reference from the same speaker_id.
                # Track scope is safer than work scope when cast metadata is unknown.
                speaker_id=source_id,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                duration=float(info.duration),
                sample_rate=int(info.samplerate),
                channels=int(info.channels),
                frames=int(info.frames),
            )
        )
    return sources


def raw_response_path(output_dir: Path, source: AudioSource) -> Path:
    return output_dir / "raw_responses" / f"{source.source_id}.json"


def _cache_matches(payload: Mapping[str, Any], source: AudioSource) -> bool:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    source_metadata = metadata.get("source")
    if not isinstance(source_metadata, Mapping):
        return False
    return (
        metadata.get("strategy") == STT_CACHE_STRATEGY
        and source_metadata.get("relative_path") == source.relative_path
        and source_metadata.get("size_bytes") == source.size_bytes
        and source_metadata.get("mtime_ns") == source.mtime_ns
    )


def load_cached_response(output_dir: Path, source: AudioSource) -> dict[str, Any] | None:
    """Load a response only when its source fingerprint still matches."""
    path = raw_response_path(output_dir, source)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not _cache_matches(payload, source):
        return None
    response = payload.get("response")
    return response if isinstance(response, dict) else None


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def save_raw_response(
    output_dir: Path,
    source: AudioSource,
    response: Mapping[str, Any],
    *,
    endpoint: str,
    options: TranscriptionOptions,
    processing: Mapping[str, Any] | None = None,
) -> Path:
    """Persist the complete provider response before any downstream processing."""
    path = raw_response_path(output_dir, source)
    payload = {
        "metadata": {
            "provider": "xai",
            "endpoint": endpoint,
            "strategy": STT_CACHE_STRATEGY,
            "transcribed_at": datetime.now(timezone.utc).isoformat(),
            "request": options.public_metadata(),
            "processing": dict(processing or {}),
            "source": source.metadata(),
        },
        "response": dict(response),
    }
    _atomic_write_json(path, payload)
    return path


def _chunk_ranges(duration: float, config: ChunkingConfig) -> list[tuple[float, float]]:
    config.validate()
    ranges: list[tuple[float, float]] = []
    start = 0.0
    step = config.seconds - config.overlap_seconds
    while start < duration:
        end = min(duration, start + config.seconds)
        ranges.append((start, end))
        if end >= duration:
            break
        start += step
    return ranges


def _chunk_cache_path(
    output_dir: Path,
    source: AudioSource,
    *,
    start: float,
    end: float,
) -> Path:
    start_ms = round(start * 1000)
    end_ms = round(end * 1000)
    return (
        output_dir / "raw_chunks" / source.source_id / f"chunk_{start_ms:010d}_{end_ms:010d}.json"
    )


def _load_cached_chunk(
    output_dir: Path,
    source: AudioSource,
    *,
    start: float,
    end: float,
    options: TranscriptionOptions,
) -> dict[str, Any] | None:
    path = _chunk_cache_path(output_dir, source, start=start, end=end)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    source_metadata = metadata.get("source")
    if not isinstance(source_metadata, Mapping):
        return None
    if (
        metadata.get("strategy") != STT_CACHE_STRATEGY
        or source_metadata.get("relative_path") != source.relative_path
        or source_metadata.get("size_bytes") != source.size_bytes
        or source_metadata.get("mtime_ns") != source.mtime_ns
        or metadata.get("start") != start
        or metadata.get("end") != end
        or metadata.get("request") != options.public_metadata()
    ):
        return None
    response = payload.get("response")
    return response if isinstance(response, dict) else None


def _save_chunk_response(
    output_dir: Path,
    source: AudioSource,
    response: Mapping[str, Any],
    *,
    start: float,
    end: float,
    endpoint: str,
    options: TranscriptionOptions,
) -> Path:
    path = _chunk_cache_path(output_dir, source, start=start, end=end)
    payload = {
        "metadata": {
            "provider": "xai",
            "endpoint": endpoint,
            "strategy": STT_CACHE_STRATEGY,
            "transcribed_at": datetime.now(timezone.utc).isoformat(),
            "request": options.public_metadata(),
            "source": source.metadata(),
            "start": start,
            "end": end,
        },
        "response": dict(response),
    }
    _atomic_write_json(path, payload)
    return path


def _write_upload_chunk(
    reader: sf.SoundFile,
    path: Path,
    *,
    start: float,
    end: float,
) -> AudioSource:
    sample_rate = int(reader.samplerate)
    start_frame = max(0, int(math.floor(start * sample_rate)))
    end_frame = min(len(reader), int(math.ceil(end * sample_rate)))
    reader.seek(start_frame)
    audio = reader.read(end_frame - start_frame, dtype="float32", always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("STT upload chunk is empty or contains non-finite samples")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, mono, sample_rate, format="FLAC", subtype="PCM_16")
    stat = path.stat()
    info = sf.info(path)
    return AudioSource(
        path=path.resolve(),
        relative_path=path.name,
        source_id=path.stem,
        speaker_id="stt-upload",
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        duration=float(info.duration),
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
    )


def load_silero_vad(
    repo: str = SILERO_VAD_REPO,
    *,
    device: str | torch.device = "cpu",
) -> tuple[Any, Callable[..., Any]]:
    """Load the pinned official Silero VAD JIT model without its pip dependency cap."""
    torch.set_num_threads(1)
    model, utils = torch.hub.load(
        repo,
        "silero_vad",
        trust_repo=True,
        onnx=False,
        force_reload=False,
    )
    get_speech_timestamps = utils[0]
    model = model.to(torch.device(device)).eval()
    return model, get_speech_timestamps


def _load_vad_waveform(source: AudioSource, *, target_sample_rate: int = 16_000) -> torch.Tensor:
    """Decode/downmix incrementally so long stereo sources do not exhaust RAM."""
    chunks: list[torch.Tensor] = []
    with sf.SoundFile(source.path) as reader:
        source_rate = int(reader.samplerate)
        block_frames = source_rate * 60
        while True:
            audio = reader.read(block_frames, dtype="float32", always_2d=True)
            if audio.size == 0:
                break
            mono = np.mean(audio, axis=1, dtype=np.float32)
            if source_rate == target_sample_rate:
                chunk = torch.from_numpy(mono.copy())
            elif source_rate > target_sample_rate and source_rate % target_sample_rate == 0:
                chunk = torch.from_numpy(mono[:: source_rate // target_sample_rate].copy())
            else:
                chunk = torchaudio.functional.resample(
                    torch.from_numpy(mono.copy()),
                    source_rate,
                    target_sample_rate,
                )
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"Could not decode audio for VAD: {source.path}")
    return torch.cat(chunks).contiguous()


def detect_speech_regions(
    source: AudioSource,
    *,
    config: SileroVADConfig,
    model: Any,
    get_speech_timestamps: Callable[..., Any],
) -> list[tuple[float, float]]:
    """Run Silero at 16 kHz and return precise source-relative seconds."""
    config.validate()
    sample_rate = 16_000
    waveform = _load_vad_waveform(source, target_sample_rate=sample_rate)
    try:
        model_device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        model_device = torch.device("cpu")
    waveform = waveform.to(model_device)
    timestamps = get_speech_timestamps(
        waveform,
        model,
        threshold=config.threshold,
        neg_threshold=config.neg_threshold,
        sampling_rate=sample_rate,
        min_speech_duration_ms=config.min_speech_duration_ms,
        max_speech_duration_s=config.max_speech_duration_s,
        min_silence_duration_ms=config.min_silence_duration_ms,
        speech_pad_ms=config.speech_pad_ms,
        return_seconds=False,
    )
    regions: list[tuple[float, float]] = []
    for timestamp in timestamps:
        if not isinstance(timestamp, Mapping):
            continue
        start_sample = int(timestamp.get("start", 0))
        end_sample = int(timestamp.get("end", 0))
        start = max(0.0, start_sample / sample_rate)
        end = min(source.duration, end_sample / sample_rate)
        if end > start:
            regions.append((start, end))
    return regions


def pack_speech_regions(
    regions: Sequence[tuple[float, float]],
    *,
    max_duration: float,
    max_gap: float,
) -> list[tuple[float, float]]:
    """Pack nearby VAD utterances without exceeding the model-aligned duration cap."""
    if max_duration <= 0 or max_gap < 0:
        raise ValueError("Invalid VAD packing settings")
    normalized: list[tuple[float, float]] = []
    for start, end in sorted(regions):
        cursor = start
        while end - cursor > max_duration:
            normalized.append((cursor, cursor + max_duration))
            cursor += max_duration
        if end > cursor:
            normalized.append((cursor, end))
    if not normalized:
        return []

    packed: list[tuple[float, float]] = []
    current_start, current_end = normalized[0]
    for start, end in normalized[1:]:
        combined_duration = end - current_start
        gap = max(0.0, start - current_end)
        if gap <= max_gap and combined_duration <= max_duration:
            current_end = max(current_end, end)
        else:
            packed.append((current_start, current_end))
            current_start, current_end = start, end
    packed.append((current_start, current_end))
    return packed


def _vad_plan_path(output_dir: Path, source: AudioSource) -> Path:
    return output_dir / "vad_plans" / f"{source.source_id}.json"


def _parse_vad_ranges(value: Any) -> tuple[tuple[float, float], ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    parsed: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        start = _safe_float(item.get("start"))
        end = _safe_float(item.get("end"))
        if start is None or end is None or start < 0 or end <= start:
            return None
        parsed.append((start, end))
    return tuple(parsed)


def prepare_vad_plan(
    source: AudioSource,
    *,
    output_dir: Path,
    config: SileroVADConfig,
    model: Any,
    get_speech_timestamps: Callable[..., Any],
) -> VADPlan:
    """Load or compute a resumable VAD plan before any network STT work."""
    config.validate()
    path = _vad_plan_path(output_dir, source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, Mapping):
        metadata = payload.get("metadata")
        regions = _parse_vad_ranges(payload.get("regions"))
        upload_ranges = _parse_vad_ranges(payload.get("upload_ranges"))
        if (
            isinstance(metadata, Mapping)
            and metadata.get("strategy") == "silero-vad-plan-v1"
            and metadata.get("source") == source.metadata()
            and metadata.get("config") == asdict(config)
            and regions is not None
            and upload_ranges is not None
        ):
            return VADPlan(regions=regions, upload_ranges=upload_ranges, cached=True)

    regions = tuple(
        detect_speech_regions(
            source,
            config=config,
            model=model,
            get_speech_timestamps=get_speech_timestamps,
        )
    )
    upload_ranges = tuple(
        pack_speech_regions(
            regions,
            max_duration=config.max_upload_duration_s,
            max_gap=config.max_join_gap_s,
        )
    )
    _atomic_write_json(
        path,
        {
            "metadata": {
                "strategy": "silero-vad-plan-v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": source.metadata(),
                "config": asdict(config),
            },
            "regions": [{"start": round(start, 6), "end": round(end, 6)} for start, end in regions],
            "upload_ranges": [
                {"start": round(start, 6), "end": round(end, 6)} for start, end in upload_ranges
            ],
        },
    )
    return VADPlan(regions=regions, upload_ranges=upload_ranges, cached=False)


def _word_payload(word: Word) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": word.text,
        "start": round(word.start, 6),
        "end": round(word.end, 6),
    }
    if word.speaker is not None:
        payload["speaker"] = word.speaker
    if word.confidence is not None:
        payload["confidence"] = word.confidence
    return payload


def transcribe_source_chunked(
    client: STTClient,
    source: AudioSource,
    options: TranscriptionOptions,
    *,
    output_dir: Path,
    config: ChunkingConfig,
    ranges: Sequence[tuple[float, float]] | None = None,
    processing: Mapping[str, Any] | None = None,
    response_metadata: Mapping[str, Any] | None = None,
    reuse_chunks: bool = True,
    progress: Callable[[int, int, bool], None] | None = None,
) -> Path:
    """Transcribe resumable overlapping chunks and save one merged response."""
    config.validate()
    upload_ranges = list(ranges) if ranges is not None else _chunk_ranges(source.duration, config)
    overlap_seconds = config.overlap_seconds if ranges is None else 0.0
    chunk_records: list[dict[str, Any]] = []
    merged_words: list[Word] = []
    languages: list[str] = []
    upload_dir = output_dir / ".stt_uploads" / source.source_id

    with sf.SoundFile(source.path) as reader:
        for index, (start, end) in enumerate(upload_ranges):
            response = None
            if reuse_chunks:
                response = _load_cached_chunk(
                    output_dir,
                    source,
                    start=start,
                    end=end,
                    options=options,
                )
            cached = response is not None
            if response is None:
                upload_path = upload_dir / f"chunk_{index:05d}.flac"
                upload_source = _write_upload_chunk(
                    reader,
                    upload_path,
                    start=start,
                    end=end,
                )
                try:
                    response = client.transcribe(upload_source, options)
                    _save_chunk_response(
                        output_dir,
                        source,
                        response,
                        start=start,
                        end=end,
                        endpoint=client.endpoint,
                        options=options,
                    )
                finally:
                    upload_path.unlink(missing_ok=True)

            language = response.get("language")
            if isinstance(language, str) and language:
                languages.append(language)
            local_words = [
                replace(word, start=word.start + start, end=word.end + start)
                for word in parse_words(response)
                if (word.start + word.end) / 2.0 <= (end - start) + 0.25
            ]
            if index > 0 and overlap_seconds > 0:
                boundary = start + overlap_seconds / 2.0
                merged_words = [
                    word for word in merged_words if (word.start + word.end) / 2.0 < boundary
                ]
                local_words = [
                    word for word in local_words if (word.start + word.end) / 2.0 >= boundary
                ]
            merged_words.extend(local_words)
            chunk_records.append(
                {
                    "index": index,
                    "start": start,
                    "end": end,
                    "cached": cached,
                    "response": response,
                }
            )
            if progress is not None:
                progress(index + 1, len(upload_ranges), cached)

    if upload_dir.is_dir():
        try:
            upload_dir.rmdir()
            upload_dir.parent.rmdir()
        except OSError:
            pass
    merged_words.sort(key=lambda word: (word.start, word.end))
    merged_response: dict[str, Any] = {
        "text": join_word_tokens([word.text for word in merged_words]),
        "language": max(set(languages), key=languages.count) if languages else options.language,
        "duration": source.duration,
        "words": [_word_payload(word) for word in merged_words],
        "chunks": chunk_records,
    }
    if response_metadata:
        merged_response.update(response_metadata)
    processing_payload = {
        "chunk_seconds": config.seconds,
        "chunk_overlap_seconds": overlap_seconds,
        "downmix": "mono",
        "upload_sample_rate": source.sample_rate,
    }
    if processing:
        processing_payload.update(processing)
    return save_raw_response(
        output_dir,
        source,
        merged_response,
        endpoint=client.endpoint,
        options=options,
        processing=processing_payload,
    )


def transcribe_source_from_vad_plan(
    client: STTClient,
    source: AudioSource,
    options: TranscriptionOptions,
    *,
    output_dir: Path,
    config: SileroVADConfig,
    plan: VADPlan,
    reuse_chunks: bool = True,
    progress: Callable[[int, int, bool], None] | None = None,
) -> Path:
    """Consume a prepared GPU-VAD plan without blocking further VAD inference."""
    chunking = ChunkingConfig(seconds=config.max_upload_duration_s, overlap_seconds=0.0)
    return transcribe_source_chunked(
        client,
        source,
        options,
        output_dir=output_dir,
        config=chunking,
        ranges=plan.upload_ranges,
        processing={
            "vad": asdict(config),
            "vad_regions": len(plan.regions),
            "vad_plan_cached": plan.cached,
        },
        response_metadata={
            "vad_regions": [
                {"start": round(start, 6), "end": round(end, 6)} for start, end in plan.regions
            ],
            "upload_regions": [
                {"start": round(start, 6), "end": round(end, 6)}
                for start, end in plan.upload_ranges
            ],
        },
        reuse_chunks=reuse_chunks,
        progress=progress,
    )


def transcribe_source_with_vad(
    client: STTClient,
    source: AudioSource,
    options: TranscriptionOptions,
    *,
    output_dir: Path,
    config: SileroVADConfig,
    model: Any,
    get_speech_timestamps: Callable[..., Any],
    reuse_chunks: bool = True,
    progress: Callable[[int, int, bool], None] | None = None,
) -> Path:
    """Prepare a resumable Silero VAD plan, then transcribe its packed regions."""
    plan = prepare_vad_plan(
        source,
        output_dir=output_dir,
        config=config,
        model=model,
        get_speech_timestamps=get_speech_timestamps,
    )
    return transcribe_source_from_vad_plan(
        client,
        source,
        options,
        output_dir=output_dir,
        config=config,
        plan=plan,
        reuse_chunks=reuse_chunks,
        progress=progress,
    )


class STTClient(Protocol):
    """Common interface used by the REST and Grok-subscription STT transports."""

    endpoint: str

    def transcribe(
        self,
        source: AudioSource,
        options: TranscriptionOptions,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class _GrokCredential:
    access_token: str = field(repr=False)
    expires_at: datetime | None = None
    auth_mode: str | None = None


def _parse_auth_timestamp(value: Any) -> datetime | None:
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    # Grok CLI currently serializes nanoseconds while datetime accepts
    # microseconds. Truncation is sufficient for refresh-margin checks.
    normalized = re.sub(
        r"(\.\d{6})\d+(?=(?:[+-]\d{2}:\d{2})?$)",
        r"\1",
        normalized,
    )
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class GrokSubscriptionAuth:
    """Read and refresh the OAuth session managed by the installed Grok CLI."""

    def __init__(
        self,
        *,
        auth_file: Path | None = None,
        cli_command: str = "grok",
        refresh_margin_seconds: float = 300.0,
        cli_timeout_seconds: float = 60.0,
    ) -> None:
        if refresh_margin_seconds < 0:
            raise ValueError("refresh_margin_seconds must be non-negative")
        if cli_timeout_seconds <= 0:
            raise ValueError("cli_timeout_seconds must be positive")
        if not cli_command.strip():
            raise ValueError("The Grok CLI command is empty")
        self.auth_file = (
            auth_file.expanduser().resolve()
            if auth_file is not None
            else (Path.home() / ".grok" / "auth.json").resolve()
        )
        self.cli_command = cli_command.strip()
        self.refresh_margin = timedelta(seconds=refresh_margin_seconds)
        self.cli_timeout_seconds = cli_timeout_seconds
        self._credential: _GrokCredential | None = None
        self._lock = Lock()

    def _read_credential(self) -> _GrokCredential:
        try:
            payload = json.loads(self.auth_file.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Grok CLI login file was not found: {self.auth_file}. Run `grok login`."
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read Grok CLI login state: {self.auth_file}") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Unexpected Grok CLI login format: {self.auth_file}")

        candidates: list[tuple[bool, float, Mapping[str, Any]]] = []
        for value in payload.values():
            if not isinstance(value, Mapping):
                continue
            token = value.get("key")
            if not isinstance(token, str) or not token.strip():
                continue
            expires_at = _parse_auth_timestamp(value.get("expires_at"))
            expiry_score = expires_at.timestamp() if expires_at is not None else float("inf")
            is_subscription = bool(value.get("refresh_token")) or str(
                value.get("auth_mode", "")
            ).lower() in {"oauth", "subscription", "grok.com"}
            candidates.append((is_subscription, expiry_score, value))
        if not candidates:
            raise RuntimeError(
                f"No usable Grok CLI credential was found in {self.auth_file}. Run `grok login`."
            )
        _, _, selected = max(candidates, key=lambda item: (item[0], item[1]))
        return _GrokCredential(
            access_token=str(selected["key"]).strip(),
            expires_at=_parse_auth_timestamp(selected.get("expires_at")),
            auth_mode=str(selected.get("auth_mode")) if selected.get("auth_mode") else None,
        )

    def _is_fresh(self, credential: _GrokCredential) -> bool:
        return credential.expires_at is None or (
            credential.expires_at > datetime.now(timezone.utc) + self.refresh_margin
        )

    def _refresh_with_cli(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        try:
            result = subprocess.run(
                [self.cli_command, "models"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.cli_timeout_seconds,
                env=environment,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Grok CLI executable was not found: {self.cli_command!r}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Timed out while refreshing the Grok CLI login") from exc
        if result.returncode != 0:
            diagnostic = (result.stderr or result.stdout).strip().replace("\n", " ")[:500]
            suffix = f": {diagnostic}" if diagnostic else ""
            raise RuntimeError(f"`grok models` could not validate the subscription login{suffix}")

    def access_token(self, *, force_refresh: bool = False) -> str:
        """Return a fresh bearer without ever exposing it in logs or metadata."""
        with self._lock:
            if (
                not force_refresh
                and self._credential is not None
                and self._is_fresh(self._credential)
            ):
                return self._credential.access_token
            if not force_refresh:
                credential = self._read_credential()
                if self._is_fresh(credential):
                    self._credential = credential
                    return credential.access_token
            self._refresh_with_cli()
            credential = self._read_credential()
            if not self._is_fresh(credential):
                raise RuntimeError(
                    "Grok CLI login remained expired after refresh; run `grok login` interactively"
                )
            self._credential = credential
            return credential.access_token

    def validate(self) -> dict[str, Any]:
        """Refresh once and return only non-secret diagnostics for preflight output."""
        self.access_token(force_refresh=True)
        assert self._credential is not None
        return {
            "auth_file": self.auth_file.as_posix(),
            "auth_mode": self._credential.auth_mode,
            "expires_at": (
                self._credential.expires_at.isoformat()
                if self._credential.expires_at is not None
                else None
            ),
        }


def _load_pcm16_mono(source: AudioSource, *, sample_rate: int = 16_000) -> bytes:
    with sf.SoundFile(source.path) as reader:
        audio = reader.read(dtype="float32", always_2d=True)
        source_rate = int(reader.samplerate)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"STT audio is empty or non-finite: {source.path}")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    waveform = torch.from_numpy(mono.copy())
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    values = waveform.detach().cpu().numpy()
    pcm = np.clip(np.rint(values * 32768.0), -32768, 32767).astype("<i2")
    return pcm.tobytes()


def streaming_events_to_response(
    events: Sequence[Mapping[str, Any]],
    *,
    duration: float,
    default_language: str,
) -> dict[str, Any]:
    """Collapse repeated streaming finals into one REST-compatible response."""
    partials: dict[tuple[int, float], Mapping[str, Any]] = {}
    completed: dict[int, Mapping[str, Any]] = {}
    languages: list[str] = []
    response_duration = duration
    for event in events:
        language = event.get("language")
        if isinstance(language, str) and language:
            languages.append(language)
        event_duration = _safe_float(event.get("duration"))
        if event.get("type") == "transcript.done" and event_duration is not None:
            response_duration = max(response_duration, event_duration)
        channel = int(event.get("channel_index", 0))
        words = event.get("words")
        has_words = (
            isinstance(words, Sequence) and not isinstance(words, (str, bytes)) and bool(words)
        )
        has_text = isinstance(event.get("text"), str) and bool(str(event.get("text")).strip())
        if event.get("type") == "transcript.done":
            if has_words or has_text:
                completed[channel] = event
            continue
        if event.get("type") != "transcript.partial" or event.get("is_final") is not True:
            continue
        if not has_words and not has_text:
            continue
        start = _safe_float(event.get("start"))
        if start is None and has_words:
            first = words[0]
            start = _safe_float(first.get("start")) if isinstance(first, Mapping) else None
        partials[(channel, round(start or 0.0, 4))] = event

    selected: list[Mapping[str, Any]] = []
    channels = {channel for channel, _ in partials} | set(completed)
    for channel in sorted(channels):
        if channel in completed:
            selected.append(completed[channel])
        else:
            selected.extend(
                event
                for (event_channel, _), event in sorted(partials.items())
                if event_channel == channel
            )

    words_out: list[Word] = []
    seen: set[tuple[float, float, str, str]] = set()
    for event in selected:
        parsed = parse_words(event)
        if not parsed:
            text = str(event.get("text", "")).strip()
            start = _safe_float(event.get("start")) or 0.0
            event_duration = _safe_float(event.get("duration")) or 0.0
            if text and event_duration > 0:
                parsed = [Word(text=text, start=start, end=start + event_duration)]
        for word in parsed:
            key = (
                round(word.start, 4),
                round(word.end, 4),
                word.text,
                str(word.speaker),
            )
            if key not in seen:
                seen.add(key)
                words_out.append(word)
    words_out.sort(key=lambda word: (word.start, word.end, word.text))
    return {
        "text": join_word_tokens([word.text for word in words_out]),
        "language": max(set(languages), key=languages.count) if languages else default_language,
        "duration": round(response_duration, 6),
        "words": [_word_payload(word) for word in words_out],
        "transport": "grok-cli-subscription-websocket",
    }


class _GrokStreamingAuthError(RuntimeError):
    pass


class _GrokStreamingRetryableError(RuntimeError):
    pass


class GrokSubscriptionSTTClient:
    """Streaming STT client authenticated by the user's Grok CLI subscription."""

    def __init__(
        self,
        auth: GrokSubscriptionAuth,
        *,
        endpoint: str = XAI_STT_WEBSOCKET_ENDPOINT,
        endpointing_ms: int = 350,
        frame_milliseconds: int = 100,
        realtime_factor: float = 0.0,
        timeout_seconds: float = 3600.0,
        max_retries: int = 4,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("The xAI streaming STT endpoint is empty")
        if not 0 <= endpointing_ms <= 5000:
            raise ValueError("endpointing_ms must be between 0 and 5000")
        if frame_milliseconds <= 0:
            raise ValueError("frame_milliseconds must be positive")
        if realtime_factor < 0:
            raise ValueError("realtime_factor must be non-negative")
        if timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("Invalid streaming STT retry settings")
        self.auth = auth
        self.endpoint = endpoint.strip()
        self.endpointing_ms = endpointing_ms
        self.frame_milliseconds = frame_milliseconds
        self.realtime_factor = realtime_factor
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def _url(self, options: TranscriptionOptions) -> str:
        if options.multichannel:
            raise ValueError(
                "Grok subscription STT receives mono VAD chunks; use --auth-mode api-key "
                "for --multichannel"
            )
        params: list[tuple[str, str]] = [
            ("sample_rate", "16000"),
            ("encoding", "pcm"),
            ("interim_results", "false"),
            ("endpointing", str(self.endpointing_ms)),
            ("language", options.language),
            ("diarize", str(options.diarize).lower()),
            ("filler_words", str(options.filler_words).lower()),
        ]
        params.extend(("keyterm", term) for term in options.keyterms)
        separator = "&" if "?" in self.endpoint else "?"
        return f"{self.endpoint}{separator}{urlencode(params)}"

    async def _transcribe_async(
        self,
        pcm: bytes,
        *,
        token: str,
        options: TranscriptionOptions,
        duration: float,
    ) -> dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds, connect=30.0)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(
                    self._url(options),
                    headers={"Authorization": f"Bearer {token}"},
                    max_msg_size=16 * 1024 * 1024,
                ) as websocket:
                    created = await websocket.receive_json(timeout=min(30.0, self.timeout_seconds))
                    if (
                        not isinstance(created, Mapping)
                        or created.get("type") != "transcript.created"
                    ):
                        raise _GrokStreamingRetryableError(
                            "xAI streaming STT did not send transcript.created"
                        )
                    bytes_per_frame = 16_000 * 2 * self.frame_milliseconds // 1000
                    for offset in range(0, len(pcm), bytes_per_frame):
                        await websocket.send_bytes(pcm[offset : offset + bytes_per_frame])
                        if self.realtime_factor > 0:
                            await asyncio.sleep(
                                self.frame_milliseconds / 1000.0 / self.realtime_factor
                            )
                    await websocket.send_json({"type": "audio.done"})
                    events: list[Mapping[str, Any]] = []
                    while True:
                        message = await websocket.receive(timeout=self.timeout_seconds)
                        if message.type == aiohttp.WSMsgType.TEXT:
                            try:
                                event = json.loads(message.data)
                            except json.JSONDecodeError as exc:
                                raise _GrokStreamingRetryableError(
                                    "xAI streaming STT sent invalid JSON"
                                ) from exc
                            if not isinstance(event, Mapping):
                                continue
                            event_type = event.get("type")
                            if event_type == "error":
                                detail = str(event.get("message", "unknown error"))[:500]
                                raise _GrokStreamingRetryableError(
                                    f"xAI streaming STT error: {detail}"
                                )
                            if event_type in {"transcript.partial", "transcript.done"}:
                                events.append(event)
                            if event_type == "transcript.done":
                                break
                        elif message.type in {
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        }:
                            raise _GrokStreamingRetryableError(
                                "xAI streaming STT closed before transcript.done"
                            )
        except aiohttp.WSServerHandshakeError as exc:
            if exc.status in {401, 403}:
                raise _GrokStreamingAuthError(
                    f"xAI streaming STT rejected the Grok CLI login (HTTP {exc.status})"
                ) from exc
            if exc.status in {408, 409, 425, 429, 500, 502, 503, 504}:
                raise _GrokStreamingRetryableError(
                    f"xAI streaming STT handshake returned HTTP {exc.status}"
                ) from exc
            raise RuntimeError(f"xAI streaming STT handshake returned HTTP {exc.status}") from exc
        return streaming_events_to_response(
            events,
            duration=duration,
            default_language=options.language,
        )

    def transcribe(
        self,
        source: AudioSource,
        options: TranscriptionOptions,
    ) -> dict[str, Any]:
        pcm = _load_pcm16_mono(source)
        duration = len(pcm) / (16_000 * 2)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                token = self.auth.access_token(
                    force_refresh=isinstance(last_error, _GrokStreamingAuthError)
                )
                return asyncio.run(
                    self._transcribe_async(
                        pcm,
                        token=token,
                        options=options,
                        duration=duration,
                    )
                )
            except _GrokStreamingAuthError as exc:
                last_error = exc
            except (
                _GrokStreamingRetryableError,
                aiohttp.ClientError,
                asyncio.TimeoutError,
                OSError,
            ) as exc:
                last_error = exc
            if attempt >= self.max_retries:
                break
            time.sleep(min(2**attempt, 30))
        raise RuntimeError(
            f"Grok subscription STT failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error


class GrokSTTClient:
    """Small retrying client for xAI's multipart STT endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = XAI_STT_ENDPOINT,
        timeout_seconds: float = 3600.0,
        max_retries: int = 4,
    ) -> None:
        if not api_key.strip():
            raise ValueError("The xAI API key is empty")
        self.api_key = api_key.strip()
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def transcribe(
        self,
        source: AudioSource,
        options: TranscriptionOptions,
    ) -> dict[str, Any]:
        if source.size_bytes > XAI_MAX_FILE_BYTES:
            raise ValueError(
                f"File exceeds xAI's 500 MB limit ({source.size_bytes} bytes): {source.path}"
            )

        retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                mime_type = _MIME_TYPES.get(source.path.suffix.lower())
                if mime_type is None:
                    mime_type = (
                        mimetypes.guess_type(source.path.name)[0] or "application/octet-stream"
                    )
                with source.path.open("rb") as audio_handle:
                    # requests serializes normal form fields before files. This keeps
                    # `file` last, as required by the xAI STT API.
                    response = requests.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        data=options.request_fields(),
                        files={"file": (source.path.name, audio_handle, mime_type)},
                        timeout=(30.0, self.timeout_seconds),
                    )
                if response.status_code in retryable_statuses:
                    raise _RetryableHTTPError(response)
                if not response.ok:
                    body = response.text.replace("\n", " ")[:1000]
                    raise RuntimeError(f"xAI STT returned HTTP {response.status_code}: {body}")
                payload = response.json()
                if not isinstance(payload, dict):
                    raise RuntimeError("xAI STT returned a non-object JSON response")
                return payload
            except (_RetryableHTTPError, requests.RequestException) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                retry_after = _retry_after_seconds(exc)
                delay = retry_after if retry_after is not None else min(2**attempt, 30)
                time.sleep(max(0.0, min(float(delay), 30.0)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid xAI STT response for {source.relative_path}: {exc}"
                ) from exc
        raise RuntimeError(
            f"xAI STT request failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error


class _RetryableHTTPError(RuntimeError):
    def __init__(self, response: requests.Response) -> None:
        self.response = response
        super().__init__(f"Retryable HTTP {response.status_code}")


def _retry_after_seconds(error: Exception) -> float | None:
    if not isinstance(error, _RetryableHTTPError):
        return None
    value = error.response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_words(response: Mapping[str, Any]) -> list[Word]:
    """Validate and normalize the endpoint's word-level timestamps."""
    raw_words = response.get("words")
    if not isinstance(raw_words, list):
        return []

    words: list[Word] = []
    for item in raw_words:
        if not isinstance(item, Mapping):
            continue
        text = item.get("text")
        start = _safe_float(item.get("start"))
        end = _safe_float(item.get("end"))
        if not isinstance(text, str) or not text.strip() or start is None or end is None:
            continue
        if start < 0 or end <= start:
            continue
        confidence = _safe_float(item.get("confidence"))
        words.append(
            Word(
                text=text,
                start=start,
                end=end,
                speaker=item.get("speaker"),
                confidence=confidence,
            )
        )
    words.sort(key=lambda item: (item.start, item.end))
    return words


def _contains_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3040 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0xFF66 <= codepoint <= 0xFF9F
    )


def join_word_tokens(tokens: Sequence[str]) -> str:
    """Join Japanese tokens without injecting English-style spaces."""
    if not tokens:
        return ""
    if any(token[:1].isspace() for token in tokens[1:] if token):
        return re.sub(r"\s+", " ", "".join(tokens)).strip()

    joined = ""
    for raw_token in tokens:
        token = raw_token.strip()
        if not token:
            continue
        if not joined:
            joined = token
            continue
        previous = joined[-1]
        current = token[0]
        needs_space = (
            previous not in _OPEN_PUNCTUATION
            and current not in _CLOSE_PUNCTUATION
            and previous.isalnum()
            and current.isalnum()
            and not _contains_cjk(previous)
            and not _contains_cjk(current)
        )
        joined += (" " if needs_space else "") + token
    return re.sub(r"\s+", " ", joined).strip()


def _is_sentence_boundary(word: Word) -> bool:
    stripped = word.text.rstrip()
    return bool(stripped) and stripped[-1] in _SENTENCE_END


def _split_hard_blocks(
    words: Sequence[Word], hard_gap: float, sentence_min_seconds: float
) -> list[tuple[int, int]]:
    if not words:
        return []
    blocks: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(words)):
        previous = words[index - 1]
        current = words[index]
        speaker_changed = (
            previous.speaker is not None
            and current.speaker is not None
            and previous.speaker != current.speaker
        )
        sentence_boundary = (
            _is_sentence_boundary(previous)
            and previous.end - words[start].start >= sentence_min_seconds
        )
        if current.start - previous.end >= hard_gap or speaker_changed or sentence_boundary:
            blocks.append((start, index))
            start = index
    blocks.append((start, len(words)))
    return blocks


def _choose_split(
    words: Sequence[Word],
    start: int,
    block_end: int,
    config: SegmentationConfig,
) -> int:
    segment_start = words[start].start
    maximum_end = start + 1
    candidates: list[tuple[float, int]] = []
    for end in range(start + 1, block_end + 1):
        duration = words[end - 1].end - segment_start
        if duration <= config.max_seconds + 1e-6:
            maximum_end = end
        else:
            break
        if duration < config.min_seconds:
            continue
        next_gap = 0.0
        if end < block_end:
            next_gap = max(0.0, words[end].start - words[end - 1].end)
        if _is_sentence_boundary(words[end - 1]) or next_gap >= config.soft_gap_seconds:
            sentence_bonus = -0.75 if _is_sentence_boundary(words[end - 1]) else 0.0
            candidates.append((abs(duration - config.target_seconds) + sentence_bonus, end))

    if candidates:
        return min(candidates, key=lambda item: (item[0], -item[1]))[1]
    return maximum_end


def _segment_block(
    words: Sequence[Word],
    block_start: int,
    block_end: int,
    config: SegmentationConfig,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = block_start
    while cursor < block_end:
        remaining_duration = words[block_end - 1].end - words[cursor].start
        if remaining_duration <= config.max_seconds:
            ranges.append((cursor, block_end))
            break
        split = _choose_split(words, cursor, block_end, config)
        if split <= cursor:
            split = cursor + 1
        ranges.append((cursor, split))
        cursor = split

    if len(ranges) >= 2:
        last_start, last_end = ranges[-1]
        last_duration = words[last_end - 1].end - words[last_start].start
        previous_start, _ = ranges[-2]
        merged_duration = words[last_end - 1].end - words[previous_start].start
        if last_duration < config.min_seconds and merged_duration <= config.max_seconds:
            ranges[-2] = (previous_start, last_end)
            ranges.pop()
    return ranges


def segment_words(
    words: Sequence[Word],
    *,
    source_duration: float,
    config: SegmentationConfig,
) -> list[Segment]:
    """Split words at speaker/long-silence boundaries and cap clip length."""
    config.validate()
    if not words:
        return []

    ranges: list[tuple[int, int]] = []
    for block_start, block_end in _split_hard_blocks(
        words, config.hard_gap_seconds, config.min_seconds
    ):
        ranges.extend(_segment_block(words, block_start, block_end, config))

    segments: list[Segment] = []
    for word_start, word_end in ranges:
        selected = words[word_start:word_end]
        spoken = [
            word
            for word in selected
            if any(unicodedata.category(character)[0] in {"L", "M", "N"} for character in word.text)
        ]
        acoustic = spoken or list(selected)
        confidences = [word.confidence for word in selected if word.confidence is not None]
        speakers = {word.speaker for word in selected if word.speaker is not None}
        speaker = next(iter(speakers)) if len(speakers) == 1 else None
        segments.append(
            Segment(
                word_start=word_start,
                word_end=word_end,
                start=max(0.0, acoustic[0].start - config.padding_seconds),
                end=min(source_duration, acoustic[-1].end + config.padding_seconds),
                text=join_word_tokens([word.text for word in selected]),
                speaker=speaker,
                confidence_mean=(sum(confidences) / len(confidences) if confidences else None),
            )
        )

    # Padding must not create overlapping clips. Split the available boundary
    # at its midpoint while keeping the actual word timestamps intact.
    for index in range(1, len(segments)):
        previous = segments[index - 1]
        current = segments[index]
        if previous.end <= current.start:
            continue
        midpoint = (previous.end + current.start) / 2.0
        segments[index - 1] = replace(previous, end=min(previous.end, midpoint))
        segments[index] = replace(current, start=max(current.start, midpoint))
    return segments


def _meaningful_character_count(text: str) -> int:
    return sum(
        1
        for character in text
        if not unicodedata.category(character).startswith(("P", "S", "Z", "C"))
    )


def review_reasons(
    segment: Segment,
    config: SegmentationConfig,
    *,
    audio_stats: AudioStats | None = None,
) -> list[str]:
    """Return conservative reasons for excluding a segment from train.jsonl."""
    reasons: list[str] = []
    meaningful = _meaningful_character_count(segment.text)
    if meaningful == 0:
        reasons.append("empty_text")
    if segment.duration < config.min_seconds:
        reasons.append("too_short")
    if segment.duration > config.max_seconds + 0.05:
        reasons.append("too_long")
    if segment.duration > 0:
        characters_per_second = meaningful / segment.duration
        if meaningful > 0 and characters_per_second < config.min_chars_per_second:
            reasons.append("text_too_sparse")
        if characters_per_second > config.max_chars_per_second:
            reasons.append("text_too_dense")
    compact = re.sub(r"\s+", "", segment.text)
    if re.search(r"(.{1,6})\1{5,}", compact):
        reasons.append("repeated_text")
    if audio_stats is not None:
        if audio_stats.rms < 1e-5:
            reasons.append("near_silent_audio")
        if audio_stats.clipping_ratio > 0.01:
            reasons.append("clipping_audio")
    return reasons


def extract_clip(
    reader: sf.SoundFile,
    output_path: Path,
    *,
    start: float,
    end: float,
    rebuild: bool = False,
) -> AudioStats:
    """Read a range, downmix like DACVAE, and write mono PCM-16 FLAC atomically."""
    sample_rate = int(reader.samplerate)
    start_frame = max(0, int(math.floor(start * sample_rate)))
    end_frame = min(len(reader), int(math.ceil(end * sample_rate)))
    if end_frame <= start_frame:
        raise ValueError(f"Invalid clip frame range: {start_frame}:{end_frame}")

    reader.seek(start_frame)
    audio = reader.read(end_frame - start_frame, dtype="float32", always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Decoded clip is empty or contains non-finite samples")

    mono = np.mean(audio, axis=1, dtype=np.float32)
    absolute = np.abs(mono)
    stats = AudioStats(
        peak=float(absolute.max(initial=0.0)),
        rms=float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))),
        clipping_ratio=float(np.mean(absolute >= 0.999)),
    )
    if output_path.is_file() and not rebuild:
        existing = sf.info(output_path)
        if existing.channels == 1 and existing.samplerate == sample_rate:
            return stats

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp.flac")
    sf.write(temporary, mono, sample_rate, format="FLAC", subtype="PCM_16")
    os.replace(temporary, output_path)
    return stats


def _manifest_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write JSON Lines atomically for CLI status/error files."""
    _atomic_write_jsonl(path, rows)


def build_dataset(
    sources: Sequence[AudioSource],
    output_dir: Path,
    *,
    config: SegmentationConfig,
    project_root: Path,
    rebuild_clips: bool = False,
) -> dict[str, Any]:
    """Create FLAC clips plus train/review manifests from cached STT responses."""
    output_dir = output_dir.expanduser().resolve()
    all_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    source_errors: list[dict[str, Any]] = []

    for source in sources:
        response = load_cached_response(output_dir, source)
        source_row = source.metadata()
        if response is None:
            source_row["status"] = "missing_transcript"
            source_rows.append(source_row)
            continue

        words = parse_words(response)
        if not words:
            source_row["status"] = "review"
            source_row["reason"] = "missing_word_timestamps"
            source_rows.append(source_row)
            source_errors.append(
                {
                    "source_uid": source.source_id,
                    "source_audio": _manifest_path(source.path, project_root),
                    "reason": "missing_word_timestamps",
                    "response_text": str(response.get("text", "")),
                }
            )
            continue

        segments = segment_words(words, source_duration=source.duration, config=config)
        written_for_source = 0
        try:
            with sf.SoundFile(source.path) as reader:
                for segment in segments:
                    start_ms = round(segment.start * 1000)
                    end_ms = round(segment.end * 1000)
                    clip_id = f"{source.source_id}_{start_ms:010d}_{end_ms:010d}"
                    clip_path = output_dir / "clips" / source.source_id / f"{clip_id}.flac"
                    try:
                        stats = extract_clip(
                            reader,
                            clip_path,
                            start=segment.start,
                            end=segment.end,
                            rebuild=rebuild_clips,
                        )
                    except (OSError, RuntimeError, ValueError) as exc:
                        source_errors.append(
                            {
                                "source_uid": source.source_id,
                                "segment_id": clip_id,
                                "reason": "clip_extract_error",
                                "error": str(exc),
                            }
                        )
                        continue

                    reasons = review_reasons(segment, config, audio_stats=stats)
                    speaker_id = source.speaker_id
                    if segment.speaker is not None:
                        speaker_component = _sanitize_component(
                            str(segment.speaker), fallback="unknown", max_length=32
                        )
                        speaker_id = f"{speaker_id}:stt-{speaker_component}"
                    row: dict[str, Any] = {
                        "id": clip_id,
                        "audio": _manifest_path(clip_path, project_root),
                        "text": segment.text,
                        "source_uid": source.source_id,
                        "speaker_id": speaker_id,
                        "duration": round(segment.duration, 6),
                        "start": round(segment.start, 6),
                        "end": round(segment.end, 6),
                        "source_audio": _manifest_path(source.path, project_root),
                        "word_start": segment.word_start,
                        "word_end": segment.word_end,
                        "word_count": segment.word_end - segment.word_start,
                        "status": "review" if reasons else "train",
                        "review_reasons": reasons,
                        "sample_rate": source.sample_rate,
                        "channels": 1,
                        "peak": round(stats.peak, 7),
                        "rms": round(stats.rms, 7),
                        "clipping_ratio": round(stats.clipping_ratio, 9),
                    }
                    if segment.speaker is not None:
                        row["stt_speaker"] = segment.speaker
                    if segment.confidence_mean is not None:
                        row["confidence_mean"] = round(segment.confidence_mean, 6)
                    all_rows.append(row)
                    written_for_source += 1
        except (OSError, RuntimeError) as exc:
            source_errors.append(
                {
                    "source_uid": source.source_id,
                    "source_audio": _manifest_path(source.path, project_root),
                    "reason": "source_decode_error",
                    "error": str(exc),
                }
            )

        source_row["status"] = "processed" if written_for_source else "review"
        source_row["segments"] = written_for_source
        source_row["response_language"] = response.get("language")
        source_row["response_duration"] = response.get("duration")
        source_rows.append(source_row)

    all_rows.sort(key=lambda row: (str(row["source_uid"]), float(row["start"])))
    train_rows = [row for row in all_rows if row["status"] == "train"]
    review_rows = [row for row in all_rows if row["status"] == "review"]
    _atomic_write_jsonl(output_dir / "all.jsonl", all_rows)
    _atomic_write_jsonl(output_dir / "train.jsonl", train_rows)
    _atomic_write_jsonl(output_dir / "review.jsonl", review_rows)
    _atomic_write_jsonl(output_dir / "sources.jsonl", source_rows)
    _atomic_write_jsonl(output_dir / "source_errors.jsonl", source_errors)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources_selected": len(sources),
        "sources_with_transcript": sum(
            1 for source in sources if load_cached_response(output_dir, source) is not None
        ),
        "segments_total": len(all_rows),
        "segments_train": len(train_rows),
        "segments_review": len(review_rows),
        "source_errors": len(source_errors),
        "train_audio_hours": round(
            sum(float(row["duration"]) for row in train_rows) / 3600.0,
            6,
        ),
        "segmentation": asdict(config),
        "model_alignment": {
            "model_family": "Irodori-TTS v3",
            "codec": "Aratako/Semantic-DACVAE-Japanese-32dim",
            "sample_rate": 48000,
            "channels": 1,
            "latent_fps": 25,
            "max_latent_steps": 750,
        },
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary


def write_output_readme(output_dir: Path, *, input_dir: Path, project_root: Path) -> None:
    """Write a short local hand-off beside the generated artifacts."""
    input_display = _manifest_path(input_dir, project_root)
    output_display = _manifest_path(output_dir, project_root)
    text = f"""# Grok STT 学習データ

入力: `{input_display}`

## ファイル

- `raw_responses/`: xAI STT の生レスポンス。再実行時に再利用されます。
- `raw_chunks/`: Silero VADで分けた各STT送信区間のレスポンス。途中再開に使います。
- `clips/`: 単語タイムスタンプから切り出した 48 kHz・モノラル・PCM-16 FLAC。
- `train.jsonl`: 自動チェックを通過した学習候補。
- `review.jsonl`: 短すぎる・文字密度が異常など、要試聴の候補。
- `all.jsonl`: 上記 2 種類を合わせた完全な一覧。
- `sources.jsonl`: 入力ファイルごとの処理状態。
- `source_errors.jsonl`: タイムスタンプ不足やデコード失敗など。
- `summary.json`: 件数と設定の要約。

`speaker_id` は録音トラック単位です。学習時の参照音声は同じトラック内から選ばれます。

## DACVAE latent の生成

リポジトリルートから実行します。

```powershell
uv run --no-sync irodori-prepare-manifest `
  --dataset json `
  --data-files train={output_display}/train.jsonl `
  --split train `
  --audio-column audio `
  --text-column text `
  --speaker-column speaker_id `
  --speaker-id-prefix grok-stt `
  --target-sample-rate 48000 `
  --output-manifest data/manifests/grok_stt.jsonl `
  --latent-dir data/latents/grok_stt `
  --device cuda `
  --num-gpus 2 `
  --merge-output `
  --prefetch 8 `
  --prefetch-workers 2
```
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / ".README.md.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, output_dir / "README.md")
