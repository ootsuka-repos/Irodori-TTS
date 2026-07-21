"""Local speech clip pipeline: Silero VAD segmentation without any cloud STT.

Replaces the former Grok STT module. Sources are scanned, Silero VAD detects
speech, nearby utterances are packed into 5-20 s clips, and FLAC clips plus
train/review manifests are written. Text starts empty; the transcribe stage
fills it, and the Grok CLI is used later only as a plain LLM text corrector.

The per-source VAD cache (``vad_responses/{source_id}.json``) keeps the same
payload shape the nonverbal pipeline consumes (``metadata.source`` plus
``response.vad_regions``), so acoustic candidate discovery keeps working.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

from dataset._io_utils import (
    atomic_write_flac,
    atomic_write_json,
    atomic_write_jsonl,
)
from dataset._textnorm import sanitize_source_component

SILERO_VAD_REPO = "snakers4/silero-vad:v6.2.1"
VAD_CACHE_SCHEMA = "silero-vad-local-v1"

# Only containers libsndfile can actually decode.
AUDIO_SUFFIXES = {
    ".aif",
    ".aiff",
    ".caf",
    ".flac",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wv",
}

_CPU_VAD_THREAD_LOCK = Lock()


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
class Segment:
    """A contiguous speech span selected for one training clip."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class SileroVADConfig:
    """Pinned Silero settings tuned for whispery ASMR speech."""

    repo: str = SILERO_VAD_REPO
    threshold: float = 0.35
    neg_threshold: float | None = 0.20
    min_speech_duration_ms: int = 180
    min_silence_duration_ms: int = 450
    speech_pad_ms: int = 120
    max_speech_duration_s: float = 29.0

    def validate(self) -> None:
        if not 0 < self.threshold < 1:
            raise ValueError("VAD threshold must be between zero and one")
        if self.neg_threshold is not None and not 0 <= self.neg_threshold < self.threshold:
            raise ValueError("VAD neg_threshold must be below threshold")
        if self.min_speech_duration_ms <= 0 or self.min_silence_duration_ms < 0:
            raise ValueError("VAD duration settings are invalid")
        if self.speech_pad_ms < 0 or self.max_speech_duration_s <= 0:
            raise ValueError("VAD padding/max speech settings are invalid")


@dataclass(frozen=True)
class SpeechSegmentationConfig:
    """Rules used to turn VAD speech regions into 5-20 s training clips."""

    min_seconds: float = 5.0
    max_seconds: float = 20.0
    # ASMR dialogue contains deliberate pauses; join utterances across gaps up
    # to this long so a sentence with breathing room stays one clip.
    join_gap_seconds: float = 2.0
    # Preserve breath attacks and low-energy decay VAD underestimates.
    padding_seconds: float = 0.35
    tail_extension_seconds: float = 1.5
    tail_silence_dbfs: float = -55.0
    tail_silence_hold_seconds: float = 0.20
    # Fixed digital silence appended to every written clip.
    tail_pad_silence_seconds: float = 0.5
    # Perceptual edge fades so clip boundaries never sound abruptly cut.
    # The fade-out is longer than the fade-in because trailing breath/decay
    # cut mid-energy is what reads as a hard chop.
    fade_in_seconds: float = 0.1
    fade_out_seconds: float = 0.8

    def validate(self) -> None:
        if self.min_seconds <= 0 or self.max_seconds <= self.min_seconds:
            raise ValueError("Expected 0 < min_seconds < max_seconds")
        if self.join_gap_seconds < 0 or self.padding_seconds < 0:
            raise ValueError("join_gap_seconds/padding_seconds must be non-negative")
        if self.tail_extension_seconds < 0 or self.tail_pad_silence_seconds < 0:
            raise ValueError("tail settings must be non-negative")
        if self.tail_silence_hold_seconds <= 0:
            raise ValueError("tail_silence_hold_seconds must be positive")
        if self.fade_in_seconds < 0 or self.fade_out_seconds < 0:
            raise ValueError("fade_in_seconds/fade_out_seconds must be non-negative")

    @property
    def max_span_seconds(self) -> float:
        """Region-span cap that keeps clips within ``max_seconds`` after padding."""
        return max(self.min_seconds, self.max_seconds - 2.0 * self.padding_seconds)


@dataclass(frozen=True)
class AudioStats:
    """Lightweight diagnostics gathered while writing a clip."""

    peak: float
    rms: float
    clipping_ratio: float


def _source_id(relative_path: str) -> str:
    relative = Path(relative_path)
    top = sanitize_source_component(relative.parts[0], fallback="source", max_length=16)
    stem = sanitize_source_component(relative.stem, fallback="audio", max_length=24)
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:10]
    return f"{top}_{stem}_{digest}"


def discover_audio_sources(
    input_dir: Path,
    *,
    output_dir: Path | None = None,
    excluded_dirs: Sequence[Path] = (),
) -> list[AudioSource]:
    """Find supported audio recursively while excluding output/excluded directories."""
    input_dir = input_dir.expanduser().resolve()
    excluded = [directory.expanduser().resolve() for directory in excluded_dirs]
    if output_dir is not None:
        excluded.append(output_dir.expanduser().resolve())
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    sources: list[AudioSource] = []
    for path in sorted(input_dir.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        resolved = path.resolve()
        if any(resolved.is_relative_to(directory) for directory in excluded):
            continue

        relative_path = resolved.relative_to(input_dir).as_posix()
        try:
            stat = resolved.stat()
            info = sf.info(resolved)
            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError("no decodable frames or sample rate")
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"skipping unreadable audio {relative_path}: {exc}", flush=True)
            continue
        source_id = _source_id(relative_path)
        sources.append(
            AudioSource(
                path=resolved,
                relative_path=relative_path,
                source_id=source_id,
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


def load_silero_vad(
    repo: str = SILERO_VAD_REPO,
    *,
    device: str | torch.device = "cpu",
) -> tuple[Any, Callable[..., Any]]:
    """Load the pinned official Silero VAD JIT model."""
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
            audio = reader.read(block_frames, dtype=("float" + "32"), always_2d=True)
            if audio.size == 0:
                break
            mono = np.mean(audio, axis=1, dtype=np.single)
            if source_rate == target_sample_rate:
                chunk = torch.from_numpy(mono.copy())
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
    vad_kwargs = {
        "threshold": config.threshold,
        "neg_threshold": config.neg_threshold,
        "sampling_rate": sample_rate,
        "min_speech_duration_ms": config.min_speech_duration_ms,
        "max_speech_duration_s": config.max_speech_duration_s,
        "min_silence_duration_ms": config.min_silence_duration_ms,
        "speech_pad_ms": config.speech_pad_ms,
        "return_seconds": False,
    }
    if model_device.type == "cpu":
        with _CPU_VAD_THREAD_LOCK:
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                timestamps = get_speech_timestamps(waveform, model, **vad_kwargs)
            finally:
                torch.set_num_threads(previous_threads)
    else:
        timestamps = get_speech_timestamps(waveform, model, **vad_kwargs)
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
    """Pack nearby VAD utterances without exceeding the duration cap."""
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


# --- VAD cache (consumed by both this module and the nonverbal pipeline) ---


def vad_response_path(output_dir: Path, source: AudioSource) -> Path:
    return output_dir / "vad_responses" / f"{source.source_id}.json"


def load_cached_vad(
    output_dir: Path,
    source: AudioSource,
    *,
    config: SileroVADConfig,
) -> list[tuple[float, float]] | None:
    path = vad_response_path(output_dir, source)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    if metadata.get("schema") != VAD_CACHE_SCHEMA:
        return None
    if metadata.get("source") != source.metadata():
        return None
    if metadata.get("vad") != asdict(config):
        return None
    response = payload.get("response")
    if not isinstance(response, Mapping):
        return None
    regions = response.get("vad_regions")
    if not isinstance(regions, list):
        return None
    result: list[tuple[float, float]] = []
    for item in regions:
        if (
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes))
            and len(item) == 2
        ):
            result.append((float(item[0]), float(item[1])))
    return result


def save_cached_vad(
    output_dir: Path,
    source: AudioSource,
    regions: Sequence[tuple[float, float]],
    *,
    config: SileroVADConfig,
) -> None:
    payload = {
        "metadata": {
            "schema": VAD_CACHE_SCHEMA,
            "source": source.metadata(),
            "vad": asdict(config),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "response": {
            "vad_regions": [[round(start, 6), round(end, 6)] for start, end in regions],
        },
    }
    atomic_write_json(vad_response_path(output_dir, source), payload)


# --- clip extraction ---


def _audio_stats(mono: np.ndarray) -> AudioStats:
    absolute = np.abs(mono)
    return AudioStats(
        peak=float(absolute.max(initial=0.0)),
        rms=float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))),
        clipping_ratio=float(np.mean(absolute >= 0.999)),
    )


def _cosine_ramp(length: int) -> np.ndarray:
    """Raised-cosine 0->1 ramp; smoother at the endpoints than a linear fade."""
    phase = np.linspace(0.0, np.pi, length, endpoint=False, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(phase)).astype(np.single)


def _apply_edge_fades(
    mono: np.ndarray,
    sample_rate: int,
    *,
    fade_in_seconds: float = 0.1,
    fade_out_seconds: float = 0.8,
) -> None:
    """Apply in-place cosine edge fades so cut boundaries do not sound chopped.

    The two edges are faded independently: a short fade-in keeps consonant
    attacks intact while the longer fade-out lands trailing breath/decay softly
    instead of cutting it mid-energy. Both are clamped so they never overlap.
    """
    half = int(mono.size) // 2
    fade_in = min(int(round(sample_rate * fade_in_seconds)), half)
    fade_out = min(int(round(sample_rate * fade_out_seconds)), half)
    if fade_in > 0:
        mono[:fade_in] *= _cosine_ramp(fade_in)
    if fade_out > 0:
        mono[-fade_out:] *= _cosine_ramp(fade_out)[::-1]


def _frame_rms(mono: np.ndarray, frame: int) -> np.ndarray:
    usable = (mono.size // frame) * frame
    if usable == 0:
        return np.zeros(0, dtype=np.float64)
    frames = mono[:usable].reshape(-1, frame)
    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))


def extend_segment_tails(
    reader: sf.SoundFile,
    segments: list[Segment],
    config: SpeechSegmentationConfig,
) -> list[Segment]:
    """Push each clip's edges outward until the audio stays quiet.

    VAD boundaries underestimate breath attacks and audible decay, so
    ``padding_seconds`` alone leaves sound running into the cut. Extensions
    never cross the neighbouring clip, the source bounds, or ``max_seconds``.
    """
    if config.tail_extension_seconds <= 0 or not segments:
        return segments
    sample_rate = int(reader.samplerate)
    source_end = len(reader) / sample_rate
    frame = max(1, int(round(0.02 * sample_rate)))
    hold_frames = max(1, int(round(config.tail_silence_hold_seconds / 0.02)))
    floor = 10.0 ** (config.tail_silence_dbfs / 20.0)

    def read_span(begin: float, finish: float) -> np.ndarray:
        begin_frame = max(0, int(begin * sample_rate))
        reader.seek(begin_frame)
        span = reader.read(
            max(0, int(finish * sample_rate) - begin_frame), dtype=("float" + "32"), always_2d=True
        )
        return span.mean(axis=1)

    def snap_edge(edge: float, limit: float, threshold: float, direction: int) -> float:
        """Move edge outward toward limit; stop after a sustained-quiet run."""
        window = read_span(min(edge, limit), max(edge, limit))
        if direction < 0:
            window = window[::-1]
        rms = _frame_rms(window, frame)
        if rms.size == 0:
            return edge
        run = 0
        for position, value in enumerate(rms):
            run = run + 1 if value < threshold else 0
            if run >= hold_frames:
                run_start = position - hold_frames + 1
                return edge + direction * ((run_start * frame) / sample_rate + 0.10)
        return limit

    refined = list(segments)
    for index, segment in enumerate(refined):
        reference = read_span(segment.start, segment.end)
        ref_rms = (
            float(np.sqrt(np.mean(np.square(reference, dtype=np.float64))))
            if reference.size
            else 0.0
        )
        threshold = max(floor, 0.03 * ref_rms)

        end_limit = min(
            source_end,
            segment.start + config.max_seconds,
            segment.end + config.tail_extension_seconds,
        )
        if index + 1 < len(refined):
            end_limit = min(end_limit, refined[index + 1].start)
        start_limit = max(0.0, segment.start - config.tail_extension_seconds)
        if index > 0:
            start_limit = max(start_limit, refined[index - 1].end)

        new_end = segment.end
        if end_limit > segment.end + 0.02:
            new_end = min(
                max(snap_edge(segment.end, end_limit, threshold, +1), segment.end), end_limit
            )
        new_start = segment.start
        if start_limit < segment.start - 0.02:
            new_start = max(
                min(snap_edge(segment.start, start_limit, threshold, -1), segment.start),
                start_limit,
            )
        if new_end - new_start > config.max_seconds:
            new_start = new_end - config.max_seconds
        if new_end > segment.end + 1e-3 or new_start < segment.start - 1e-3:
            refined[index] = replace(segment, start=new_start, end=new_end)
    return refined


def extract_clip(
    reader: sf.SoundFile,
    output_path: Path,
    *,
    start: float,
    end: float,
    rebuild: bool = False,
    tail_pad_seconds: float = 0.0,
    fade_in_seconds: float = 0.1,
    fade_out_seconds: float = 0.8,
) -> AudioStats:
    """Read a range, downmix, fade the edges, and write mono FLAC."""
    sample_rate = int(reader.samplerate)
    start_frame = max(0, int(math.floor(start * sample_rate)))
    end_frame = min(len(reader), int(math.ceil(end * sample_rate)))
    if end_frame <= start_frame:
        raise ValueError(f"Invalid clip frame range: {start_frame}:{end_frame}")
    pad_frames = max(0, int(round(tail_pad_seconds * sample_rate)))
    expected_frames = end_frame - start_frame + pad_frames

    if output_path.is_file() and not rebuild:
        try:
            existing = sf.info(output_path)
        except (OSError, RuntimeError):
            existing = None
        if (
            existing is not None
            and int(existing.channels) == 1
            and int(existing.samplerate) == sample_rate
            and int(existing.frames) == expected_frames
        ):
            mono = sf.read(output_path, dtype=("float" + "32"), always_2d=True)[0][:, 0]
            if mono.size and np.isfinite(mono).all():
                return _audio_stats(mono)

    reader.seek(start_frame)
    audio = reader.read(end_frame - start_frame, dtype=("float" + "32"), always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Decoded clip is empty or contains non-finite samples")

    mono = np.mean(audio, axis=1, dtype=np.single)
    _apply_edge_fades(
        mono,
        sample_rate,
        fade_in_seconds=fade_in_seconds,
        fade_out_seconds=fade_out_seconds,
    )
    if pad_frames:
        mono = np.concatenate([mono, np.zeros(pad_frames, dtype=np.single)])
    stats = _audio_stats(mono)
    atomic_write_flac(output_path, mono, sample_rate)
    return stats


def _manifest_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def segment_vad_regions(
    regions: Sequence[tuple[float, float]],
    *,
    source_duration: float,
    config: SpeechSegmentationConfig,
) -> list[Segment]:
    """Turn raw VAD speech regions into padded, non-overlapping clip spans."""
    config.validate()
    packed = pack_speech_regions(
        regions,
        max_duration=config.max_span_seconds,
        max_gap=config.join_gap_seconds,
    )
    segments = [
        Segment(
            start=max(0.0, start - config.padding_seconds),
            end=min(source_duration, end + config.padding_seconds),
        )
        for start, end in packed
    ]
    # Padding must not create overlapping clips; cut at the midpoint.
    for index in range(1, len(segments)):
        previous = segments[index - 1]
        current = segments[index]
        if previous.end > current.start:
            midpoint = (previous.end + current.start) / 2.0
            segments[index - 1] = replace(previous, end=midpoint)
            segments[index] = replace(current, start=midpoint)
    return segments


def review_reasons(
    segment: Segment,
    config: SpeechSegmentationConfig,
    *,
    audio_stats: AudioStats | None = None,
) -> list[str]:
    """Return conservative reasons for excluding a clip from train.jsonl."""
    reasons: list[str] = []
    if segment.duration < config.min_seconds:
        reasons.append("too_short")
    if segment.duration > config.max_seconds + 0.05:
        reasons.append("too_long")
    if audio_stats is not None:
        if audio_stats.rms < 1e-5:
            reasons.append("near_silent_audio")
        if audio_stats.clipping_ratio > 0.01:
            reasons.append("clipping_audio")
    return reasons


def _process_one_source(
    source: AudioSource,
    *,
    output_dir: Path,
    vad_config: SileroVADConfig,
    segmentation_config: SpeechSegmentationConfig,
    project_root: Path,
    rebuild_clips: bool,
    vad_model: Any,
    get_speech_timestamps: Callable[..., Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """VAD + clip extraction for one source; returns (source_row, rows, errors)."""
    source_row = source.metadata()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    regions = load_cached_vad(output_dir, source, config=vad_config)
    if regions is None:
        try:
            regions = detect_speech_regions(
                source,
                config=vad_config,
                model=vad_model,
                get_speech_timestamps=get_speech_timestamps,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            source_row["status"] = "error"
            source_row["reason"] = "vad_error"
            errors.append(
                {
                    "source_uid": source.source_id,
                    "source_audio": _manifest_path(source.path, project_root),
                    "reason": "vad_error",
                    "error": str(exc),
                }
            )
            return source_row, rows, errors
        save_cached_vad(output_dir, source, regions, config=vad_config)

    segments = segment_vad_regions(
        regions,
        source_duration=source.duration,
        config=segmentation_config,
    )
    written_for_source = 0
    try:
        with sf.SoundFile(source.path) as reader:
            segments = extend_segment_tails(reader, segments, segmentation_config)
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
                        tail_pad_seconds=segmentation_config.tail_pad_silence_seconds,
                        fade_in_seconds=segmentation_config.fade_in_seconds,
                        fade_out_seconds=segmentation_config.fade_out_seconds,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    errors.append(
                        {
                            "source_uid": source.source_id,
                            "segment_id": clip_id,
                            "reason": "clip_extract_error",
                            "error": str(exc),
                        }
                    )
                    continue

                reasons = review_reasons(segment, segmentation_config, audio_stats=stats)
                rows.append(
                    {
                        "id": clip_id,
                        "audio": _manifest_path(clip_path, project_root),
                        "text": "",
                        "source_uid": source.source_id,
                        "speaker_id": source.speaker_id,
                        "duration": round(segment.duration, 6),
                        "start": round(segment.start, 6),
                        "end": round(segment.end, 6),
                        "source_audio": _manifest_path(source.path, project_root),
                        "status": "review" if reasons else "train",
                        "review_reasons": reasons,
                        "sample_rate": source.sample_rate,
                        "channels": 1,
                        "peak": round(stats.peak, 7),
                        "rms": round(stats.rms, 7),
                        "clipping_ratio": round(stats.clipping_ratio, 9),
                    }
                )
                written_for_source += 1
    except (OSError, RuntimeError) as exc:
        errors.append(
            {
                "source_uid": source.source_id,
                "source_audio": _manifest_path(source.path, project_root),
                "reason": "source_decode_error",
                "error": str(exc),
            }
        )

    source_row["status"] = "processed" if written_for_source else "review"
    source_row["segments"] = written_for_source
    return source_row, rows, errors


_POOL_STATE: dict[str, Any] = {}


def _pool_init(
    output_dir: str,
    vad_config: SileroVADConfig,
    segmentation_config: SpeechSegmentationConfig,
    project_root: str,
    vad_devices: tuple[str, ...],
    rebuild_clips: bool,
) -> None:
    import multiprocessing
    import os

    # WindowsのPIDはほぼ常に4の倍数で pid % n が偏る（n=2で全ワーカーがGPU0に集中）。
    # プール内ワーカー番号（1始まりの連番）で割り当てて均等に分散させる。
    identity = multiprocessing.current_process()._identity
    worker_index = identity[0] if identity else os.getpid()
    _POOL_STATE.update(
        output_dir=Path(output_dir),
        vad_config=vad_config,
        segmentation_config=segmentation_config,
        project_root=Path(project_root),
        vad_device=vad_devices[worker_index % len(vad_devices)],
        rebuild_clips=rebuild_clips,
        model=None,
        get_speech_timestamps=None,
    )


def _pool_process(source: AudioSource) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    state = _POOL_STATE
    if state["model"] is None:
        # Lazy per-worker Silero load; cached sources never pay for it.
        needs_vad = (
            load_cached_vad(state["output_dir"], source, config=state["vad_config"]) is None
        )
        if needs_vad:
            state["model"], state["get_speech_timestamps"] = load_silero_vad(
                state["vad_config"].repo, device=state["vad_device"]
            )
    return _process_one_source(
        source,
        output_dir=state["output_dir"],
        vad_config=state["vad_config"],
        segmentation_config=state["segmentation_config"],
        project_root=state["project_root"],
        rebuild_clips=state["rebuild_clips"],
        vad_model=state["model"],
        get_speech_timestamps=state["get_speech_timestamps"],
    )


def build_dataset(
    sources: Sequence[AudioSource],
    output_dir: Path,
    *,
    vad_config: SileroVADConfig,
    segmentation_config: SpeechSegmentationConfig,
    project_root: Path,
    vad_device: str = "cpu",
    vad_devices: Sequence[str] | None = None,
    workers: int = 1,
    rebuild_clips: bool = False,
    manifest_suffix: str = "",
) -> dict[str, Any]:
    """Run VAD, cut 5-20 s FLAC clips, and write all/train/review manifests.

    ``workers > 1`` uses a dynamic process pool: every worker pulls the next
    source as soon as it finishes, so uneven source durations cannot leave a
    long single-worker tail the way static sharding does. Manifest rows start
    with an empty ``text``; the ASR stage is the single source of transcripts.
    """
    output_dir = output_dir.expanduser().resolve()
    segmentation_config.validate()
    devices = tuple(vad_devices) if vad_devices else (vad_device,)

    all_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    source_errors: list[dict[str, Any]] = []

    def consume(index: int, result: tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]) -> None:
        source_row, rows, errors = result
        source_rows.append(source_row)
        all_rows.extend(rows)
        source_errors.extend(errors)
        print(
            f"speech {index}/{len(sources)} source={source_row['source_id']} "
            f"segments={source_row.get('segments', 0)}",
            flush=True,
        )

    if workers <= 1:
        model: Any = None
        get_speech_timestamps: Callable[..., Any] | None = None
        for index, source in enumerate(sources, 1):
            if model is None and load_cached_vad(output_dir, source, config=vad_config) is None:
                model, get_speech_timestamps = load_silero_vad(
                    vad_config.repo, device=devices[0]
                )
            consume(
                index,
                _process_one_source(
                    source,
                    output_dir=output_dir,
                    vad_config=vad_config,
                    segmentation_config=segmentation_config,
                    project_root=project_root,
                    rebuild_clips=rebuild_clips,
                    vad_model=model,
                    get_speech_timestamps=get_speech_timestamps,
                ),
            )
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_pool_init,
            initargs=(
                str(output_dir),
                vad_config,
                segmentation_config,
                str(project_root),
                devices,
                rebuild_clips,
            ),
        ) as pool:
            for index, result in enumerate(
                pool.map(_pool_process, sources, chunksize=1), 1
            ):
                consume(index, result)

    all_rows.sort(key=lambda row: (str(row["source_uid"]), float(row["start"])))
    train_rows = [row for row in all_rows if row["status"] == "train"]
    review_rows = [row for row in all_rows if row["status"] == "review"]
    atomic_write_jsonl(output_dir / f"all{manifest_suffix}.jsonl", all_rows)
    atomic_write_jsonl(output_dir / f"train{manifest_suffix}.jsonl", train_rows)
    atomic_write_jsonl(output_dir / f"review{manifest_suffix}.jsonl", review_rows)
    atomic_write_jsonl(output_dir / f"sources{manifest_suffix}.jsonl", source_rows)
    if source_errors:
        atomic_write_jsonl(output_dir / f"source_errors{manifest_suffix}.jsonl", source_errors)

    return {
        "sources": len(sources),
        "rows": len(all_rows),
        "train": len(train_rows),
        "review": len(review_rows),
        "errors": len(source_errors),
        "train_seconds": round(sum(float(row["duration"]) for row in train_rows), 3),
    }
