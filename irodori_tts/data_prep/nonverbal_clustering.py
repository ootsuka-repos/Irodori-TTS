"""Cluster audio outside cached Silero VAD regions with BEATs embeddings."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA

from irodori_tts.data_prep.acoustic_segmentation import (
    ACOUSTIC_SEGMENTATION_VERSION,
    AcousticPrimitive,
    AcousticSegmentationConfig,
    segment_acoustic_primitives,
)

BEATS_CHECKPOINT_FILENAME = "BEATs_iter3_plus_AS2M.pt"
BEATS_CHECKPOINT_SHA256 = "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34"
BEATS_MODEL_NAME = "Microsoft BEATs Iter3+ AS2M (pre-trained)"
BEATS_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 48_000


@dataclass(frozen=True)
class CandidateWindow:
    id: str
    source_key: str
    source_uid: str
    speaker_id: str
    source_audio: str
    source_relative_path: str
    gap_index: int
    window_index: int
    start: float
    end: float
    duration: float
    segmentation_mode: str = "acoustic"
    segmentation_version: str = ACOUSTIC_SEGMENTATION_VERSION
    left_boundary_reason: str = "gap_start"
    left_boundary_score: float = 1.0
    right_boundary_reason: str = "gap_end"
    right_boundary_score: float = 1.0


@dataclass(frozen=True)
class FeatureConfig:
    embedding_chunk_seconds: float = 5.0
    min_gap_seconds: float = 0.08
    sample_rate: int = BEATS_SAMPLE_RATE
    segmentation_mode: str = "acoustic"
    fixed_window_seconds: float = 5.0
    acoustic: AcousticSegmentationConfig = AcousticSegmentationConfig()

    def __post_init__(self) -> None:
        if self.embedding_chunk_seconds <= 0:
            raise ValueError("embedding_chunk_seconds must be positive")
        if self.min_gap_seconds < 0:
            raise ValueError("min_gap_seconds cannot be negative")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.segmentation_mode not in {"acoustic", "fixed"}:
            raise ValueError("segmentation_mode must be 'acoustic' or 'fixed'")
        if self.fixed_window_seconds <= 0:
            raise ValueError("fixed_window_seconds must be positive")


@dataclass(frozen=True)
class ClusterConfig:
    pca_components: int = 64
    hdbscan_min_cluster_size: int = 30
    hdbscan_min_samples: int = 10
    kmeans_clusters: int = 96
    representatives_near: int = 3
    representatives_boundary: int = 2
    random_state: int = 42


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
    )


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=_json_default) + "\n"
        for row in rows
    )
    _atomic_write_text(path, text)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_intervals(
    intervals: Sequence[tuple[float, float]],
    *,
    duration: float,
) -> list[tuple[float, float]]:
    """Clamp, sort, and merge overlapping intervals inside ``[0, duration]``."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    normalized: list[tuple[float, float]] = []
    for start, end in intervals:
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        bounded_start = min(duration, max(0.0, float(start)))
        bounded_end = min(duration, max(0.0, float(end)))
        if bounded_end > bounded_start:
            normalized.append((bounded_start, bounded_end))
    normalized.sort()
    merged: list[tuple[float, float]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def complement_intervals(
    intervals: Sequence[tuple[float, float]],
    *,
    duration: float,
) -> list[tuple[float, float]]:
    """Return the exact complement of intervals inside ``[0, duration]``."""
    merged = merge_intervals(intervals, duration=duration)
    complement: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            complement.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        complement.append((cursor, duration))
    return complement


def split_interval_evenly(
    start: float,
    end: float,
    *,
    max_seconds: float,
) -> list[tuple[float, float]]:
    """Split without overlap while avoiding a tiny final remainder."""
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    duration = end - start
    if duration <= 0:
        return []
    count = max(1, math.ceil(duration / max_seconds))
    step = duration / count
    return [
        (start + step * index, end if index == count - 1 else start + step * (index + 1))
        for index in range(count)
    ]


def _source_key(index: int, source_uid: str) -> str:
    suffix = hashlib.sha1(source_uid.encode("utf-8")).hexdigest()[:10]
    return f"s{index:03d}_{suffix}"


def _candidate_id(source_key: str, start: float, end: float) -> str:
    return f"nv_{source_key}_{round(start * 1000):09d}_{round(end * 1000):09d}"


def _ffprobe_duration(path: Path) -> float | None:
    executable = shutil.which("ffprobe")
    if executable is None:
        return None
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=creation_flags,
        )
        value = float(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _read_gap_waveform(
    reader: sf.SoundFile,
    start: float,
    end: float,
) -> torch.Tensor:
    sample_rate = int(reader.samplerate)
    start_frame = max(0, int(math.floor(start * sample_rate)))
    end_frame = min(len(reader), int(math.ceil(end * sample_rate)))
    reader.seek(start_frame)
    audio = reader.read(end_frame - start_frame, dtype="float32", always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio gap at {start:.6f}-{end:.6f}s")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(mono))


def _fixed_primitives(duration: float, max_seconds: float) -> list[AcousticPrimitive]:
    intervals = split_interval_evenly(0.0, duration, max_seconds=max_seconds)
    primitives: list[AcousticPrimitive] = []
    for index, (start, end) in enumerate(intervals):
        primitives.append(
            AcousticPrimitive(
                start=start,
                end=end,
                left_boundary_reason="gap_start" if index == 0 else "fixed_even",
                left_boundary_score=1.0 if index == 0 else 0.0,
                right_boundary_reason=(
                    "gap_end" if index == len(intervals) - 1 else "fixed_even"
                ),
                right_boundary_score=1.0 if index == len(intervals) - 1 else 0.0,
            )
        )
    return primitives


def load_vad_complements(
    raw_response_dir: Path,
    *,
    data_root: Path,
    feature_config: FeatureConfig,
    max_sources: int | None = None,
    source_shard_index: int = 0,
    source_shard_count: int = 1,
    segmentation_device: str | torch.device = "cpu",
) -> tuple[list[CandidateWindow], list[dict[str, Any]], dict[str, Any]]:
    """Build exact-cover natural candidates from cached raw VAD responses."""
    raw_paths = sorted(raw_response_dir.glob("*.json"), key=lambda path: path.name.casefold())
    if max_sources is not None:
        raw_paths = raw_paths[:max_sources]
    if not raw_paths:
        raise RuntimeError(f"No raw response JSON files found under: {raw_response_dir}")
    if source_shard_count <= 0:
        raise ValueError("source_shard_count must be positive")
    if not 0 <= source_shard_index < source_shard_count:
        raise ValueError("source_shard_index must be in [0, source_shard_count)")
    indexed_raw_paths = list(enumerate(raw_paths))[source_shard_index::source_shard_count]
    analysis_device = torch.device(segmentation_device)
    if analysis_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA acoustic segmentation was requested, but CUDA is unavailable")

    candidates: list[CandidateWindow] = []
    excluded: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    total_duration = 0.0
    total_vad_seconds = 0.0
    total_outside_seconds = 0.0

    for source_index, raw_path in indexed_raw_paths:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata")
        response = payload.get("response")
        if not isinstance(metadata, dict) or not isinstance(response, dict):
            raise ValueError(f"Invalid cached response structure: {raw_path}")
        source_meta = metadata.get("source")
        vad_payload = response.get("vad_regions")
        if not isinstance(source_meta, dict) or not isinstance(vad_payload, list):
            raise ValueError(f"Cached response has no original vad_regions: {raw_path}")

        cached_duration = float(source_meta["duration"])
        relative_path = str(source_meta["relative_path"])
        source_audio = (data_root / Path(relative_path)).resolve()
        if not source_audio.is_file():
            raise FileNotFoundError(f"Source audio from cache is missing: {source_audio}")
        audio_info = sf.info(source_audio)
        soundfile_duration = float(audio_info.frames) / float(audio_info.samplerate)
        probed_duration = _ffprobe_duration(source_audio)
        decodable_duration = min(
            soundfile_duration,
            probed_duration if probed_duration is not None else soundfile_duration,
        )
        duration = min(cached_duration, decodable_duration)
        source_uid = str(source_meta.get("source_id") or raw_path.stem)
        speaker_id = str(source_meta.get("speaker_id") or source_uid)
        source_key = _source_key(source_index, source_uid)
        vad_regions = [
            (float(item["start"]), float(item["end"]))
            for item in vad_payload
            if isinstance(item, dict) and "start" in item and "end" in item
        ]
        merged_vad = merge_intervals(vad_regions, duration=duration)
        outside_regions = complement_intervals(merged_vad, duration=duration)

        source_candidate_count = 0
        with sf.SoundFile(source_audio) as segmentation_reader:
            for gap_index, (gap_start, gap_end) in enumerate(outside_regions):
                gap_duration = gap_end - gap_start
                terminal_mp3_padding = (
                    source_audio.suffix.casefold() == ".mp3"
                    and math.isclose(gap_end, duration, abs_tol=1e-5)
                    and gap_duration < 0.25
                )
                if gap_duration < feature_config.min_gap_seconds or terminal_mp3_padding:
                    excluded.append(
                        {
                            "source_uid": source_uid,
                            "source_audio": source_audio.as_posix(),
                            "start": round(gap_start, 6),
                            "end": round(gap_end, 6),
                            "duration": round(gap_duration, 6),
                            "reason": (
                                "mp3_terminal_padding"
                                if terminal_mp3_padding
                                else "shorter_than_min_gap"
                            ),
                        }
                    )
                    continue

                if feature_config.segmentation_mode == "acoustic":
                    waveform = _read_gap_waveform(segmentation_reader, gap_start, gap_end)
                    primitives = segment_acoustic_primitives(
                        waveform.to(analysis_device, non_blocking=True),
                        int(segmentation_reader.samplerate),
                        config=feature_config.acoustic,
                    )
                    segmentation_version = ACOUSTIC_SEGMENTATION_VERSION
                else:
                    primitives = _fixed_primitives(
                        gap_duration,
                        feature_config.fixed_window_seconds,
                    )
                    segmentation_version = "fixed_even_v1"

                previous_end = gap_start
                for window_index, primitive in enumerate(primitives):
                    start = previous_end
                    end = (
                        gap_end
                        if window_index == len(primitives) - 1
                        else min(gap_end, max(start, gap_start + primitive.end))
                    )
                    previous_end = end
                    if end <= start:
                        continue
                    candidate = CandidateWindow(
                        id=_candidate_id(source_key, start, end),
                        source_key=source_key,
                        source_uid=source_uid,
                        speaker_id=speaker_id,
                        source_audio=source_audio.as_posix(),
                        source_relative_path=relative_path.replace("\\", "/"),
                        gap_index=gap_index,
                        window_index=window_index,
                        start=round(start, 6),
                        end=round(end, 6),
                        duration=round(end - start, 6),
                        segmentation_mode=feature_config.segmentation_mode,
                        segmentation_version=segmentation_version,
                        left_boundary_reason=primitive.left_boundary_reason,
                        left_boundary_score=round(float(primitive.left_boundary_score), 6),
                        right_boundary_reason=primitive.right_boundary_reason,
                        right_boundary_score=round(float(primitive.right_boundary_score), 6),
                    )
                    candidates.append(candidate)
                    source_candidate_count += 1

        vad_seconds = sum(end - start for start, end in merged_vad)
        outside_seconds = sum(end - start for start, end in outside_regions)
        total_duration += duration
        total_vad_seconds += vad_seconds
        total_outside_seconds += outside_seconds
        source_summaries.append(
            {
                "source_key": source_key,
                "source_uid": source_uid,
                "source_audio": source_audio.as_posix(),
                "duration": round(duration, 6),
                "cached_duration": round(cached_duration, 6),
                "decodable_duration": round(decodable_duration, 6),
                "soundfile_duration": round(soundfile_duration, 6),
                "ffprobe_duration": None if probed_duration is None else round(probed_duration, 6),
                "vad_regions": len(merged_vad),
                "vad_seconds": round(vad_seconds, 6),
                "outside_regions": len(outside_regions),
                "outside_seconds": round(outside_seconds, 6),
                "candidate_windows": source_candidate_count,
                "segmentation_mode": feature_config.segmentation_mode,
            }
        )

    summary = {
        "sources": len(source_summaries),
        "source_audio_hours": total_duration / 3600,
        "vad_audio_hours": total_vad_seconds / 3600,
        "outside_vad_hours": total_outside_seconds / 3600,
        "candidate_windows": len(candidates),
        "excluded_short_gaps": len(excluded),
        "sources_detail": source_summaries,
    }
    return candidates, excluded, summary


def _load_beats_model(
    *,
    beats_code_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    beats_code_dir = beats_code_dir.resolve()
    if not (beats_code_dir / "BEATs.py").is_file():
        raise FileNotFoundError(f"Microsoft BEATs.py not found under: {beats_code_dir}")
    sys.path.insert(0, str(beats_code_dir))
    try:
        from BEATs import BEATs, BEATsConfig
    finally:
        try:
            sys.path.remove(str(beats_code_dir))
        except ValueError:
            pass

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "cfg" not in checkpoint or "model" not in checkpoint:
        raise ValueError(f"Unexpected BEATs checkpoint structure: {checkpoint_path}")
    model = BEATs(BEATsConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    return model


def _audio_metrics(mono: np.ndarray) -> dict[str, float]:
    if mono.size == 0:
        return {"rms_dbfs": -120.0, "peak_dbfs": -120.0, "crest_factor_db": 0.0}
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    peak = float(np.max(np.abs(mono)))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-6))
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-6))
    return {
        "rms_dbfs": round(max(-120.0, rms_dbfs), 4),
        "peak_dbfs": round(max(-120.0, peak_dbfs), 4),
        "crest_factor_db": round(max(0.0, peak_dbfs - rms_dbfs), 4),
    }


def _read_candidate_audio(
    reader: sf.SoundFile,
    candidate: CandidateWindow,
    *,
    resamplers: dict[int, torchaudio.transforms.Resample],
) -> tuple[torch.Tensor, dict[str, float]]:
    sample_rate = int(reader.samplerate)
    start_frame = max(0, int(math.floor(candidate.start * sample_rate)))
    end_frame = min(len(reader), int(math.ceil(candidate.end * sample_rate)))
    reader.seek(start_frame)
    audio = reader.read(end_frame - start_frame, dtype="float32", always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio window: {candidate.id}")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    metrics = _audio_metrics(mono)
    waveform = torch.from_numpy(mono.copy())
    if sample_rate != BEATS_SAMPLE_RATE:
        resampler = resamplers.get(sample_rate)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(sample_rate, BEATS_SAMPLE_RATE)
            resamplers[sample_rate] = resampler
        waveform = resampler(waveform)
    return waveform.contiguous(), metrics


def _pool_beats_batch(
    model: torch.nn.Module,
    waveforms: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> np.ndarray:
    """Pool one model-sized batch without normalizing individual rows."""
    if not waveforms:
        raise ValueError("waveforms cannot be empty")
    batch_samples = max(int(waveform.numel()) for waveform in waveforms)
    if batch_samples <= 0:
        raise ValueError("waveforms must contain at least one sample")
    batch = torch.zeros((len(waveforms), batch_samples), dtype=torch.float32)
    padding_mask = torch.ones((len(waveforms), batch_samples), dtype=torch.bool)
    for index, waveform in enumerate(waveforms):
        valid_samples = int(waveform.numel())
        if valid_samples <= 0:
            continue
        batch[index, :valid_samples] = waveform
        padding_mask[index, :valid_samples] = False
    batch = batch.to(device, non_blocking=True)
    padding_mask = padding_mask.to(device, non_blocking=True)
    with torch.inference_mode():
        features, feature_padding = model.extract_features(batch, padding_mask=padding_mask)
        if feature_padding is None:
            pooled = features.mean(dim=1)
        else:
            valid = (~feature_padding).unsqueeze(-1)
            counts = valid.sum(dim=1).clamp_min(1)
            pooled = (features * valid).sum(dim=1) / counts
    return pooled.float().cpu().numpy().astype(np.float32, copy=False)


def _embed_batch(
    model: torch.nn.Module,
    waveforms: Sequence[torch.Tensor],
    *,
    device: torch.device,
    model_samples: int,
) -> np.ndarray:
    """Embed natural units, pooling internal model-sized chunks by valid duration.

    ``model_samples`` limits only the hidden BEATs forward-pass size.  A visible
    natural event is never truncated or exposed as several training segments.
    """
    if model_samples <= 0:
        raise ValueError("model_samples must be positive")
    if not waveforms:
        return np.empty((0, 0), dtype=np.float32)

    chunks: list[torch.Tensor] = []
    owners: list[int] = []
    weights: list[int] = []
    for owner, waveform in enumerate(waveforms):
        sample_count = int(waveform.numel())
        if sample_count <= 0:
            raise ValueError(f"waveform {owner} contains no samples")
        for start in range(0, sample_count, model_samples):
            chunk = waveform[start : start + model_samples].contiguous()
            chunks.append(chunk)
            owners.append(owner)
            weights.append(int(chunk.numel()))

    # A long natural unit can expand into several internal chunks.  Limiting
    # each forward pass to the original candidate batch size prevents a 15 s
    # event from silently tripling peak VRAM relative to the former 5 s path.
    chunk_batch_size = max(1, len(waveforms))
    chunk_embeddings: list[np.ndarray] = []
    for offset in range(0, len(chunks), chunk_batch_size):
        chunk_embeddings.append(
            _pool_beats_batch(
                model,
                chunks[offset : offset + chunk_batch_size],
                device=device,
            )
        )
    pooled_chunks = np.concatenate(chunk_embeddings, axis=0)

    weighted = np.zeros((len(waveforms), pooled_chunks.shape[1]), dtype=np.float64)
    total_weights = np.zeros(len(waveforms), dtype=np.float64)
    for index, (owner, weight) in enumerate(zip(owners, weights, strict=True)):
        weighted[owner] += pooled_chunks[index].astype(np.float64) * weight
        total_weights[owner] += weight
    combined = weighted / np.maximum(total_weights[:, None], 1.0)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / np.maximum(norms, 1e-12)
    return combined.astype(np.float32, copy=False)


def _flush_embedding_batch(
    *,
    model: torch.nn.Module,
    batch_waveforms: list[torch.Tensor],
    batch_rows: list[dict[str, Any]],
    source_embeddings: list[np.ndarray],
    source_rows: list[dict[str, Any]],
    device: torch.device,
    model_samples: int,
) -> None:
    if not batch_waveforms:
        return
    embedded = _embed_batch(
        model,
        batch_waveforms,
        device=device,
        model_samples=model_samples,
    )
    source_embeddings.append(embedded)
    source_rows.extend(batch_rows)
    batch_waveforms.clear()
    batch_rows.clear()


def extract_beats_embeddings(
    candidates: Sequence[CandidateWindow],
    *,
    output_dir: Path,
    feature_config: FeatureConfig,
    beats_code_dir: Path,
    checkpoint_path: Path,
    device_name: str,
    batch_size: int,
    force: bool = False,
    write_combined: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Extract resumable, per-source BEATs embedding shards."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    expected_sha = BEATS_CHECKPOINT_SHA256
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"Unexpected BEATs checkpoint SHA256: {actual_sha}; expected {expected_sha}"
        )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch environment has no CUDA support")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    grouped: dict[str, list[CandidateWindow]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.source_key].append(candidate)
    shard_dir = output_dir / "embeddings" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    model: torch.nn.Module | None = None
    all_embeddings: list[np.ndarray] = []
    all_rows: list[dict[str, Any]] = []
    model_samples = math.ceil(feature_config.embedding_chunk_seconds * BEATS_SAMPLE_RATE)
    resamplers: dict[int, torchaudio.transforms.Resample] = {}

    for source_number, source_key in enumerate(sorted(grouped), 1):
        source_candidates = sorted(grouped[source_key], key=lambda item: (item.start, item.end))
        embedding_path = shard_dir / f"{source_key}.npy"
        metadata_path = shard_dir / f"{source_key}.jsonl"
        if not force and embedding_path.is_file() and metadata_path.is_file():
            cached_embeddings = np.load(embedding_path, allow_pickle=False)
            cached_rows = _read_jsonl(metadata_path)
            expected_ids = [candidate.id for candidate in source_candidates]
            cached_ids = [str(row.get("id")) for row in cached_rows]
            if cached_embeddings.shape[0] == len(expected_ids) and cached_ids == expected_ids:
                all_embeddings.append(cached_embeddings.astype(np.float32, copy=False))
                all_rows.extend(cached_rows)
                print(
                    f"features source={source_number}/{len(grouped)} cached=true "
                    f"windows={len(source_candidates)} key={source_key}",
                    flush=True,
                )
                continue

        if model is None:
            model = _load_beats_model(
                beats_code_dir=beats_code_dir,
                checkpoint_path=checkpoint_path,
                device=device,
            )
        source_path = Path(source_candidates[0].source_audio)
        source_embeddings: list[np.ndarray] = []
        source_rows: list[dict[str, Any]] = []
        batch_waveforms: list[torch.Tensor] = []
        batch_rows: list[dict[str, Any]] = []

        with sf.SoundFile(source_path) as reader:
            for candidate in source_candidates:
                waveform, metrics = _read_candidate_audio(
                    reader,
                    candidate,
                    resamplers=resamplers,
                )
                row = {
                    **asdict(candidate),
                    "outside_vad": True,
                    "sample_type": "nonverbal_candidate",
                    "reference_eligible": False,
                    "embedding_chunks": max(
                        1,
                        math.ceil(int(waveform.numel()) / model_samples),
                    ),
                    **metrics,
                }
                batch_waveforms.append(waveform)
                batch_rows.append(row)
                if len(batch_waveforms) >= batch_size:
                    _flush_embedding_batch(
                        model=model,
                        batch_waveforms=batch_waveforms,
                        batch_rows=batch_rows,
                        source_embeddings=source_embeddings,
                        source_rows=source_rows,
                        device=device,
                        model_samples=model_samples,
                    )
            _flush_embedding_batch(
                model=model,
                batch_waveforms=batch_waveforms,
                batch_rows=batch_rows,
                source_embeddings=source_embeddings,
                source_rows=source_rows,
                device=device,
                model_samples=model_samples,
            )

        embedding_array = np.concatenate(source_embeddings, axis=0)
        if embedding_array.shape[0] != len(source_rows):
            raise RuntimeError(f"Embedding/metadata mismatch for {source_key}")
        _atomic_save_npy(embedding_path, embedding_array)
        _atomic_write_jsonl(metadata_path, source_rows)
        all_embeddings.append(embedding_array)
        all_rows.extend(source_rows)
        print(
            f"features source={source_number}/{len(grouped)} cached=false "
            f"windows={len(source_rows)} key={source_key}",
            flush=True,
        )

    embeddings = np.concatenate(all_embeddings, axis=0)
    if embeddings.shape[0] != len(all_rows):
        raise RuntimeError("Combined embedding/metadata row count mismatch")
    if write_combined:
        _atomic_save_npy(output_dir / "embeddings" / "beats.npy", embeddings)
    return embeddings, all_rows


def _remap_labels_by_size(labels: np.ndarray) -> np.ndarray:
    counts = Counter(int(label) for label in labels if int(label) >= 0)
    mapping = {
        original: new
        for new, (original, _count) in enumerate(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )
    }
    return np.asarray([mapping.get(int(label), -1) for label in labels], dtype=np.int32)


def cluster_embeddings(
    embeddings: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    config: ClusterConfig,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Run PCA, HDBSCAN, and a complete-coverage MiniBatchKMeans fallback."""
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        raise ValueError("embeddings must be [rows, dimensions] and align with metadata")
    sample_count = embeddings.shape[0]
    if sample_count < 2:
        raise ValueError("At least two candidate windows are required for clustering")
    pca_components = min(config.pca_components, embeddings.shape[1], sample_count - 1)
    pca = PCA(
        n_components=pca_components,
        svd_solver="randomized",
        random_state=config.random_state,
    )
    reduced = pca.fit_transform(embeddings).astype(np.float32, copy=False)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    reduced = reduced / np.maximum(norms, 1e-12)
    _atomic_save_npy(output_dir / "embeddings" / "beats_pca.npy", reduced)
    np.savez_compressed(
        output_dir / "embeddings" / "pca_model.npz",
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )

    minimum_cluster = min(config.hdbscan_min_cluster_size, max(2, sample_count // 2))
    minimum_samples = min(config.hdbscan_min_samples, minimum_cluster)
    hdbscan = HDBSCAN(
        min_cluster_size=minimum_cluster,
        min_samples=minimum_samples,
        metric="euclidean",
        n_jobs=-1,
        cluster_selection_method="eom",
        copy=True,
    )
    hdbscan_labels = _remap_labels_by_size(hdbscan.fit_predict(reduced))
    hdbscan_affinity = np.asarray(hdbscan.probabilities_, dtype=np.float32)

    kmeans_count = min(config.kmeans_clusters, sample_count)
    kmeans = MiniBatchKMeans(
        n_clusters=kmeans_count,
        batch_size=min(2048, max(256, sample_count)),
        n_init="auto",
        random_state=config.random_state,
        reassignment_ratio=0.005,
    )
    raw_kmeans_labels = kmeans.fit_predict(reduced)
    kmeans_labels = _remap_labels_by_size(raw_kmeans_labels)

    hdbscan_distances = np.full(sample_count, np.nan, dtype=np.float32)
    for label in sorted(set(hdbscan_labels) - {-1}):
        indices = np.flatnonzero(hdbscan_labels == label)
        centroid = reduced[indices].mean(axis=0)
        hdbscan_distances[indices] = np.linalg.norm(reduced[indices] - centroid, axis=1)

    kmeans_distances = np.zeros(sample_count, dtype=np.float32)
    for label in sorted(set(kmeans_labels)):
        indices = np.flatnonzero(kmeans_labels == label)
        centroid = reduced[indices].mean(axis=0)
        kmeans_distances[indices] = np.linalg.norm(reduced[indices] - centroid, axis=1)

    enriched_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        hdb_label = int(hdbscan_labels[index])
        enriched = {
            **row,
            "hdbscan_cluster": None if hdb_label < 0 else f"c{hdb_label:04d}",
            "hdbscan_affinity": round(float(hdbscan_affinity[index]), 6),
            "hdbscan_distance": (
                None
                if not np.isfinite(hdbscan_distances[index])
                else round(float(hdbscan_distances[index]), 6)
            ),
            "kmeans_cluster": f"k{int(kmeans_labels[index]):04d}",
            "kmeans_distance": round(float(kmeans_distances[index]), 6),
        }
        enriched_rows.append(enriched)
    _atomic_write_jsonl(output_dir / "windows.jsonl", enriched_rows)

    hdbscan_cluster_count = len(set(hdbscan_labels) - {-1})
    noise_count = int(np.sum(hdbscan_labels < 0))
    summary = {
        "pca_components": pca_components,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "hdbscan_clusters": hdbscan_cluster_count,
        "hdbscan_noise_windows": noise_count,
        "hdbscan_noise_ratio": noise_count / sample_count,
        "kmeans_clusters": len(set(kmeans_labels)),
    }
    label_matrix = np.column_stack(
        [hdbscan_labels, hdbscan_affinity, hdbscan_distances, kmeans_labels, kmeans_distances]
    ).astype(np.float32)
    _atomic_save_npy(output_dir / "embeddings" / "cluster_assignments.npy", label_matrix)
    return reduced, enriched_rows, summary


def _select_representatives(
    indices: np.ndarray,
    distances: np.ndarray,
    rows: Sequence[dict[str, Any]],
    *,
    near_count: int,
    boundary_count: int,
) -> list[tuple[int, str]]:
    selected: list[tuple[int, str]] = []
    selected_indices: set[int] = set()
    used_sources: set[str] = set()

    def select_from(order: Sequence[int], count: int, role: str) -> None:
        candidates = [int(index) for index in order if int(index) not in selected_indices]
        for require_new_source in (True, False):
            for index in candidates:
                if len([item for item in selected if item[1] == role]) >= count:
                    return
                source_uid = str(rows[index]["source_uid"])
                if require_new_source and source_uid in used_sources:
                    continue
                selected.append((index, role))
                selected_indices.add(index)
                used_sources.add(source_uid)

    ordered_near = indices[np.argsort(distances[indices])]
    ordered_boundary = indices[np.argsort(distances[indices])[::-1]]
    select_from(ordered_near, near_count, "near")
    select_from(ordered_boundary, boundary_count, "boundary")
    return selected


def _export_audio_window(row: dict[str, Any], output_path: Path) -> np.ndarray:
    source_path = Path(str(row["source_audio"]))
    with sf.SoundFile(source_path) as reader:
        sample_rate = int(reader.samplerate)
        start_frame = max(0, int(math.floor(float(row["start"]) * sample_rate)))
        end_frame = min(len(reader), int(math.ceil(float(row["end"]) * sample_rate)))
        reader.seek(start_frame)
        audio = reader.read(end_frame - start_frame, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    waveform = torch.from_numpy(mono.copy())
    if sample_rate != OUTPUT_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, OUTPUT_SAMPLE_RATE)
    output = waveform.numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, output, OUTPUT_SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return output


def _cluster_summary(
    cluster_id: str,
    indices: np.ndarray,
    rows: Sequence[dict[str, Any]],
    distances: np.ndarray,
) -> dict[str, Any]:
    source_counts = Counter(str(rows[index]["source_uid"]) for index in indices)
    total_seconds = sum(float(rows[index]["duration"]) for index in indices)
    rms_values = [float(rows[index]["rms_dbfs"]) for index in indices]
    return {
        "cluster_id": cluster_id,
        "member_count": int(indices.size),
        "total_seconds": round(total_seconds, 3),
        "source_count": len(source_counts),
        "dominant_source_ratio": max(source_counts.values()) / int(indices.size),
        "mean_rms_dbfs": round(float(np.mean(rms_values)), 3),
        "mean_distance": round(float(np.mean(distances[indices])), 6),
    }


def _export_cluster_set(
    *,
    kind: str,
    labels: np.ndarray,
    distances: np.ndarray,
    rows: list[dict[str, Any]],
    output_dir: Path,
    config: ClusterConfig,
) -> list[dict[str, Any]]:
    root = output_dir / "clusters" / kind
    if root.is_dir():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for label in sorted({int(value) for value in labels if int(value) >= 0}):
        cluster_id = ("c" if kind == "hdbscan" else "k") + f"{label:04d}"
        indices = np.flatnonzero(labels == label)
        summary = _cluster_summary(cluster_id, indices, rows, distances)
        cluster_dir = root / cluster_id
        representatives_dir = cluster_dir / "representatives"
        representatives = _select_representatives(
            indices,
            distances,
            rows,
            near_count=config.representatives_near,
            boundary_count=config.representatives_boundary,
        )
        preview_parts: list[np.ndarray] = []
        playlist: list[str] = ["#EXTM3U"]
        representative_rows: list[dict[str, Any]] = []
        for rank, (index, role) in enumerate(representatives, 1):
            row = rows[index]
            filename = f"{rank:02d}_{role}_{row['id']}.flac"
            audio = _export_audio_window(row, representatives_dir / filename)
            preview_parts.append(audio)
            playlist.extend([f"#EXTINF:{float(row['duration']):.3f},{row['id']}", filename])
            representative_rows.append(
                {
                    "rank": rank,
                    "role": role,
                    "id": row["id"],
                    "source_uid": row["source_uid"],
                    "source_audio": row["source_audio"],
                    "start": row["start"],
                    "end": row["end"],
                    "duration": row["duration"],
                    "distance": round(float(distances[index]), 6),
                    "audio": (representatives_dir / filename).as_posix(),
                }
            )
        silence = np.zeros(round(0.35 * OUTPUT_SAMPLE_RATE), dtype=np.float32)
        if preview_parts:
            preview = np.concatenate(
                [part for audio in preview_parts for part in (audio, silence)][:-1]
            )
            sf.write(
                cluster_dir / "preview.flac",
                preview,
                OUTPUT_SAMPLE_RATE,
                format="FLAC",
                subtype="PCM_16",
            )
        _atomic_write_text(representatives_dir / "representatives.m3u8", "\n".join(playlist) + "\n")
        summary["representatives"] = representative_rows
        _atomic_write_json(cluster_dir / "summary.json", summary)
        summaries.append(summary)
    return summaries


def export_cluster_review(
    reduced: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    config: ClusterConfig,
) -> list[dict[str, Any]]:
    hdbscan_labels = np.asarray(
        [
            -1 if row["hdbscan_cluster"] is None else int(str(row["hdbscan_cluster"])[1:])
            for row in rows
        ],
        dtype=np.int32,
    )
    hdbscan_distances = np.asarray(
        [np.nan if row["hdbscan_distance"] is None else row["hdbscan_distance"] for row in rows],
        dtype=np.float32,
    )
    kmeans_labels = np.asarray(
        [int(str(row["kmeans_cluster"])[1:]) for row in rows], dtype=np.int32
    )
    kmeans_distances = np.asarray([row["kmeans_distance"] for row in rows], dtype=np.float32)

    all_summaries: list[dict[str, Any]] = []
    if np.any(hdbscan_labels >= 0):
        all_summaries.extend(
            {"kind": "hdbscan", **summary}
            for summary in _export_cluster_set(
                kind="hdbscan",
                labels=hdbscan_labels,
                distances=hdbscan_distances,
                rows=rows,
                output_dir=output_dir,
                config=config,
            )
        )
    all_summaries.extend(
        {"kind": "kmeans", **summary}
        for summary in _export_cluster_set(
            kind="kmeans",
            labels=kmeans_labels,
            distances=kmeans_distances,
            rows=rows,
            output_dir=output_dir,
            config=config,
        )
    )

    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    generated_path = review_dir / "cluster_review.generated.csv"
    fields = [
        "kind",
        "cluster_id",
        "member_count",
        "total_seconds",
        "source_count",
        "dominant_source_ratio",
        "mean_rms_dbfs",
        "preview_audio",
        "event_label",
        "tts_tag",
        "decision",
        "notes",
    ]
    with generated_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for summary in all_summaries:
            kind = str(summary["kind"])
            cluster_id = str(summary["cluster_id"])
            writer.writerow(
                {
                    **summary,
                    "preview_audio": (
                        output_dir / "clusters" / kind / cluster_id / "preview.flac"
                    ).as_posix(),
                    "event_label": "",
                    "tts_tag": "",
                    "decision": "review",
                    "notes": "",
                }
            )

    user_path = review_dir / "cluster_labels.user.csv"
    if not user_path.exists():
        shutil.copyfile(generated_path, user_path)
    override_path = review_dir / "segment_overrides.user.csv"
    if not override_path.exists():
        with override_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "event_label", "tts_tag", "decision", "notes"])
    _atomic_write_json(output_dir / "clusters" / "cluster_summary.json", all_summaries)
    return all_summaries


def _read_mono_window(reader: sf.SoundFile, row: dict[str, Any]) -> torch.Tensor:
    sample_rate = int(reader.samplerate)
    start_frame = max(0, int(math.floor(float(row["start"]) * sample_rate)))
    end_frame = min(len(reader), int(math.ceil(float(row["end"]) * sample_rate)))
    reader.seek(start_frame)
    audio = reader.read(end_frame - start_frame, dtype="float32", always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio window while exporting: {row['id']}")
    waveform = torch.from_numpy(np.mean(audio, axis=1, dtype=np.float32).copy())
    if sample_rate != OUTPUT_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, OUTPUT_SAMPLE_RATE)
    return waveform


def _replace_generated_members(root: Path) -> None:
    if not root.is_dir():
        return
    for members in root.glob("*/members"):
        if members.is_dir():
            shutil.rmtree(members)
    for playlist in root.glob("*/all_members.m3u8"):
        playlist.unlink(missing_ok=True)


def export_clustered_audio_folders(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Materialize every candidate once and hard-link it into both cluster views."""
    canonical_root = output_dir / "clips"
    canonical_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_key"])].append(row)

    clips_written = 0
    clips_reused = 0
    for source_number, source_key in enumerate(sorted(grouped), 1):
        source_rows = sorted(grouped[source_key], key=lambda row: (row["start"], row["end"]))
        source_path = Path(str(source_rows[0]["source_audio"]))
        with sf.SoundFile(source_path) as reader:
            for row in source_rows:
                clip_path = canonical_root / source_key / f"{row['id']}.flac"
                if clip_path.is_file() and clip_path.stat().st_size > 44:
                    clips_reused += 1
                else:
                    waveform = _read_mono_window(reader, row)
                    clip_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(
                        clip_path,
                        waveform.numpy(),
                        OUTPUT_SAMPLE_RATE,
                        format="FLAC",
                        subtype="PCM_16",
                    )
                    clips_written += 1
                row["audio"] = clip_path.relative_to(output_dir).as_posix()
        print(
            f"clips source={source_number}/{len(grouped)} windows={len(source_rows)} "
            f"key={source_key}",
            flush=True,
        )

    hdbscan_root = output_dir / "clusters" / "hdbscan"
    kmeans_root = output_dir / "clusters" / "kmeans"
    _replace_generated_members(hdbscan_root)
    _replace_generated_members(kmeans_root)
    playlists: dict[tuple[str, str], list[str]] = defaultdict(lambda: ["#EXTM3U"])
    hardlinks_created = 0
    copies_created = 0

    for row in rows:
        canonical_path = output_dir / str(row["audio"])
        assignments = [
            ("kmeans", str(row["kmeans_cluster"])),
            (
                "hdbscan",
                "noise" if row["hdbscan_cluster"] is None else str(row["hdbscan_cluster"]),
            ),
        ]
        for kind, cluster_id in assignments:
            cluster_dir = output_dir / "clusters" / kind / cluster_id
            members_dir = cluster_dir / "members"
            members_dir.mkdir(parents=True, exist_ok=True)
            member_path = members_dir / canonical_path.name
            if not member_path.exists():
                try:
                    os.link(canonical_path, member_path)
                    hardlinks_created += 1
                except OSError:
                    shutil.copy2(canonical_path, member_path)
                    copies_created += 1
            playlists[(kind, cluster_id)].extend(
                [
                    f"#EXTINF:{float(row['duration']):.3f},{row['id']}",
                    f"members/{member_path.name}",
                ]
            )

    for (kind, cluster_id), lines in playlists.items():
        _atomic_write_text(
            output_dir / "clusters" / kind / cluster_id / "all_members.m3u8",
            "\n".join(lines) + "\n",
        )
    _atomic_write_jsonl(output_dir / "windows.jsonl", rows)
    return {
        "canonical_clips": len(rows),
        "clips_written": clips_written,
        "clips_reused": clips_reused,
        "hardlinks_created": hardlinks_created,
        "fallback_copies_created": copies_created,
        "cluster_member_folders": len(playlists),
    }


def _embedding_map_html(points_json: str) -> str:
    template = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VAD外音声クラスターマップ</title>
<style>
:root{color-scheme:dark;background:#101218;color:#e9eef7;font-family:system-ui,sans-serif}
body{margin:0;padding:18px}h1{font-size:20px;margin:0 0 8px}.note{color:#aeb8c8;font-size:13px}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:14px 0}
button{background:#263247;color:#fff;border:1px solid #52627c;border-radius:7px;padding:7px 12px;cursor:pointer}
button.active{background:#3867d6}.layout{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:14px}
.panel{background:#171b24;border:1px solid #293142;border-radius:10px;padding:10px}
canvas{width:100%;height:72vh;display:block;background:#0b0d12;border-radius:7px;cursor:crosshair}
#details{white-space:pre-wrap;word-break:break-word;font-size:13px;min-height:150px}audio{width:100%;margin-top:10px}
#tooltip{position:fixed;display:none;pointer-events:none;background:#050608e8;border:1px solid #66738a;border-radius:6px;padding:6px 8px;font-size:12px;max-width:360px;z-index:10}
@media(max-width:900px){.layout{grid-template-columns:1fr}canvas{height:60vh}}
</style>
</head>
<body>
<h1>VAD外音声クラスターマップ</h1>
<div class="note">BEATs埋め込みのPCA二次元投影です。近さは確認用の目安で、実際のクラスタリングは64次元で行っています。点をクリックすると音声を再生できます。</div>
<div class="toolbar"><button id="hdb" class="active">HDBSCAN</button><button id="km">KMeans</button><span id="count"></span></div>
<div class="layout"><div class="panel"><canvas id="map"></canvas></div><div class="panel"><strong>選択した候補</strong><div id="details">点にカーソルを合わせてクリックしてください。</div><audio id="player" controls preload="none"></audio></div></div>
<div id="tooltip"></div>
<script>
const points=__POINTS__;
const canvas=document.getElementById('map'),ctx=canvas.getContext('2d'),tip=document.getElementById('tooltip');
const details=document.getElementById('details'),player=document.getElementById('player');let mode='h',screen=[];
function hashColor(label){if(label===null||label==='noise')return '#69707c';let text=String(label),match=text.match(/\d+/),n=match?Number(match[0]):[...text].reduce((v,c)=>(v*31+c.charCodeAt(0))>>>0,0);return `hsl(${(n*137.508+18)%360} 72% 58%)`}
function resize(){const r=canvas.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);ctx.setTransform(d,0,0,d,0,0);draw()}
function draw(){const w=canvas.clientWidth,h=canvas.clientHeight,pad=18;ctx.clearRect(0,0,w,h);let xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);screen=[];for(let i=0;i<points.length;i++){let p=points[i],x=pad+(p[0]-xmin)/(xmax-xmin||1)*(w-2*pad),y=h-pad-(p[1]-ymin)/(ymax-ymin||1)*(h-2*pad),lab=mode==='h'?(p[2]??'noise'):p[3];screen.push([x,y]);ctx.fillStyle=hashColor(lab);ctx.globalAlpha=.72;ctx.beginPath();ctx.arc(x,y,2.15,0,Math.PI*2);ctx.fill()}ctx.globalAlpha=1;document.getElementById('count').textContent=`${points.length.toLocaleString()} windows`}
function nearest(ev){const r=canvas.getBoundingClientRect(),x=ev.clientX-r.left,y=ev.clientY-r.top;let best=-1,bd=64;for(let i=0;i<screen.length;i++){let dx=screen[i][0]-x,dy=screen[i][1]-y,d=dx*dx+dy*dy;if(d<bd){bd=d;best=i}}return best}
canvas.addEventListener('mousemove',ev=>{let i=nearest(ev);if(i<0){tip.style.display='none';return}let p=points[i],lab=mode==='h'?(p[2]??'noise'):p[3];tip.textContent=`${lab} | ${p[4]} | ${p[7].toFixed(3)}–${p[8].toFixed(3)}s | ${p[9].toFixed(1)} dBFS`;tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px'});
canvas.addEventListener('mouseleave',()=>tip.style.display='none');
canvas.addEventListener('click',ev=>{let i=nearest(ev);if(i<0)return;let p=points[i];details.textContent=`ID: ${p[4]}\nHDBSCAN: ${p[2]??'noise'}\nKMeans: ${p[3]}\nSource: ${p[6]}\nTime: ${p[7].toFixed(3)}–${p[8].toFixed(3)} sec\nRMS: ${p[9].toFixed(2)} dBFS`;player.src='../'+p[5];player.play().catch(()=>{})});
for(const [id,value] of [['hdb','h'],['km','k']])document.getElementById(id).onclick=()=>{mode=value;document.getElementById('hdb').classList.toggle('active',mode==='h');document.getElementById('km').classList.toggle('active',mode==='k');draw()};
addEventListener('resize',resize);resize();
</script>
</body></html>"""
    return template.replace("__POINTS__", points_json)


def _cluster_dashboard_html(summaries_json: str) -> str:
    template = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VAD外クラスタ一覧</title>
<style>:root{color-scheme:dark;background:#101218;color:#edf2fa;font-family:system-ui,sans-serif}body{margin:0;padding:18px}h1{font-size:21px}.note{color:#aeb8c8}.tools{display:flex;gap:9px;margin:14px 0}button{padding:7px 12px;border-radius:7px;border:1px solid #53627a;background:#263247;color:#fff}button.active{background:#3867d6}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:11px}.card{background:#171b24;border:1px solid #2c3547;border-radius:10px;padding:12px}.meta{color:#b9c4d4;font-size:13px;line-height:1.55}audio{width:100%;margin:9px 0}a{color:#7fb0ff}</style></head>
<body><h1>VAD外音声クラスタ一覧</h1><div class="note">代表音は中心3件＋境界2件です。ラベルは <code>review/cluster_labels.user.csv</code> に記入してください。</div>
<div class="tools"><button id="h" class="active">HDBSCAN</button><button id="k">KMeans</button><a href="embedding_map.html">散布図を開く</a></div><div id="grid" class="grid"></div>
<script>const data=__SUMMARIES__;let mode='hdbscan';const grid=document.getElementById('grid');function render(){grid.innerHTML='';for(const s of data.filter(x=>x.kind===mode)){let card=document.createElement('div');card.className='card';let base=`../clusters/${s.kind}/${s.cluster_id}`;card.innerHTML=`<strong>${s.kind} / ${s.cluster_id}</strong><div class="meta">${s.member_count.toLocaleString()}件・${s.total_seconds.toFixed(1)}秒・${s.source_count}音源<br>平均RMS ${s.mean_rms_dbfs.toFixed(1)} dBFS・最大音源比 ${(s.dominant_source_ratio*100).toFixed(1)}%</div><audio controls preload="none" src="${base}/preview.flac"></audio><a href="${base}/members/">全メンバーのフォルダ</a>`;grid.appendChild(card)}}for(const [id,m] of [['h','hdbscan'],['k','kmeans']])document.getElementById(id).onclick=()=>{mode=m;document.getElementById('h').classList.toggle('active',mode==='hdbscan');document.getElementById('k').classList.toggle('active',mode==='kmeans');render()};render();</script></body></html>"""
    return template.replace("__SUMMARIES__", summaries_json)


def write_visualizations(
    reduced: np.ndarray,
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
) -> None:
    if reduced.shape[1] < 2:
        coordinates = np.column_stack([reduced[:, 0], np.zeros(reduced.shape[0])])
    else:
        coordinates = reduced[:, :2]
    points = [
        [
            round(float(coordinates[index, 0]), 6),
            round(float(coordinates[index, 1]), 6),
            row.get("hdbscan_cluster"),
            row.get("kmeans_cluster"),
            row["id"],
            row.get("audio", ""),
            row["source_uid"],
            float(row["start"]),
            float(row["end"]),
            float(row["rms_dbfs"]),
        ]
        for index, row in enumerate(rows)
    ]
    points_json = json.dumps(points, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    summaries_json = json.dumps(list(summaries), ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    visualization_dir = output_dir / "visualization"
    _atomic_write_text(
        visualization_dir / "embedding_map.html",
        _embedding_map_html(points_json),
    )
    _atomic_write_text(
        visualization_dir / "cluster_dashboard.html",
        _cluster_dashboard_html(summaries_json),
    )


def write_output_readme(output_dir: Path) -> None:
    content = """# VAD外音声の自然イベント・BEATsクラスタリング

Silero VADの `vad_regions` に含まれない全区間を、log-mel周波数変化と相対的な
音量の谷から自然イベントへ分割し、Microsoft BEATs Iter3+ AS2Mの埋め込みで
クラスタリングした結果です。低音量を理由に音声を除外していません。

長い自然イベントは、BEATs内部だけ5秒以下へ分けて長さ加重平均しています。
ユーザーが確認・学習に使う音声は自然イベントのままで、5秒で切り捨てていません。

## 最初に確認するもの

1. `visualization/cluster_dashboard.html`: 代表音を比較する一覧画面
2. `visualization/embedding_map.html`: 点をクリックして再生できる二次元散布図
3. `clusters/hdbscan/*/members/`: HDBSCAN別に分けた全音声
4. `clusters/kmeans/*/members/`: 外れ値も含めKMeans別に分けた全音声
5. `review/cluster_labels.user.csv`: 人が編集するラベル表

各previewは「クラスタ中心3件 + 境界2件」を0.35秒の無音で連結しています。
`*.user.csv` は再実行しても上書きされません。

全候補のFLAC本体は `clips/` に一度だけ保存しています。クラスタ別の `members/` は
同一ファイルへのNTFSハードリンクなので、HDBSCANとKMeansの両方へ分けても音声容量は
三重にはなりません。

## ファイル

- `windows.jsonl`: 全候補の音声、元音源、開始・終了秒、音量、クラスタ番号
- `embeddings/beats.npy`: L2正規化済みBEATs埋め込み
- `embeddings/beats_pca.npy`: クラスタリングに使用したPCA特徴
- `clusters/cluster_summary.json`: 全クラスタの統計
- `excluded/excluded.jsonl`: 短すぎてBEATsへ渡せなかった隙間
- `review/cluster_review.generated.csv`: 毎回再生成される確認表
- `review/cluster_labels.user.csv`: 人手ラベルの正本

HDBSCANのcluster IDとKMeansのcluster IDは別物です。最初はHDBSCANを確認し、
`hdbscan_cluster=null` の候補はKMeans側で確認してください。
"""
    _atomic_write_text(output_dir / "README.md", content)


def run_nonverbal_clustering(
    *,
    data_root: Path,
    raw_response_dir: Path,
    output_dir: Path,
    beats_code_dir: Path,
    checkpoint_path: Path,
    device: str,
    batch_size: int,
    feature_config: FeatureConfig,
    cluster_config: ClusterConfig,
    max_sources: int | None = None,
    force_features: bool = False,
    source_shard_index: int = 0,
    source_shard_count: int = 1,
    features_only: bool = False,
    export_all_clips: bool = True,
) -> dict[str, Any]:
    """Execute candidate discovery, embedding extraction, clustering, and review export."""
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates, excluded, inventory = load_vad_complements(
        raw_response_dir,
        data_root=data_root,
        feature_config=feature_config,
        max_sources=max_sources,
        source_shard_index=source_shard_index,
        source_shard_count=source_shard_count,
        segmentation_device=device,
    )
    metadata_dir = (
        output_dir / "workers" / f"shard_{source_shard_index:02d}_of_{source_shard_count:02d}"
        if features_only
        else output_dir
    )
    _atomic_write_jsonl(metadata_dir / "candidates.jsonl", (asdict(item) for item in candidates))
    _atomic_write_jsonl(metadata_dir / "excluded" / "excluded.jsonl", excluded)
    print(
        f"inventory sources={inventory['sources']} candidates={len(candidates)} "
        f"outside_vad_hours={inventory['outside_vad_hours']:.3f}",
        flush=True,
    )
    embeddings, rows = extract_beats_embeddings(
        candidates,
        output_dir=output_dir,
        feature_config=feature_config,
        beats_code_dir=beats_code_dir,
        checkpoint_path=checkpoint_path,
        device_name=device,
        batch_size=batch_size,
        force=force_features,
        write_combined=not features_only,
    )
    if features_only:
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "features_only",
            "source_shard_index": source_shard_index,
            "source_shard_count": source_shard_count,
            "model": BEATS_MODEL_NAME,
            "checkpoint_sha256": BEATS_CHECKPOINT_SHA256,
            "device": device,
            "feature_config": asdict(feature_config),
            **inventory,
            "embedding_rows": int(embeddings.shape[0]),
            "embedding_dimensions": int(embeddings.shape[1]),
        }
        _atomic_write_json(metadata_dir / "summary.json", summary)
        return summary
    reduced, enriched_rows, cluster_summary = cluster_embeddings(
        embeddings,
        rows,
        output_dir=output_dir,
        config=cluster_config,
    )
    summaries = export_cluster_review(
        reduced,
        enriched_rows,
        output_dir=output_dir,
        config=cluster_config,
    )
    clip_export_summary: dict[str, Any] = {}
    if export_all_clips:
        clip_export_summary = export_clustered_audio_folders(
            enriched_rows,
            output_dir=output_dir,
        )
    write_visualizations(
        reduced,
        enriched_rows,
        summaries,
        output_dir=output_dir,
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": BEATS_MODEL_NAME,
        "checkpoint_sha256": BEATS_CHECKPOINT_SHA256,
        "device": device,
        "feature_config": asdict(feature_config),
        "cluster_config": asdict(cluster_config),
        **inventory,
        **cluster_summary,
        **clip_export_summary,
        "review_clusters": len(summaries),
        "visualizations": [
            "visualization/embedding_map.html",
            "visualization/cluster_dashboard.html",
        ],
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    _atomic_write_json(
        output_dir / "config.json",
        {
            "data_root": data_root.resolve().as_posix(),
            "raw_response_dir": raw_response_dir.resolve().as_posix(),
            "beats_code_dir": beats_code_dir.resolve().as_posix(),
            "checkpoint_path": checkpoint_path.resolve().as_posix(),
            "feature_config": asdict(feature_config),
            "cluster_config": asdict(cluster_config),
        },
    )
    write_output_readme(output_dir)
    return summary
