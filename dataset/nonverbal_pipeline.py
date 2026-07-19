"""Mine nonverbal event clips from the VAD complement, without any classifier.

Silero VAD only marks voice-like regions, so moans and lip noise largely live
in the gaps between speech regions. This module cuts those gaps at natural
acoustic boundaries, drops silence, and emits 5-20 s FLAC clips shaped exactly
like speech rows (empty ``text``); the shared ASR + LLM-correction stages then
transcribe and classify them, making the LLM ``category`` the single label
authority for the whole dataset.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from dataset._io_utils import atomic_write_jsonl
from dataset.acoustic_segmentation import (
    AcousticSegmentationConfig,
    segment_acoustic_primitives,
)
from dataset.speech_pipeline import (
    AudioSource,
    SileroVADConfig,
    SpeechSegmentationConfig,
    _manifest_path,
    extract_clip,
    load_cached_vad,
    review_reasons,
)
from dataset.speech_pipeline import (
    Segment as ClipSegment,
)

ANALYSIS_SAMPLE_RATE = 16_000
# Events quieter than this (relative to full scale at 16 kHz) are treated as
# silence and never become clips; ASMR runs quiet, so keep the floor low.
MINIMUM_EVENT_RMS = 10.0 ** (-60.0 / 20.0)
MINIMUM_GAP_SECONDS = 5.0


def _vad_complement(
    regions: Sequence[tuple[float, float]],
    *,
    duration: float,
    minimum_gap_seconds: float,
) -> list[tuple[float, float]]:
    """Return the parts of [0, duration] that VAD did not mark as speech."""
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in sorted(regions):
        if start - cursor >= minimum_gap_seconds:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor >= minimum_gap_seconds:
        gaps.append((cursor, duration))
    return gaps


def _read_analysis_segment(path: Path, *, start: float, end: float) -> np.ndarray:
    """Read [start, end] as mono 16 kHz float32 for boundary/energy analysis."""
    import torchaudio

    with sf.SoundFile(path) as reader:
        rate = int(reader.samplerate)
        start_frame = max(0, int(start * rate))
        reader.seek(start_frame)
        audio = reader.read(
            max(0, int(end * rate) - start_frame), dtype="float32", always_2d=True
        )
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if rate == ANALYSIS_SAMPLE_RATE:
        return mono
    resampled = torchaudio.functional.resample(
        torch.from_numpy(mono.copy()), rate, ANALYSIS_SAMPLE_RATE
    )
    return resampled.numpy()


def _pack_events(
    kept: Sequence[tuple[float, float]],
    *,
    max_span: float,
    min_seconds: float,
) -> list[tuple[float, float]]:
    """Greedily join consecutive loud primitives into >=min, <=max_span events."""
    events: list[tuple[float, float]] = []
    current: tuple[float, float] | None = None
    previous_end: float | None = None
    for start, end in kept:
        contiguous = previous_end is not None and math.isclose(
            start, previous_end, abs_tol=0.05
        )
        if current is not None and contiguous and end - current[0] <= max_span:
            current = (current[0], end)
        else:
            if current is not None and current[1] - current[0] >= min_seconds:
                events.append(current)
            cursor = start
            while end - cursor > max_span:
                events.append((cursor, cursor + max_span))
                cursor += max_span
            current = (cursor, end)
        previous_end = end
    if current is not None and current[1] - current[0] >= min_seconds:
        events.append(current)
    return events


def process_source(
    source: AudioSource,
    *,
    output_dir: Path,
    vad_config: SileroVADConfig,
    segmentation_config: SpeechSegmentationConfig,
    acoustic_config: AcousticSegmentationConfig,
    project_root: Path,
    rebuild_clips: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Cut nonverbal event clips for one source; returns (source_row, rows, errors)."""
    source_row = source.metadata()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    regions = load_cached_vad(output_dir, source, config=vad_config)
    if regions is None:
        source_row["status"] = "error"
        source_row["reason"] = "missing_vad_cache"
        errors.append(
            {
                "source_uid": source.source_id,
                "source_audio": _manifest_path(source.path, project_root),
                "reason": "missing_vad_cache",
                "error": "run the speech stage first to populate vad_responses",
            }
        )
        return source_row, rows, errors

    gaps = _vad_complement(
        regions,
        duration=source.duration,
        minimum_gap_seconds=MINIMUM_GAP_SECONDS,
    )
    events: list[tuple[float, float]] = []
    try:
        for gap_start, gap_end in gaps:
            waveform = _read_analysis_segment(source.path, start=gap_start, end=gap_end)
            if waveform.size == 0:
                continue
            primitives = segment_acoustic_primitives(
                torch.from_numpy(waveform),
                ANALYSIS_SAMPLE_RATE,
                config=acoustic_config,
            )
            kept: list[tuple[float, float]] = []
            for primitive in primitives:
                begin = int(primitive.start * ANALYSIS_SAMPLE_RATE)
                finish = int(primitive.end * ANALYSIS_SAMPLE_RATE)
                piece = waveform[begin:finish]
                if piece.size == 0:
                    continue
                rms = float(np.sqrt(np.mean(np.square(piece, dtype=np.float64))))
                if rms < MINIMUM_EVENT_RMS:
                    continue
                kept.append((gap_start + primitive.start, gap_start + primitive.end))
            events.extend(
                _pack_events(
                    kept,
                    max_span=segmentation_config.max_span_seconds,
                    min_seconds=segmentation_config.min_seconds,
                )
            )
    except (OSError, RuntimeError, ValueError) as exc:
        source_row["status"] = "error"
        source_row["reason"] = "segmentation_error"
        errors.append(
            {
                "source_uid": source.source_id,
                "source_audio": _manifest_path(source.path, project_root),
                "reason": "segmentation_error",
                "error": str(exc),
            }
        )
        return source_row, rows, errors

    written = 0
    try:
        with sf.SoundFile(source.path) as reader:
            for event_start, event_end in events:
                start = max(0.0, event_start - segmentation_config.padding_seconds)
                end = min(source.duration, event_end + segmentation_config.padding_seconds)
                start_ms = round(start * 1000)
                end_ms = round(end * 1000)
                clip_id = f"{source.source_id}_nv_{start_ms:010d}_{end_ms:010d}"
                clip_path = (
                    output_dir / "clips_nonverbal" / source.source_id / f"{clip_id}.flac"
                )
                try:
                    stats = extract_clip(
                        reader,
                        clip_path,
                        start=start,
                        end=end,
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
                segment = ClipSegment(start=start, end=end)
                reasons = review_reasons(segment, segmentation_config, audio_stats=stats)
                rows.append(
                    {
                        "id": clip_id,
                        "audio": _manifest_path(clip_path, project_root),
                        "text": "",
                        "source_uid": source.source_id,
                        "speaker_id": source.speaker_id,
                        "duration": round(segment.duration, 6),
                        "start": round(start, 6),
                        "end": round(end, 6),
                        "source_audio": _manifest_path(source.path, project_root),
                        "origin": "vad_complement",
                        "status": "review" if reasons else "train",
                        "review_reasons": reasons,
                        "sample_rate": source.sample_rate,
                        "channels": 1,
                        "peak": round(stats.peak, 7),
                        "rms": round(stats.rms, 7),
                        "clipping_ratio": round(stats.clipping_ratio, 9),
                    }
                )
                written += 1
    except (OSError, RuntimeError) as exc:
        errors.append(
            {
                "source_uid": source.source_id,
                "source_audio": _manifest_path(source.path, project_root),
                "reason": "source_decode_error",
                "error": str(exc),
            }
        )

    source_row["status"] = "processed" if written else "empty"
    source_row["events"] = written
    return source_row, rows, errors


_POOL_STATE: dict[str, Any] = {}


def _pool_init(
    output_dir: str,
    vad_config: SileroVADConfig,
    segmentation_config: SpeechSegmentationConfig,
    acoustic_config: AcousticSegmentationConfig,
    project_root: str,
    rebuild_clips: bool,
) -> None:
    _POOL_STATE.update(
        output_dir=Path(output_dir),
        vad_config=vad_config,
        segmentation_config=segmentation_config,
        acoustic_config=acoustic_config,
        project_root=Path(project_root),
        rebuild_clips=rebuild_clips,
    )


def _pool_process(
    source: AudioSource,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    state = _POOL_STATE
    return process_source(
        source,
        output_dir=state["output_dir"],
        vad_config=state["vad_config"],
        segmentation_config=state["segmentation_config"],
        acoustic_config=state["acoustic_config"],
        project_root=state["project_root"],
        rebuild_clips=state["rebuild_clips"],
    )


def build_nonverbal_dataset(
    sources: Sequence[AudioSource],
    output_dir: Path,
    *,
    vad_config: SileroVADConfig | None = None,
    segmentation_config: SpeechSegmentationConfig | None = None,
    acoustic_config: AcousticSegmentationConfig | None = None,
    project_root: Path,
    workers: int = 1,
    rebuild_clips: bool = False,
) -> dict[str, Any]:
    """Cut VAD-complement event clips and write ``nonverbal_events.jsonl``."""
    output_dir = output_dir.expanduser().resolve()
    vad_config = vad_config or SileroVADConfig()
    segmentation_config = segmentation_config or SpeechSegmentationConfig()
    acoustic_config = acoustic_config or AcousticSegmentationConfig()

    all_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    source_errors: list[dict[str, Any]] = []

    def consume(index: int, result: tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]) -> None:
        source_row, rows, errors = result
        source_rows.append(source_row)
        all_rows.extend(rows)
        source_errors.extend(errors)
        print(
            f"nonverbal {index}/{len(sources)} source={source_row['source_id']} "
            f"events={source_row.get('events', 0)}",
            flush=True,
        )

    if workers <= 1:
        for index, source in enumerate(sources, 1):
            consume(
                index,
                process_source(
                    source,
                    output_dir=output_dir,
                    vad_config=vad_config,
                    segmentation_config=segmentation_config,
                    acoustic_config=acoustic_config,
                    project_root=project_root,
                    rebuild_clips=rebuild_clips,
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
                acoustic_config,
                str(project_root),
                rebuild_clips,
            ),
        ) as pool:
            for index, result in enumerate(
                pool.map(_pool_process, sources, chunksize=1), 1
            ):
                consume(index, result)

    all_rows.sort(key=lambda row: (str(row["source_uid"]), float(row["start"])))
    train_rows = [row for row in all_rows if row["status"] == "train"]
    atomic_write_jsonl(output_dir / "nonverbal_events.jsonl", all_rows)
    atomic_write_jsonl(output_dir / "nonverbal_sources.jsonl", source_rows)
    if source_errors:
        atomic_write_jsonl(output_dir / "nonverbal_errors.jsonl", source_errors)

    return {
        "sources": len(sources),
        "rows": len(all_rows),
        "train": len(train_rows),
        "review": len(all_rows) - len(train_rows),
        "errors": len(source_errors),
        "train_seconds": round(sum(float(row["duration"]) for row in train_rows), 3),
    }
