"""Finalize natural nonverbal primitives into reviewable training events.

The expensive BEATs extraction is intentionally kept outside this module.  This
pipeline consumes its resumable per-source shards, applies a local weak classifier,
uses conservative source-scoped label propagation, joins only weakly separated
``aegi``/``chupa`` primitives, and atomically publishes one self-contained dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf
import torch
import torchaudio

from irodori_tts.data_prep.ero_voice_classifier import (
    ERO_LABELS,
    SPACE_REVISION,
    JapaneseEroVoiceClassifier,
)
from irodori_tts.data_prep.nonverbal_labeling import (
    FINAL_LABELS,
    TARGET_LABELS,
    NonverbalLabelingConfig,
    predict_segment_rows,
    propagate_nonverbal_labels,
)
from irodori_tts.data_prep.nonverbal_subclustering import (
    SubclusteringConfig,
    select_representatives_by_cluster,
    subcluster_event_embeddings,
)

PIPELINE_SCHEMA_VERSION = 2
CLASSIFIER_CACHE_SCHEMA_VERSION = 1
PUBLIC_TARGET_LABELS = ("aegi", "chupa")
CLASSIFIER_CLIP_SAMPLE_RATE = 16_000
FINAL_CLIP_SAMPLE_RATE = 48_000


class WeakClassifier(Protocol):
    """Small protocol shared by the real model and test doubles."""

    def predict(self, audio_paths: Sequence[Path], *, batch_size: int) -> list[Any]: ...


VisualizationHook = Callable[
    [Path, Sequence[Mapping[str, Any]], np.ndarray, Mapping[str, Any]],
    Mapping[str, Any] | None,
]


@dataclass(frozen=True)
class ManualSeedConfig:
    """Controls interval-to-primitive matching for human supplied labels."""

    minimum_overlap_ratio: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_overlap_ratio <= 1.0:
            raise ValueError("minimum_overlap_ratio must be in (0, 1]")


@dataclass(frozen=True)
class EventMergeConfig:
    """Conservative rules for rejoining adjacent natural primitives."""

    labels: tuple[str, ...] = ("aegi", "chupa")
    minimum_seconds: float = 5.0
    max_seconds: float = 12.0
    adjacency_tolerance_seconds: float = 0.005
    strong_boundary_score: float = 2.0
    hard_boundary_reasons: tuple[str, ...] = (
        "gap_start",
        "gap_end",
        "pause_valley",
    )

    def __post_init__(self) -> None:
        if not self.labels or any(label not in FINAL_LABELS for label in self.labels):
            raise ValueError("labels must contain supported final labels")
        if self.minimum_seconds <= 0.0 or not math.isfinite(self.minimum_seconds):
            raise ValueError("minimum_seconds must be positive and finite")
        if self.max_seconds <= self.minimum_seconds or not math.isfinite(self.max_seconds):
            raise ValueError("max_seconds must be finite and greater than minimum_seconds")
        if self.adjacency_tolerance_seconds < 0.0:
            raise ValueError("adjacency_tolerance_seconds cannot be negative")
        if not math.isfinite(self.strong_boundary_score):
            raise ValueError("strong_boundary_score must be finite")


@dataclass(frozen=True)
class PipelineConfig:
    """I/O and orchestration settings that are not model thresholds."""

    classifier_batch_size: int = 64
    propagation_scope: str = "source"
    propagate_manual_seeds: bool = False
    lock_confident_usual: bool = True
    reject_target_class_conflicts: bool = True
    keep_work_clips: bool = False
    force_classifier: bool = False
    force_finalize: bool = False
    final_clip_padding_seconds: float = 0.35

    def __post_init__(self) -> None:
        if self.classifier_batch_size <= 0:
            raise ValueError("classifier_batch_size must be positive")
        if self.propagation_scope not in {"source", "global"}:
            raise ValueError("propagation_scope must be 'source' or 'global'")
        if self.final_clip_padding_seconds < 0.0 or not math.isfinite(
            self.final_clip_padding_seconds
        ):
            raise ValueError("final_clip_padding_seconds must be finite and non-negative")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            # On Windows a short-lived reader (preview, indexer, antivirus) can
            # briefly deny replace even though both files are local. Retrying
            # keeps resumable classifier checkpoints from aborting a long run.
            if attempt == 7:
                raise
            time.sleep(0.025 * (attempt + 1))


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
    )


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(
            json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"), default=_json_default)
            + "\n"
            for row in rows
        ),
    )


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


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("source_key", "")).casefold(),
        int(row.get("gap_index", 0)),
        float(row.get("start", 0.0)),
        float(row.get("end", 0.0)),
        str(row.get("id", "")),
    )


def load_feature_shards(
    feature_dir: str | Path,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Load matching JSONL/NPY BEATs shards and restore a stable global order."""
    root = Path(feature_dir)
    shard_dir = root / "embeddings" / "shards"
    canonical_path = root / "windows.jsonl"
    canonical_ids: set[str] | None = None
    if canonical_path.is_file():
        canonical_rows = _read_jsonl(canonical_path)
        canonical_id_list = [str(row.get("id", "")) for row in canonical_rows]
        if any(not segment_id for segment_id in canonical_id_list):
            raise ValueError(f"Missing id in canonical feature manifest: {canonical_path}")
        canonical_ids = set(canonical_id_list)
        if len(canonical_ids) != len(canonical_id_list):
            raise ValueError(f"Duplicate id in canonical feature manifest: {canonical_path}")
    metadata_paths = sorted(shard_dir.glob("*.jsonl"), key=lambda path: path.name.casefold())
    if not metadata_paths:
        raise RuntimeError(f"No feature metadata shards found under: {shard_dir}")
    orphan_arrays = sorted(
        path.name for path in shard_dir.glob("*.npy") if not path.with_suffix(".jsonl").is_file()
    )
    if orphan_arrays:
        raise ValueError(f"Embedding shards without metadata: {orphan_arrays[:10]}")

    records: list[tuple[dict[str, Any], np.ndarray]] = []
    dimensions: set[int] = set()
    seen_ids: set[str] = set()
    shard_summaries: list[dict[str, Any]] = []
    ignored_rows = 0
    ignored_shards = 0
    for metadata_path in metadata_paths:
        embedding_path = metadata_path.with_suffix(".npy")
        if not embedding_path.is_file():
            raise FileNotFoundError(f"Missing embedding shard for {metadata_path.name}")
        rows = _read_jsonl(metadata_path)
        selected_indices = [
            index
            for index, row in enumerate(rows)
            if canonical_ids is None or str(row.get("id", "")) in canonical_ids
        ]
        ignored_rows += len(rows) - len(selected_indices)
        if not selected_indices:
            ignored_shards += 1
            continue
        embeddings = np.load(embedding_path, allow_pickle=False)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(rows) or embeddings.shape[1] < 1:
            raise ValueError(
                f"Shard shape mismatch: {metadata_path.name} has {len(rows)} rows, "
                f"embedding shape is {embeddings.shape}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError(f"Embedding shard contains non-finite values: {embedding_path}")
        dimensions.add(int(embeddings.shape[1]))
        for index in selected_indices:
            row = rows[index]
            segment_id = str(row.get("id", ""))
            if not segment_id:
                raise ValueError(f"Missing id at {metadata_path}:{index + 1}")
            if segment_id in seen_ids:
                raise ValueError(f"Duplicate feature id: {segment_id}")
            seen_ids.add(segment_id)
            records.append((row, np.asarray(embeddings[index], dtype=np.float32)))
        shard_summaries.append(
            {
                "metadata": metadata_path.relative_to(root).as_posix(),
                "embeddings": embedding_path.relative_to(root).as_posix(),
                "rows": len(selected_indices),
            }
        )
    if canonical_ids is not None:
        missing_ids = canonical_ids - seen_ids
        if missing_ids:
            raise ValueError(
                f"Canonical feature ids missing from embedding shards: {sorted(missing_ids)[:10]}"
            )
    if len(dimensions) != 1:
        raise ValueError(f"Feature dimensions differ across shards: {sorted(dimensions)}")
    records.sort(key=lambda item: _row_sort_key(item[0]))
    rows = [row for row, _embedding in records]
    embeddings = np.stack([embedding for _row, embedding in records]).astype(np.float32)
    return (
        rows,
        embeddings,
        {
            "shards": shard_summaries,
            "shard_count": len(shard_summaries),
            "rows": len(rows),
            "dimensions": int(embeddings.shape[1]),
            "canonical_manifest": canonical_path.relative_to(root).as_posix()
            if canonical_ids is not None
            else None,
            "ignored_stale_shards": ignored_shards,
            "ignored_stale_rows": ignored_rows,
        },
    )


def _normalize_path(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.strip()).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.casefold()


def load_manual_seed_intervals(path: str | Path | None) -> list[dict[str, Any]]:
    """Read the documented manual seed schema from JSONL."""
    if path is None:
        return []
    seed_path = Path(path)
    if not seed_path.is_file():
        return []
    intervals = _read_jsonl(seed_path)
    seen_ids: set[str] = set()
    output: list[dict[str, Any]] = []
    for index, interval in enumerate(intervals):
        required = ("source_relative_path", "start", "end", "label")
        missing = [field for field in required if field not in interval]
        if missing:
            raise ValueError(f"Manual interval {index + 1} is missing: {missing}")
        interval_id = str(interval.get("id") or f"manual_{index:05d}")
        if interval_id in seen_ids:
            raise ValueError(f"Duplicate manual interval id: {interval_id}")
        seen_ids.add(interval_id)
        start = float(interval["start"])
        end = float(interval["end"])
        label = str(interval["label"]).strip().lower()
        if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end <= start:
            raise ValueError(f"Manual interval {interval_id} has an invalid time range")
        if label not in FINAL_LABELS:
            raise ValueError(f"Manual interval {interval_id} has unsupported label {label!r}")
        output.append(
            {
                "id": interval_id,
                "source_relative_path": str(interval["source_relative_path"]).replace("\\", "/"),
                "start": start,
                "end": end,
                "label": label,
                "provenance": str(interval.get("provenance") or "manual"),
                "note": str(interval.get("note") or ""),
            }
        )
    return output


def match_manual_seed_intervals(
    primitive_rows: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    *,
    config: ManualSeedConfig | None = None,
) -> tuple[dict[str, str], dict[str, list[str]], list[dict[str, Any]]]:
    """Match intervals by source path and candidate-duration overlap ratio."""
    settings = config or ManualSeedConfig()
    by_source: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(primitive_rows):
        by_source[_normalize_path(str(row.get("source_relative_path", "")))].append((index, row))

    labels: dict[str, str] = {}
    provenance: dict[str, list[str]] = defaultdict(list)
    applications: list[dict[str, Any]] = []
    for interval in intervals:
        source_key = _normalize_path(str(interval["source_relative_path"]))
        matches: list[dict[str, Any]] = []
        for _index, row in by_source.get(source_key, []):
            start = float(row["start"])
            end = float(row["end"])
            overlap = max(
                0.0, min(end, float(interval["end"])) - max(start, float(interval["start"]))
            )
            duration = end - start
            ratio = overlap / duration if duration > 0.0 else 0.0
            if ratio + 1e-12 < settings.minimum_overlap_ratio:
                continue
            segment_id = str(row["id"])
            label = str(interval["label"])
            previous = labels.get(segment_id)
            if previous is not None and previous != label:
                raise ValueError(
                    f"Conflicting manual labels for {segment_id}: {previous!r} and {label!r}"
                )
            labels[segment_id] = label
            provenance[segment_id].append(str(interval["id"]))
            matches.append(
                {
                    "primitive_id": segment_id,
                    "overlap_seconds": round(overlap, 6),
                    "primitive_overlap_ratio": round(ratio, 6),
                }
            )
        applications.append(
            {
                **dict(interval),
                "minimum_overlap_ratio": settings.minimum_overlap_ratio,
                "matched_primitive_count": len(matches),
                "matches": matches,
            }
        )
    return labels, dict(provenance), applications


def propagate_labels_scoped(
    prediction_rows: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
    primitive_rows: Sequence[Mapping[str, Any]],
    *,
    manual_seeds: Mapping[str, str] | None = None,
    config: NonverbalLabelingConfig | None = None,
    scope: str = "source",
    propagate_manual_seeds: bool = False,
    lock_confident_usual: bool = True,
    reject_target_class_conflicts: bool = True,
) -> list[dict[str, Any]]:
    """Run propagation globally or, by default, independently per source file."""
    if scope not in {"source", "global"}:
        raise ValueError("scope must be 'source' or 'global'")
    settings = config or NonverbalLabelingConfig()
    if len(prediction_rows) != len(primitive_rows) or len(prediction_rows) != len(embeddings):
        raise ValueError("predictions, embeddings, and primitive rows must align")
    ids = [str(row.get("id", "")) for row in prediction_rows]
    primitive_ids = [str(row.get("id", "")) for row in primitive_rows]
    if ids != primitive_ids:
        raise ValueError("prediction ids do not match primitive ids in order")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(primitive_rows):
        key = "__global__" if scope == "global" else str(row.get("source_key", ""))
        grouped[key].append(index)
    output: list[dict[str, Any] | None] = [None] * len(prediction_rows)
    manual = dict(manual_seeds or {})
    for key in sorted(grouped, key=str.casefold):
        indices = grouped[key]
        group_ids = {ids[index] for index in indices}
        group_manual = (
            {segment_id: label for segment_id, label in manual.items() if segment_id in group_ids}
            if propagate_manual_seeds
            else {}
        )
        decisions = propagate_nonverbal_labels(
            [prediction_rows[index] for index in indices],
            np.asarray(embeddings[indices], dtype=np.float32),
            manual_seeds=group_manual,
            config=settings,
        )
        for index, decision in zip(indices, decisions, strict=True):
            output[index] = decision
    if any(row is None for row in output):  # pragma: no cover - invariant
        raise RuntimeError("not every primitive received a scoped label")
    finalized = [row for row in output if row is not None]
    for index, (segment_id, prediction) in enumerate(zip(ids, prediction_rows, strict=True)):
        if segment_id in manual:
            decision = finalized[index]
            evidence = dict(decision.get("evidence") or {})
            evidence["direct_seed"] = {
                "seed_id": segment_id,
                "label": manual[segment_id],
                "source": "manual",
                "propagates": propagate_manual_seeds,
            }
            evidence["manual_interval_only"] = not propagate_manual_seeds
            finalized[index] = {
                **decision,
                "final_label": manual[segment_id],
                "label_status": "manual_override",
                "evidence": evidence,
            }
            continue
        raw_label = str(prediction.get("top_label") or "")
        decision = finalized[index]
        if (
            reject_target_class_conflicts
            and decision.get("label_status") == "neighbor_supported"
            and raw_label in TARGET_LABELS
            and decision.get("final_label") in TARGET_LABELS
            and decision.get("final_label") != raw_label
        ):
            evidence = dict(decision.get("evidence") or {})
            evidence["classifier_target_conflict"] = {
                "classifier_label": raw_label,
                "rejected_neighbor_label": decision.get("final_label"),
            }
            finalized[index] = {
                **decision,
                "final_label": "uncertain",
                "label_status": "classifier_target_conflict",
                "evidence": evidence,
            }
            continue
        if (
            lock_confident_usual
            and prediction.get("status", "ok") == "ok"
            and prediction.get("top_label") == "usual"
            and float(prediction.get("top_probability") or 0.0) >= settings.other_prob
            and float(prediction.get("margin") or 0.0) >= settings.other_margin
        ):
            decision = finalized[index]
            evidence = dict(decision.get("evidence") or {})
            evidence["classifier_usual_locked"] = True
            if decision.get("label_status") == "neighbor_supported":
                evidence["rejected_neighbor_label"] = decision.get("final_label")
            finalized[index] = {
                **decision,
                "final_label": "other",
                "label_status": "classifier_other_locked",
                "evidence": evidence,
            }
    return finalized


def _feature_fingerprint(rows: Sequence[Mapping[str, Any]], embeddings: np.ndarray) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = {
            key: row.get(key)
            for key in (
                "id",
                "source_key",
                "source_relative_path",
                "gap_index",
                "start",
                "end",
                "left_boundary_reason",
                "left_boundary_score",
                "right_boundary_reason",
                "right_boundary_score",
            )
        }
        digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        source = Path(str(row.get("source_audio", "")))
        if source.is_file():
            stat = source.stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    digest.update(np.ascontiguousarray(embeddings, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_source_path(row: Mapping[str, Any], data_root: Path) -> Path:
    direct = Path(str(row.get("source_audio", "")))
    if direct.is_file():
        return direct
    relative = Path(str(row.get("source_relative_path", "")))
    resolved = data_root / relative
    if not resolved.is_file():
        raise FileNotFoundError(f"Source audio is missing for {row.get('id')}: {resolved}")
    return resolved


def _read_window(reader: sf.SoundFile, start: float, end: float, sample_rate: int) -> np.ndarray:
    source_rate = int(reader.samplerate)
    start_frame = max(0, int(math.floor(start * source_rate)))
    end_frame = min(len(reader), int(math.ceil(end * source_rate)))
    reader.seek(start_frame)
    samples = reader.read(end_frame - start_frame, dtype="float32", always_2d=True)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError(f"Invalid source window {start:.6f}-{end:.6f}")
    waveform = torch.from_numpy(np.mean(samples, axis=1, dtype=np.float32).copy())
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform.numpy()


def _valid_audio_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 44:
        return False
    try:
        return sf.info(path).frames > 0
    except (RuntimeError, TypeError):
        return False


def _atomic_write_flac(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.flac")
    sf.write(temporary, samples, sample_rate, format="FLAC", subtype="PCM_16")
    temporary.replace(path)


def export_audio_windows(
    rows: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    data_root: Path,
    relative_prefix: Path,
    sample_rate: int,
    padding_seconds: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Export source crops atomically and return rows with root-relative audio paths."""
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row.get("source_key", ""))].append((index, row))
    output: list[dict[str, Any] | None] = [None] * len(rows)
    written = reused = 0
    for source_number, source_key in enumerate(sorted(grouped, key=str.casefold), 1):
        source_rows = sorted(grouped[source_key], key=lambda item: _row_sort_key(item[1]))
        source_path = _resolve_source_path(source_rows[0][1], data_root)
        with sf.SoundFile(source_path) as reader:
            for index, row in source_rows:
                relative = relative_prefix / source_key / f"{row['id']}.flac"
                output_path = root / relative
                if _valid_audio_file(output_path):
                    reused += 1
                else:
                    samples = _read_window(
                        reader,
                        max(0.0, float(row["start"]) - padding_seconds),
                        float(row["end"]) + padding_seconds,
                        sample_rate,
                    )
                    _atomic_write_flac(output_path, samples, sample_rate)
                    written += 1
                output[index] = {**dict(row), "audio": relative.as_posix()}
        print(
            f"event clips source={source_number}/{len(grouped)} "
            f"windows={len(source_rows)} key={source_key}",
            flush=True,
        )
    return [row for row in output if row is not None], {"written": written, "reused": reused}


def attach_audio_paths(
    rows: Sequence[Mapping[str, Any]],
    *,
    relative_prefix: Path,
) -> list[dict[str, Any]]:
    """Attach deterministic relative clip paths without touching the filesystem."""
    return [
        {
            **dict(row),
            "audio": (
                relative_prefix / str(row.get("source_key", "")) / f"{row['id']}.flac"
            ).as_posix(),
        }
        for row in rows
    ]


def filter_public_target_primitives(
    labeled_rows: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
    *,
    target_labels: Sequence[str] = PUBLIC_TARGET_LABELS,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Filter labeled rows and BEATs with one shared index mask for publication."""
    values = np.asarray(embeddings)
    if values.ndim != 2 or values.shape[0] != len(labeled_rows):
        raise ValueError("labeled rows and embeddings must align as [rows, dimensions]")
    labels = tuple(str(label).strip().lower() for label in target_labels)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("target_labels must contain unique non-empty labels")
    invalid = [label for label in labels if label not in PUBLIC_TARGET_LABELS]
    if invalid:
        raise ValueError(f"Only public nonverbal target labels are supported: {invalid}")

    row_labels = [str(row.get("final_label", "uncertain")).strip().lower() for row in labeled_rows]
    mask = np.asarray([label in labels for label in row_labels], dtype=bool)
    internal_counts = Counter(row_labels)
    discarded_counts = Counter(
        label for label, keep in zip(row_labels, mask, strict=True) if not keep
    )
    selected_rows = [dict(row) for row, keep in zip(labeled_rows, mask, strict=True) if keep]
    selected_embeddings = np.asarray(values[mask], dtype=np.float32)
    return (
        selected_rows,
        selected_embeddings,
        {
            "input_primitive_count": len(labeled_rows),
            "internal_label_counts": dict(sorted(internal_counts.items())),
            "published_labels": list(labels),
            "published_primitive_count": len(selected_rows),
            "discarded_primitive_count": len(labeled_rows) - len(selected_rows),
            "discarded_by_label": dict(sorted(discarded_counts.items())),
        },
    )


def filter_manual_applications_for_publication(
    applications: Sequence[Mapping[str, Any]],
    published_ids: set[str],
) -> list[dict[str, Any]]:
    """Remove references to internally rejected primitives from public provenance."""
    output: list[dict[str, Any]] = []
    for application in applications:
        matches = [
            dict(match)
            for match in application.get("matches", [])
            if str(match.get("primitive_id", "")) in published_ids
        ]
        if matches:
            output.append(
                {
                    **dict(application),
                    "matched_primitive_count": len(matches),
                    "matches": matches,
                }
            )
    return output


def _classifier_metadata(classifier: WeakClassifier) -> dict[str, Any]:
    metadata = getattr(classifier, "run_metadata", None)
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return {
        "schema_version": 1,
        "role": "weak_closed_set_cluster_labeler",
        "classifier_type": type(classifier).__name__,
    }


def classify_primitives_resumable(
    primitive_rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    feature_fingerprint: str,
    batch_size: int,
    classifier: WeakClassifier | None = None,
    classifier_factory: Callable[[], WeakClassifier] | None = None,
    prepare_missing_audio: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
    force: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Classify clips in restartable chunks while keeping all stored paths relative."""
    work_dir = output_dir / "_work"
    final_path = work_dir / "classifier_predictions.jsonl"
    partial_path = work_dir / "classifier_predictions.partial.jsonl"
    state_path = work_dir / "classifier_state.json"
    classifier_fingerprint = _canonical_hash(
        {
            "feature_fingerprint": feature_fingerprint,
            "space_revision": SPACE_REVISION,
            "schema": CLASSIFIER_CACHE_SCHEMA_VERSION,
        }
    )
    expected_ids = [str(row["id"]) for row in primitive_rows]
    expected_audio = {str(row["id"]): str(row["audio"]) for row in primitive_rows}

    state: dict[str, Any] = {}
    if state_path.is_file():
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            state = loaded
    cached: dict[str, dict[str, Any]] = {}
    if not force and state.get("classifier_fingerprint") == classifier_fingerprint:
        for candidate_path in (final_path, partial_path):
            if not candidate_path.is_file():
                continue
            for row in _read_jsonl(candidate_path):
                segment_id = str(row.get("id", ""))
                if segment_id in expected_audio:
                    cached[segment_id] = {**row, "audio": expected_audio[segment_id]}

    missing = [segment_id for segment_id in expected_ids if segment_id not in cached]
    if missing and prepare_missing_audio is not None:
        row_by_id = {str(row["id"]): row for row in primitive_rows}
        prepare_missing_audio([row_by_id[segment_id] for segment_id in missing])
    metadata = state.get("model") if isinstance(state.get("model"), dict) else None
    instance = classifier
    if missing and instance is None:
        if classifier_factory is None:
            raise RuntimeError("A classifier or classifier_factory is required for uncached rows")
        instance = classifier_factory()
    if instance is not None:
        metadata = _classifier_metadata(instance)
    metadata = metadata or {"space_revision": SPACE_REVISION}
    state = {
        "schema_version": CLASSIFIER_CACHE_SCHEMA_VERSION,
        "classifier_fingerprint": classifier_fingerprint,
        "feature_fingerprint": feature_fingerprint,
        "model": metadata,
    }
    _atomic_write_json(state_path, state)

    row_by_id = {str(row["id"]): row for row in primitive_rows}
    for offset in range(0, len(missing), batch_size):
        ids = missing[offset : offset + batch_size]
        batch_rows = [row_by_id[segment_id] for segment_id in ids]
        assert instance is not None
        predicted = predict_segment_rows(
            batch_rows,
            instance,  # type: ignore[arg-type]
            audio_root=output_dir,
            batch_size=batch_size,
        )
        for row in predicted:
            segment_id = str(row["id"])
            cached[segment_id] = {**row, "audio": expected_audio[segment_id]}
        _atomic_write_jsonl(
            partial_path,
            (cached[segment_id] for segment_id in expected_ids if segment_id in cached),
        )
        print(
            f"classifier rows={min(offset + len(ids), len(missing))}/{len(missing)} "
            f"cached={len(expected_ids) - len(missing)}",
            flush=True,
        )

    if set(cached) != set(expected_ids):
        absent = [segment_id for segment_id in expected_ids if segment_id not in cached]
        raise RuntimeError(f"Classifier cache is incomplete: {absent[:10]}")
    predictions = [cached[segment_id] for segment_id in expected_ids]
    _atomic_write_jsonl(final_path, predictions)
    partial_path.unlink(missing_ok=True)
    return predictions, metadata


def _prediction_probabilities(row: Mapping[str, Any]) -> dict[str, float] | None:
    prediction = row.get("prediction")
    if not isinstance(prediction, Mapping) or prediction.get("status", "ok") != "ok":
        return None
    probabilities = prediction.get("probabilities")
    if not isinstance(probabilities, Mapping):
        return None
    try:
        values = {label: float(probabilities[label]) for label in ERO_LABELS}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value >= 0.0 for value in values.values()):
        return None
    total = sum(values.values())
    if total <= 0.0:
        return None
    return {label: value / total for label, value in values.items()}


def _aggregate_prediction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    weighted = np.zeros(len(ERO_LABELS), dtype=np.float64)
    total_weight = 0.0
    valid_count = 0
    for row in rows:
        probabilities = _prediction_probabilities(row)
        if probabilities is None:
            continue
        weight = max(0.0, float(row["duration"]))
        weighted += weight * np.asarray([probabilities[label] for label in ERO_LABELS])
        total_weight += weight
        valid_count += 1
    if total_weight <= 0.0:
        return {
            "probabilities": dict.fromkeys(ERO_LABELS),
            "top_label": None,
            "top_probability": None,
            "margin": None,
            "normalized_entropy": None,
            "status": "error",
            "valid_primitive_count": 0,
        }
    values = weighted / total_weight
    order = np.argsort(-values, kind="stable")
    positive = values[values > 0.0]
    entropy = -float(np.sum(positive * np.log(positive))) / math.log(len(ERO_LABELS))
    return {
        "probabilities": {label: float(values[index]) for index, label in enumerate(ERO_LABELS)},
        "top_label": ERO_LABELS[int(order[0])],
        "top_probability": float(values[int(order[0])]),
        "margin": float(values[int(order[0])] - values[int(order[1])]),
        "normalized_entropy": entropy,
        "status": "ok",
        "valid_primitive_count": valid_count,
    }


def _shared_boundary(row: Mapping[str, Any], next_row: Mapping[str, Any]) -> dict[str, Any]:
    right_reason = str(row.get("right_boundary_reason", "unknown"))
    left_reason = str(next_row.get("left_boundary_reason", "unknown"))
    right_score = float(row.get("right_boundary_score", math.inf))
    left_score = float(next_row.get("left_boundary_score", math.inf))
    return {
        "time": float(row["end"]),
        "right_reason": right_reason,
        "left_reason": left_reason,
        "score": max(right_score, left_score),
    }


def _boundary_is_strong(boundary: Mapping[str, Any], config: EventMergeConfig) -> bool:
    reasons = {str(boundary["right_reason"]), str(boundary["left_reason"])}
    if reasons.intersection(config.hard_boundary_reasons):
        return True
    known_weak_or_scored = {
        "long_local_valley",
        "local_valley",
        "spectral_change",
        "fixed_even",
    }
    if not reasons.issubset(known_weak_or_scored):
        return True
    return float(boundary["score"]) >= config.strong_boundary_score


def _event_id(row: Mapping[str, Any], end: float) -> str:
    source_key = str(row["source_key"])
    start_ms = round(float(row["start"]) * 1_000)
    end_ms = round(end * 1_000)
    return f"nve_{source_key}_{start_ms:09d}_{end_ms:09d}"


def _aggregate_audio_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    weights = np.asarray([max(0.0, float(row["duration"])) for row in rows], dtype=np.float64)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(rows), dtype=np.float64)
    rms_amplitudes = np.asarray(
        [10.0 ** (float(row.get("rms_dbfs", -120.0)) / 20.0) for row in rows],
        dtype=np.float64,
    )
    rms = math.sqrt(float(np.average(np.square(rms_amplitudes), weights=weights)))
    peak = max(float(row.get("peak_dbfs", -120.0)) for row in rows)
    rms_dbfs = max(-120.0, 20.0 * math.log10(max(rms, 1e-6)))
    return {
        "rms_dbfs": round(rms_dbfs, 4),
        "peak_dbfs": round(peak, 4),
        "crest_factor_db": round(max(0.0, peak - rms_dbfs), 4),
    }


def _merge_group(
    rows: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    first = rows[0]
    last = rows[-1]
    duration_weights = np.asarray([float(row["duration"]) for row in rows], dtype=np.float64)
    combined = np.average(
        np.asarray(embeddings, dtype=np.float64), axis=0, weights=duration_weights
    )
    norm = float(np.linalg.norm(combined))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"Merged embedding has zero norm for {first['id']}")
    combined = (combined / norm).astype(np.float32)
    statuses = Counter(str(row["label_status"]) for row in rows)
    manual_ids = sorted(
        {str(interval_id) for row in rows for interval_id in row.get("manual_interval_ids", [])}
    )
    boundaries = [
        _shared_boundary(left, right) for left, right in zip(rows[:-1], rows[1:], strict=True)
    ]
    end = float(last["end"])
    event = {
        **dict(first),
        "id": _event_id(first, end),
        "start": float(first["start"]),
        "end": end,
        "duration": round(end - float(first["start"]), 6),
        "right_boundary_reason": last.get("right_boundary_reason"),
        "right_boundary_score": last.get("right_boundary_score"),
        "embedding_chunks": sum(int(row.get("embedding_chunks", 1)) for row in rows),
        "primitive_count": len(rows),
        "primitive_ids": [str(row["id"]) for row in rows],
        "primitive_label_statuses": dict(sorted(statuses.items())),
        "label_status": (str(first["label_status"]) if len(statuses) == 1 else "merged_same_label"),
        "manual_interval_ids": manual_ids,
        "prediction": _aggregate_prediction(rows),
        "crossed_boundaries": boundaries,
        **_aggregate_audio_metrics(rows),
    }
    event.pop("audio", None)
    event.pop("evidence", None)
    probability_label = str(event["final_label"])
    if probability_label == "other":
        probability_label = "usual"
    probabilities = event["prediction"]["probabilities"]
    event["confidence"] = (
        probabilities.get(probability_label)
        if isinstance(probabilities, Mapping) and probability_label in ERO_LABELS
        else None
    )
    return event, combined


def merge_labeled_primitives(
    labeled_rows: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
    *,
    config: EventMergeConfig | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Merge adjacent same-label primitives into natural clips of about five seconds.

    Strong acoustic boundaries are respected after the accumulated clip reaches the
    requested minimum.  Before that point they may be crossed, but speech gaps,
    source boundaries, label changes, and the maximum duration are never crossed.
    Isolated runs shorter than the minimum remain visible for the training quality
    gate to reject instead of being padded with unrelated audio.
    """
    settings = config or EventMergeConfig()
    values_input = np.asarray(embeddings)
    if values_input.ndim != 2 or len(labeled_rows) != len(values_input):
        raise ValueError("labeled rows and embeddings must align")
    if not labeled_rows:
        return [], np.empty((0, values_input.shape[1]), dtype=np.float32)
    order = sorted(range(len(labeled_rows)), key=lambda index: _row_sort_key(labeled_rows[index]))
    rows = [labeled_rows[index] for index in order]
    values = np.asarray(values_input[order], dtype=np.float32)
    groups: list[list[int]] = []
    current: list[int] = []
    for index, row in enumerate(rows):
        if not current:
            current = [index]
            continue
        previous = rows[current[-1]]
        same_stream = str(previous.get("source_key")) == str(row.get("source_key")) and int(
            previous.get("gap_index", -1)
        ) == int(row.get("gap_index", -2))
        adjacent = abs(float(previous["end"]) - float(row["start"])) <= (
            settings.adjacency_tolerance_seconds
        )
        same_target_label = (
            str(previous.get("final_label")) == str(row.get("final_label"))
            and str(row.get("final_label")) in settings.labels
        )
        within_maximum = float(row["end"]) - float(rows[current[0]]["start"]) <= (
            settings.max_seconds + 1e-9
        )
        strong = _boundary_is_strong(_shared_boundary(previous, row), settings)
        current_duration = float(previous["end"]) - float(rows[current[0]]["start"])
        needs_minimum = current_duration < settings.minimum_seconds - 1e-9
        if (
            same_stream
            and adjacent
            and same_target_label
            and within_maximum
            and (needs_minimum or not strong)
        ):
            current.append(index)
        else:
            groups.append(current)
            current = [index]
    if current:
        groups.append(current)

    event_rows: list[dict[str, Any]] = []
    event_embeddings: list[np.ndarray] = []
    for group in groups:
        event, embedding = _merge_group(
            [rows[index] for index in group],
            values[group],
        )
        event_rows.append(event)
        event_embeddings.append(embedding)
    return event_rows, np.stack(event_embeddings).astype(np.float32)


def _safe_remove_tree(path: Path, *, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Refusing to remove path outside generated root: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def _relative_link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "reused"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def materialize_class_folders(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_dir: Path,
) -> dict[str, Any]:
    """Create class/cluster review folders from canonical final clips."""
    classes_root = dataset_dir / "classes"
    if classes_root.exists():
        _safe_remove_tree(classes_root, allowed_root=dataset_dir)
    class_playlists: dict[str, list[str]] = defaultdict(lambda: ["#EXTM3U"])
    cluster_playlists: dict[tuple[str, str], list[str]] = defaultdict(lambda: ["#EXTM3U"])
    link_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()

    for row in rows:
        label = str(row["final_label"])
        source = dataset_dir / str(row["audio"])
        member = classes_root / label / "members" / source.name
        link_counts[_relative_link_or_copy(source, member)] += 1
        class_counts[label] += 1
        class_playlists[label].extend(
            [f"#EXTINF:{float(row['duration']):.3f},{row['id']}", f"members/{source.name}"]
        )
        cluster = str(row.get("event_cluster") or row.get("fallback_cluster") or "")
        if cluster:
            cluster_member = classes_root / label / "clusters" / cluster / "members" / source.name
            link_counts[_relative_link_or_copy(source, cluster_member)] += 1
            cluster_counts[cluster] += 1
            cluster_playlists[(label, cluster)].extend(
                [
                    f"#EXTINF:{float(row['duration']):.3f},{row['id']}",
                    f"members/{source.name}",
                ]
            )

    for label, lines in class_playlists.items():
        _atomic_write_text(classes_root / label / "all_members.m3u8", "\n".join(lines) + "\n")
    for (label, cluster), lines in cluster_playlists.items():
        _atomic_write_text(
            classes_root / label / "clusters" / cluster / "all_members.m3u8",
            "\n".join(lines) + "\n",
        )

    representatives = select_representatives_by_cluster(rows)
    for cluster, selected in representatives.items():
        for index, role in selected:
            row = rows[index]
            label = str(row["final_label"])
            source = dataset_dir / str(row["audio"])
            destination = (
                classes_root
                / label
                / "clusters"
                / cluster
                / "representatives"
                / f"{role}_{source.name}"
            )
            link_counts[_relative_link_or_copy(source, destination)] += 1

    return {
        "class_counts": dict(sorted(class_counts.items())),
        "cluster_counts": dict(sorted(cluster_counts.items())),
        "class_folder_count": len(class_counts),
        "cluster_folder_count": len(cluster_counts),
        "hardlinks_created": link_counts["hardlink"],
        "copies_created": link_counts["copy"],
        "links_reused": link_counts["reused"],
    }


def _sanitize_source_path(
    row: Mapping[str, Any],
    *,
    keep_audio: bool = True,
) -> dict[str, Any]:
    output = dict(row)
    relative = str(output.get("source_relative_path", "")).replace("\\", "/")
    output["source_audio"] = relative
    if not keep_audio:
        output.pop("audio", None)
    prediction = output.get("prediction")
    if isinstance(prediction, Mapping):
        clean_prediction = dict(prediction)
        clean_prediction.pop("audio", None)
        output["prediction"] = clean_prediction
    output.pop("_source_audio_abs", None)
    return output


def _dataset_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_summary: Mapping[str, Any],
    feature_fingerprint: str,
    run_fingerprint: str,
    classifier_metadata: Mapping[str, Any],
    selection_summary: Mapping[str, Any],
    manual_applications: Sequence[Mapping[str, Any]],
    clip_summary: Mapping[str, Any],
    folder_summary: Mapping[str, Any],
    configs: Mapping[str, Any],
) -> dict[str, Any]:
    counts = Counter(str(row["final_label"]) for row in rows)
    seconds: dict[str, float] = defaultdict(float)
    for row in rows:
        seconds[str(row["final_label"])] += float(row["duration"])
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "run_fingerprint": run_fingerprint,
        "feature_fingerprint": feature_fingerprint,
        "input_features": dict(feature_summary),
        "classifier": dict(classifier_metadata),
        "configs": dict(configs),
        "selection": dict(selection_summary),
        "input_primitive_count": int(selection_summary["input_primitive_count"]),
        "published_primitive_count": int(selection_summary["published_primitive_count"]),
        "discarded_primitive_count": int(selection_summary["discarded_primitive_count"]),
        "discarded_by_label": dict(selection_summary["discarded_by_label"]),
        "published_labels": list(selection_summary["published_labels"]),
        "event_count": len(rows),
        "published_primitives_joined": (
            int(selection_summary["published_primitive_count"]) - len(rows)
        ),
        "class_counts": dict(sorted(counts.items())),
        "class_hours": {label: seconds[label] / 3_600 for label in sorted(seconds)},
        "manual_interval_count": len(manual_applications),
        "manual_seed_count": len(
            {
                str(match["primitive_id"])
                for application in manual_applications
                for match in application["matches"]
            }
        ),
        "clips": {"published": dict(clip_summary)},
        "folders": dict(folder_summary),
        "manifests": {
            "events": "manifests/events.jsonl",
            "labeled_primitives": "manifests/labeled_primitives.jsonl",
            "manual_seed_applications": "manifests/manual_seed_applications.jsonl",
            "event_embeddings": "embeddings/beats.npy",
        },
    }


def _prepare_stage(output_dir: Path, run_fingerprint: str) -> Path:
    stage = output_dir / "_work" / "dataset_stage"
    state_path = stage / ".stage.json"
    if stage.is_dir():
        state: dict[str, Any] = {}
        if state_path.is_file():
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                state = loaded
        if state.get("run_fingerprint") != run_fingerprint:
            _safe_remove_tree(stage, allowed_root=output_dir)
    stage.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        state_path,
        {"schema_version": PIPELINE_SCHEMA_VERSION, "run_fingerprint": run_fingerprint},
    )
    return stage


def _publish_stage(stage: Path, destination: Path, *, output_dir: Path) -> None:
    """Replace the whole dataset directory with one same-volume rename."""
    backup = output_dir / "_work" / "dataset_previous"
    if backup.exists():
        _safe_remove_tree(backup, allowed_root=output_dir)
    if destination.exists():
        destination.replace(backup)
    try:
        stage.replace(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    if backup.exists():
        _safe_remove_tree(backup, allowed_root=output_dir)


def _existing_complete_summary(output_dir: Path, run_fingerprint: str) -> dict[str, Any] | None:
    path = output_dir / "dataset" / "summary.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        isinstance(value, dict)
        and value.get("status") == "complete"
        and value.get("run_fingerprint") == run_fingerprint
    ):
        return value
    return None


def _relative_config_path(path: Path | None, output_dir: Path) -> str | None:
    if path is None:
        return None
    return Path(os.path.relpath(path, output_dir)).as_posix()


def run_nonverbal_event_pipeline(
    *,
    feature_dir: str | Path,
    output_dir: str | Path,
    data_root: str | Path,
    manual_seed_path: str | Path | None = None,
    device: str = "cuda:0",
    hf_cache_dir: str | Path | None = None,
    local_files_only: bool = True,
    classifier: WeakClassifier | None = None,
    manual_config: ManualSeedConfig | None = None,
    labeling_config: NonverbalLabelingConfig | None = None,
    merge_config: EventMergeConfig | None = None,
    subclustering_config: SubclusteringConfig | None = None,
    pipeline_config: PipelineConfig | None = None,
    visualization_hook: VisualizationHook | None = None,
) -> dict[str, Any]:
    """Build and atomically publish ``output_dir/dataset`` from BEATs shards."""
    feature_root = Path(feature_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    source_root = Path(data_root).expanduser().resolve()
    seed_path = Path(manual_seed_path).expanduser().resolve() if manual_seed_path else None
    cache_path = Path(hf_cache_dir).expanduser().resolve() if hf_cache_dir else None
    output_root.mkdir(parents=True, exist_ok=True)
    settings = pipeline_config or PipelineConfig()
    manual_settings = manual_config or ManualSeedConfig()
    label_settings = labeling_config or NonverbalLabelingConfig(
        seed_prob=0.80,
        seed_margin=0.20,
        k=7,
        min_cosine=0.92,
        neighbor_agreement=0.75,
        min_neighbors=3,
        other_prob=0.75,
        other_margin=0.20,
    )
    merge_settings = merge_config or EventMergeConfig()
    subcluster_settings = subclustering_config or SubclusteringConfig()

    primitive_rows, primitive_embeddings, feature_summary = load_feature_shards(feature_root)
    feature_fingerprint = _feature_fingerprint(primitive_rows, primitive_embeddings)
    intervals = load_manual_seed_intervals(seed_path)
    configs = {
        "pipeline": asdict(settings),
        "manual_seed": asdict(manual_settings),
        "labeling": asdict(label_settings),
        "merge": asdict(merge_settings),
        "subclustering": asdict(subcluster_settings),
        "feature_dir": _relative_config_path(feature_root, output_root),
        "data_root": _relative_config_path(source_root, output_root),
        "manual_seed_path": _relative_config_path(seed_path, output_root),
    }
    fingerprint_configs = {
        "propagation_scope": settings.propagation_scope,
        "propagate_manual_seeds": settings.propagate_manual_seeds,
        "lock_confident_usual": settings.lock_confident_usual,
        "reject_target_class_conflicts": settings.reject_target_class_conflicts,
        "final_clip_padding_seconds": settings.final_clip_padding_seconds,
        "manual_seed": asdict(manual_settings),
        "labeling": asdict(label_settings),
        "merge": asdict(merge_settings),
        "subclustering": asdict(subcluster_settings),
    }
    run_fingerprint = _canonical_hash(
        {
            "schema": PIPELINE_SCHEMA_VERSION,
            "feature_fingerprint": feature_fingerprint,
            "manual_intervals": intervals,
            "configs": fingerprint_configs,
            "space_revision": SPACE_REVISION,
        }
    )
    if not settings.force_finalize:
        existing = _existing_complete_summary(output_root, run_fingerprint)
        if existing is not None:
            return existing

    work_prefix = Path("_work") / "primitive_clips"
    work_rows = attach_audio_paths(
        primitive_rows,
        relative_prefix=work_prefix,
    )

    def prepare_missing_audio(missing_rows: Sequence[Mapping[str, Any]]) -> None:
        export_audio_windows(
            missing_rows,
            root=output_root,
            data_root=source_root,
            relative_prefix=work_prefix,
            sample_rate=CLASSIFIER_CLIP_SAMPLE_RATE,
        )

    def classifier_factory() -> WeakClassifier:
        return JapaneseEroVoiceClassifier(
            device=device,
            cache_dir=cache_path,
            local_files_only=local_files_only,
        )

    prediction_rows, classifier_metadata = classify_primitives_resumable(
        work_rows,
        output_dir=output_root,
        feature_fingerprint=feature_fingerprint,
        batch_size=settings.classifier_batch_size,
        classifier=classifier,
        classifier_factory=classifier_factory,
        prepare_missing_audio=prepare_missing_audio,
        force=settings.force_classifier,
    )
    manual_seeds, manual_provenance, applications = match_manual_seed_intervals(
        primitive_rows,
        intervals,
        config=manual_settings,
    )
    decisions = propagate_labels_scoped(
        prediction_rows,
        primitive_embeddings,
        primitive_rows,
        manual_seeds=manual_seeds,
        config=label_settings,
        scope=settings.propagation_scope,
        propagate_manual_seeds=settings.propagate_manual_seeds,
        lock_confident_usual=settings.lock_confident_usual,
        reject_target_class_conflicts=settings.reject_target_class_conflicts,
    )
    labeled_primitives: list[dict[str, Any]] = []
    for primitive, decision in zip(primitive_rows, decisions, strict=True):
        segment_id = str(primitive["id"])
        labeled_primitives.append(
            {
                **dict(primitive),
                **decision,
                "manual_interval_ids": manual_provenance.get(segment_id, []),
            }
        )

    published_primitives, published_embeddings, selection_summary = filter_public_target_primitives(
        labeled_primitives, primitive_embeddings
    )
    event_rows, event_embeddings = merge_labeled_primitives(
        published_primitives,
        published_embeddings,
        config=merge_settings,
    )
    duration_mask = np.asarray(
        [float(row["duration"]) + 1e-9 >= merge_settings.minimum_seconds for row in event_rows],
        dtype=bool,
    )
    short_event_rows = [
        row for row, keep in zip(event_rows, duration_mask, strict=True) if not keep
    ]
    event_rows = [row for row, keep in zip(event_rows, duration_mask, strict=True) if keep]
    event_embeddings = np.asarray(event_embeddings[duration_mask], dtype=np.float32)
    retained_primitive_ids = {
        str(primitive_id) for row in event_rows for primitive_id in row.get("primitive_ids", [])
    }
    target_primitive_count = len(published_primitives)
    published_primitives = [
        row for row in published_primitives if str(row["id"]) in retained_primitive_ids
    ]
    selection_summary = {
        **selection_summary,
        "target_primitive_count": target_primitive_count,
        "published_primitive_count": len(published_primitives),
        "discarded_short_event_count": len(short_event_rows),
        "discarded_short_primitive_count": (target_primitive_count - len(published_primitives)),
        "minimum_published_event_seconds": merge_settings.minimum_seconds,
        "discarded_primitive_count": (
            int(selection_summary["input_primitive_count"]) - len(published_primitives)
        ),
    }
    published_ids = {str(row["id"]) for row in published_primitives}
    public_applications = filter_manual_applications_for_publication(
        applications,
        published_ids,
    )
    event_rows = subcluster_event_embeddings(
        event_embeddings,
        event_rows,
        config=subcluster_settings,
    )
    stage = _prepare_stage(output_root, run_fingerprint)
    event_rows, final_clip_summary = export_audio_windows(
        event_rows,
        root=stage,
        data_root=source_root,
        relative_prefix=Path("clips"),
        sample_rate=FINAL_CLIP_SAMPLE_RATE,
        padding_seconds=settings.final_clip_padding_seconds,
    )
    folder_summary = materialize_class_folders(event_rows, dataset_dir=stage)
    clean_primitives = [
        _sanitize_source_path(row, keep_audio=False) for row in published_primitives
    ]
    clean_events = [_sanitize_source_path(row) for row in event_rows]
    _atomic_write_jsonl(stage / "manifests" / "labeled_primitives.jsonl", clean_primitives)
    _atomic_write_jsonl(stage / "manifests" / "events.jsonl", clean_events)
    _atomic_write_jsonl(
        stage / "manifests" / "manual_seed_applications.jsonl",
        public_applications,
    )
    _atomic_save_npy(stage / "embeddings" / "beats.npy", event_embeddings)
    summary = _dataset_summary(
        clean_events,
        feature_summary=feature_summary,
        feature_fingerprint=feature_fingerprint,
        run_fingerprint=run_fingerprint,
        classifier_metadata=classifier_metadata,
        selection_summary=selection_summary,
        manual_applications=public_applications,
        clip_summary=final_clip_summary,
        folder_summary=folder_summary,
        configs=configs,
    )
    if visualization_hook is not None:
        visualization = visualization_hook(stage, clean_events, event_embeddings, summary)
        if visualization:
            summary["visualization"] = dict(visualization)
    _atomic_write_json(stage / "run_config.json", configs)
    _atomic_write_json(stage / "summary.json", summary)
    (stage / ".stage.json").unlink(missing_ok=True)
    _publish_stage(stage, output_root / "dataset", output_dir=output_root)

    if not settings.keep_work_clips:
        primitive_clip_root = output_root / "_work" / "primitive_clips"
        if primitive_clip_root.exists():
            _safe_remove_tree(primitive_clip_root, allowed_root=output_root)
    return summary
