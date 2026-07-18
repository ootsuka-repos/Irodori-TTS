"""Build Irodori-TTS input manifests from finalized nonverbal events.

The output is the same *pre-codec* JSONL shape used by ``prepare_manifest``:
``audio``, ``text``, and ``speaker_id`` are the training-facing fields, while
all event/classifier metadata is retained for audit and later relabeling.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset._io_utils import atomic_write_text
from dataset.acoustic_segmentation import ACOUSTIC_SEGMENTATION_VERSION
from dataset.ero_voice_classifier import SPACE_REPO_ID, SPACE_REVISION
from dataset.speech_pipeline import SILERO_VAD_REPO
from dataset.nonverbal_clustering import BEATS_MODEL_NAME

SCHEMA_VERSION = 3
TEXT_TAG_SCHEMA = "ja_nonverbal_transcript_only_v4"
TARGET_CLASSES = ("aegi", "chupa")
CLASS_TEXT = {
    "aegi": "喘ぎ声",
    "chupa": "フェラ音",
}
MANUAL_STATUSES = frozenset({"manual", "manual_seed", "manual_override"})
TRUSTED_PRIMITIVE_STATUSES = frozenset({*MANUAL_STATUSES, "classifier_seed", "neighbor_supported"})


@dataclass(frozen=True)
class NonverbalTrainingManifestConfig:
    """Conservative eligibility rules for direct training candidates.

    These defaults are the single source of truth; the CLI only overrides
    fields that were explicitly given on the command line.
    """

    target_classes: tuple[str, ...] = TARGET_CLASSES
    minimum_seconds: float = 5.0
    # 20.0 matches the acoustic segmentation cap (SegmentationConfig.max_seconds)
    # and the speech clip range (5-20 s), so both streams share one duration policy.
    maximum_seconds: float = 20.0
    classifier_minimum_probability: float = 0.75
    classifier_minimum_margin: float = 0.20
    neighbor_minimum_count: int = 3
    neighbor_minimum_support: float = 0.80
    neighbor_minimum_similarity: float = 0.92
    # Speech clips deliberately carry about 120 ms of boundary padding.
    # Requiring 500 ms avoids rejecting a clean nonverbal event merely because
    # that padding touches its edge, while still catching material duplication.
    speech_overlap_minimum_seconds: float = 0.50
    require_cluster: bool = True
    require_speaker_id: bool = True
    require_audio_file: bool = True

    def __post_init__(self) -> None:
        classes = tuple(str(value).strip().lower() for value in self.target_classes)
        if not classes or len(classes) != len(set(classes)):
            raise ValueError("target_classes must contain unique, non-empty labels")
        if any(value not in CLASS_TEXT for value in classes):
            raise ValueError(f"target_classes must be selected from {sorted(CLASS_TEXT)}")
        object.__setattr__(self, "target_classes", classes)
        if not math.isfinite(self.minimum_seconds) or self.minimum_seconds < 0.0:
            raise ValueError("minimum_seconds must be finite and non-negative")
        if not math.isfinite(self.maximum_seconds) or self.maximum_seconds <= self.minimum_seconds:
            raise ValueError("maximum_seconds must be finite and greater than minimum_seconds")
        for field_name in (
            "classifier_minimum_probability",
            "classifier_minimum_margin",
            "neighbor_minimum_support",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if not -1.0 <= self.neighbor_minimum_similarity <= 1.0:
            raise ValueError("neighbor_minimum_similarity must be in [-1, 1]")
        if self.neighbor_minimum_count <= 0:
            raise ValueError("neighbor_minimum_count must be positive")
        if (
            not math.isfinite(self.speech_overlap_minimum_seconds)
            or self.speech_overlap_minimum_seconds < 0.0
        ):
            raise ValueError("speech_overlap_minimum_seconds must be finite and non-negative")


@dataclass(frozen=True)
class NonverbalTrainingManifestResult:
    """Paths and counts published by :func:`write_nonverbal_training_manifests`."""

    train_manifest: Path
    combined_manifest: Path
    review_manifest: Path
    metadata_path: Path
    readme_path: Path
    train_count: int
    review_count: int
    discarded_count: int
    speech_count: int
    combined_count: int
    train_audio_seconds: float


@dataclass(frozen=True)
class LatentManifestMergeResult:
    """Audit information for a merged, model-ready latent manifest."""

    output_manifest: Path
    input_manifests: tuple[Path, ...]
    input_counts: tuple[int, ...]
    combined_count: int


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _normalized_source_audio(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or "").strip()).replace("\\", "/").casefold()


def _source_key(row: Mapping[str, Any]) -> str:
    source_uid = str(row.get("source_uid", "") or "").strip()
    if source_uid:
        return f"uid:{source_uid}"
    source_audio = _normalized_source_audio(row.get("source_audio"))
    return f"audio:{source_audio}" if source_audio else ""


def _interval(row: Mapping[str, Any]) -> tuple[float, float] | None:
    start = _finite_float(row.get("start"))
    end = _finite_float(row.get("end"))
    if start is None or end is None or start < 0.0 or end <= start:
        return None
    return start, end


def detect_speech_overlaps(
    nonverbal_rows: Sequence[Mapping[str, Any]],
    speech_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_overlap_seconds: float = 0.05,
) -> dict[str, list[dict[str, Any]]]:
    """Return deterministic timeline overlaps without reading or mutating files.

    Rows are matched by ``source_uid`` first and by normalized ``source_audio``
    only when the UID is absent. Invalid intervals cannot overlap and are left to
    the manifest validator. Touching boundaries are not overlaps.
    """
    if not math.isfinite(minimum_overlap_seconds) or minimum_overlap_seconds < 0.0:
        raise ValueError("minimum_overlap_seconds must be finite and non-negative")

    speech_by_source: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for index, row in enumerate(speech_rows):
        source = _source_key(row)
        interval = _interval(row)
        if not source or interval is None:
            continue
        speech_id = str(row.get("id") or f"speech_{index:08d}")
        speech_by_source[source].append((interval[0], interval[1], speech_id))
    for intervals in speech_by_source.values():
        intervals.sort(key=lambda value: (value[0], value[1], value[2]))

    output: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    for index, row in enumerate(nonverbal_rows):
        segment_id = str(row.get("id") or "").strip()
        if not segment_id:
            raise ValueError(f"nonverbal row {index} is missing a non-empty id")
        if segment_id in seen_ids:
            raise ValueError(f"duplicate nonverbal id: {segment_id}")
        seen_ids.add(segment_id)
        source = _source_key(row)
        interval = _interval(row)
        if not source or interval is None:
            continue
        start, end = interval
        duration = end - start
        matches: list[dict[str, Any]] = []
        for speech_start, speech_end, speech_id in speech_by_source.get(source, []):
            if speech_start >= end:
                break
            if speech_end <= start:
                continue
            overlap_start = max(start, speech_start)
            overlap_end = min(end, speech_end)
            overlap = overlap_end - overlap_start
            if overlap + 1e-12 < minimum_overlap_seconds or overlap <= 0.0:
                continue
            matches.append(
                {
                    "speech_id": speech_id,
                    "start": round(overlap_start, 6),
                    "end": round(overlap_end, 6),
                    "overlap_seconds": round(overlap, 6),
                    "nonverbal_ratio": round(overlap / duration, 6),
                }
            )
        if matches:
            output[segment_id] = matches
    return output


def _effective_cluster(row: Mapping[str, Any]) -> str | None:
    for field_name in ("event_cluster", "fallback_cluster"):
        value = str(row.get(field_name, "") or "").strip()
        if value:
            return value
    return None


def _cluster_token(final_label: str, cluster: str | None) -> tuple[str | None, bool]:
    """Return a human-typeable token and whether a known class prefix conflicts."""
    if cluster is None:
        return None, False
    normalized = unicodedata.normalize("NFKC", cluster.strip())
    lower = normalized.casefold()
    mismatch = any(
        lower.startswith(f"{candidate}_") and candidate != final_label for candidate in CLASS_TEXT
    )
    prefix = f"{final_label}_"
    if lower.startswith(prefix):
        normalized = normalized[len(prefix) :]
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = re.sub(r"[^\w.-]+", "-", normalized, flags=re.UNICODE).strip("-._")
    if not normalized:
        return None, mismatch
    if len(normalized) > 32:
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
        normalized = f"{normalized[:23]}-{digest}"
    return normalized, mismatch


def _clean_transcript_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).strip()
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def render_nonverbal_text(
    final_label: str,
    cluster: str | None,
    transcript: str | None = None,
) -> str:
    """Render the plain-text condition: the transcript content only.

    The class (aegi/chupa) and cluster are NOT embedded in the text anymore;
    they are published as separate manifest fields (``category`` /
    ``cluster_token``). Returns an empty string when no usable transcript
    exists — callers must route such rows to review.
    """
    label = str(final_label).strip().lower()
    if label not in CLASS_TEXT:
        raise ValueError(f"unsupported nonverbal class: {final_label!r}")
    _token, mismatch = _cluster_token(label, cluster)
    if mismatch:
        raise ValueError(f"cluster {cluster!r} does not belong to class {label!r}")
    transcript_text = _clean_transcript_text(transcript)
    if transcript_text and transcript_text[-1] not in "。！？!?…":
        transcript_text += "。"
    return transcript_text


def _usable_transcript(row: Mapping[str, Any]) -> str | None:
    """Return corrected local-ASR text only when its provenance is usable."""
    text = _clean_transcript_text(row.get("transcript_text"))
    if not text or str(row.get("transcript_status", "ok")) == "error":
        return None
    correction = row.get("transcript_correction")
    if isinstance(correction, Mapping) and correction.get("status") in {
        "uncertain",
        "rejected_unsafe",
        "missing",
    }:
        return None
    return text


def _probability_payload(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates: list[Any] = [row.get("probabilities")]
    prediction = row.get("prediction")
    if isinstance(prediction, Mapping):
        candidates.append(prediction.get("probabilities"))
    evidence = row.get("evidence")
    if isinstance(evidence, Mapping):
        raw = evidence.get("raw_classifier")
        if isinstance(raw, Mapping):
            candidates.append(raw.get("probabilities"))
    return next((value for value in candidates if isinstance(value, Mapping)), None)


def _probability_and_margin(
    row: Mapping[str, Any],
    final_label: str,
) -> tuple[float | None, float | None]:
    payload = _probability_payload(row)
    if payload is None:
        return None, None
    probability = _finite_float(payload.get(final_label))
    competing = [
        value
        for key, raw in payload.items()
        if str(key) != final_label and (value := _finite_float(raw)) is not None
    ]
    margin = probability - max(competing) if probability is not None and competing else None
    return probability, margin


def _neighbor_evidence_reasons(
    row: Mapping[str, Any],
    final_label: str,
    config: NonverbalTrainingManifestConfig,
) -> list[str]:
    evidence = row.get("evidence")
    if not isinstance(evidence, Mapping):
        return ["neighbor_evidence_missing"]
    reasons: list[str] = []
    if evidence.get("neighbor_supported") is not True:
        reasons.append("neighbor_not_accepted")
    count = _finite_float(evidence.get("accepted_neighbor_count"))
    if count is None or count < config.neighbor_minimum_count:
        reasons.append("neighbor_count_low")
    candidate = str(evidence.get("candidate_label", "") or "").strip().lower()
    if candidate and candidate != final_label:
        reasons.append("neighbor_label_mismatch")
    support = _finite_float(evidence.get("candidate_support"))
    support_table = evidence.get("support")
    class_support: Mapping[str, Any] | None = None
    if isinstance(support_table, Mapping) and isinstance(support_table.get(final_label), Mapping):
        class_support = support_table[final_label]
        if support is None:
            support = _finite_float(class_support.get("fraction"))
    if support is None or support < config.neighbor_minimum_support:
        reasons.append("neighbor_support_low")
    similarity = (
        _finite_float(class_support.get("nearest_similarity"))
        if class_support is not None
        else None
    )
    if similarity is None:
        neighbors = evidence.get("neighbors")
        if isinstance(neighbors, Sequence) and not isinstance(neighbors, (str, bytes)):
            similarities = [
                parsed
                for neighbor in neighbors
                if isinstance(neighbor, Mapping)
                and str(neighbor.get("label", "")) == final_label
                and (parsed := _finite_float(neighbor.get("similarity"))) is not None
            ]
            similarity = max(similarities, default=None)
    if similarity is None or similarity < config.neighbor_minimum_similarity:
        reasons.append("neighbor_similarity_low")
    return reasons


def _classifier_gate_reasons(
    row: Mapping[str, Any],
    final_label: str,
    config: NonverbalTrainingManifestConfig,
) -> list[str]:
    probability, margin = _probability_and_margin(row, final_label)
    reasons: list[str] = []
    if probability is None:
        reasons.append("classifier_probability_missing")
    elif probability < config.classifier_minimum_probability:
        reasons.append("classifier_probability_low")
    if margin is None:
        reasons.append("classifier_margin_missing")
    elif margin < config.classifier_minimum_margin:
        reasons.append("classifier_margin_low")
    return reasons


def _confidence_reasons(
    row: Mapping[str, Any],
    final_label: str,
    config: NonverbalTrainingManifestConfig,
) -> tuple[list[str], bool]:
    """Return ``(review reasons, gates_evaluated)`` for one final event row.

    ``gates_evaluated`` is false only when a trusted row carries no usable
    prediction payload at all, so the numeric gates could not be applied and
    the row passed through on provenance alone.
    """
    status = str(row.get("label_status", "") or "").strip().lower()
    raw_statuses = row.get("primitive_label_statuses")
    if isinstance(raw_statuses, Mapping) and raw_statuses:
        statuses = {
            str(value).strip().lower()
            for value, count in raw_statuses.items()
            if (_finite_float(count) or 0.0) > 0.0
        }
        if not statuses or not statuses.issubset(TRUSTED_PRIMITIVE_STATUSES):
            return ["merged_provenance_untrusted"], True
        if statuses & MANUAL_STATUSES:
            # A human decision anywhere in the merged event outranks the
            # classifier's aggregated probabilities.
            return [], True
        # Purely automatic provenance: re-apply the classifier gates to the
        # duration-weighted prediction the finalizer keeps on the event.
        # Per-primitive neighbor evidence is dropped during merging, so the
        # neighbor gates cannot be re-evaluated here.
        if _probability_payload(row) is None:
            return [], False
        return _classifier_gate_reasons(row, final_label, config), True
    if status in MANUAL_STATUSES:
        return [], True
    if status == "classifier_seed":
        return _classifier_gate_reasons(row, final_label, config), True
    if status == "neighbor_supported":
        return _neighbor_evidence_reasons(row, final_label, config), True
    if status == "merged_same_label":
        return ["merged_provenance_missing"], True
    return ["untrusted_label_status"], True


def _duration_for_row(row: Mapping[str, Any]) -> tuple[float | None, bool]:
    duration = _finite_float(row.get("duration"))
    interval = _interval(row)
    if duration is None and interval is not None:
        return interval[1] - interval[0], False
    if duration is None:
        return None, False
    mismatch = interval is not None and abs(duration - (interval[1] - interval[0])) > 0.05
    return duration, mismatch


def _audio_paths(
    raw_audio: Any,
    *,
    audio_base_dir: Path,
    path_base: Path,
) -> tuple[str, Path] | None:
    value = str(raw_audio or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    absolute = path.resolve() if path.is_absolute() else (audio_base_dir / path).resolve()
    try:
        relative = os.path.relpath(absolute, start=path_base)
    except ValueError as exc:
        raise ValueError(
            f"audio path and path_base must be on the same filesystem: {absolute}, {path_base}"
        ) from exc
    return Path(relative).as_posix(), absolute


def _stable_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("source_uid", "")).casefold(),
        _normalized_source_audio(row.get("source_audio")),
        _finite_float(row.get("start")) or 0.0,
        _finite_float(row.get("end")) or 0.0,
        str(row.get("id", "")),
    )


def prepare_nonverbal_training_rows(
    final_rows: Sequence[Mapping[str, Any]],
    *,
    manifest_dir: str | Path,
    config: NonverbalTrainingManifestConfig | None = None,
    speech_rows: Sequence[Mapping[str, Any]] = (),
    path_base: str | Path | None = None,
    audio_base_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify final event rows into train/review lists without writing files.

    ``audio`` is rewritten relative to ``path_base``. Relative input paths are
    resolved from ``audio_base_dir``, which defaults to the manifest directory.
    Repository callers can pass the repository root as ``path_base`` to emit
    root-relative paths compatible with their existing ``train.jsonl``.
    """
    settings = config or NonverbalTrainingManifestConfig()
    output_dir = Path(manifest_dir).expanduser().resolve()
    relative_root = Path(path_base).expanduser().resolve() if path_base else output_dir
    audio_root = Path(audio_base_dir).expanduser().resolve() if audio_base_dir else output_dir
    overlaps = detect_speech_overlaps(
        final_rows,
        speech_rows,
        minimum_overlap_seconds=settings.speech_overlap_minimum_seconds,
    )
    target_classes = set(settings.target_classes)
    seen_ids: set[str] = set()
    prepared: list[dict[str, Any]] = []

    for index, source_row in enumerate(final_rows):
        row = dict(source_row)
        segment_id = str(row.get("id") or "").strip()
        if not segment_id:
            raise ValueError(f"final row {index} is missing a non-empty id")
        if segment_id in seen_ids:
            raise ValueError(f"duplicate final event id: {segment_id}")
        seen_ids.add(segment_id)

        final_label = str(row.get("final_label", "") or "").strip().lower()
        # The classifier's closed-set leftovers are neither useful review work
        # nor safe training data.  Review is reserved for aegi/chupa rows that
        # can become trainable after a concrete quality fix.
        if final_label not in target_classes:
            continue
        cluster = _effective_cluster(row)
        cluster_token, cluster_mismatch = _cluster_token(final_label, cluster)
        raw_review_reasons = row.get("review_reasons")
        if isinstance(raw_review_reasons, Sequence) and not isinstance(
            raw_review_reasons, (str, bytes)
        ):
            reasons = [str(reason) for reason in raw_review_reasons]
        elif raw_review_reasons:
            reasons = [str(raw_review_reasons)]
        else:
            reasons = []
        transcript_text = _usable_transcript(source_row)
        text = render_nonverbal_text(
            final_label,
            cluster if not cluster_mismatch else None,
            transcript_text,
        )
        # Text is now transcript-only; rows without a usable transcript have
        # nothing to train on and must be reviewed instead.
        if not text:
            reasons.append("missing_transcript")
        confidence_details, confidence_gates_evaluated = _confidence_reasons(
            row, final_label, settings
        )
        row["confidence_gates_evaluated"] = confidence_gates_evaluated
        if confidence_details:
            reasons.append("low_confidence")
        if settings.require_cluster and cluster_token is None:
            reasons.append("missing_cluster")
        if cluster_mismatch:
            reasons.append("cluster_class_mismatch")

        duration, duration_mismatch = _duration_for_row(row)
        if duration is None or duration <= 0.0:
            reasons.append("invalid_duration")
        else:
            if duration < settings.minimum_seconds:
                reasons.append("too_short")
            if duration > settings.maximum_seconds:
                reasons.append("too_long")
            row["duration"] = round(duration, 6)
        if _interval(row) is None:
            reasons.append("invalid_timeline")
        elif duration_mismatch:
            reasons.append("duration_mismatch")

        audio_pair = _audio_paths(
            row.get("audio"),
            audio_base_dir=audio_root,
            path_base=relative_root,
        )
        if audio_pair is None:
            reasons.append("missing_audio")
        else:
            row["audio"] = audio_pair[0]
            if settings.require_audio_file and not audio_pair[1].is_file():
                reasons.append("audio_not_found")
        if not str(row.get("source_uid", "") or "").strip():
            reasons.append("missing_source_uid")
        if settings.require_speaker_id and not str(row.get("speaker_id", "") or "").strip():
            reasons.append("missing_speaker_id")
        if segment_id in overlaps:
            reasons.append("speech_overlap")
            row["speech_overlaps"] = overlaps[segment_id]

        if "text" in source_row and source_row.get("text") != text:
            row.setdefault("input_text", source_row.get("text"))
        if "status" in source_row:
            row.setdefault("input_status", source_row.get("status"))
        row["text"] = text
        row["text_tag_schema"] = TEXT_TAG_SCHEMA
        # The LLM correction stage classifies every segment from transcript and
        # timeline context; its verdict wins over the acoustic classifier for
        # the published label. An LLM "speech" verdict on a nonverbal event is
        # a leak signal, so route it to review instead of training.
        llm_category = str(row.get("llm_category", "") or "").strip().lower() or None
        if llm_category in {"aegi", "chupa", "mixed"}:
            row["category"] = llm_category
        else:
            row["category"] = final_label
            if llm_category in {"speech", "other"}:
                reasons.append(f"llm_category_{llm_category}")
        row["transcript_used"] = transcript_text is not None
        if transcript_text is not None:
            row["transcript_text"] = transcript_text
        row["effective_cluster"] = cluster
        row["cluster_token"] = cluster_token
        if confidence_details:
            row["confidence_review_reasons"] = confidence_details
        row["review_reasons"] = list(dict.fromkeys(reasons))
        row["status"] = "review" if row["review_reasons"] else "train"
        prepared.append(row)

    prepared.sort(key=_stable_row_sort_key)
    train_rows = [row for row in prepared if row["status"] == "train"]
    review_rows = [row for row in prepared if row["status"] == "review"]
    return train_rows, review_rows


# Shared fsync-and-retry writer from _io_utils.
_atomic_write_text = atomic_write_text


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
        for row in rows
    )


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def merge_latent_training_manifests(
    input_manifests: Sequence[str | Path],
    output_manifest: str | Path,
    *,
    require_latent_files: bool = True,
) -> LatentManifestMergeResult:
    """Safely combine DACVAE latent manifests without changing their sources.

    Relative ``latent_path`` values are resolved from each source manifest and
    rewritten relative to the output manifest, so inputs from different
    directories retain the same file meaning. All row metadata is preserved.
    """
    sources = tuple(Path(path).expanduser().resolve() for path in input_manifests)
    if not sources:
        raise ValueError("input_manifests must contain at least one manifest")
    if len(sources) != len(set(sources)):
        raise ValueError("input_manifests contains a duplicate path")
    output = Path(output_manifest).expanduser().resolve()
    if output in set(sources):
        raise ValueError("output_manifest must not overwrite an input manifest")

    merged: list[dict[str, Any]] = []
    input_counts: list[int] = []
    seen_latents: dict[str, str] = {}
    seen_ids: dict[str, str] = {}
    for source in sources:
        rows = _read_jsonl_objects(source)
        input_counts.append(len(rows))
        for line_index, source_row in enumerate(rows, 1):
            row = dict(source_row)
            raw_latent_path = str(row.get("latent_path") or "").strip()
            if not raw_latent_path:
                raise ValueError(f"latent_path is missing at {source}:{line_index}")
            if "text" not in row:
                raise ValueError(f"text is missing at {source}:{line_index}")
            latent_path = Path(raw_latent_path).expanduser()
            absolute_latent = (
                latent_path.resolve()
                if latent_path.is_absolute()
                else (source.parent / latent_path).resolve()
            )
            latent_key = os.path.normcase(str(absolute_latent))
            previous_latent = seen_latents.get(latent_key)
            location = f"{source}:{line_index}"
            if previous_latent is not None:
                raise ValueError(
                    f"duplicate latent_path: {absolute_latent} "
                    f"({previous_latent} and {location})"
                )
            seen_latents[latent_key] = location
            if require_latent_files and not absolute_latent.is_file():
                raise FileNotFoundError(f"latent file not found at {location}: {absolute_latent}")

            row_id = str(row.get("id") or "").strip()
            if row_id:
                previous_id = seen_ids.get(row_id)
                if previous_id is not None:
                    raise ValueError(
                        f"duplicate latent manifest id: {row_id} "
                        f"({previous_id} and {location})"
                    )
                seen_ids[row_id] = location
            try:
                relative_latent = os.path.relpath(absolute_latent, start=output.parent)
            except ValueError as exc:
                raise ValueError(
                    "latent file and output manifest must be on the same filesystem: "
                    f"{absolute_latent}, {output}"
                ) from exc
            row["latent_path"] = Path(relative_latent).as_posix()
            merged.append(row)

    merged.sort(
        key=lambda row: (
            str(row["latent_path"]).casefold(),
            str(row.get("speaker_id", "")).casefold(),
            str(row.get("text", "")),
            str(row.get("id", "")),
        )
    )
    _atomic_write_text(output, _jsonl_text(merged))
    return LatentManifestMergeResult(
        output_manifest=output,
        input_manifests=sources,
        input_counts=tuple(input_counts),
        combined_count=len(merged),
    )


def _combined_training_rows(
    speech_rows: Sequence[Mapping[str, Any]],
    nonverbal_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy, validate, and deterministically order the combined training rows."""
    combined: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}
    for kind, rows in (("speech", speech_rows), ("nonverbal", nonverbal_rows)):
        for index, source_row in enumerate(rows):
            row = dict(source_row)
            row_id = str(row.get("id") or "").strip()
            if not row_id:
                raise ValueError(f"{kind} training row {index} is missing a non-empty id")
            previous = seen_ids.get(row_id)
            if previous is not None:
                raise ValueError(
                    f"duplicate combined training id: {row_id} "
                    f"({previous} and {kind})"
                )
            seen_ids[row_id] = kind
            combined.append(row)
    combined.sort(key=_stable_row_sort_key)
    return combined


def _readme_text(
    combined_path: Path,
    nonverbal_path: Path,
    *,
    path_base: Path,
) -> str:
    try:
        combined_display = Path(os.path.relpath(combined_path, start=path_base)).as_posix()
    except ValueError:
        combined_display = combined_path.as_posix()
    try:
        nonverbal_display = Path(os.path.relpath(nonverbal_path, start=path_base)).as_posix()
    except ValueError:
        nonverbal_display = nonverbal_path.as_posix()
    return f"""# 非言語音の学習マニフェスト

`train_combined.jsonl`（`{combined_display}`）は既存speechと採用済み非言語音を
統合した、DACVAE変換前の完全な監査・再構築用マニフェストです。
`train_nonverbal.jsonl` は採用された非言語音だけの監査用部分集合です
（`{nonverbal_display}`）。`review_nonverbal.jsonl` には品質条件を満たさなかった
`aegi / chupa` だけが入り、試聴・ラベル修正の対象になります。

入力したspeechマニフェストは読み取り専用で、上書きしません。対象外classは
学習にもreviewにも含めず、選別時に破棄します。

## text の規則

既存の日本語 tokenizer をそのまま使い、特殊トークンは追加しません。
`text` は anime-whisper と文脈校正による文字起こし内容のみです。class と
クラスタは text に埋め込まず、`category`（aegi/chupa）と `cluster_token` の
別フィールドとして保存します。

- `aegi_c0007` + `あっ、んっ……` → text=`あっ、んっ……。` / category=`aegi` / cluster_token=`c0007`
- 文字起こしが空・不確実・棄却の行は `missing_transcript` として review 側へ回します。

低信頼、長さ範囲外、speech区間と重複する `aegi / chupa` は review 側だけです。

## 音声区切りと分類モデル

- speech VAD: `{SILERO_VAD_REPO}`（JIT版、ONNXではありません）
- VAD外の自然区切り: `{ACOUSTIC_SEGMENTATION_VERSION}`（log-mel、音量の谷、周波数変化）
- 音響埋め込み: `{BEATS_MODEL_NAME}`
- 上位class分類: `{SPACE_REPO_ID}@{SPACE_REVISION}`

固定5秒では切らず、自然区切りを先に作ります。同じclassが連続する区間を
5〜20秒程度へまとめ、5秒に届かない孤立音は混ぜ物を避けるため公開しません。
公開する `aegi` と `chupa` の中だけをBEATsで細分類しています。

## DACVAE latent の生成と最終統合

音声パスは、このREADMEを生成した `path_base` からの相対パスです。同じ場所を
カレントディレクトリにします。既存speech latentを再計算せず、まず採用済み
非言語音だけをDACVAE変換します。

```powershell
uv run --no-sync irodori-prepare-manifest `
  --dataset json `
  --data-files train={nonverbal_display} `
  --split train `
  --audio-column audio `
  --text-column text `
  --speaker-column speaker_id `
  --speaker-id-prefix full-data-pipeline `
  --target-sample-rate 48000 `
  --output-manifest dataset/data/manifests/nonverbal.jsonl `
  --latent-dir dataset/data/latents/nonverbal `
  --device cuda
```

次に、既存のspeech latentと非言語latentを検証し、モデルが直接読む最終
`dataset/data/manifests/train.jsonl` を生成します。入力マニフェストは上書きされません。

```powershell
uv run --no-sync python -m dataset.cli.merge_latent_manifests `
  --input dataset/data/manifests/speech.jsonl `
  --input dataset/data/manifests/nonverbal.jsonl `
  --output dataset/data/manifests/train.jsonl
```

入力行の `speaker_id` は変更しません。speechマニフェストと同じ
`--speaker-id-prefix` を使うことで、同じキャラクター/音源の参照IDを分断しません。
`source_uid`、時刻、分類確率、evidence、クラスタ情報もJSONLに残るため、
後から判定根拠を追跡できます。
"""


def write_nonverbal_training_manifests(
    final_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    config: NonverbalTrainingManifestConfig | None = None,
    speech_rows: Sequence[Mapping[str, Any]] = (),
    path_base: str | Path | None = None,
    audio_base_dir: str | Path | None = None,
) -> NonverbalTrainingManifestResult:
    """Atomically publish train/review JSONL, metadata, and a usage README."""
    settings = config or NonverbalTrainingManifestConfig()
    destination = Path(output_dir).expanduser().resolve()
    relative_root = Path(path_base).expanduser().resolve() if path_base else destination
    train_rows, review_rows = prepare_nonverbal_training_rows(
        final_rows,
        manifest_dir=destination,
        config=settings,
        speech_rows=speech_rows,
        path_base=relative_root,
        audio_base_dir=audio_base_dir,
    )
    train_path = destination / "train_nonverbal.jsonl"
    combined_path = destination / "train_combined.jsonl"
    review_path = destination / "review_nonverbal.jsonl"
    metadata_path = destination / "nonverbal_training_metadata.json"
    readme_path = destination / "README_nonverbal_training.md"

    reason_counts = Counter(
        str(reason) for row in review_rows for reason in row.get("review_reasons", [])
    )
    confidence_gate_unevaluated = {
        "train": sum(1 for row in train_rows if row.get("confidence_gates_evaluated") is False),
        "review": sum(1 for row in review_rows if row.get("confidence_gates_evaluated") is False),
    }
    combined_rows = _combined_training_rows(speech_rows, train_rows)
    target_classes = set(settings.target_classes)
    discarded_rows = [
        row
        for row in final_rows
        if str(row.get("final_label", "") or "").strip().lower() not in target_classes
    ]
    discarded_labels = Counter(
        str(row.get("final_label", "") or "<missing>").strip().lower() or "<missing>"
        for row in discarded_rows
    )
    label_train = Counter(str(row.get("final_label", "")) for row in train_rows)
    label_review = Counter(str(row.get("final_label", "")) for row in review_rows)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "train": train_path.name,
            "nonverbal_train": train_path.name,
            "combined_train": combined_path.name,
            "review": review_path.name,
            "readme": readme_path.name,
        },
        "counts": {
            "input": len(final_rows),
            "train": len(train_rows),
            "review": len(review_rows),
            "nonverbal_train": len(train_rows),
            "nonverbal_review": len(review_rows),
            "discarded_non_target": len(discarded_rows),
            "discarded_by_class": dict(sorted(discarded_labels.items())),
            "speech": len(speech_rows),
            "combined": len(combined_rows),
            "combined_train": len(combined_rows),
            "train_by_class": dict(sorted(label_train.items())),
            "review_by_class": dict(sorted(label_review.items())),
            "review_reasons": dict(sorted(reason_counts.items())),
            # Rows whose numeric confidence gates could not be applied because
            # no prediction payload survived; they passed on provenance alone.
            "confidence_gate_unevaluated": confidence_gate_unevaluated,
            "train_audio_seconds": round(sum(float(row["duration"]) for row in train_rows), 6),
        },
        "eligibility": asdict(settings),
        "audio_paths": {
            "absolute": False,
            "relative_to": "caller supplied path_base" if path_base else "output directory",
        },
        "text_conditioning": {
            "schema": TEXT_TAG_SCHEMA,
            "tokenizer_change_required": False,
            "cluster_precedence": ["event_cluster", "fallback_cluster"],
            "examples": [
                {
                    "final_label": "aegi",
                    "cluster": "aegi_c0007",
                    "transcript": "あっ、んっ……",
                    "text": "あっ、んっ……。",
                    "category": "aegi",
                    "cluster_token": "c0007",
                },
            ],
        },
        "source_pipeline": {
            "vad": SILERO_VAD_REPO,
            "segmentation": ACOUSTIC_SEGMENTATION_VERSION,
            "embedding": BEATS_MODEL_NAME,
            "classifier": f"{SPACE_REPO_ID}@{SPACE_REVISION}",
        },
        "metadata_policy": {
            "input_fields_preserved": True,
            "speaker_id_rewritten": False,
            "published_classes": list(settings.target_classes),
            "non_target_rows_written_to_review": False,
            "speech_input_mutated": False,
        },
    }

    _atomic_write_text(train_path, _jsonl_text(train_rows))
    _atomic_write_text(combined_path, _jsonl_text(combined_rows))
    _atomic_write_text(review_path, _jsonl_text(review_rows))
    _atomic_write_text(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
    )
    _atomic_write_text(
        readme_path,
        _readme_text(combined_path, train_path, path_base=relative_root),
    )
    return NonverbalTrainingManifestResult(
        train_manifest=train_path,
        combined_manifest=combined_path,
        review_manifest=review_path,
        metadata_path=metadata_path,
        readme_path=readme_path,
        train_count=len(train_rows),
        review_count=len(review_rows),
        discarded_count=len(discarded_rows),
        speech_count=len(speech_rows),
        combined_count=len(combined_rows),
        train_audio_seconds=float(metadata["counts"]["train_audio_seconds"]),
    )
