"""Weak nonverbal labels from a local classifier and BEATs neighbors.

This module deliberately contains no JSON/NPY I/O.  Prediction rows and L2 BEATs
embeddings are passed in by callers so the propagation logic remains deterministic,
testable, and independent from a particular dataset layout.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dataset.ero_voice_classifier import (
    ERO_LABELS,
    EroVoicePrediction,
    JapaneseEroVoiceClassifier,
)

TARGET_LABELS = ("aegi", "chupa")
FINAL_LABELS = ("aegi", "chupa", "other", "uncertain")


@dataclass(frozen=True)
class NonverbalLabelingConfig:
    """Thresholds for conservative seed selection and rejected kNN propagation.

    These defaults are the single source of truth for production behavior.  The
    event pipeline and the ``prepare_nonverbal_events`` CLI no longer carry
    their own copies; CLI flags override individual fields only when given.
    """

    seed_prob: float = 0.80
    seed_margin: float = 0.50
    k: int = 7
    min_cosine: float = 0.92
    neighbor_agreement: float = 0.80
    min_neighbors: int = 3
    other_prob: float = 0.75
    other_margin: float = 0.20
    query_chunk_size: int = 2_048
    norm_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "seed_prob",
            "seed_margin",
            "neighbor_agreement",
            "other_prob",
            "other_margin",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not -1.0 <= self.min_cosine <= 1.0:
            raise ValueError("min_cosine must be in [-1, 1]")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.min_neighbors <= 0 or self.min_neighbors > self.k:
            raise ValueError("min_neighbors must be in [1, k]")
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if not math.isfinite(self.norm_epsilon) or self.norm_epsilon <= 0.0:
            raise ValueError("norm_epsilon must be positive and finite")


@dataclass(frozen=True)
class _ParsedPrediction:
    probabilities: dict[str, float]
    top_label: str
    top_probability: float
    margin: float
    normalized_entropy: float


@dataclass(frozen=True)
class _TargetSeed:
    row_index: int
    segment_id: str
    label: str
    source: str


def _segment_ids(rows: Sequence[Mapping[str, Any]], *, id_key: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if id_key not in row:
            raise ValueError(f"segment row {index} is missing {id_key!r}")
        segment_id = str(row[id_key])
        if not segment_id:
            raise ValueError(f"segment row {index} has an empty id")
        if segment_id in seen:
            raise ValueError(f"duplicate segment id: {segment_id}")
        seen.add(segment_id)
        ids.append(segment_id)
    return ids


def _parse_probabilities(probabilities: Mapping[str, Any]) -> _ParsedPrediction:
    missing = [label for label in ERO_LABELS if label not in probabilities]
    if missing:
        raise ValueError(f"missing classifier probabilities: {missing}")
    values = np.asarray([float(probabilities[label]) for label in ERO_LABELS], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("classifier probabilities must be finite and non-negative")
    total = float(np.sum(values))
    if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"classifier probabilities sum to {total}, not 1")
    values /= total

    order = np.argsort(-values, kind="stable")
    top_index = int(order[0])
    second_index = int(order[1])
    positive = values[values > 0.0]
    entropy = -float(np.sum(positive * np.log(positive))) / math.log(len(ERO_LABELS))
    return _ParsedPrediction(
        probabilities={label: float(values[index]) for index, label in enumerate(ERO_LABELS)},
        top_label=ERO_LABELS[top_index],
        top_probability=float(values[top_index]),
        margin=float(values[top_index] - values[second_index]),
        normalized_entropy=entropy,
    )


def _prediction_row(
    *,
    segment_id: str,
    audio: str,
    prediction: EroVoicePrediction,
) -> dict[str, Any]:
    parsed = _parse_probabilities(prediction.probabilities)
    return {
        "id": segment_id,
        "audio": audio,
        "probabilities": parsed.probabilities,
        "top_label": parsed.top_label,
        "top_probability": parsed.top_probability,
        "margin": parsed.margin,
        "normalized_entropy": parsed.normalized_entropy,
        "status": "ok",
        "error": None,
    }


def _error_prediction_row(
    *,
    segment_id: str,
    audio: str | None,
    error: BaseException | str,
) -> dict[str, Any]:
    message = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    return {
        "id": segment_id,
        "audio": audio,
        "probabilities": dict.fromkeys(ERO_LABELS),
        "top_label": None,
        "top_probability": None,
        "margin": None,
        "normalized_entropy": None,
        "status": "error",
        "error": message,
    }


def predict_segment_rows(
    segment_rows: Sequence[Mapping[str, Any]],
    classifier: JapaneseEroVoiceClassifier,
    *,
    id_key: str = "id",
    audio_key: str = "audio",
    audio_root: str | Path | None = None,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Classify generic segment rows while isolating individual file failures.

    A failed batch is bisected until the failing file is isolated.  This preserves
    successful rows without paying one classifier call per file in the normal case.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = list(segment_rows)
    ids = _segment_ids(rows, id_key=id_key)
    root = Path(audio_root) if audio_root is not None else None
    results: list[dict[str, Any] | None] = [None] * len(rows)
    paths: dict[int, Path] = {}

    for index, (segment_id, row) in enumerate(zip(ids, rows, strict=True)):
        if audio_key not in row or row[audio_key] in (None, ""):
            results[index] = _error_prediction_row(
                segment_id=segment_id,
                audio=None,
                error=f"segment row is missing {audio_key!r}",
            )
            continue
        path = Path(str(row[audio_key]))
        if not path.is_absolute() and root is not None:
            path = root / path
        paths[index] = path

    def predict_indices(indices: list[int]) -> None:
        if not indices:
            return
        try:
            predictions = classifier.predict(
                [paths[index] for index in indices],
                batch_size=min(batch_size, len(indices)),
            )
            if len(predictions) != len(indices):
                raise RuntimeError(
                    f"classifier returned {len(predictions)} rows for {len(indices)} files"
                )
        except Exception as exc:
            if len(indices) == 1:
                index = indices[0]
                results[index] = _error_prediction_row(
                    segment_id=ids[index],
                    audio=str(paths[index]),
                    error=exc,
                )
                return
            midpoint = len(indices) // 2
            predict_indices(indices[:midpoint])
            predict_indices(indices[midpoint:])
            return

        for index, prediction in zip(indices, predictions, strict=True):
            try:
                results[index] = _prediction_row(
                    segment_id=ids[index],
                    audio=str(paths[index]),
                    prediction=prediction,
                )
            except Exception as exc:
                results[index] = _error_prediction_row(
                    segment_id=ids[index],
                    audio=str(paths[index]),
                    error=exc,
                )

    valid_indices = list(paths)
    for offset in range(0, len(valid_indices), batch_size):
        predict_indices(valid_indices[offset : offset + batch_size])

    if any(result is None for result in results):  # pragma: no cover - defensive invariant
        raise RuntimeError("internal error: not every segment received a prediction row")
    return [result for result in results if result is not None]


def _prediction_from_row(row: Mapping[str, Any]) -> _ParsedPrediction | None:
    if row.get("status", "ok") != "ok":
        return None
    nested = row.get("probabilities")
    if isinstance(nested, Mapping):
        probabilities = nested
    elif all(label in row for label in ERO_LABELS):
        probabilities = {label: row[label] for label in ERO_LABELS}
    else:
        return None
    try:
        return _parse_probabilities(probabilities)
    except (TypeError, ValueError):
        return None


def _normalize_embeddings(embeddings: np.ndarray, *, epsilon: float) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("embeddings must have shape [segments, dimensions]")
    if not np.all(np.isfinite(values)):
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    invalid = np.flatnonzero(norms[:, 0] <= epsilon)
    if invalid.size:
        raise ValueError(f"embeddings contain zero-norm rows: {invalid[:10].tolist()}")
    return values / norms


def _empty_support() -> dict[str, dict[str, Any]]:
    return {
        label: {
            "weight": 0.0,
            "fraction": 0.0,
            "count": 0,
            "nearest_similarity": None,
            "seed_ids": [],
        }
        for label in TARGET_LABELS
    }


def _base_evidence(config: NonverbalLabelingConfig) -> dict[str, Any]:
    return {
        "neighbors": [],
        "support": _empty_support(),
        "accepted_neighbor_count": 0,
        "candidate_label": None,
        "candidate_support": 0.0,
        "neighbor_supported": False,
        "rejection_reason": "no_target_seeds",
        "thresholds": {
            "k": config.k,
            "min_cosine": config.min_cosine,
            "neighbor_agreement": config.neighbor_agreement,
            "min_neighbors": config.min_neighbors,
        },
    }


def _support_from_neighbors(
    neighbors: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    support = _empty_support()
    # A negative cosine is evidence against similarity, never positive support.
    total_weight = sum(max(0.0, float(neighbor["similarity"])) for neighbor in neighbors)
    for label in TARGET_LABELS:
        selected = [neighbor for neighbor in neighbors if neighbor["label"] == label]
        weight = sum(max(0.0, float(neighbor["similarity"])) for neighbor in selected)
        support[label] = {
            "weight": weight,
            "fraction": weight / total_weight if total_weight > 0.0 else 0.0,
            "count": len(selected),
            "nearest_similarity": (
                max(float(neighbor["similarity"]) for neighbor in selected) if selected else None
            ),
            "seed_ids": [str(neighbor["seed_id"]) for neighbor in selected],
        }
    return support


def _raw_prediction_evidence(prediction: _ParsedPrediction | None) -> dict[str, Any] | None:
    if prediction is None:
        return None
    return {
        "probabilities": dict(prediction.probabilities),
        "top_label": prediction.top_label,
        "top_probability": prediction.top_probability,
        "margin": prediction.margin,
        "normalized_entropy": prediction.normalized_entropy,
    }


def propagate_nonverbal_labels(
    prediction_rows: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
    *,
    manual_seeds: Mapping[str, str] | None = None,
    config: NonverbalLabelingConfig | None = None,
    id_key: str = "id",
    propagate_manual_seeds: bool = True,
) -> list[dict[str, Any]]:
    """Propagate ``aegi/chupa`` seeds through BEATs cosine neighbors with rejection.

    Priority is fixed and deterministic:

    1. a manual label always wins and the row is never used as a classifier
       seed; manual ``aegi/chupa`` labels also become propagating seeds only
       when ``propagate_manual_seeds`` is true;
    2. a high-confidence classifier ``aegi/chupa`` prediction is adopted as a seed;
    3. a non-seed is labeled only when accepted neighbors meet weighted agreement;
    4. a high-confidence raw ``usual`` prediction without accepted support is ``other``;
    5. every remaining row is ``uncertain``.

    Input predictions are copied into each output under ``prediction`` and all neighbor
    IDs, similarities, and weighted support are retained under ``evidence``.
    """
    rows = list(prediction_rows)
    settings = config or NonverbalLabelingConfig()
    ids = _segment_ids(rows, id_key=id_key)
    normalized = _normalize_embeddings(embeddings, epsilon=settings.norm_epsilon)
    if normalized.shape[0] != len(rows):
        raise ValueError(
            f"embedding row count {normalized.shape[0]} does not match predictions {len(rows)}"
        )

    manual = {str(segment_id): str(label) for segment_id, label in (manual_seeds or {}).items()}
    unknown_manual_ids = sorted(set(manual) - set(ids))
    if unknown_manual_ids:
        raise ValueError(f"manual seed IDs are not present: {unknown_manual_ids[:10]}")
    invalid_manual = {
        segment_id: label for segment_id, label in manual.items() if label not in FINAL_LABELS
    }
    if invalid_manual:
        raise ValueError(f"invalid manual labels: {invalid_manual}")

    parsed_predictions = [_prediction_from_row(row) for row in rows]
    final_labels: list[str | None] = [None] * len(rows)
    decisions: list[str | None] = [None] * len(rows)
    seeds: list[_TargetSeed] = []

    for index, (segment_id, prediction) in enumerate(zip(ids, parsed_predictions, strict=True)):
        if segment_id in manual:
            label = manual[segment_id]
            final_labels[index] = label
            decisions[index] = "manual_override"
            # A manually labeled row must never act as a classifier seed: the
            # human decision replaces the classifier signal for this row even
            # when the classifier disagrees confidently.  Whether the manual
            # label itself spreads to neighbors is a separate policy switch.
            if propagate_manual_seeds and label in TARGET_LABELS:
                seeds.append(
                    _TargetSeed(
                        row_index=index,
                        segment_id=segment_id,
                        label=label,
                        source="manual",
                    )
                )
            continue
        if (
            prediction is not None
            and prediction.top_label in TARGET_LABELS
            and prediction.top_probability >= settings.seed_prob
            and prediction.margin >= settings.seed_margin
        ):
            final_labels[index] = prediction.top_label
            decisions[index] = "classifier_seed"
            seeds.append(
                _TargetSeed(
                    row_index=index,
                    segment_id=segment_id,
                    label=prediction.top_label,
                    source="classifier",
                )
            )

    # Sorting seed columns by ID makes equal-similarity top-k ties reproducible.
    seeds.sort(key=lambda seed: seed.segment_id)
    output: list[dict[str, Any] | None] = [None] * len(rows)
    for index, (segment_id, label, decision, prediction) in enumerate(
        zip(ids, final_labels, decisions, parsed_predictions, strict=True)
    ):
        if label is None or decision is None:
            continue
        evidence = _base_evidence(settings)
        evidence["raw_classifier"] = _raw_prediction_evidence(prediction)
        is_manual = decision == "manual_override"
        propagates = label in TARGET_LABELS and (propagate_manual_seeds or not is_manual)
        evidence["direct_seed"] = {
            "seed_id": segment_id if propagates else None,
            "label": label,
            "source": "manual" if is_manual else "classifier",
            "propagates": propagates,
        }
        evidence["candidate_label"] = label
        evidence["candidate_support"] = 1.0
        evidence["rejection_reason"] = None
        output[index] = {
            "id": segment_id,
            "final_label": label,
            "label_status": decision,
            "prediction": dict(rows[index]),
            "evidence": evidence,
        }

    query_indices = [index for index, label in enumerate(final_labels) if label is None]
    if seeds:
        seed_embeddings = normalized[[seed.row_index for seed in seeds]]
        top_k = min(settings.k, len(seeds))
        for chunk_start in range(0, len(query_indices), settings.query_chunk_size):
            chunk_indices = query_indices[chunk_start : chunk_start + settings.query_chunk_size]
            similarities = normalized[chunk_indices] @ seed_embeddings.T
            # Stable sort preserves seed-ID order when two cosine scores are equal.
            neighbor_order = np.argsort(-similarities, axis=1, kind="stable")[:, :top_k]
            for local_index, row_index in enumerate(chunk_indices):
                nearest_neighbors: list[dict[str, Any]] = []
                accepted_neighbors: list[dict[str, Any]] = []
                for seed_column in neighbor_order[local_index]:
                    similarity = float(similarities[local_index, seed_column])
                    seed = seeds[int(seed_column)]
                    neighbor = {
                        "seed_id": seed.segment_id,
                        "label": seed.label,
                        "similarity": similarity,
                        "seed_source": seed.source,
                        "accepted": similarity >= settings.min_cosine,
                    }
                    nearest_neighbors.append(neighbor)
                    if neighbor["accepted"]:
                        accepted_neighbors.append(neighbor)
                support = _support_from_neighbors(accepted_neighbors)
                best_label = max(
                    TARGET_LABELS,
                    key=lambda candidate: support[candidate]["fraction"],
                )
                accepted = (
                    len(accepted_neighbors) >= settings.min_neighbors
                    and support[best_label]["fraction"] >= settings.neighbor_agreement
                )
                if len(accepted_neighbors) < settings.min_neighbors:
                    rejection_reason = "insufficient_neighbors"
                elif support[best_label]["fraction"] < settings.neighbor_agreement:
                    rejection_reason = "insufficient_agreement"
                else:
                    rejection_reason = None
                evidence = _base_evidence(settings)
                evidence["neighbors"] = nearest_neighbors
                evidence["support"] = support
                evidence["accepted_neighbor_count"] = len(accepted_neighbors)
                evidence["candidate_label"] = best_label
                evidence["candidate_support"] = support[best_label]["fraction"]
                evidence["neighbor_supported"] = accepted
                evidence["rejection_reason"] = rejection_reason
                evidence["raw_classifier"] = _raw_prediction_evidence(parsed_predictions[row_index])
                evidence["direct_seed"] = None
                if accepted:
                    final_labels[row_index] = best_label
                    decisions[row_index] = "neighbor_supported"
                output[row_index] = {
                    "id": ids[row_index],
                    "final_label": best_label if accepted else "",
                    "label_status": "neighbor_supported" if accepted else "",
                    "prediction": dict(rows[row_index]),
                    "evidence": evidence,
                }

    for row_index in query_indices:
        if final_labels[row_index] is not None:
            continue
        prediction = parsed_predictions[row_index]
        if (
            prediction is not None
            and prediction.top_label == "usual"
            and prediction.top_probability >= settings.other_prob
            and prediction.margin >= settings.other_margin
        ):
            label = "other"
            decision = "classifier_other"
        else:
            label = "uncertain"
            decision = "uncertain"
        final_labels[row_index] = label
        decisions[row_index] = decision
        if output[row_index] is None:
            evidence = _base_evidence(settings)
            evidence["raw_classifier"] = _raw_prediction_evidence(prediction)
            evidence["direct_seed"] = None
        else:
            evidence = output[row_index]["evidence"]
        output[row_index] = {
            "id": ids[row_index],
            "final_label": label,
            "label_status": decision,
            "prediction": dict(rows[row_index]),
            "evidence": evidence,
        }

    if any(row is None for row in output):  # pragma: no cover - defensive invariant
        raise RuntimeError("internal error: not every row received a final label")
    return [row for row in output if row is not None]
