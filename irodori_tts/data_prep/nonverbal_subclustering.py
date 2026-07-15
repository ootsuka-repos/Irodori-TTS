"""Deterministic BEATs subclustering inside finalized nonverbal event classes."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA

TARGET_EVENT_CLASSES = ("aegi", "chupa")


@dataclass(frozen=True)
class SubclusteringConfig:
    """Controls class-local dimensionality reduction and clustering."""

    label_field: str = "final_label"
    target_classes: tuple[str, ...] = TARGET_EVENT_CLASSES
    pca_components: int = 32
    hdbscan_min_samples_to_run: int = 5
    hdbscan_min_cluster_floor: int = 5
    hdbscan_min_cluster_ceiling: int = 30
    hdbscan_min_samples_ceiling: int = 10
    fallback_min_group_size: int = 40
    fallback_target_group_size: int = 60
    fallback_max_group_size: int = 80
    fallback_min_clusters: int = 1
    fallback_max_clusters: int = 128
    random_state: int = 42


def choose_hdbscan_min_cluster_size(
    sample_count: int,
    *,
    minimum: int = 5,
    maximum: int = 30,
) -> int:
    """Scale HDBSCAN's minimum cluster size conservatively with class size."""
    if sample_count < 0:
        raise ValueError("sample_count must not be negative")
    if minimum < 2 or maximum < minimum:
        raise ValueError("HDBSCAN bounds must satisfy 2 <= minimum <= maximum")
    if sample_count == 0:
        return 0
    scaled = math.ceil(math.sqrt(sample_count) / 2.0)
    return min(sample_count, max(2, min(maximum, max(minimum, scaled))))


def choose_fallback_cluster_count(
    sample_count: int,
    *,
    minimum_group_size: int = 40,
    target_group_size: int = 60,
    maximum_group_size: int = 80,
    minimum_clusters: int = 1,
    maximum_clusters: int = 128,
) -> int:
    """Choose K-means K, targeting 40--80 rows per cluster when feasible."""
    if sample_count < 0:
        raise ValueError("sample_count must not be negative")
    if not 1 <= minimum_group_size <= target_group_size <= maximum_group_size:
        raise ValueError("group sizes must satisfy 1 <= minimum <= target <= maximum")
    if minimum_clusters < 1 or maximum_clusters < minimum_clusters:
        raise ValueError("cluster bounds must satisfy 1 <= minimum <= maximum")
    if sample_count == 0:
        return 0

    lower_for_size = math.ceil(sample_count / maximum_group_size)
    upper_for_size = max(1, sample_count // minimum_group_size)
    rounded_target = math.floor(sample_count / target_group_size + 0.5)
    candidate = min(max(1, rounded_target), sample_count)

    feasible_lower = max(1, minimum_clusters, lower_for_size)
    feasible_upper = min(sample_count, maximum_clusters, upper_for_size)
    if feasible_lower <= feasible_upper:
        return min(feasible_upper, max(feasible_lower, candidate))

    # A hard cluster-count bound or a sub-minimum sample count can make the
    # desired average size impossible.  Honor the hard bounds and stay usable.
    return min(sample_count, maximum_clusters, max(minimum_clusters, candidate))


def _remap_labels_by_size(labels: np.ndarray) -> np.ndarray:
    counts = Counter(int(label) for label in labels if int(label) >= 0)
    mapping = {
        old: new
        for new, (old, _count) in enumerate(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )
    }
    return np.asarray([mapping.get(int(label), -1) for label in labels], dtype=np.int32)


def _reduce_class_embeddings(
    embeddings: np.ndarray,
    *,
    maximum_components: int,
    random_state: int,
) -> np.ndarray:
    sample_count, dimension = embeddings.shape
    if sample_count == 0:
        return np.empty((0, 0), dtype=np.float32)
    if sample_count == 1:
        reduced = embeddings.astype(np.float32, copy=True)
    else:
        component_count = min(maximum_components, dimension, sample_count - 1)
        if component_count < 1:
            reduced = embeddings.astype(np.float32, copy=True)
        else:
            solver = "randomized" if component_count < min(sample_count, dimension) else "full"
            reduced = PCA(
                n_components=component_count,
                svd_solver=solver,
                random_state=random_state,
            ).fit_transform(embeddings)
            reduced = reduced.astype(np.float32, copy=False)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    return reduced / np.maximum(norms, 1e-12)


def _centroid_distances(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    distances = np.full(len(labels), np.nan, dtype=np.float32)
    for label in sorted({int(value) for value in labels if int(value) >= 0}):
        indices = np.flatnonzero(labels == label)
        centroid = features[indices].mean(axis=0)
        distances[indices] = np.linalg.norm(features[indices] - centroid, axis=1)
    return distances


def _fallback_kmeans(
    features: np.ndarray,
    *,
    cluster_count: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = features.shape[0]
    if sample_count == 0 or cluster_count == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    unique_count = np.unique(features, axis=0).shape[0]
    cluster_count = min(cluster_count, sample_count, max(1, unique_count))
    if cluster_count == 1:
        labels = np.zeros(sample_count, dtype=np.int32)
    else:
        model = MiniBatchKMeans(
            n_clusters=cluster_count,
            batch_size=min(2048, max(256, sample_count)),
            n_init=10,
            random_state=random_state,
            reassignment_ratio=0.0,
        )
        labels = _remap_labels_by_size(model.fit_predict(features))
    return labels, _centroid_distances(features, labels)


def _hdbscan_class(
    features: np.ndarray,
    *,
    config: SubclusteringConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_count = features.shape[0]
    labels = np.full(sample_count, -1, dtype=np.int32)
    affinity = np.zeros(sample_count, dtype=np.float32)
    distances = np.full(sample_count, np.nan, dtype=np.float32)
    if sample_count < config.hdbscan_min_samples_to_run:
        return labels, affinity, distances

    minimum_cluster = choose_hdbscan_min_cluster_size(
        sample_count,
        minimum=config.hdbscan_min_cluster_floor,
        maximum=config.hdbscan_min_cluster_ceiling,
    )
    minimum_samples = min(
        minimum_cluster,
        config.hdbscan_min_samples_ceiling,
        max(2, minimum_cluster // 3),
    )
    model = HDBSCAN(
        min_cluster_size=minimum_cluster,
        min_samples=minimum_samples,
        metric="euclidean",
        n_jobs=-1,
        cluster_selection_method="eom",
        allow_single_cluster=True,
        copy=True,
    )
    labels = _remap_labels_by_size(model.fit_predict(features))
    affinity = np.asarray(model.probabilities_, dtype=np.float32)
    distances = _centroid_distances(features, labels)
    return labels, affinity, distances


def _validate_config(config: SubclusteringConfig) -> tuple[str, ...]:
    if not config.label_field:
        raise ValueError("label_field must not be empty")
    if config.pca_components < 1:
        raise ValueError("pca_components must be positive")
    if config.hdbscan_min_samples_to_run < 2:
        raise ValueError("hdbscan_min_samples_to_run must be at least 2")
    if config.hdbscan_min_samples_ceiling < 1:
        raise ValueError("hdbscan_min_samples_ceiling must be positive")
    classes = tuple(str(value).strip().lower() for value in config.target_classes)
    if not classes or any(not value for value in classes) or len(classes) != len(set(classes)):
        raise ValueError("target_classes must contain unique non-empty labels")
    # Reuse the public validators even when no target rows are present.
    choose_hdbscan_min_cluster_size(
        0,
        minimum=config.hdbscan_min_cluster_floor,
        maximum=config.hdbscan_min_cluster_ceiling,
    )
    choose_fallback_cluster_count(
        0,
        minimum_group_size=config.fallback_min_group_size,
        target_group_size=config.fallback_target_group_size,
        maximum_group_size=config.fallback_max_group_size,
        minimum_clusters=config.fallback_min_clusters,
        maximum_clusters=config.fallback_max_clusters,
    )
    return classes


def subcluster_event_embeddings(
    embeddings: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    config: SubclusteringConfig | None = None,
) -> list[dict[str, Any]]:
    """Subcluster aegi and chupa independently while retaining every input row.

    ``event_cluster`` is the density cluster and may be ``None`` for HDBSCAN
    noise or small classes.  ``fallback_cluster`` covers every aegi/chupa row.
    ``distance`` is measured to the density centroid when assigned, otherwise
    to the fallback centroid.  Other labels are retained but never clustered.
    """
    config = config or SubclusteringConfig()
    target_classes = _validate_config(config)
    array = np.asarray(embeddings)
    if array.ndim != 2:
        raise ValueError("embeddings must have shape (N, D)")
    if array.shape[0] != len(rows):
        raise ValueError("embeddings and rows must have the same length")
    if array.shape[1] < 1:
        raise ValueError("embeddings must have at least one feature")
    if not np.isfinite(array).all():
        raise ValueError("embeddings contain non-finite values")

    event_classes = [
        str(row.get(config.label_field, "uncertain") or "uncertain").strip().lower() or "uncertain"
        for row in rows
    ]
    output = [
        {
            **dict(row),
            "event_class": event_class,
            "event_cluster": None,
            "event_affinity": 0.0,
            "fallback_cluster": None,
            "distance": None,
        }
        for row, event_class in zip(rows, event_classes, strict=True)
    ]

    for event_class in target_classes:
        global_indices = np.asarray(
            [index for index, value in enumerate(event_classes) if value == event_class],
            dtype=np.int64,
        )
        if global_indices.size == 0:
            continue
        features = _reduce_class_embeddings(
            np.asarray(array[global_indices], dtype=np.float32),
            maximum_components=config.pca_components,
            random_state=config.random_state,
        )
        density_labels, affinity, density_distances = _hdbscan_class(
            features,
            config=config,
        )
        fallback_count = choose_fallback_cluster_count(
            len(global_indices),
            minimum_group_size=config.fallback_min_group_size,
            target_group_size=config.fallback_target_group_size,
            maximum_group_size=config.fallback_max_group_size,
            minimum_clusters=config.fallback_min_clusters,
            maximum_clusters=config.fallback_max_clusters,
        )
        fallback_labels, fallback_distances = _fallback_kmeans(
            features,
            cluster_count=fallback_count,
            random_state=config.random_state,
        )

        for local_index, global_index_value in enumerate(global_indices):
            global_index = int(global_index_value)
            density_label = int(density_labels[local_index])
            fallback_label = int(fallback_labels[local_index])
            distance = (
                density_distances[local_index]
                if density_label >= 0
                else fallback_distances[local_index]
            )
            output[global_index].update(
                {
                    "event_cluster": (
                        f"{event_class}_c{density_label:04d}" if density_label >= 0 else None
                    ),
                    "event_affinity": float(affinity[local_index]),
                    "fallback_cluster": f"{event_class}_k{fallback_label:04d}",
                    "distance": float(distance) if np.isfinite(distance) else None,
                }
            )
    return output


def _source_identity(row: Mapping[str, Any], index: int, source_field: str) -> str:
    value = str(row.get(source_field, "")).strip()
    if value:
        return value
    for fallback_field in ("source_uid", "source_key", "source_audio"):
        value = str(row.get(fallback_field, "")).strip()
        if value:
            return value
    return f"__row_{index}"


def select_diverse_representatives(
    indices: Sequence[int] | np.ndarray,
    distances: Sequence[float] | np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    near_count: int = 3,
    boundary_count: int = 2,
    source_field: str = "source_uid",
) -> list[tuple[int, str]]:
    """Select central and boundary rows, preferring unused source identities."""
    if near_count < 0 or boundary_count < 0:
        raise ValueError("representative counts must not be negative")
    distance_array = np.asarray(distances, dtype=np.float64).reshape(-1)
    if len(distance_array) != len(rows):
        raise ValueError("distances and rows must have the same length")
    candidate_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if len({int(index) for index in candidate_indices}) != len(candidate_indices):
        raise ValueError("indices must not contain duplicates")
    if np.any(candidate_indices < 0) or np.any(candidate_indices >= len(rows)):
        raise ValueError("indices contain an out-of-range row")
    if not np.isfinite(distance_array[candidate_indices]).all():
        raise ValueError("candidate distances must be finite")

    ascending = sorted(
        (int(index) for index in candidate_indices),
        key=lambda index: (distance_array[index], index),
    )
    descending = sorted(
        (int(index) for index in candidate_indices),
        key=lambda index: (-distance_array[index], index),
    )
    selected: list[tuple[int, str]] = []
    selected_indices: set[int] = set()
    used_sources: set[str] = set()

    def choose(order: Sequence[int], count: int, role: str) -> None:
        chosen_for_role = 0
        for require_new_source in (True, False):
            for index in order:
                if chosen_for_role >= count:
                    return
                if index in selected_indices:
                    continue
                source = _source_identity(rows[index], index, source_field)
                if require_new_source and source in used_sources:
                    continue
                selected.append((index, role))
                selected_indices.add(index)
                used_sources.add(source)
                chosen_for_role += 1

    choose(ascending, near_count, "near")
    choose(descending, boundary_count, "boundary")
    return selected


def select_representatives_by_cluster(
    rows: Sequence[Mapping[str, Any]],
    *,
    near_count: int = 3,
    boundary_count: int = 2,
    source_field: str = "source_uid",
) -> dict[str, list[tuple[int, str]]]:
    """Select representatives for each effective density-or-fallback cluster."""
    grouped: dict[str, list[int]] = defaultdict(list)
    distances: list[float] = []
    for index, row in enumerate(rows):
        cluster = str(row.get("event_cluster") or row.get("fallback_cluster") or "").strip()
        if cluster:
            grouped[cluster].append(index)
        value = row.get("distance")
        distances.append(float(value) if value is not None else math.nan)
    return {
        cluster: select_diverse_representatives(
            indices,
            distances,
            rows,
            near_count=near_count,
            boundary_count=boundary_count,
            source_field=source_field,
        )
        for cluster, indices in sorted(grouped.items())
    }
