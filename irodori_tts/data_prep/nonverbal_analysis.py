"""Secondary analysis for nonverbal segments and cluster representatives.

The VA embedding path intentionally accepts a generic segment manifest.  The
representative-manifest adapter is kept separate so future natural event
segments can reuse inference without depending on the current BEATs output.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ANIME_VA_MODEL_ID = "litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm"
ANIME_VA_VARIANT = "va"
ANIME_VA_EMBEDDING_DIM = 192
ANIME_VA_SAMPLE_RATE = 16_000

REPRESENTATIVE_MANIFEST_FILENAME = "representative_manifest.jsonl"
VA_EMBEDDINGS_FILENAME = "va_embeddings.npy"
VA_INDEX_FILENAME = "va_embedding_index.jsonl"
VA_FAILURES_FILENAME = "va_embedding_failures.jsonl"
VA_CLUSTER_METRICS_FILENAME = "va_cluster_metrics.json"
VA_SUMMARY_FILENAME = "va_analysis_summary.json"

_REPRESENTATIVE_FIELDS = (
    "source_uid",
    "source_audio",
    "speaker_id",
    "start",
    "end",
    "duration",
)


@dataclass(frozen=True)
class VAEmbeddingResult:
    """Successful embeddings and auditable rows for one inference pass."""

    embeddings: np.ndarray
    index_rows: list[dict[str, Any]]
    failure_rows: list[dict[str, Any]]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _atomic_path(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    return descriptor, Path(temporary_name)


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary = _atomic_path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
    )


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"), default=_json_default)
        + "\n"
        for row in rows
    )
    _atomic_write_text(path, text)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    descriptor, temporary = _atomic_path(path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL manifest and require one object per non-empty line."""
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


def _cluster_rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        payload = payload.get("clusters")
    if not isinstance(payload, list) or not all(isinstance(row, Mapping) for row in payload):
        raise ValueError("cluster_summary must be a list, or an object containing a clusters list")
    return payload


def _kind_sort_key(kind: str) -> tuple[int, str]:
    priorities = {"hdbscan": 0, "kmeans": 1}
    return priorities.get(kind, 2), kind


def _metadata_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    return left == right


def build_representative_manifest(
    cluster_summary: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build one manifest row per representative ID.

    An ID may represent both an HDBSCAN and K-means cluster, and the upstream
    selector can occasionally repeat one ID at multiple ranks.  The audio is
    embedded only once while every occurrence remains in ``memberships`` for
    later cluster-level aggregation.
    """
    clusters = _cluster_rows(cluster_summary)
    occurrences: list[tuple[tuple[Any, ...], Mapping[str, Any], Mapping[str, Any]]] = []
    for cluster_position, cluster in enumerate(clusters):
        kind = str(cluster.get("kind", "")).strip()
        cluster_id = str(cluster.get("cluster_id", "")).strip()
        if not kind or not cluster_id:
            raise ValueError(f"Cluster at index {cluster_position} needs kind and cluster_id")
        representatives = cluster.get("representatives", [])
        if not isinstance(representatives, list):
            raise ValueError(f"representatives for {kind}/{cluster_id} must be a list")
        for representative_position, representative in enumerate(representatives):
            if not isinstance(representative, Mapping):
                raise ValueError(
                    f"Representative {representative_position} in {kind}/{cluster_id} is not an object"
                )
            segment_id = str(representative.get("id", "")).strip()
            audio = str(representative.get("audio", "")).strip()
            if not segment_id or not audio:
                raise ValueError(
                    f"Representative {representative_position} in {kind}/{cluster_id} "
                    "needs id and audio"
                )
            rank_value = representative.get("rank", representative_position + 1)
            try:
                rank = int(rank_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid rank for representative {segment_id}: {rank_value!r}"
                ) from exc
            sort_key = (
                *_kind_sort_key(kind),
                cluster_id,
                rank,
                representative_position,
                segment_id,
            )
            occurrence_cluster = {
                "kind": kind,
                "cluster_id": cluster_id,
                "rank": rank,
                "role": str(representative.get("role", "")).strip(),
                "distance": representative.get("distance"),
                "representative_audio": audio,
            }
            occurrences.append((sort_key, representative, occurrence_cluster))

    rows_by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for _, representative, membership in sorted(occurrences, key=lambda item: item[0]):
        segment_id = str(representative["id"]).strip()
        audio = str(representative["audio"]).strip()
        row = rows_by_id.get(segment_id)
        if row is None:
            row = {
                "id": segment_id,
                "sample_type": "cluster_representative",
                "audio": audio,
                "audio_alternatives": [],
                "memberships": [],
            }
            for field in _REPRESENTATIVE_FIELDS:
                if field in representative:
                    row[field] = representative[field]
            rows_by_id[segment_id] = row
            order.append(segment_id)
        else:
            for field in _REPRESENTATIVE_FIELDS:
                if field not in representative or field not in row:
                    continue
                if not _metadata_equal(row[field], representative[field]):
                    raise ValueError(
                        f"Conflicting {field!r} for duplicate representative ID {segment_id!r}"
                    )
        if audio != row["audio"] and audio not in row["audio_alternatives"]:
            row["audio_alternatives"].append(audio)
        row["memberships"].append(membership)

    manifest: list[dict[str, Any]] = []
    for segment_id in order:
        row = rows_by_id[segment_id]
        row["occurrence_count"] = len(row["memberships"])
        manifest.append(row)
    return manifest


def create_representative_manifest(
    cluster_summary_path: Path,
    *,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Read ``cluster_summary.json`` and optionally atomically write its manifest."""
    payload = json.loads(cluster_summary_path.read_text(encoding="utf-8"))
    manifest = build_representative_manifest(payload)
    if output_path is not None:
        _atomic_write_jsonl(output_path, manifest)
    return manifest


def _load_runtime_modules() -> tuple[Any, Any, Any, Any]:
    try:
        soundfile = importlib.import_module("soundfile")
        torch = importlib.import_module("torch")
        torch_f = importlib.import_module("torch.nn.functional")
        torchaudio_f = importlib.import_module("torchaudio.functional")
    except ImportError as exc:
        raise RuntimeError(
            "VA inference requires soundfile, torch, and torchaudio in the runtime environment"
        ) from exc
    return soundfile, torch, torch_f, torchaudio_f


def load_anime_va_model(
    *,
    device: str = "cuda",
    variant: str = ANIME_VA_VARIANT,
    sample_rate: int = ANIME_VA_SAMPLE_RATE,
    checkpoint_path: Path | None = None,
) -> Any:
    """Dynamically import and load the anime-domain ECAPA-TDNN model."""
    try:
        module = importlib.import_module("anime_speaker_embedding")
        model_class = module.AnimeSpeakerEmbedding
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Install anime-speaker-embedding to enable VA inference; it is intentionally "
            "an optional runtime dependency"
        ) from exc

    torch = importlib.import_module("torch")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} was requested, but CUDA is unavailable")
    kwargs: dict[str, Any] = {
        "device": device,
        "variant": variant,
        "sr": sample_rate,
    }
    if checkpoint_path is not None:
        kwargs["ckpt_path"] = checkpoint_path
    model = model_class(**kwargs)
    model.eval()
    return model


def _audio_candidates(row: Mapping[str, Any]) -> list[str]:
    values: list[Any] = [row.get("audio")]
    alternatives = row.get("audio_alternatives", [])
    if isinstance(alternatives, list):
        values.extend(alternatives)
    memberships = row.get("memberships", [])
    if isinstance(memberships, list):
        values.extend(
            membership.get("representative_audio")
            for membership in memberships
            if isinstance(membership, Mapping)
        )
    result: list[str] = []
    for value in values:
        candidate = str(value or "").strip()
        if candidate and candidate not in result:
            result.append(candidate)
    return result


def _resolve_audio_path(row: Mapping[str, Any], audio_root: Path | None) -> Path:
    candidates = _audio_candidates(row)
    if not candidates:
        raise ValueError("segment row needs a non-empty audio path")
    resolved: list[Path] = []
    for value in candidates:
        path = Path(value)
        if not path.is_absolute() and audio_root is not None:
            path = audio_root / path
        resolved.append(path)
    return next((path for path in resolved if path.is_file()), resolved[0])


def _load_waveform(
    path: Path,
    *,
    device: str,
    sample_rate: int,
    soundfile: Any,
    torch: Any,
    torchaudio_f: Any,
) -> Any:
    audio, source_rate = soundfile.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    if audio.shape[0] == 0 or audio.shape[1] == 0:
        raise ValueError("audio contains no samples")
    if not np.isfinite(audio).all():
        raise ValueError("audio contains non-finite samples")
    mono = np.asarray(audio.mean(axis=1, dtype=np.float32), dtype=np.float32)
    waveform = torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0).to(device)
    if int(source_rate) != sample_rate:
        waveform = torchaudio_f.resample(waveform, int(source_rate), sample_rate)
    return waveform


def extract_va_embeddings(
    segments: Sequence[Mapping[str, Any]],
    *,
    device: str = "cuda",
    variant: str = ANIME_VA_VARIANT,
    sample_rate: int = ANIME_VA_SAMPLE_RATE,
    expected_dimension: int = ANIME_VA_EMBEDDING_DIM,
    checkpoint_path: Path | None = None,
    audio_root: Path | None = None,
    model: Any | None = None,
) -> VAEmbeddingResult:
    """Extract one VA embedding per successful generic segment row.

    Broken or unsupported audio is recorded in ``failure_rows`` and does not
    shift the relationship between the NPY rows and ``embedding_index`` in the
    returned index rows.  Model-loading errors remain fatal configuration errors.
    """
    if expected_dimension <= 0:
        raise ValueError("expected_dimension must be positive")
    segment_ids = [str(row.get("id", "")).strip() for row in segments]
    if any(not segment_id for segment_id in segment_ids):
        raise ValueError("Every segment row needs a non-empty id")
    if len(segment_ids) != len(set(segment_ids)):
        raise ValueError("Segment IDs must be unique before VA inference")
    if not segments:
        return VAEmbeddingResult(
            embeddings=np.empty((0, expected_dimension), dtype=np.float32),
            index_rows=[],
            failure_rows=[],
        )

    soundfile, torch, torch_f, torchaudio_f = _load_runtime_modules()
    if model is None:
        model = load_anime_va_model(
            device=device,
            variant=variant,
            sample_rate=sample_rate,
            checkpoint_path=checkpoint_path,
        )

    vectors: list[np.ndarray] = []
    index_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for manifest_index, row in enumerate(segments):
        segment_id = segment_ids[manifest_index]
        resolved_path: Path | None = None
        try:
            resolved_path = _resolve_audio_path(row, audio_root)
            waveform = _load_waveform(
                resolved_path,
                device=device,
                sample_rate=sample_rate,
                soundfile=soundfile,
                torch=torch,
                torchaudio_f=torchaudio_f,
            )
            with torch.inference_mode():
                embedding = model(waveform).reshape(-1)
                if embedding.numel() != expected_dimension:
                    raise ValueError(
                        f"model returned {embedding.numel()} values; expected {expected_dimension}"
                    )
                if not torch.isfinite(embedding).all().item():
                    raise ValueError("model returned a non-finite embedding")
                norm = torch.linalg.vector_norm(embedding)
                if not torch.isfinite(norm).item() or norm.item() <= 1e-12:
                    raise ValueError("model returned a zero-norm embedding")
                embedding = torch_f.normalize(embedding, dim=0)
            vector = np.asarray(embedding.detach().cpu().numpy(), dtype=np.float32)
            output_row = dict(row)
            output_row["manifest_index"] = manifest_index
            output_row["embedding_index"] = len(vectors)
            output_row["resolved_audio"] = resolved_path.as_posix()
            vectors.append(vector)
            index_rows.append(output_row)
        except Exception as exc:
            failure: dict[str, Any] = {
                "manifest_index": manifest_index,
                "id": segment_id,
                "audio": str(row.get("audio", "")),
                "stage": "audio_or_inference",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if resolved_path is not None:
                failure["resolved_audio"] = resolved_path.as_posix()
            if "memberships" in row:
                failure["memberships"] = row["memberships"]
            failure_rows.append(failure)

    embeddings = (
        np.stack(vectors).astype(np.float32, copy=False)
        if vectors
        else np.empty((0, expected_dimension), dtype=np.float32)
    )
    return VAEmbeddingResult(
        embeddings=embeddings,
        index_rows=index_rows,
        failure_rows=failure_rows,
    )


def summarize_cosines(values: np.ndarray | Sequence[float]) -> dict[str, float | int | None]:
    """Return JSON-safe descriptive statistics for cosine values."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    if not np.isfinite(array).all():
        raise ValueError("Cosine values must be finite")
    array = np.clip(array, -1.0, 1.0)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def _row_memberships(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    memberships = row.get("memberships")
    if isinstance(memberships, list):
        return [membership for membership in memberships if isinstance(membership, Mapping)]
    cluster_id = str(row.get("cluster_id", "")).strip()
    if not cluster_id:
        return []
    return [
        {
            "kind": row.get("cluster_kind", row.get("kind", "cluster")),
            "cluster_id": cluster_id,
            "role": row.get("role", ""),
        }
    ]


def _normalized_rows(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("embeddings must have shape (N, D)")
    if not np.isfinite(array).all():
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1e-12):
        bad = np.flatnonzero(norms <= 1e-12).tolist()
        raise ValueError(f"embeddings contain zero-norm rows: {bad}")
    return array / norms[:, None]


def aggregate_cluster_cosine_metrics(
    embeddings: np.ndarray,
    index_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate near/boundary cosine diagnostics for each cluster membership.

    Membership occurrences are collapsed by embedding index and role, so a
    representative repeated at two ranks cannot create a self-pair or bias a
    centroid.  This function is pure and independent of the embedding model.
    """
    normalized = _normalized_rows(embeddings)
    if len(index_rows) != normalized.shape[0]:
        raise ValueError("index_rows length must equal the number of embedding rows")

    rows_by_index: dict[int, Mapping[str, Any]] = {}
    roles: dict[tuple[str, str], dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row_position, row in enumerate(index_rows):
        embedding_index = int(row.get("embedding_index", row_position))
        if not 0 <= embedding_index < normalized.shape[0]:
            raise ValueError(f"embedding_index is outside the NPY array: {embedding_index}")
        if embedding_index in rows_by_index:
            raise ValueError(f"Duplicate embedding_index in index_rows: {embedding_index}")
        rows_by_index[embedding_index] = row
        for membership in _row_memberships(row):
            kind = str(membership.get("kind", "cluster")).strip() or "cluster"
            cluster_id = str(membership.get("cluster_id", "")).strip()
            if not cluster_id:
                continue
            role = str(membership.get("role", "")).strip().lower()
            roles[(kind, cluster_id)][embedding_index].add(role)

    metrics: list[dict[str, Any]] = []
    for (kind, cluster_id), role_by_index in sorted(
        roles.items(), key=lambda item: (*_kind_sort_key(item[0][0]), item[0][1])
    ):
        near_indices = sorted(index for index, values in role_by_index.items() if "near" in values)
        boundary_indices = sorted(
            index
            for index, values in role_by_index.items()
            if "boundary" in values and "near" not in values
        )
        near = normalized[near_indices]
        boundary = normalized[boundary_indices]

        if len(near) >= 2:
            pairwise_matrix = near @ near.T
            triangle = np.triu_indices(len(near), 1)
            near_pairwise = pairwise_matrix[triangle]
        else:
            near_pairwise = np.empty(0, dtype=np.float64)

        centroid: np.ndarray | None = None
        resultant_norm: float | None = None
        near_to_centroid = np.empty(0, dtype=np.float64)
        if len(near):
            centroid_mean = near.mean(axis=0)
            resultant_norm = float(np.linalg.norm(centroid_mean))
            if resultant_norm > 1e-12:
                centroid = centroid_mean / resultant_norm
                near_to_centroid = near @ centroid

        if len(boundary) and len(near):
            boundary_to_near_matrix = boundary @ near.T
            boundary_to_near = boundary_to_near_matrix.reshape(-1)
        else:
            boundary_to_near_matrix = np.empty((len(boundary), len(near)), dtype=np.float64)
            boundary_to_near = np.empty(0, dtype=np.float64)
        boundary_to_centroid = (
            boundary @ centroid if centroid is not None and len(boundary) else np.empty(0)
        )

        boundary_rows: list[dict[str, Any]] = []
        for position, embedding_index in enumerate(boundary_indices):
            pair_values = (
                boundary_to_near_matrix[position]
                if boundary_to_near_matrix.shape[1]
                else np.empty(0)
            )
            row = rows_by_index[embedding_index]
            boundary_rows.append(
                {
                    "id": str(row.get("id", "")),
                    "embedding_index": embedding_index,
                    "to_near_cosine": summarize_cosines(pair_values),
                    "to_near_centroid_cosine": (
                        float(boundary_to_centroid[position]) if boundary_to_centroid.size else None
                    ),
                }
            )

        metrics.append(
            {
                "kind": kind,
                "cluster_id": cluster_id,
                "near_count": len(near_indices),
                "boundary_count": len(boundary_indices),
                "near_ids": [str(rows_by_index[index].get("id", "")) for index in near_indices],
                "boundary_ids": [
                    str(rows_by_index[index].get("id", "")) for index in boundary_indices
                ],
                "near_pairwise_cosine": summarize_cosines(near_pairwise),
                "near_centroid": centroid.tolist() if centroid is not None else None,
                "near_centroid_resultant_norm": resultant_norm,
                "near_to_centroid_cosine": summarize_cosines(near_to_centroid),
                "boundary_to_near_cosine": summarize_cosines(boundary_to_near),
                "boundary_to_near_centroid_cosine": summarize_cosines(boundary_to_centroid),
                "boundaries": boundary_rows,
            }
        )
    return metrics


def write_va_analysis_artifacts(
    output_dir: Path,
    result: VAEmbeddingResult,
    cluster_metrics: Sequence[Mapping[str, Any]],
    *,
    input_count: int,
    device: str,
    variant: str,
    sample_rate: int,
) -> dict[str, Path]:
    """Atomically commit VA artifacts, writing the summary marker last."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "embeddings": output_dir / VA_EMBEDDINGS_FILENAME,
        "index": output_dir / VA_INDEX_FILENAME,
        "failures": output_dir / VA_FAILURES_FILENAME,
        "cluster_metrics": output_dir / VA_CLUSTER_METRICS_FILENAME,
        "summary": output_dir / VA_SUMMARY_FILENAME,
    }
    _atomic_save_npy(paths["embeddings"], result.embeddings)
    _atomic_write_jsonl(paths["index"], result.index_rows)
    _atomic_write_jsonl(paths["failures"], result.failure_rows)
    _atomic_write_json(paths["cluster_metrics"], list(cluster_metrics))
    _atomic_write_json(
        paths["summary"],
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_id": ANIME_VA_MODEL_ID,
            "variant": variant,
            "device": device,
            "sample_rate": sample_rate,
            "embedding_dimension": int(result.embeddings.shape[1]),
            "input_count": input_count,
            "success_count": int(result.embeddings.shape[0]),
            "failure_count": len(result.failure_rows),
            "cluster_count": len(cluster_metrics),
            "files": {key: path.name for key, path in paths.items() if key != "summary"},
        },
    )
    return paths


def analyze_segment_manifest(
    segments: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    device: str = "cuda",
    variant: str = ANIME_VA_VARIANT,
    sample_rate: int = ANIME_VA_SAMPLE_RATE,
    expected_dimension: int = ANIME_VA_EMBEDDING_DIM,
    checkpoint_path: Path | None = None,
    audio_root: Path | None = None,
    model: Any | None = None,
) -> dict[str, Path]:
    """Run generic segment VA inference, aggregation, and atomic artifact writes."""
    result = extract_va_embeddings(
        segments,
        device=device,
        variant=variant,
        sample_rate=sample_rate,
        expected_dimension=expected_dimension,
        checkpoint_path=checkpoint_path,
        audio_root=audio_root,
        model=model,
    )
    cluster_metrics = aggregate_cluster_cosine_metrics(result.embeddings, result.index_rows)
    return write_va_analysis_artifacts(
        output_dir,
        result,
        cluster_metrics,
        input_count=len(segments),
        device=device,
        variant=variant,
        sample_rate=sample_rate,
    )


def analyze_cluster_representatives(
    cluster_summary_path: Path,
    output_dir: Path,
    **embedding_options: Any,
) -> dict[str, Path]:
    """Compatibility adapter from the current representative summary to generic analysis."""
    manifest_path = output_dir / REPRESENTATIVE_MANIFEST_FILENAME
    manifest = create_representative_manifest(
        cluster_summary_path,
        output_path=manifest_path,
    )
    paths = analyze_segment_manifest(manifest, output_dir, **embedding_options)
    return {"manifest": manifest_path, **paths}
