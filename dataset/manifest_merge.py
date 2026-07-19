"""Safely combine model-ready DACVAE latent manifests."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset._io_utils import atomic_write_text

# One DLsite work (RJ code) is voiced by one CV in this corpus, so the RJ code
# is the true speaker unit. Raw pipeline rows carry per-source-file speaker ids
# (namespace:RJxxx_track_hash); collapsing them here gives cross-track reference
# sampling and leak-free speaker-grouped validation splits.
_RJ_CODE_RE = re.compile(r"RJ\d+", re.IGNORECASE)


@dataclass(frozen=True)
class LatentManifestMergeResult:
    """Audit information for a merged, model-ready latent manifest."""

    output_manifest: Path
    input_manifests: tuple[Path, ...]
    input_counts: tuple[int, ...]
    combined_count: int
    speaker_normalized_count: int = 0


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


def normalize_speaker_id_to_work(speaker_id: str) -> str:
    """Collapse a per-source-file speaker id to its RJ work code when present."""
    match = _RJ_CODE_RE.search(speaker_id)
    if match is None:
        return speaker_id
    return match.group(0).upper()


def merge_latent_training_manifests(
    input_manifests: Sequence[str | Path],
    output_manifest: str | Path,
    *,
    require_latent_files: bool = True,
    normalize_speaker_rj: bool = False,
) -> LatentManifestMergeResult:
    """Safely combine DACVAE latent manifests without changing their sources.

    Relative ``latent_path`` values are resolved from each source manifest and
    rewritten relative to the output manifest, so inputs from different
    directories retain the same file meaning. All row metadata is preserved,
    except that ``normalize_speaker_rj=True`` collapses ``speaker_id`` to its
    RJ work code (rows without an RJ code keep their original id).
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
    speaker_normalized_count = 0
    for source in sources:
        rows = _read_jsonl_objects(source)
        input_counts.append(len(rows))
        for line_index, source_row in enumerate(rows, 1):
            row = dict(source_row)
            if normalize_speaker_rj:
                raw_speaker = row.get("speaker_id")
                if raw_speaker is not None:
                    normalized_speaker = normalize_speaker_id_to_work(str(raw_speaker))
                    if normalized_speaker != raw_speaker:
                        row["speaker_id"] = normalized_speaker
                        speaker_normalized_count += 1
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
    atomic_write_text(output, _jsonl_text(merged))
    return LatentManifestMergeResult(
        output_manifest=output,
        input_manifests=sources,
        input_counts=tuple(input_counts),
        combined_count=len(merged),
        speaker_normalized_count=speaker_normalized_count,
    )
