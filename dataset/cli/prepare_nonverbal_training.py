#!/usr/bin/env python3
"""Create conservative Irodori-TTS manifests from finalized nonverbal events."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dataset.nonverbal_training_manifest import (
    NonverbalTrainingManifestConfig,
    write_nonverbal_training_manifests,
)

DEFAULT_DATASET_ROOT = Path("data/grok_stt/nonverbal_events/dataset")
DEFAULT_EVENTS_MANIFEST = DEFAULT_DATASET_ROOT / "manifests/events.jsonl"
DEFAULT_SPEECH_MANIFEST = Path("data/grok_stt/train.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL without accepting non-object rows or hiding line errors."""
    rows: list[dict[str, Any]] = []
    try:
        handle = path.expanduser().open("r", encoding="utf-8-sig")
    except OSError as exc:
        raise OSError(f"Could not open JSONL manifest: {path}") from exc

    with handle:
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate finalized aegi/chupa events and publish train/review manifests "
            "for Irodori-TTS."
        )
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--speech-manifest", type=Path, default=DEFAULT_SPEECH_MANIFEST)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Base used for audio paths written to the generated manifests.",
    )
    parser.add_argument(
        "--audio-base-dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Base for relative audio paths in --events (default: dataset root).",
    )

    # Quality gates default to None: only explicitly given flags override the
    # canonical NonverbalTrainingManifestConfig dataclass defaults.
    quality = parser.add_argument_group("training quality gates")
    quality.add_argument("--minimum-seconds", type=float, default=None)
    quality.add_argument("--maximum-seconds", type=float, default=None)
    quality.add_argument("--classifier-minimum-probability", type=float, default=None)
    quality.add_argument("--classifier-minimum-margin", type=float, default=None)
    quality.add_argument("--neighbor-minimum-count", type=int, default=None)
    quality.add_argument("--neighbor-minimum-support", type=float, default=None)
    quality.add_argument("--neighbor-minimum-similarity", type=float, default=None)
    quality.add_argument("--speech-overlap-minimum-seconds", type=float, default=None)
    quality.add_argument(
        "--require-cluster",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    quality.add_argument(
        "--require-speaker-id",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    quality.add_argument(
        "--require-audio-file",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args(argv)


def _manifest_config_from_args(args: argparse.Namespace) -> NonverbalTrainingManifestConfig:
    """Build the manifest config only from flags the user actually passed."""
    overrides = {
        "minimum_seconds": args.minimum_seconds,
        "maximum_seconds": args.maximum_seconds,
        "classifier_minimum_probability": args.classifier_minimum_probability,
        "classifier_minimum_margin": args.classifier_minimum_margin,
        "neighbor_minimum_count": args.neighbor_minimum_count,
        "neighbor_minimum_support": args.neighbor_minimum_support,
        "neighbor_minimum_similarity": args.neighbor_minimum_similarity,
        "speech_overlap_minimum_seconds": args.speech_overlap_minimum_seconds,
        "require_cluster": args.require_cluster,
        "require_speaker_id": args.require_speaker_id,
        "require_audio_file": args.require_audio_file,
    }
    provided = {key: value for key, value in overrides.items() if value is not None}
    return NonverbalTrainingManifestConfig(**provided)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    event_rows = _read_jsonl(args.events)
    speech_rows = _read_jsonl(args.speech_manifest)
    config = _manifest_config_from_args(args)
    result = write_nonverbal_training_manifests(
        event_rows,
        args.output_dir,
        config=config,
        speech_rows=speech_rows,
        path_base=args.project_root,
        audio_base_dir=args.audio_base_dir,
    )
    print(
        f"complete events={len(event_rows)} nonverbal_train={result.train_count} "
        f"nonverbal_review={result.review_count} discarded={result.discarded_count} "
        f"speech={result.speech_count} combined={result.combined_count} "
        f"train_audio_seconds={result.train_audio_seconds:.3f} "
        f"combined_manifest={result.combined_manifest}"
    )


if __name__ == "__main__":
    main()
