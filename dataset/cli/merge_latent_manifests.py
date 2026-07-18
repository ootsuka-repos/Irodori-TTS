#!/usr/bin/env python3
"""Combine model-ready DACVAE latent manifests without touching their sources."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dataset.nonverbal_training_manifest import (
    merge_latent_training_manifests,
)

DEFAULT_INPUT_MANIFESTS = (
    Path("dataset/data/manifests/speech.jsonl"),
    Path("dataset/data/manifests/nonverbal.jsonl"),
)
DEFAULT_OUTPUT_MANIFEST = Path("dataset/data/manifests/train.jsonl")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely combine speech and nonverbal DACVAE latent JSONL manifests "
            "into the model-ready training manifest."
        )
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        help=(
            "Input latent manifest; repeat in any order. Defaults to "
            "dataset/data/manifests/speech.jsonl and dataset/data/manifests/nonverbal.jsonl."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument(
        "--require-latent-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject rows whose latent_path does not exist (enabled by default).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    inputs = tuple(args.inputs) if args.inputs else DEFAULT_INPUT_MANIFESTS
    result = merge_latent_training_manifests(
        inputs,
        args.output,
        require_latent_files=args.require_latent_files,
    )
    counts = "+".join(str(count) for count in result.input_counts)
    print(
        f"complete inputs={len(result.input_manifests)} rows={counts} "
        f"combined={result.combined_count} output={result.output_manifest}"
    )


if __name__ == "__main__":
    main()
