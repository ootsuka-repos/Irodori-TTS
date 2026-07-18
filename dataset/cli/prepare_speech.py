#!/usr/bin/env python3
"""Cut 5-20 s speech clips with local Silero VAD (no cloud STT involved)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dataset.speech_pipeline import (
    SileroVADConfig,
    SpeechSegmentationConfig,
    build_dataset,
    discover_audio_sources,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover source audio, run Silero VAD, and cut padded 5-20 s FLAC "
            "clips plus all/train/review manifests. Text starts empty; the "
            "anime-whisper stage supplies transcripts."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("dataset/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/data/pipeline"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--min-source-seconds", type=float, default=300.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--vad-device", default="cpu")
    parser.add_argument("--vad-threshold", type=float, default=None)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=None)
    parser.add_argument("--min-seconds", type=float, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--join-gap-seconds", type=float, default=None)
    parser.add_argument("--padding-seconds", type=float, default=None)
    parser.add_argument("--rebuild-clips", action="store_true")
    parser.add_argument(
        "--max-source-errors",
        type=int,
        default=0,
        help="Fail when more than this many source errors are recorded.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        type=Path,
        default=None,
        help="Directory below --input-dir to skip during discovery (repeatable).",
    )
    return parser.parse_args(argv)


def _override(config, **updates):
    values = {key: value for key, value in updates.items() if value is not None}
    return type(config)(**{**config.__dict__, **values}) if values else config


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = _parse_args(argv)

    vad_config = _override(
        SileroVADConfig(),
        threshold=args.vad_threshold,
        speech_pad_ms=args.vad_speech_pad_ms,
    )
    segmentation_config = _override(
        SpeechSegmentationConfig(),
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        join_gap_seconds=args.join_gap_seconds,
        padding_seconds=args.padding_seconds,
    )

    excluded = tuple(args.exclude_dir or ())
    sources = discover_audio_sources(
        args.input_dir,
        output_dir=args.output_dir,
        excluded_dirs=excluded,
    )
    sources = [source for source in sources if source.duration > args.min_source_seconds]
    if args.max_files is not None:
        sources = sources[: args.max_files]
    if not sources:
        raise SystemExit(f"No eligible source audio found under {args.input_dir}")

    summary = build_dataset(
        sources,
        args.output_dir,
        vad_config=vad_config,
        segmentation_config=segmentation_config,
        project_root=args.project_root.expanduser().resolve(),
        vad_device=args.vad_device,
        rebuild_clips=args.rebuild_clips,
    )
    print(
        f"speech complete sources={summary['sources']} rows={summary['rows']} "
        f"train={summary['train']} review={summary['review']} "
        f"train_hours={summary['train_seconds'] / 3600.0:.2f} errors={summary['errors']}"
    )
    if summary["errors"] > args.max_source_errors:
        raise SystemExit(
            f"speech preparation recorded {summary['errors']} source error(s) "
            f"(limit {args.max_source_errors}); see source_errors.jsonl"
        )


if __name__ == "__main__":
    main()
