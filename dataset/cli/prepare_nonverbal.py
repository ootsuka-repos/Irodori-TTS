#!/usr/bin/env python3
"""Cut 5-20 s nonverbal event clips from the VAD complement (no classifier)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dataset.nonverbal_pipeline import build_nonverbal_dataset
from dataset.speech_pipeline import SileroVADConfig, discover_audio_sources


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cut natural 5-20 s event clips from the regions Silero VAD did not "
            "mark as speech. Rows share the speech manifest shape with empty "
            "text; ASR + LLM correction assign transcript and category."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("dataset/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/data/pipeline"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--min-source-seconds", type=float, default=300.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--vad-speech-pad-ms",
        type=int,
        default=None,
        help="Must match the speech stage's value; part of the VAD cache key.",
    )
    parser.add_argument("--rebuild-clips", action="store_true")
    parser.add_argument("--max-source-errors", type=int, default=0)
    parser.add_argument(
        "--exclude-dir",
        action="append",
        type=Path,
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = _parse_args(argv)
    sources = discover_audio_sources(
        args.input_dir,
        output_dir=args.output_dir,
        excluded_dirs=tuple(args.exclude_dir or ()),
    )
    sources = [source for source in sources if source.duration > args.min_source_seconds]
    if args.max_files is not None:
        sources = sources[: args.max_files]
    if not sources:
        raise SystemExit(f"No eligible source audio found under {args.input_dir}")

    vad_config = SileroVADConfig()
    if args.vad_speech_pad_ms is not None:
        vad_config = SileroVADConfig(
            **{**vad_config.__dict__, "speech_pad_ms": int(args.vad_speech_pad_ms)}
        )
    summary = build_nonverbal_dataset(
        sources,
        args.output_dir,
        vad_config=vad_config,
        project_root=args.project_root.expanduser().resolve(),
        workers=args.workers,
        rebuild_clips=args.rebuild_clips,
    )
    print(
        f"nonverbal complete sources={summary['sources']} rows={summary['rows']} "
        f"train={summary['train']} review={summary['review']} "
        f"train_hours={summary['train_seconds'] / 3600.0:.2f} errors={summary['errors']}"
    )
    if summary["errors"] > args.max_source_errors:
        raise SystemExit(
            f"nonverbal preparation recorded {summary['errors']} error(s) "
            f"(limit {args.max_source_errors}); see nonverbal_errors.jsonl"
        )


if __name__ == "__main__":
    main()
