#!/usr/bin/env python3
"""Transcribe finalized VAD-excluded events with anime-whisper on CUDA."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dataset._io_utils import (
    atomic_write_json as _atomic_write_json,
)
from dataset._io_utils import (
    atomic_write_jsonl as _atomic_write_jsonl,
)
from dataset.local_asr import (
    ANIME_WHISPER_MODEL,
    ANIME_WHISPER_REVISION,
    AnimeWhisperConfig,
    AnimeWhisperTranscriber,
    transcribe_manifest_rows,
)

DEFAULT_OUTPUT_MANIFEST = Path(
    "data/grok_stt/nonverbal_events_new/dataset/manifests/events_transcribed.jsonl"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pinned litagin/anime-whisper locally on finalized VAD-excluded events. "
            "Each result is cached by audio fingerprint and model revision."
        )
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("data/grok_stt/nonverbal_events_new/dataset/manifests/events.jsonl"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help=(
            f"Output manifest path (default: {DEFAULT_OUTPUT_MANIFEST.as_posix()}). "
            "Must be passed explicitly when --max-rows is used, because the output "
            "would otherwise truncate the published manifest."
        ),
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("data/grok_stt/nonverbal_events_new/dataset"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/grok_stt/nonverbal_events_new/anime_whisper_cache"),
    )
    parser.add_argument("--model", default=ANIME_WHISPER_MODEL)
    parser.add_argument("--revision", default=ANIME_WHISPER_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-length-seconds", type=float, default=30.0)
    parser.add_argument("--language", default="Japanese")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = _parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    if args.output_manifest is None:
        if args.max_rows is not None:
            raise ValueError(
                "--max-rows truncates the written manifest; pass an explicit "
                "--output-manifest (for example a scratch path) when smoke-testing"
            )
        args.output_manifest = DEFAULT_OUTPUT_MANIFEST
    input_path = args.input_manifest.expanduser().resolve()
    rows = _read_jsonl(input_path)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    config = AnimeWhisperConfig(
        model=args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        chunk_length_seconds=args.chunk_length_seconds,
        language=args.language,
    )
    transcriber = AnimeWhisperTranscriber(config)
    enriched, summary = transcribe_manifest_rows(
        rows,
        audio_root=args.audio_root,
        cache_dir=args.cache_dir,
        transcriber=transcriber,
        batch_size=args.batch_size,
        force=args.force,
    )
    output_path = args.output_manifest.expanduser().resolve()
    _atomic_write_jsonl(output_path, enriched)
    summary_path = output_path.with_suffix(".summary.json")
    _atomic_write_json(summary_path, summary)
    print(
        f"anime-whisper complete rows={summary['rows']} ok={summary['ok']} "
        f"review={summary['review']} errors={summary['errors']} output={output_path}"
    )
    if summary["errors"]:
        raise SystemExit(
            f"anime-whisper completed with {summary['errors']} error(s); see {summary_path}"
        )


if __name__ == "__main__":
    main()
