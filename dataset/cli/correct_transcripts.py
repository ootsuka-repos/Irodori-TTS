#!/usr/bin/env python3
"""Correct Grok STT and anime-whisper text with shared temporal context."""

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
from dataset.transcript_correction import (
    CorrectionConfig,
    correct_transcript_rows,
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


def _parse_priority(value: str) -> tuple[str, ...]:
    result = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("agent priority cannot be empty")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correct both Grok STT and anime-whisper results in overlapping temporal "
            "context windows. Codex, Claude, then Grok CLI are tried by default."
        )
    )
    parser.add_argument("--speech-all", type=Path, default=Path("dataset/data/grok_stt/all.jsonl"))
    parser.add_argument(
        "--speech-output-dir",
        type=Path,
        default=Path("dataset/data/grok_stt"),
        help="Atomically rewrites all/train/review JSONL under this directory.",
    )
    parser.add_argument(
        "--nonverbal-input",
        type=Path,
        default=Path(
            "dataset/data/grok_stt/nonverbal_events_new/dataset/manifests/events_transcribed.jsonl"
        ),
    )
    parser.add_argument(
        "--nonverbal-output",
        type=Path,
        default=Path("dataset/data/grok_stt/nonverbal_events_new/dataset/manifests/events_corrected.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("dataset/data/grok_stt/text_correction"),
    )
    parser.add_argument(
        "--agent-priority",
        type=_parse_priority,
        default=("codex", "claude", "grok"),
        help="Comma-separated CLI fallback order (default: codex,claude,grok).",
    )
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--context-segments", type=int, default=10)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--attempts-per-agent", type=int, default=2)
    parser.add_argument("--minimum-similarity", type=float, default=0.60)
    parser.add_argument("--minimum-length-ratio", type=float, default=0.45)
    parser.add_argument("--maximum-length-ratio", type=float, default=1.65)
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--claude-model", default=None)
    parser.add_argument("--grok-model", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-batch-failures",
        action="store_true",
        help="Publish unchanged text even if every configured CLI fails for a batch.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = _parse_args(argv)
    speech_rows = _read_jsonl(args.speech_all.expanduser().resolve())
    nonverbal_rows = _read_jsonl(args.nonverbal_input.expanduser().resolve())
    config = CorrectionConfig(
        agent_priority=tuple(args.agent_priority),
        target_batch_size=args.target_batch_size,
        context_segments=args.context_segments,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        attempts_per_agent=args.attempts_per_agent,
        minimum_similarity=args.minimum_similarity,
        minimum_length_ratio=args.minimum_length_ratio,
        maximum_length_ratio=args.maximum_length_ratio,
        codex_model=args.codex_model,
        claude_model=args.claude_model,
        grok_model=args.grok_model,
    )
    speech, nonverbal, summary = correct_transcript_rows(
        speech_rows,
        nonverbal_rows,
        cache_dir=args.cache_dir,
        config=config,
        force=args.force,
    )
    if summary["batches_failed"] and not args.allow_batch_failures:
        failure_path = args.cache_dir.expanduser().resolve() / "last_failures.json"
        _atomic_write_json(failure_path, summary)
        raise SystemExit(
            f"Transcript correction failed for {summary['batches_failed']} batch(es); "
            f"unchanged manifests were not published. See {failure_path}"
        )

    speech_dir = args.speech_output_dir.expanduser().resolve()
    speech.sort(key=lambda row: (str(row.get("source_uid", "")), float(row.get("start", 0))))
    _atomic_write_jsonl(speech_dir / "all.jsonl", speech)
    _atomic_write_jsonl(
        speech_dir / "train.jsonl",
        [row for row in speech if row.get("status") == "train"],
    )
    _atomic_write_jsonl(
        speech_dir / "review.jsonl",
        [row for row in speech if row.get("status") == "review"],
    )
    nonverbal_path = args.nonverbal_output.expanduser().resolve()
    _atomic_write_jsonl(nonverbal_path, nonverbal)
    summary_path = args.cache_dir.expanduser().resolve() / "last_summary.json"
    _atomic_write_json(summary_path, summary)
    _atomic_write_jsonl(
        args.cache_dir.expanduser().resolve() / "last_changes.jsonl",
        summary["changes"],
    )
    print(
        f"correction complete batches={summary['batches_completed']}/{summary['batches']} "
        f"accepted={summary['accepted']} unchanged={summary['unchanged']} "
        f"uncertain={summary['uncertain']} rejected={summary['rejected_unsafe']} "
        f"speech={speech_dir / 'train.jsonl'} nonverbal={nonverbal_path}"
    )


if __name__ == "__main__":
    main()
