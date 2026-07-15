#!/usr/bin/env python3
"""Transcribe local audio with Grok STT and build an Irodori-TTS dataset."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from typing import Any

from irodori_tts.data_prep.grok_stt import (
    XAI_BATCH_PRICE_USD_PER_HOUR,
    XAI_MAX_FILE_BYTES,
    XAI_STT_ENDPOINT,
    AudioSource,
    ChunkingConfig,
    GrokSTTClient,
    SegmentationConfig,
    SileroVADConfig,
    TranscriptionOptions,
    atomic_write_jsonl,
    build_dataset,
    discover_audio_sources,
    load_cached_response,
    load_silero_vad,
    transcribe_source_chunked,
    transcribe_source_with_vad,
    write_output_readme,
)

_THREAD_LOCAL = local()


def _load_grok_cli_oauth_token() -> str:
    """Read Grok CLI OAuth without logging or copying credentials elsewhere."""
    auth_path = Path.home() / ".grok" / "auth.json"
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    candidates = []
    for value in payload.values():
        if not isinstance(value, dict) or value.get("auth_mode") not in {"oauth", "oidc"}:
            continue
        token = value.get("key")
        if isinstance(token, str) and token.strip():
            candidates.append((str(value.get("expires_at", "")), token.strip()))
    return max(candidates, default=("", ""))[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe local long-form audio with xAI Grok Speech-to-Text, then create "
            "timestamp-aligned FLAC clips and JSONL manifests for Irodori-TTS."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/grok_stt"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect files, cache coverage, and estimated STT cost without writing or uploading.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Only process the first N files in deterministic path order (useful for a pilot).",
    )
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Save raw STT responses but do not create clips or manifests.",
    )
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Ignore valid cached responses and submit selected files again (incurs API cost).",
    )
    parser.add_argument(
        "--rebuild-clips",
        action="store_true",
        help="Rewrite FLAC clips even when matching files already exist.",
    )

    api = parser.add_argument_group("xAI Grok STT")
    api.add_argument("--api-key-env", default="XAI_API_KEY")
    api.add_argument("--endpoint", default=XAI_STT_ENDPOINT)
    api.add_argument("--language", default="ja")
    api.add_argument(
        "--format-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable inverse text normalization (disabled by default for closer speech labels).",
    )
    api.add_argument(
        "--filler-words",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep filler words when supported (enabled by default for TTS alignment).",
    )
    api.add_argument(
        "--diarize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request speaker labels; enable only for genuinely multi-speaker tracks.",
    )
    api.add_argument(
        "--multichannel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Transcribe channels independently. Keep disabled for binaural/stereo ASMR.",
    )
    api.add_argument(
        "--keyterm",
        action="append",
        default=[],
        help="Bias term for uncommon names/words; repeat up to 100 times.",
    )
    api.add_argument("--workers", type=int, default=2, help="Concurrent uploads (default: 2).")
    api.add_argument("--timeout-seconds", type=float, default=3600.0)
    api.add_argument("--max-retries", type=int, default=4)
    api.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pinned Silero VAD before STT (enabled by default).",
    )
    api.add_argument("--vad-repo", default="snakers4/silero-vad:v6.2.1")
    api.add_argument("--vad-threshold", type=float, default=0.35)
    api.add_argument("--vad-neg-threshold", type=float, default=0.20)
    api.add_argument("--vad-min-speech-ms", type=int, default=180)
    api.add_argument("--vad-min-silence-ms", type=int, default=450)
    api.add_argument("--vad-speech-pad-ms", type=int, default=120)
    api.add_argument("--vad-max-speech-seconds", type=float, default=29.0)
    api.add_argument("--vad-max-join-gap-seconds", type=float, default=1.2)
    api.add_argument("--vad-max-upload-seconds", type=float, default=29.5)
    api.add_argument(
        "--chunk-seconds",
        type=float,
        default=60.0,
        help="Internal STT upload chunk duration (default: 60 seconds).",
    )
    api.add_argument(
        "--chunk-overlap-seconds",
        type=float,
        default=1.0,
        help="Overlap used to avoid losing boundary words (default: 1 second).",
    )

    segmentation = parser.add_argument_group("clip segmentation")
    segmentation.add_argument("--min-seconds", type=float, default=1.0)
    segmentation.add_argument("--target-seconds", type=float, default=15.0)
    segmentation.add_argument("--max-seconds", type=float, default=29.5)
    segmentation.add_argument(
        "--hard-gap-seconds",
        type=float,
        default=2.0,
        help=(
            "Split at an unpunctuated word gap this long (default: 2.0). "
            "The ASMR-oriented default preserves sentence-internal pauses."
        ),
    )
    segmentation.add_argument("--soft-gap-seconds", type=float, default=0.20)
    segmentation.add_argument("--padding-seconds", type=float, default=0.12)
    segmentation.add_argument("--min-chars-per-second", type=float, default=0.20)
    segmentation.add_argument("--max-chars-per-second", type=float, default=20.0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be greater than zero")
    if args.workers <= 0:
        raise ValueError("--workers must be greater than zero")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be greater than zero")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative")
    ChunkingConfig(args.chunk_seconds, args.chunk_overlap_seconds).validate()
    SileroVADConfig(
        repo=args.vad_repo,
        threshold=args.vad_threshold,
        neg_threshold=args.vad_neg_threshold,
        min_speech_duration_ms=args.vad_min_speech_ms,
        min_silence_duration_ms=args.vad_min_silence_ms,
        speech_pad_ms=args.vad_speech_pad_ms,
        max_speech_duration_s=args.vad_max_speech_seconds,
        max_join_gap_s=args.vad_max_join_gap_seconds,
        max_upload_duration_s=args.vad_max_upload_seconds,
    ).validate()
    if len(args.keyterm) > 100:
        raise ValueError("xAI STT accepts at most 100 --keyterm values")
    too_long = [term for term in args.keyterm if len(term) > 50]
    if too_long:
        raise ValueError("Each --keyterm must contain at most 50 characters")


def _print_inventory(sources: list[AudioSource], output_dir: Path) -> None:
    total_bytes = sum(source.size_bytes for source in sources)
    total_seconds = sum(source.duration for source in sources)
    cached = sum(1 for source in sources if load_cached_response(output_dir, source) is not None)
    over_limit = sum(1 for source in sources if source.size_bytes > XAI_MAX_FILE_BYTES)
    print(
        f"sources={len(sources)} duration={total_seconds / 3600:.3f}h "
        f"size={total_bytes / 1024**3:.3f}GiB cached={cached}"
    )
    print(
        f"estimated_batch_cost=${total_seconds / 3600 * XAI_BATCH_PRICE_USD_PER_HOUR:.3f} "
        f"at ${XAI_BATCH_PRICE_USD_PER_HOUR:.2f}/hour"
    )
    print(f"over_500MB_limit={over_limit}")


def _transcribe_one(
    source: AudioSource,
    *,
    client: GrokSTTClient,
    options: TranscriptionOptions,
    output_dir: Path,
    chunking: ChunkingConfig,
    vad: SileroVADConfig | None,
    reuse_chunks: bool,
) -> Path:
    def report_chunk(done: int, total: int, cached: bool) -> None:
        print(
            f"stt-chunk={done}/{total} cached={str(cached).lower()} source={source.relative_path}",
            flush=True,
        )

    if vad is not None:
        cached_repo = getattr(_THREAD_LOCAL, "vad_repo", None)
        if cached_repo != vad.repo:
            model, get_speech_timestamps = load_silero_vad(vad.repo)
            _THREAD_LOCAL.vad_repo = vad.repo
            _THREAD_LOCAL.vad_model = model
            _THREAD_LOCAL.get_speech_timestamps = get_speech_timestamps
        return transcribe_source_with_vad(
            client,
            source,
            options,
            output_dir=output_dir,
            config=vad,
            model=_THREAD_LOCAL.vad_model,
            get_speech_timestamps=_THREAD_LOCAL.get_speech_timestamps,
            reuse_chunks=reuse_chunks,
            progress=report_chunk,
        )
    return transcribe_source_chunked(
        client,
        source,
        options=options,
        output_dir=output_dir,
        config=chunking,
        reuse_chunks=reuse_chunks,
        progress=report_chunk,
    )


def _write_transcription_errors(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_jsonl(output_dir / "transcription_errors.jsonl", rows)


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    project_root = Path.cwd().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    all_sources = discover_audio_sources(input_dir, output_dir=output_dir)
    if not all_sources:
        raise RuntimeError(f"No supported audio files found under: {input_dir}")

    selected = all_sources[: args.max_files] if args.max_files is not None else all_sources
    _print_inventory(selected, output_dir)
    if len(selected) != len(all_sources):
        print(f"pilot_selection={len(selected)}/{len(all_sources)} files")
    if args.dry_run:
        print("dry-run complete; no files were written and no audio was uploaded")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_output_readme(output_dir, input_dir=input_dir, project_root=project_root)

    options = TranscriptionOptions(
        language=args.language,
        format_text=args.format_text,
        filler_words=args.filler_words,
        diarize=args.diarize,
        multichannel=args.multichannel,
        keyterms=tuple(args.keyterm),
    )
    chunking = ChunkingConfig(
        seconds=args.chunk_seconds,
        overlap_seconds=args.chunk_overlap_seconds,
    )
    vad = None
    if args.vad:
        vad = SileroVADConfig(
            repo=args.vad_repo,
            threshold=args.vad_threshold,
            neg_threshold=args.vad_neg_threshold,
            min_speech_duration_ms=args.vad_min_speech_ms,
            min_silence_duration_ms=args.vad_min_silence_ms,
            speech_pad_ms=args.vad_speech_pad_ms,
            max_speech_duration_s=args.vad_max_speech_seconds,
            max_join_gap_s=args.vad_max_join_gap_seconds,
            max_upload_duration_s=args.vad_max_upload_seconds,
        )
    missing = [
        source
        for source in selected
        if args.force_transcribe or load_cached_response(output_dir, source) is None
    ]
    transcribable = missing
    transcription_errors: list[dict[str, Any]] = []

    if transcribable:
        api_key = os.environ.get(args.api_key_env, "").strip() or _load_grok_cli_oauth_token()
        if not api_key:
            _write_transcription_errors(output_dir, transcription_errors)
            raise SystemExit(
                f"{args.api_key_env} is not set and Grok CLI OAuth was not found. "
                "Authenticate Grok CLI or set the environment variable, then rerun. "
                "Existing raw responses will be resumed."
            )
        client = GrokSTTClient(
            api_key,
            endpoint=args.endpoint,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
        print(
            f"submitting={len(transcribable)} cached={len(selected) - len(missing)} "
            f"workers={min(args.workers, len(transcribable))} vad={args.vad}"
        )
        with ThreadPoolExecutor(max_workers=min(args.workers, len(transcribable))) as executor:
            future_to_source = {
                executor.submit(
                    _transcribe_one,
                    source,
                    client=client,
                    options=options,
                    output_dir=output_dir,
                    chunking=chunking,
                    vad=vad,
                    reuse_chunks=not args.force_transcribe,
                ): source
                for source in transcribable
            }
            completed = 0
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    path = future.result()
                    completed += 1
                    print(
                        f"transcribed={completed}/{len(transcribable)} "
                        f"source={source.relative_path} raw={path.name}"
                    )
                except Exception as exc:
                    transcription_errors.append(
                        {
                            "source_uid": source.source_id,
                            "source_audio": source.relative_path,
                            "reason": "stt_request_error",
                            "error": str(exc),
                        }
                    )
                    print(f"failed source={source.relative_path} error={exc}")
    else:
        print("all selected transcripts are already cached")

    _write_transcription_errors(output_dir, transcription_errors)
    if args.transcribe_only:
        print(f"transcribe-only complete: {output_dir / 'raw_responses'}")
    else:
        segmentation = SegmentationConfig(
            min_seconds=args.min_seconds,
            target_seconds=args.target_seconds,
            max_seconds=args.max_seconds,
            hard_gap_seconds=args.hard_gap_seconds,
            soft_gap_seconds=args.soft_gap_seconds,
            padding_seconds=args.padding_seconds,
            min_chars_per_second=args.min_chars_per_second,
            max_chars_per_second=args.max_chars_per_second,
        )
        segmentation.validate()
        summary = build_dataset(
            selected,
            output_dir,
            config=segmentation,
            project_root=project_root,
            rebuild_clips=args.rebuild_clips,
        )
        print(
            f"dataset complete: train={summary['segments_train']} "
            f"review={summary['segments_review']} "
            f"hours={summary['train_audio_hours']:.3f} output={output_dir}"
        )

    if transcription_errors:
        raise SystemExit(
            f"Completed with {len(transcription_errors)} transcription error(s); "
            f"see {output_dir / 'transcription_errors.jsonl'}"
        )


if __name__ == "__main__":
    main()
