#!/usr/bin/env python3
"""Transcribe local audio with Grok STT and build an Irodori-TTS dataset."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any

from irodori_tts.data_prep.grok_stt import (
    XAI_BATCH_PRICE_USD_PER_HOUR,
    XAI_MAX_FILE_BYTES,
    XAI_STT_ENDPOINT,
    XAI_STT_WEBSOCKET_ENDPOINT,
    AudioSource,
    ChunkingConfig,
    GrokSTTClient,
    GrokSubscriptionAuth,
    GrokSubscriptionSTTClient,
    SegmentationConfig,
    SileroVADConfig,
    STTClient,
    TranscriptionOptions,
    VADPlan,
    atomic_write_jsonl,
    build_dataset,
    discover_audio_sources,
    load_cached_response,
    load_silero_vad,
    prepare_vad_plan,
    transcribe_source_chunked,
    transcribe_source_from_vad_plan,
    write_output_readme,
)

_PRINT_LOCK = Lock()


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
        help="Inspect files and cache coverage without writing or uploading.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Only process the first N files in deterministic path order (useful for a pilot).",
    )
    parser.add_argument(
        "--min-source-seconds",
        type=float,
        default=0.0,
        help=(
            "Exclude source files whose duration is at or below this value before STT "
            "(default: 0, no duration exclusion)."
        ),
    )
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Save raw STT responses but do not create clips or manifests.",
    )
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Ignore valid cached responses and submit selected files again.",
    )
    parser.add_argument(
        "--rebuild-clips",
        action="store_true",
        help="Rewrite FLAC clips even when matching files already exist.",
    )

    api = parser.add_argument_group("xAI Grok STT")
    api.add_argument(
        "--auth-mode",
        choices=("subscription", "api-key"),
        default="subscription",
        help=(
            "Authentication transport. The default reuses `grok login`; api-key uses the "
            "multipart REST endpoint."
        ),
    )
    api.add_argument("--api-key-env", default="XAI_API_KEY")
    api.add_argument("--endpoint", default=XAI_STT_ENDPOINT)
    api.add_argument("--stream-endpoint", default=XAI_STT_WEBSOCKET_ENDPOINT)
    api.add_argument("--grok-cli", default="grok")
    api.add_argument(
        "--grok-auth-file",
        type=Path,
        default=None,
        help="Override the Grok CLI auth.json path (default: ~/.grok/auth.json).",
    )
    api.add_argument("--stream-endpointing-ms", type=int, default=350)
    api.add_argument("--stream-frame-ms", type=int, default=100)
    api.add_argument(
        "--stream-realtime-factor",
        type=float,
        default=0.0,
        help=(
            "Streaming pacing factor: 1 is real time; 0 sends immediately with WebSocket "
            "backpressure (default, verified for batch processing)."
        ),
    )
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
    api.add_argument(
        "--vad-prefetch-sources",
        type=int,
        default=8,
        help=(
            "Maximum prepared VAD source plans waiting for asynchronous STT consumers (default: 8)."
        ),
    )
    api.add_argument("--timeout-seconds", type=float, default=3600.0)
    api.add_argument("--max-retries", type=int, default=4)
    api.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pinned Silero VAD before STT (enabled by default).",
    )
    api.add_argument("--vad-repo", default="snakers4/silero-vad:v6.2.1")
    api.add_argument(
        "--vad-device",
        default="cpu",
        help="Single Silero VAD device used when --vad-devices is omitted.",
    )
    api.add_argument(
        "--vad-devices",
        default=None,
        help=(
            "Comma-separated devices for parallel VAD producers, e.g. cuda:0,cuda:1. "
            "Overrides --vad-device."
        ),
    )
    api.add_argument("--vad-threshold", type=float, default=0.35)
    api.add_argument("--vad-neg-threshold", type=float, default=0.20)
    api.add_argument("--vad-min-speech-ms", type=int, default=180)
    api.add_argument("--vad-min-silence-ms", type=int, default=450)
    api.add_argument("--vad-speech-pad-ms", type=int, default=350)
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
    segmentation.add_argument("--padding-seconds", type=float, default=0.35)
    segmentation.add_argument("--min-chars-per-second", type=float, default=0.20)
    segmentation.add_argument("--max-chars-per-second", type=float, default=20.0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be greater than zero")
    if args.min_source_seconds < 0:
        raise ValueError("--min-source-seconds must be non-negative")
    if args.workers <= 0:
        raise ValueError("--workers must be greater than zero")
    if args.vad_prefetch_sources <= 0:
        raise ValueError("--vad-prefetch-sources must be greater than zero")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be greater than zero")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative")
    if not 0 <= args.stream_endpointing_ms <= 5000:
        raise ValueError("--stream-endpointing-ms must be between 0 and 5000")
    if args.stream_frame_ms <= 0:
        raise ValueError("--stream-frame-ms must be positive")
    if args.stream_realtime_factor < 0:
        raise ValueError("--stream-realtime-factor must be non-negative")
    if args.auth_mode == "subscription" and args.multichannel:
        raise ValueError("--multichannel requires --auth-mode api-key")
    _selected_vad_devices(args)
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


def _selected_vad_devices(args: argparse.Namespace) -> tuple[str, ...]:
    raw = args.vad_devices if args.vad_devices is not None else args.vad_device
    devices = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("VAD devices must contain at least one unique device")
    return devices


def _print_inventory(sources: list[AudioSource], output_dir: Path, *, auth_mode: str) -> None:
    total_bytes = sum(source.size_bytes for source in sources)
    total_seconds = sum(source.duration for source in sources)
    cached = sum(1 for source in sources if load_cached_response(output_dir, source) is not None)
    over_limit = sum(1 for source in sources if source.size_bytes > XAI_MAX_FILE_BYTES)
    print(
        f"sources={len(sources)} duration={total_seconds / 3600:.3f}h "
        f"size={total_bytes / 1024**3:.3f}GiB cached={cached}"
    )
    if auth_mode == "api-key":
        print(
            f"estimated_batch_cost=${total_seconds / 3600 * XAI_BATCH_PRICE_USD_PER_HOUR:.3f} "
            f"at ${XAI_BATCH_PRICE_USD_PER_HOUR:.2f}/hour"
        )
    else:
        print("stt_transport=grok-cli-subscription-websocket")
    print(f"over_500MB_limit={over_limit}")


def _chunk_reporter(source: AudioSource) -> Any:
    def report_chunk(done: int, total: int, cached: bool) -> None:
        with _PRINT_LOCK:
            print(
                f"stt-chunk={done}/{total} cached={str(cached).lower()} "
                f"source={source.relative_path}",
                flush=True,
            )

    return report_chunk


def _transcribe_one_without_vad(
    source: AudioSource,
    *,
    client: STTClient,
    options: TranscriptionOptions,
    output_dir: Path,
    chunking: ChunkingConfig,
    reuse_chunks: bool,
) -> Path:
    return transcribe_source_chunked(
        client,
        source,
        options=options,
        output_dir=output_dir,
        config=chunking,
        reuse_chunks=reuse_chunks,
        progress=_chunk_reporter(source),
    )


def _transcribe_prepared_vad_source(
    source: AudioSource,
    *,
    plan: VADPlan,
    client: STTClient,
    options: TranscriptionOptions,
    output_dir: Path,
    vad: SileroVADConfig,
    reuse_chunks: bool,
) -> Path:
    return transcribe_source_from_vad_plan(
        client,
        source,
        options,
        output_dir=output_dir,
        config=vad,
        plan=plan,
        reuse_chunks=reuse_chunks,
        progress=_chunk_reporter(source),
    )


def _error_row(source: AudioSource, *, reason: str, error: Exception) -> dict[str, Any]:
    return {
        "source_uid": source.source_id,
        "source_audio": source.relative_path,
        "reason": reason,
        "error": str(error),
    }


async def _run_pipelined_vad_stt(
    sources: list[AudioSource],
    *,
    client: STTClient,
    options: TranscriptionOptions,
    output_dir: Path,
    vad: SileroVADConfig,
    vad_devices: tuple[str, ...],
    stt_workers: int,
    prefetch_sources: int,
    reuse_chunks: bool,
) -> list[dict[str, Any]]:
    """Overlap GPU VAD producers with asynchronous STT consumer workers."""
    consumer_count = min(stt_workers, len(sources))
    producer_count = min(len(vad_devices), len(sources))
    selected_devices = vad_devices[:producer_count]
    source_queue: asyncio.Queue[AudioSource | None] = asyncio.Queue()
    for source in sources:
        source_queue.put_nowait(source)
    for _ in range(producer_count):
        source_queue.put_nowait(None)
    plan_queue: asyncio.Queue[tuple[AudioSource, VADPlan] | None] = asyncio.Queue(
        maxsize=prefetch_sources
    )
    errors: list[dict[str, Any]] = []
    completed = 0
    prepared = 0
    vad_models = [load_silero_vad(vad.repo, device=device) for device in selected_devices]
    event_loop = asyncio.get_running_loop()
    vad_executors = [
        ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"gpu-vad-{index}")
        for index in range(producer_count)
    ]
    stt_executor = ThreadPoolExecutor(
        max_workers=consumer_count,
        thread_name_prefix="async-stt",
    )

    async def producer(worker_index: int) -> None:
        nonlocal prepared
        device = selected_devices[worker_index]
        model, get_speech_timestamps = vad_models[worker_index]
        executor = vad_executors[worker_index]
        while True:
            source = await source_queue.get()
            try:
                if source is None:
                    return
                try:
                    plan = await event_loop.run_in_executor(
                        executor,
                        partial(
                            prepare_vad_plan,
                            source,
                            output_dir=output_dir,
                            config=vad,
                            model=model,
                            get_speech_timestamps=get_speech_timestamps,
                        ),
                    )
                except Exception as exc:
                    errors.append(_error_row(source, reason="vad_inference_error", error=exc))
                    with _PRINT_LOCK:
                        print(
                            f"failed-vad worker={worker_index} device={device} "
                            f"source={source.relative_path} error={exc}",
                            flush=True,
                        )
                    continue
                await plan_queue.put((source, plan))
                prepared += 1
                with _PRINT_LOCK:
                    print(
                        f"vad-prepared={prepared}/{len(sources)} "
                        f"worker={worker_index} device={device} "
                        f"cached={str(plan.cached).lower()} regions={len(plan.regions)} "
                        f"uploads={len(plan.upload_ranges)} queue={plan_queue.qsize()} "
                        f"source={source.relative_path}",
                        flush=True,
                    )
            finally:
                source_queue.task_done()

    async def consumer(worker_index: int) -> None:
        nonlocal completed
        while True:
            item = await plan_queue.get()
            try:
                if item is None:
                    return
                source, plan = item
                try:
                    path = await event_loop.run_in_executor(
                        stt_executor,
                        partial(
                            _transcribe_prepared_vad_source,
                            source,
                            plan=plan,
                            client=client,
                            options=options,
                            output_dir=output_dir,
                            vad=vad,
                            reuse_chunks=reuse_chunks,
                        ),
                    )
                except Exception as exc:
                    errors.append(_error_row(source, reason="stt_request_error", error=exc))
                    with _PRINT_LOCK:
                        print(
                            f"failed-stt worker={worker_index} "
                            f"source={source.relative_path} error={exc}",
                            flush=True,
                        )
                    continue
                completed += 1
                with _PRINT_LOCK:
                    print(
                        f"transcribed={completed}/{len(sources)} worker={worker_index} "
                        f"source={source.relative_path} raw={path.name}",
                        flush=True,
                    )
            finally:
                plan_queue.task_done()

    producer_tasks = [
        asyncio.create_task(producer(index), name=f"gpu-vad-producer-{index}")
        for index in range(producer_count)
    ]
    consumer_tasks = [
        asyncio.create_task(consumer(index), name=f"stt-consumer-{index}")
        for index in range(consumer_count)
    ]
    try:
        await asyncio.gather(*producer_tasks)
        for _ in range(consumer_count):
            await plan_queue.put(None)
        await plan_queue.join()
        await asyncio.gather(*consumer_tasks)
    finally:
        for task in producer_tasks + consumer_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*producer_tasks, *consumer_tasks, return_exceptions=True)
        for executor in vad_executors:
            executor.shutdown(wait=True, cancel_futures=True)
        stt_executor.shutdown(wait=True, cancel_futures=True)
    return errors


def _write_transcription_errors(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_jsonl(output_dir / "transcription_errors.jsonl", rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = _parse_args()
    _validate_args(args)

    project_root = Path.cwd().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    discovered_sources = discover_audio_sources(input_dir, output_dir=output_dir)
    if not discovered_sources:
        raise RuntimeError(f"No supported audio files found under: {input_dir}")
    all_sources = [
        source for source in discovered_sources if source.duration > args.min_source_seconds
    ]
    excluded_short = len(discovered_sources) - len(all_sources)
    if not all_sources:
        raise RuntimeError(
            "No audio remains after the source-duration filter: "
            f"duration must be greater than {args.min_source_seconds:.3f}s"
        )

    selected = all_sources[: args.max_files] if args.max_files is not None else all_sources
    _print_inventory(selected, output_dir, auth_mode=args.auth_mode)
    print(
        f"source_duration_filter=>{args.min_source_seconds:.3f}s "
        f"excluded_at_or_below={excluded_short}"
    )
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
        client: STTClient
        if args.auth_mode == "subscription":
            auth = GrokSubscriptionAuth(
                auth_file=args.grok_auth_file,
                cli_command=args.grok_cli,
            )
            auth_metadata = auth.validate()
            client = GrokSubscriptionSTTClient(
                auth,
                endpoint=args.stream_endpoint,
                endpointing_ms=args.stream_endpointing_ms,
                frame_milliseconds=args.stream_frame_ms,
                realtime_factor=args.stream_realtime_factor,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
            print(
                "grok_subscription_login=valid "
                f"expires_at={auth_metadata.get('expires_at') or 'unspecified'}"
            )
        else:
            api_key = os.environ.get(args.api_key_env, "").strip()
            if not api_key:
                _write_transcription_errors(output_dir, transcription_errors)
                raise SystemExit(
                    f"{args.api_key_env} is not set for --auth-mode api-key. "
                    "Set it or use the default --auth-mode subscription after `grok login`."
                )
            client = GrokSTTClient(
                api_key,
                endpoint=args.endpoint,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
        print(
            f"submitting={len(transcribable)} cached={len(selected) - len(missing)} "
            f"workers={min(args.workers, len(transcribable))} vad={args.vad} "
            f"auth_mode={args.auth_mode}"
        )
        if vad is not None:
            vad_devices = _selected_vad_devices(args)
            print(
                "pipeline=gpu-vad-producers+async-stt-consumers "
                f"vad_workers={min(len(vad_devices), len(transcribable))} "
                f"vad_devices={','.join(vad_devices)} "
                f"stt_consumers={min(args.workers, len(transcribable))} "
                f"prefetch_sources={args.vad_prefetch_sources}"
            )
            transcription_errors.extend(
                asyncio.run(
                    _run_pipelined_vad_stt(
                        transcribable,
                        client=client,
                        options=options,
                        output_dir=output_dir,
                        vad=vad,
                        vad_devices=vad_devices,
                        stt_workers=args.workers,
                        prefetch_sources=args.vad_prefetch_sources,
                        reuse_chunks=not args.force_transcribe,
                    )
                )
            )
        else:
            with ThreadPoolExecutor(max_workers=min(args.workers, len(transcribable))) as executor:
                future_to_source = {
                    executor.submit(
                        _transcribe_one_without_vad,
                        source,
                        client=client,
                        options=options,
                        output_dir=output_dir,
                        chunking=chunking,
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
                            _error_row(source, reason="stt_request_error", error=exc)
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
