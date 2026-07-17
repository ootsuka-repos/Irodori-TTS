"""GPU/network-free regression tests for the ASR data-prep review fixes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch

from dataset.acoustic_segmentation import (
    _Boundary,
    _merge_short_fragments,
    segment_acoustic_primitives,
)
from dataset.ero_voice_classifier import (
    EroVoiceClassifierError,
    EroVoicePrediction,
    JapaneseEroVoiceClassifier,
)
from dataset.grok_stt import (
    AUDIO_SUFFIXES,
    AudioSource,
    SegmentationConfig,
    SileroVADConfig,
    TranscriptionOptions,
    Word,
    _drop_overlap_duplicate_words,
    _load_cached_chunk,
    _save_chunk_response,
    discover_audio_sources,
    extract_clip,
    load_cached_response,
    prepare_vad_plan,
    review_reasons,
    save_raw_response,
    segment_words,
    streaming_events_to_response,
)
from dataset.local_asr import (
    LOCAL_ASR_CACHE_SCHEMA_VERSION,
    _load_cached,
    transcript_review_reasons,
)
from dataset.transcript_correction import (
    CorrectionConfig,
    _build_timeline,
    _safe_change,
    correct_transcript_rows,
)


def _fake_source(tmp_path: Path, name: str = "source.flac", *, duration: float = 600.0) -> AudioSource:
    path = tmp_path / name
    path.write_bytes(b"placeholder")
    stat = path.stat()
    return AudioSource(
        path=path,
        relative_path=name,
        source_id=Path(name).stem,
        speaker_id="speaker",
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        duration=duration,
        sample_rate=16_000,
        channels=1,
        frames=round(duration * 16_000),
    )


# --- H-1: discovery ---------------------------------------------------------


def test_audio_suffixes_exclude_formats_libsndfile_cannot_decode() -> None:
    assert not AUDIO_SUFFIXES & {".aac", ".m4a", ".mp4", ".mka", ".mkv"}


def test_discover_skips_unreadable_and_excluded_files(tmp_path: Path) -> None:
    sf.write(tmp_path / "good.wav", np.zeros(1600, dtype=np.float32), 16_000)
    (tmp_path / "broken.flac").write_bytes(b"this is not audio")
    excluded = tmp_path / "skip"
    excluded.mkdir()
    sf.write(excluded / "excluded.wav", np.zeros(1600, dtype=np.float32), 16_000)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    sf.write(output_dir / "artifact.wav", np.zeros(1600, dtype=np.float32), 16_000)

    sources = discover_audio_sources(
        tmp_path,
        output_dir=output_dir,
        excluded_dirs=[excluded],
    )

    assert [source.relative_path for source in sources] == ["good.wav"]


# --- H-2: VAD plan / chunk-cache rounding ------------------------------------


def test_vad_plan_round_trips_to_identical_values(tmp_path: Path, monkeypatch: Any) -> None:
    source = _fake_source(tmp_path)
    # Odd sample indices on the 16 kHz grid need 7 decimal digits, which is
    # exactly the case where round(x, 6) changes the float.
    raw = [(123_457 / 16_000, 200_003 / 16_000)]
    assert round(raw[0][0], 6) != raw[0][0]
    monkeypatch.setattr(
        "dataset.grok_stt.detect_speech_regions",
        lambda *_args, **_kwargs: list(raw),
    )

    first = prepare_vad_plan(
        source,
        output_dir=tmp_path / "output",
        config=SileroVADConfig(),
        model=object(),
        get_speech_timestamps=lambda *_args, **_kwargs: [],
    )
    second = prepare_vad_plan(
        source,
        output_dir=tmp_path / "output",
        config=SileroVADConfig(),
        model=object(),
        get_speech_timestamps=lambda *_args, **_kwargs: [],
    )

    assert first.cached is False and second.cached is True
    assert first.regions == second.regions
    assert first.upload_ranges == second.upload_ranges


def test_chunk_cache_hits_entries_saved_before_rounding(tmp_path: Path) -> None:
    source = _fake_source(tmp_path)
    options = TranscriptionOptions()
    unrounded_start = 123_457 / 16_000
    unrounded_end = 200_003 / 16_000
    _save_chunk_response(
        tmp_path / "output",
        source,
        {"text": "x", "words": []},
        start=unrounded_start,
        end=unrounded_end,
        endpoint="test",
        options=options,
    )

    loaded = _load_cached_chunk(
        tmp_path / "output",
        source,
        start=round(unrounded_start, 6),
        end=round(unrounded_end, 6),
        options=options,
    )
    assert loaded == {"text": "x", "words": []}

    assert (
        _load_cached_chunk(
            tmp_path / "output",
            source,
            start=round(unrounded_start, 6) + 0.01,
            end=round(unrounded_end, 6),
            options=options,
        )
        is None
    )


# --- M-1: source-level cache checks request options --------------------------


def test_source_cache_checks_request_options(tmp_path: Path) -> None:
    source = _fake_source(tmp_path)
    options = TranscriptionOptions()
    save_raw_response(
        tmp_path / "output",
        source,
        {"text": "x"},
        endpoint="test",
        options=options,
    )

    assert load_cached_response(tmp_path / "output", source) is not None
    assert load_cached_response(tmp_path / "output", source, options) is not None
    assert (
        load_cached_response(tmp_path / "output", source, TranscriptionOptions(language="en"))
        is None
    )


# --- H-3 / M-4: segmentation ---------------------------------------------------


def test_segment_span_cap_accounts_for_padding() -> None:
    config = SegmentationConfig(
        min_seconds=1.0,
        target_seconds=5.0,
        max_seconds=10.0,
        padding_seconds=1.0,
    )
    assert config.max_span_seconds == pytest.approx(8.0)
    words = [Word(text="あ", start=float(i), end=float(i) + 0.9) for i in range(30)]

    segments = segment_words(words, source_duration=100.0, config=config)

    assert len(segments) >= 2
    for previous, current in zip(segments, segments[1:], strict=False):
        assert previous.end <= current.start + 1e-9
    for segment in segments:
        assert segment.duration <= config.max_seconds + 1e-9
        assert "too_long" not in review_reasons(segment, config)


def test_midpoint_is_clamped_into_the_word_gap() -> None:
    config = SegmentationConfig()
    words = [
        Word(text="あ", start=0.05, end=1.0, speaker=0),
        Word(text="い", start=1.1, end=4.0, speaker=1),
    ]
    # source_duration clamps the first segment's padded end, which pushes the
    # raw midpoint (0.975) below the last word end (1.0) of the first segment.
    segments = segment_words(words, source_duration=1.2, config=config)

    assert len(segments) == 2
    assert segments[0].end == pytest.approx(1.0)
    assert segments[1].start == pytest.approx(1.0)
    assert "overlapping_words" not in review_reasons(segments[0], config)


def test_overlapping_word_timestamps_are_flagged() -> None:
    config = SegmentationConfig()
    words = [
        Word(text="あ", start=0.0, end=3.0, speaker=0),
        Word(text="い", start=2.9, end=6.0, speaker=1),
    ]

    segments = segment_words(words, source_duration=10.0, config=config)

    assert len(segments) == 2
    assert segments[0].end == pytest.approx(segments[1].start)
    assert "overlapping_words" in review_reasons(segments[0], config)
    assert "overlapping_words" in review_reasons(segments[1], config)


# --- M-2: chunk-boundary duplicates ------------------------------------------


def test_drop_overlap_duplicate_words() -> None:
    kept = [Word(text="こん", start=10.0, end=10.4)]
    candidates = [
        Word(text="こん", start=10.05, end=10.45),  # jittered duplicate
        Word(text="こん", start=10.8, end=11.2),  # genuine repeat, outside tolerance
        Word(text="にちは", start=10.5, end=11.0),
    ]

    result = _drop_overlap_duplicate_words(kept, candidates, window_start=10.0)

    assert [(word.text, word.start) for word in result] == [
        ("こん", 10.8),
        ("にちは", 10.5),
    ]


# --- M-5: streaming union merge ----------------------------------------------


def test_streaming_done_does_not_drop_earlier_finals() -> None:
    events = [
        {
            "type": "transcript.partial",
            "is_final": True,
            "words": [{"text": "hello", "start": 0.1, "end": 0.5}],
        },
        {
            "type": "transcript.partial",
            "is_final": True,
            "words": [{"text": "world", "start": 1.0, "end": 1.4}],
        },
        # Non-cumulative done containing only the last utterance.
        {
            "type": "transcript.done",
            "words": [{"text": "world", "start": 1.0, "end": 1.4}],
            "duration": 2.0,
        },
    ]

    response = streaming_events_to_response(events, duration=1.8, default_language="ja")

    assert [word["text"] for word in response["words"]] == ["hello", "world"]
    assert response["duration"] == 2.0


def test_streaming_text_only_done_does_not_duplicate_words() -> None:
    events = [
        {
            "type": "transcript.partial",
            "is_final": True,
            "words": [{"text": "hello", "start": 0.1, "end": 0.5}],
        },
        {"type": "transcript.done", "text": "hello", "duration": 1.0},
    ]

    response = streaming_events_to_response(events, duration=1.0, default_language="ja")

    assert [word["text"] for word in response["words"]] == ["hello"]


# --- L-1 / L-10: clip extraction ----------------------------------------------


def test_extract_clip_fades_edges_and_reuses_existing(tmp_path: Path) -> None:
    source_path = tmp_path / "src.wav"
    sf.write(source_path, np.full(16_000, 0.5, dtype=np.float32), 16_000)
    clip_path = tmp_path / "clips" / "clip.flac"

    with sf.SoundFile(source_path) as reader:
        stats = extract_clip(reader, clip_path, start=0.0, end=0.5)
        info = sf.info(clip_path)
        assert info.channels == 1
        assert info.samplerate == 16_000
        assert info.frames == 8_000
        samples = sf.read(clip_path, dtype="float32")[0]
        assert abs(samples[0]) < 1e-4  # fade-in
        assert abs(samples[-1]) < 0.02  # fade-out
        assert samples[4_000] == pytest.approx(0.5, abs=1e-3)
        assert stats.peak == pytest.approx(0.5, abs=1e-3)
        assert stats.clipping_ratio == 0.0

        mtime = clip_path.stat().st_mtime_ns
        stats_again = extract_clip(reader, clip_path, start=0.0, end=0.5)
        assert clip_path.stat().st_mtime_ns == mtime  # reused, not rewritten
        assert stats_again.peak == pytest.approx(stats.peak, abs=1e-4)


# --- H-4 / M-9: local ASR ------------------------------------------------------


def test_local_asr_error_cache_is_not_reused(tmp_path: Path) -> None:
    audio = {"path": "x", "size_bytes": 1, "mtime_ns": 2}
    model = {"model": "anime-whisper"}
    path = tmp_path / "cache.json"

    def write(status: str) -> None:
        path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "schema_version": LOCAL_ASR_CACHE_SCHEMA_VERSION,
                        "row_id": "row",
                        "audio": audio,
                        "model": model,
                    },
                    "result": {"text": "", "status": status, "review_reasons": []},
                }
            ),
            encoding="utf-8",
        )

    write("error")
    assert _load_cached(path, row_id="row", audio=audio, model=model) is None
    write("ok")
    assert _load_cached(path, row_id="row", audio=audio, model=model) is not None


def test_transcript_review_reasons_use_duration_density() -> None:
    assert "text_too_dense" in transcript_review_reasons("あ" * 100, duration_seconds=2.0)
    assert "text_too_sparse" in transcript_review_reasons("あ", duration_seconds=30.0)
    no_duration = transcript_review_reasons("あ" * 100)
    assert "text_too_dense" not in no_duration and "text_too_sparse" not in no_duration
    empty = transcript_review_reasons("", duration_seconds=30.0)
    assert "empty_transcript" in empty and "text_too_sparse" not in empty


# --- L-4 / L-5 / L-6: transcript correction ------------------------------------


def test_safe_change_blocks_punctuation_only_inflation() -> None:
    config = CorrectionConfig()
    safe, similarity, _ = _safe_change("こんにちは", "こんにちは" + "。" * 40, config)
    assert similarity == pytest.approx(1.0)
    assert not safe
    safe, _, _ = _safe_change("こんにちは元気", "こんにちは、元気", config)
    assert safe


def test_rows_without_start_are_parked_after_timed_rows() -> None:
    speech = [
        {"id": "later", "source_uid": "s", "start": 5.0, "end": 6.0, "text": "実時刻"},
        {"id": "no-time", "source_uid": "s", "text": "時刻なし"},
    ]
    _, _, grouped = _build_timeline(speech, [])
    assert [item.row_id for item in grouped["s"]] == ["later", "no-time"]


def test_invalid_correction_cache_falls_back_to_agents(
    tmp_path: Path, monkeypatch: Any
) -> None:
    speech = [
        {
            "id": "s0",
            "source_uid": "source",
            "start": 0.0,
            "end": 1.0,
            "text": "こんにちは",
        }
    ]
    calls = 0

    def fake_call(
        prompt: str,
        *,
        target_ids: Any,
        **_kwargs: Any,
    ) -> tuple[list[dict[str, str]], str, None, list[dict[str, str]]]:
        nonlocal calls
        calls += 1
        return (
            [
                {"id": row_id, "corrected_text": "こんにちは", "status": "unchanged"}
                for row_id in target_ids
            ],
            "codex",
            None,
            [],
        )

    monkeypatch.setattr("dataset.transcript_correction._call_agents", fake_call)
    config = CorrectionConfig(workers=1, attempts_per_agent=1)
    correct_transcript_rows(speech, [], cache_dir=tmp_path, config=config)
    assert calls == 1

    cache_files = list(tmp_path.rglob("batch_*.json"))
    assert len(cache_files) == 1
    payload = json.loads(cache_files[0].read_text(encoding="utf-8"))
    payload["segments"] = [{"id": "bogus", "corrected_text": "x", "status": "unchanged"}]
    cache_files[0].write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    _, _, summary = correct_transcript_rows(speech, [], cache_dir=tmp_path, config=config)
    assert calls == 2  # invalid cache fell back to the agents
    assert summary["batches_failed"] == 0
    assert summary["batches_completed"] == 1


# --- M-10: ero voice classifier -------------------------------------------------


def test_ero_prediction_error_marker_contract() -> None:
    marker = EroVoicePrediction(audio="x.wav", usual=0.0, aegi=0.0, chupa=0.0, error="boom")
    assert marker.failed is True
    with pytest.raises(EroVoiceClassifierError, match="boom"):
        _ = marker.probabilities
    assert marker.to_dict() == {"audio": "x.wav", "error": "boom"}

    healthy = EroVoicePrediction(audio="y.wav", usual=0.6, aegi=0.3, chupa=0.1)
    assert healthy.failed is False
    assert healthy.label == "usual"
    assert healthy.confidence == pytest.approx(0.6)


def test_predict_isolates_failing_clips() -> None:
    classifier = object.__new__(JapaneseEroVoiceClassifier)
    classifier.device = torch.device("cpu")
    classifier.head = lambda tensor: torch.zeros((tensor.shape[0], 3), dtype=torch.float64)

    def fake_embedding(path: Path) -> np.ndarray:
        if "bad" in str(path):
            raise RuntimeError("unreadable clip")
        return np.zeros(256, dtype=np.float32)

    classifier._embedding_from_file = fake_embedding  # type: ignore[method-assign]

    predictions = classifier.predict(["good1.wav", "bad.wav", "good2.wav"], batch_size=2)

    assert [prediction.failed for prediction in predictions] == [False, True, False]
    assert "unreadable clip" in str(predictions[1].error)
    assert predictions[0].probabilities["usual"] == pytest.approx(1.0 / 3.0)
    assert predictions[2].audio.endswith("good2.wav")


# --- L-11: acoustic segmentation -------------------------------------------------


def test_merge_short_fragments_matches_first_short_pair_semantics() -> None:
    boundaries = [
        _Boundary(time=0.0, frame_index=None, reason="gap_start", score=1.0, hard=True),
        _Boundary(time=0.5, frame_index=50, reason="pause_valley", score=1.0),
        _Boundary(time=0.6, frame_index=60, reason="pause_valley", score=2.0),
        _Boundary(time=5.0, frame_index=None, reason="gap_end", score=1.0, hard=True),
    ]
    _merge_short_fragments(boundaries, preferred_min_seconds=1.2)
    assert [boundary.time for boundary in boundaries] == [0.0, 5.0]


def test_acoustic_primitives_remain_contiguous_and_capped() -> None:
    torch.manual_seed(0)
    sample_rate = 16_000
    waveform = 0.3 * torch.randn(sample_rate * 35)
    waveform[10 * sample_rate : 11 * sample_rate] *= 0.001
    waveform[22 * sample_rate : 23 * sample_rate] *= 0.001

    primitives = segment_acoustic_primitives(waveform, sample_rate)

    assert primitives[0].start == 0.0
    assert primitives[-1].end == pytest.approx(35.0, abs=1e-6)
    for previous, current in zip(primitives, primitives[1:], strict=False):
        assert previous.end == current.start
    for primitive in primitives:
        assert primitive.duration > 0.0
        assert primitive.duration <= 15.0 + 1e-6
