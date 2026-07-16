from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from irodori_tts.cli.prepare_all import _parse_args as parse_full_pipeline_args
from irodori_tts.cli.prepare_grok_stt import _run_pipelined_vad_stt
from irodori_tts.data_prep.grok_stt import (
    AudioSource,
    GrokSubscriptionAuth,
    SileroVADConfig,
    TranscriptionOptions,
    VADPlan,
    prepare_vad_plan,
    streaming_events_to_response,
)
from irodori_tts.data_prep.local_asr import transcribe_manifest_rows
from irodori_tts.data_prep.nonverbal_clustering import (
    BEATS_MIN_INPUT_SAMPLES,
    FeatureConfig,
    _pool_beats_batch,
    load_vad_complements,
)
from irodori_tts.data_prep.nonverbal_event_pipeline import load_feature_shards
from irodori_tts.data_prep.nonverbal_training_manifest import render_nonverbal_text
from irodori_tts.data_prep.transcript_correction import (
    CorrectionConfig,
    correct_transcript_rows,
)


class _FakeASR:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls: list[list[Path]] = []

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "test",
            "model": "fake-anime-whisper",
            "revision": "pinned",
            "device": "cuda:0",
        }

    def transcribe(self, audio_paths: list[Path]) -> list[str]:
        self.calls.append(list(audio_paths))
        return self.texts[: len(audio_paths)]


class _FakeBEATs(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.source: torch.Tensor | None = None
        self.padding_mask: torch.Tensor | None = None

    def extract_features(
        self,
        source: torch.Tensor,
        *,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        self.source = source.detach().cpu()
        self.padding_mask = padding_mask.detach().cpu()
        features = torch.ones((source.shape[0], 2, 4), device=source.device)
        return features, None


def test_beats_pool_pads_audio_shorter_than_patch_kernel() -> None:
    model = _FakeBEATs()
    waveforms = [torch.ones(1_600), torch.ones(2_400)]

    pooled = _pool_beats_batch(model, waveforms, device=torch.device("cpu"))

    assert pooled.shape == (2, 4)
    assert model.source is not None
    assert model.padding_mask is not None
    assert model.source.shape == (2, BEATS_MIN_INPUT_SAMPLES)
    assert not model.padding_mask[0, :1_600].any()
    assert model.padding_mask[0, 1_600:].all()
    assert not model.padding_mask[1, :2_400].any()
    assert model.padding_mask[1, 2_400:].all()


def test_feature_loader_ignores_shards_outside_canonical_manifest(tmp_path: Path) -> None:
    shard_dir = tmp_path / "embeddings" / "shards"
    shard_dir.mkdir(parents=True)
    active = {"id": "current", "source_key": "s000", "start": 0.0, "end": 1.0}
    stale = {"id": "stale", "source_key": "s999", "start": 0.0, "end": 1.0}
    (tmp_path / "windows.jsonl").write_text(json.dumps(active) + "\n", encoding="utf-8")
    (shard_dir / "s000.jsonl").write_text(json.dumps(active) + "\n", encoding="utf-8")
    (shard_dir / "s999.jsonl").write_text(json.dumps(stale) + "\n", encoding="utf-8")
    np.save(shard_dir / "s000.npy", np.ones((1, 4), dtype=np.float32))
    np.save(shard_dir / "s999.npy", np.zeros((1, 4), dtype=np.float32))

    rows, embeddings, summary = load_feature_shards(tmp_path)

    assert [row["id"] for row in rows] == ["current"]
    assert embeddings.shape == (1, 4)
    assert summary["ignored_stale_shards"] == 1
    assert summary["ignored_stale_rows"] == 1


def _fake_audio_source(tmp_path: Path, name: str, *, duration: float = 600.0) -> AudioSource:
    path = tmp_path / name
    path.write_bytes(b"audio-placeholder")
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


def test_vad_plan_is_resumable(tmp_path: Path, monkeypatch: Any) -> None:
    source = _fake_audio_source(tmp_path, "source.flac")
    calls = 0

    def fake_detect(*_args: Any, **_kwargs: Any) -> list[tuple[float, float]]:
        nonlocal calls
        calls += 1
        return [(1.0, 3.0), (3.2, 5.0)]

    monkeypatch.setattr("irodori_tts.data_prep.grok_stt.detect_speech_regions", fake_detect)
    config = SileroVADConfig(max_join_gap_s=0.5)
    first = prepare_vad_plan(
        source,
        output_dir=tmp_path / "output",
        config=config,
        model=object(),
        get_speech_timestamps=lambda *_args, **_kwargs: [],
    )
    second = prepare_vad_plan(
        source,
        output_dir=tmp_path / "output",
        config=config,
        model=object(),
        get_speech_timestamps=lambda *_args, **_kwargs: [],
    )
    assert calls == 1
    assert first.cached is False
    assert second.cached is True
    assert second.upload_ranges == ((1.0, 5.0),)


def test_gpu_vad_producer_overlaps_async_stt_consumers(tmp_path: Path, monkeypatch: Any) -> None:
    sources = [_fake_audio_source(tmp_path, f"source-{index}.flac") for index in range(3)]
    events: list[str] = []

    def fake_load(*_args: Any, **_kwargs: Any) -> tuple[object, Any]:
        return object(), lambda *_inner_args, **_inner_kwargs: []

    def fake_prepare(source: AudioSource, **_kwargs: Any) -> VADPlan:
        events.append(f"vad-start:{source.source_id}")
        time.sleep(0.03)
        events.append(f"vad-end:{source.source_id}")
        return VADPlan(regions=((0.0, 1.0),), upload_ranges=((0.0, 1.0),))

    def fake_transcribe(source: AudioSource, **_kwargs: Any) -> Path:
        events.append(f"stt-start:{source.source_id}")
        time.sleep(0.10)
        events.append(f"stt-end:{source.source_id}")
        return tmp_path / f"{source.source_id}.json"

    monkeypatch.setattr("irodori_tts.cli.prepare_grok_stt.load_silero_vad", fake_load)
    monkeypatch.setattr("irodori_tts.cli.prepare_grok_stt.prepare_vad_plan", fake_prepare)
    monkeypatch.setattr(
        "irodori_tts.cli.prepare_grok_stt._transcribe_prepared_vad_source",
        fake_transcribe,
    )
    errors = asyncio.run(
        _run_pipelined_vad_stt(
            sources,
            client=object(),
            options=TranscriptionOptions(),
            output_dir=tmp_path / "output",
            vad=SileroVADConfig(),
            vad_devices=("cuda:0", "cuda:1"),
            stt_workers=2,
            prefetch_sources=2,
            reuse_chunks=True,
        )
    )
    assert errors == []
    assert events.index("vad-start:source-1") < events.index("stt-end:source-0")


def test_streaming_stt_collapses_repeated_final_events() -> None:
    repeated = {
        "type": "transcript.partial",
        "text": "hello",
        "words": [{"text": "hello", "start": 0.1, "end": 0.5}],
        "is_final": True,
        "speech_final": False,
        "start": 0.0,
        "duration": 0.8,
        "language": "en",
    }
    response = streaming_events_to_response(
        [
            repeated,
            {**repeated, "speech_final": True},
            {
                "type": "transcript.partial",
                "text": "world",
                "words": [{"text": "world", "start": 1.0, "end": 1.4}],
                "is_final": True,
                "speech_final": True,
                "start": 0.8,
                "duration": 0.8,
                "language": "en",
            },
            {"type": "transcript.done", "text": "", "words": [], "duration": 2.0},
        ],
        duration=1.8,
        default_language="ja",
    )
    assert [word["text"] for word in response["words"]] == ["hello", "world"]
    assert response["text"] == "hello world"
    assert response["duration"] == 2.0
    assert response["transport"] == "grok-cli-subscription-websocket"


def test_grok_subscription_auth_reads_fresh_cli_credential(tmp_path: Path) -> None:
    secret = "must-not-appear-in-repr"
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "https://auth.x.ai::user": {
                    "key": secret,
                    "auth_mode": "oauth",
                    "refresh_token": "refresh-secret",
                    "expires_at": "2099-01-01T00:00:00.123456789Z",
                }
            }
        ),
        encoding="utf-8",
    )
    auth = GrokSubscriptionAuth(auth_file=auth_file, cli_command="unused-for-fresh-token")
    assert auth.access_token() == secret
    assert secret not in repr(auth)


def test_full_pipeline_excludes_five_minute_sources_by_default() -> None:
    args = parse_full_pipeline_args([])
    assert args.min_source_seconds == 300.0


def test_nonverbal_inventory_excludes_source_at_duration_threshold(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "short.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "source": {
                        "source_id": "short",
                        "relative_path": "short.flac",
                        "duration": 300.0,
                    }
                },
                "response": {"vad_regions": []},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="No eligible raw response"):
        load_vad_complements(
            raw_dir,
            data_root=tmp_path,
            feature_config=FeatureConfig(),
            min_source_seconds=300.0,
        )


def test_local_asr_cache_uses_audio_fingerprint(tmp_path: Path) -> None:
    audio = tmp_path / "event.flac"
    audio.write_bytes(b"fake audio")
    rows = [{"id": "event-1", "audio": audio.name}]
    first = _FakeASR(["あっ、んっ……"])
    enriched, summary = transcribe_manifest_rows(
        rows,
        audio_root=tmp_path,
        cache_dir=tmp_path / "cache",
        transcriber=first,
        batch_size=4,
    )
    assert len(first.calls) == 1
    assert enriched[0]["transcript_text_raw"] == "あっ、んっ……"
    assert summary["transcribed"] == 1

    second = _FakeASR(["should not be used"])
    cached, cached_summary = transcribe_manifest_rows(
        rows,
        audio_root=tmp_path,
        cache_dir=tmp_path / "cache",
        transcriber=second,
        batch_size=4,
    )
    assert second.calls == []
    assert cached[0]["transcript_text"] == "あっ、んっ……"
    assert cached_summary["cached"] == 1

    audio.write_bytes(b"changed audio")
    third = _FakeASR(["更新後"])
    changed, _ = transcribe_manifest_rows(
        rows,
        audio_root=tmp_path,
        cache_dir=tmp_path / "cache",
        transcriber=third,
        batch_size=4,
    )
    assert len(third.calls) == 1
    assert changed[0]["transcript_text"] == "更新後"


def test_context_correction_mixes_backends_and_reuses_cache(
    tmp_path: Path, monkeypatch: Any
) -> None:
    speech = [
        {
            "id": "s0",
            "source_uid": "source",
            "start": 0.0,
            "end": 1.0,
            "text": "今日は",
            "status": "train",
        },
        {
            "id": "s1",
            "source_uid": "source",
            "start": 3.0,
            "end": 4.0,
            "text": "こんにちわ",
            "status": "train",
        },
        {
            "id": "s2",
            "source_uid": "source",
            "start": 6.0,
            "end": 7.0,
            "text": "またね",
            "status": "train",
        },
    ]
    nonverbal = [
        {
            "id": "n0",
            "source_uid": "source",
            "start": 1.5,
            "end": 2.5,
            "transcript_text_raw": "あっ",
            "transcript_backend": "anime-whisper",
        }
    ]
    observed_inputs: list[dict[str, Any]] = []

    def fake_call(
        prompt: str,
        *,
        target_ids: list[str] | tuple[str, ...],
        **_kwargs: Any,
    ) -> tuple[list[dict[str, str]], str, None, list[dict[str, str]]]:
        payload = json.loads(prompt.split("input:\n", 1)[1])
        observed_inputs.append(payload)
        output = []
        for row_id in target_ids:
            original = next(item["text"] for item in payload["segments"] if item["id"] == row_id)
            corrected = "こんにちは" if row_id == "s1" else original
            output.append(
                {
                    "id": row_id,
                    "corrected_text": corrected,
                    "status": "corrected" if row_id == "s1" else "unchanged",
                }
            )
        return output, "codex", None, []

    monkeypatch.setattr(
        "irodori_tts.data_prep.transcript_correction._call_agents",
        fake_call,
    )
    config = CorrectionConfig(
        target_batch_size=2,
        context_segments=1,
        workers=1,
        attempts_per_agent=1,
    )
    corrected_speech, corrected_nonverbal, summary = correct_transcript_rows(
        speech,
        nonverbal,
        cache_dir=tmp_path / "corrections",
        config=config,
    )
    assert corrected_speech[1]["text"] == "こんにちは"
    assert corrected_speech[1]["asr_text_raw"] == "こんにちわ"
    assert corrected_nonverbal[0]["transcript_text"] == "あっ"
    assert summary["accepted"] == 1
    assert any(
        {segment["backend"] for segment in payload["segments"]} == {"grok-stt", "anime-whisper"}
        for payload in observed_inputs
    )
    assert any(
        any(segment["role"] == "context_only" for segment in payload["segments"])
        for payload in observed_inputs
    )

    observed_inputs.clear()
    cached_speech, _, cached_summary = correct_transcript_rows(
        speech,
        nonverbal,
        cache_dir=tmp_path / "corrections",
        config=config,
    )
    assert observed_inputs == []
    assert cached_speech[1]["text"] == "こんにちは"
    assert cached_summary["batches_completed"] == cached_summary["batches"]


def test_nonverbal_text_uses_corrected_transcript_when_present() -> None:
    assert (
        render_nonverbal_text("aegi", "aegi_c0007", "あっ、んっ……")
        == "喘ぎ声。あっ、んっ……ラベルc0007。"
    )
    assert render_nonverbal_text("chupa", "chupa_k0012") == "フェラ音。ラベルk0012。"
