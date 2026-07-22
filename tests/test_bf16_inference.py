from __future__ import annotations

import pytest
import torch

from core.codec import DACVAECodec
from core.config import ModelConfig
from core.duration import build_duration_features
from core.model import TextToLatentRFDiT, apply_rotary_emb, precompute_freqs_cis
from core.rf import sample_euler_rf_cfg
from core.speaker_inversion import SpeakerInversionEmbedding
from core.watermark import SilentCipherWatermarker
from inference.cli.convert_checkpoint import _cast_inference_state_bf16
from inference.runtime import (
    _assert_module_bf16,
    _cast_checkpoint_state_bf16,
    _load_audio,
    save_wav,
)


def _tiny_model() -> TextToLatentRFDiT:
    cfg = ModelConfig(
        latent_dim=8,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2.0,
        text_mlp_ratio=2.0,
        text_vocab_size=32,
        text_dim=16,
        text_layers=1,
        text_heads=2,
        use_speaker_condition=False,
        timestep_embed_dim=8,
        adaln_rank=4,
    )
    return TextToLatentRFDiT(cfg).to(dtype=torch.bfloat16).eval()


def test_rope_uses_real_bf16_pairs_without_complex_tensors() -> None:
    x = torch.randn(2, 7, 2, 8, dtype=torch.bfloat16)
    freqs = precompute_freqs_cis(dim=8, end=7)
    actual = apply_rotary_emb(x, freqs)

    assert freqs.shape == (7, 4, 2)
    assert freqs.dtype == actual.dtype == torch.bfloat16
    assert not freqs.is_complex()

    pairs = x.float().reshape(2, 7, 2, 4, 2)
    complex_x = torch.view_as_complex(pairs)
    complex_freqs = torch.complex(freqs[..., 0].float(), freqs[..., 1].float())
    expected = torch.view_as_real(complex_x * complex_freqs[None, :, None]).reshape_as(x)
    assert torch.allclose(actual.float(), expected, atol=0.03, rtol=0.03)


def test_tiny_model_and_rf_sampler_remain_bf16() -> None:
    model = _tiny_model()
    _assert_module_bf16(model, label="tiny model")

    text_ids = torch.randint(0, 32, (1, 5))
    text_mask = torch.ones(1, 5, dtype=torch.bool)
    with torch.inference_mode():
        latent = sample_euler_rf_cfg(
            model,
            text_ids,
            text_mask,
            None,
            None,
            sequence_length=7,
            num_steps=2,
            cfg_scale_text=0.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
        )

    assert latent.shape == (1, 7, 8)
    assert latent.dtype == torch.bfloat16
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert all(buffer.dtype == torch.bfloat16 for buffer in model.buffers())

    with pytest.raises(RuntimeError, match="non-BF16"):
        _assert_module_bf16(torch.nn.Linear(2, 2), label="FP32 model")


def test_converted_inference_checkpoint_is_bf16() -> None:
    converted = _cast_inference_state_bf16(
        {
            "weight": torch.randn(2, 3),
            "counter": torch.tensor(4, dtype=torch.int64),
        }
    )
    assert converted["weight"].dtype == torch.bfloat16
    assert converted["counter"].dtype == torch.int64

    loaded = _cast_checkpoint_state_bf16({"weight": torch.randn(2, 3)})
    assert loaded["weight"].dtype == torch.bfloat16


def test_duration_speaker_and_audio_boundaries_are_bf16(tmp_path) -> None:
    features = build_duration_features(
        ["BF16 inference"],
        token_counts=[4],
        max_text_len=32,
        has_speaker=[True],
    )
    assert features.dtype == torch.bfloat16

    speaker = SpeakerInversionEmbedding(
        num_tokens=2,
        speaker_dim=8,
        init_std=0.02,
    )
    assert speaker.embedding.dtype == torch.bfloat16

    waveform = torch.randn(1600, dtype=torch.bfloat16) * 0.01
    normalized = DACVAECodec._normalize_loudness(waveform, sample_rate=16_000, target_db=-16.0)
    assert normalized.dtype == torch.bfloat16

    output = tmp_path / "strict-bf16.wav"
    save_wav(output, normalized.unsqueeze(0), 16_000)
    loaded, sample_rate = _load_audio(output)
    assert sample_rate == 16_000
    assert loaded.dtype == torch.bfloat16

    watermarker = SilentCipherWatermarker(device="cpu")
    assert not watermarker.ready
    assert watermarker.encode_one(loaded, sample_rate=sample_rate).dtype == torch.bfloat16
