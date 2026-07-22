"""Inference precision contract: fp32 and bf16 both work (original-compatible)."""

from __future__ import annotations

import pytest
import torch

from core.codec import DACVAECodec
from core.config import ModelConfig
from core.duration import build_duration_features
from core.model import (
    RMSNorm,
    TextToLatentRFDiT,
    apply_rotary_emb,
    get_timestep_embedding,
    precompute_freqs_cis,
)
from core.rf import sample_euler_rf_cfg
from core.speaker_inversion import SpeakerInversionEmbedding
from core.watermark import SilentCipherWatermarker
from inference.cli.convert_checkpoint import _prepare_inference_state
from inference.runtime import (
    _load_audio,
    list_available_runtime_precisions,
    resolve_runtime_dtype,
    save_wav,
)


def _tiny_model(dtype: torch.dtype = torch.float32) -> TextToLatentRFDiT:
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
    return TextToLatentRFDiT(cfg).to(dtype=dtype).eval()


def test_rope_uses_complex_float_path() -> None:
    x = torch.randn(2, 7, 2, 8, dtype=torch.float32)
    freqs = precompute_freqs_cis(dim=8, end=7)
    actual = apply_rotary_emb(x, freqs)

    assert freqs.is_complex()
    assert freqs.dtype == torch.complex64
    assert actual.dtype == torch.float32

    pairs = x.float().reshape(2, 7, 2, 4, 2)
    complex_x = torch.view_as_complex(pairs)
    expected = torch.view_as_real(complex_x * freqs[None, :, None]).reshape_as(x)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_rope_and_rmsnorm_work_under_bf16_weights() -> None:
    x = torch.randn(2, 7, 2, 8, dtype=torch.bfloat16)
    freqs = precompute_freqs_cis(dim=8, end=7)
    rotated = apply_rotary_emb(x, freqs)
    assert rotated.dtype == torch.bfloat16
    assert torch.isfinite(rotated.float()).all()

    norm = RMSNorm(8)
    inp = torch.randn(2, 7, 8, dtype=torch.bfloat16)
    out = norm(inp)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_timestep_embedding_uses_float_intermediates() -> None:
    t = torch.tensor([0.25, 0.75], dtype=torch.float32)
    emb = get_timestep_embedding(t, dim=8)
    assert emb.shape == (2, 8)
    assert emb.dtype == torch.float32
    assert torch.isfinite(emb).all()

    emb_bf = get_timestep_embedding(t.to(torch.bfloat16), dim=8)
    assert emb_bf.dtype == torch.bfloat16


def test_tiny_model_and_rf_sampler_fp32_and_bf16() -> None:
    text_ids = torch.randint(0, 32, (1, 5))
    text_mask = torch.ones(1, 5, dtype=torch.bool)

    for dtype in (torch.float32, torch.bfloat16):
        model = _tiny_model(dtype=dtype)
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
        assert latent.dtype == dtype
        assert torch.isfinite(latent.float()).all()
        assert all(parameter.dtype == dtype for parameter in model.parameters())


def test_converted_inference_checkpoint_keeps_native_dtype() -> None:
    converted = _prepare_inference_state(
        {
            "weight": torch.randn(2, 3),
            "counter": torch.tensor(4, dtype=torch.int64),
        }
    )
    assert converted["weight"].dtype == torch.float32
    assert converted["counter"].dtype == torch.int64


def test_resolve_runtime_dtype_fp32_and_bf16() -> None:
    cpu = torch.device("cpu")
    assert resolve_runtime_dtype(precision="fp32", device=cpu) is torch.float32
    assert list_available_runtime_precisions(cpu) == ["fp32"]
    with pytest.raises(ValueError, match="fp32, bf16|requires CUDA"):
        resolve_runtime_dtype(precision="bf16", device=cpu)

    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        assert resolve_runtime_dtype(precision="bf16", device=cuda) is torch.bfloat16
        assert resolve_runtime_dtype(precision="fp32", device=cuda) is torch.float32
        precisions = list_available_runtime_precisions(cuda)
        assert "fp32" in precisions and "bf16" in precisions


def test_duration_speaker_and_audio_boundaries_float32(tmp_path) -> None:
    features = build_duration_features(
        ["mixed precision"],
        token_counts=[4],
        max_text_len=32,
        has_speaker=[True],
    )
    assert features.dtype == torch.float32

    speaker = SpeakerInversionEmbedding(
        num_tokens=2,
        speaker_dim=8,
        init_std=0.02,
    )
    assert speaker.embedding.dtype == torch.float32

    waveform = torch.randn(1600, dtype=torch.float32) * 0.01
    normalized = DACVAECodec._normalize_loudness(waveform, sample_rate=16_000, target_db=None)
    assert normalized.dtype == torch.float32

    output = tmp_path / "mixed-precision.wav"
    save_wav(output, normalized.unsqueeze(0), 16_000)
    loaded, sample_rate = _load_audio(output)
    assert sample_rate == 16_000
    assert loaded.is_floating_point()

    watermarker = SilentCipherWatermarker(device="cpu")
    encoded = watermarker.encode_one(loaded, sample_rate=sample_rate)
    assert encoded.is_floating_point()
