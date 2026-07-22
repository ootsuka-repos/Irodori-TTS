"""Tests for restored mixed-precision train/infer contract (original-compatible).

fp32 masters + optional bf16 compute/autocast; inference accepts fp32 and bf16.
Pure-bf16 experiment paths remain optional and are not required here.
"""

from __future__ import annotations

import pytest
import torch

from core.codec import DACVAECodec
from core.config import ModelConfig, TrainConfig
from core.duration import build_duration_features
from core.model import (
    RMSNorm,
    TextToLatentRFDiT,
    apply_rotary_emb,
    get_timestep_embedding,
    precompute_freqs_cis,
)
from core.rf import sample_euler_rf_cfg, sample_logit_normal_t
from core.speaker_inversion import SpeakerInversionEmbedding
from core.watermark import SilentCipherWatermarker
from inference.cli.convert_checkpoint import _prepare_inference_state
from inference.runtime import (
    _load_audio,
    list_available_runtime_precisions,
    resolve_runtime_dtype,
    save_wav,
)


def _tiny_cfg(**overrides) -> ModelConfig:
    base = dict(
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
    base.update(overrides)
    return ModelConfig(**base)


def _tiny_model(dtype: torch.dtype = torch.float32) -> TextToLatentRFDiT:
    return TextToLatentRFDiT(_tiny_cfg()).to(dtype=dtype).eval()


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

    norm = RMSNorm(8)
    y = norm(x.reshape(2, 7, 2, 8)[..., 0, :])  # (B,S,Dh) path not needed; just vector
    # RMSNorm on (2,7,8) bf16 input
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

    t_bf = t.to(torch.bfloat16)
    emb_bf = get_timestep_embedding(t_bf, dim=8)
    assert emb_bf.dtype == torch.bfloat16


def test_tiny_model_forward_fp32_and_bf16() -> None:
    text_ids = torch.randint(0, 32, (1, 5))
    text_mask = torch.ones(1, 5, dtype=torch.bool)
    x_t = torch.randn(1, 7, 8)
    t = torch.tensor([0.5])

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
            v = model(
                x_t=x_t.to(dtype=dtype),
                t=t.to(dtype=dtype),
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=None,
                ref_mask=None,
            )
        assert latent.shape == (1, 7, 8)
        assert latent.dtype == dtype
        assert v.dtype == dtype
        assert torch.isfinite(latent.float()).all()
        assert torch.isfinite(v.float()).all()


def test_train_config_mixed_precision_defaults() -> None:
    cfg = TrainConfig()
    assert cfg.precision in {"fp32", "bf16"}
    assert cfg.precision == "fp32"
    assert not hasattr(cfg, "pure_bf16")
    assert not hasattr(cfg, "bf16_stochastic_round")


def test_resolve_runtime_dtype_fp32_and_bf16() -> None:
    cpu = torch.device("cpu")
    assert resolve_runtime_dtype(precision="fp32", device=cpu) is torch.float32
    assert list_available_runtime_precisions(cpu) == ["fp32"]
    with pytest.raises(ValueError, match="requires CUDA|fp32, bf16"):
        resolve_runtime_dtype(precision="bf16", device=cpu)
    with pytest.raises(ValueError, match="Unsupported precision"):
        resolve_runtime_dtype(precision="fp16", device=cpu)

    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        assert resolve_runtime_dtype(precision="bf16", device=cuda) is torch.bfloat16
        assert resolve_runtime_dtype(precision="fp32", device=cuda) is torch.float32
        assert "fp32" in list_available_runtime_precisions(cuda)
        assert "bf16" in list_available_runtime_precisions(cuda)


def test_checkpoint_prepare_keeps_float32() -> None:
    prepared = _prepare_inference_state(
        {
            "weight": torch.randn(2, 3),
            "counter": torch.tensor(4, dtype=torch.int64),
        }
    )
    assert prepared["weight"].dtype == torch.float32
    assert prepared["counter"].dtype == torch.int64


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
    # Without audiotools, normalize with target_db=None path only.
    normalized = DACVAECodec._normalize_loudness(waveform, sample_rate=16_000, target_db=None)
    assert normalized.dtype == torch.float32

    output = tmp_path / "mixed-precision.wav"
    save_wav(output, normalized.unsqueeze(0), 16_000)
    loaded, sample_rate = _load_audio(output)
    assert sample_rate == 16_000
    assert loaded.dtype == torch.float32 or loaded.is_floating_point()

    watermarker = SilentCipherWatermarker(device="cpu")
    out = watermarker.encode_one(loaded if loaded.ndim >= 1 else loaded, sample_rate=sample_rate)
    assert out.is_floating_point()


def test_rf_timestep_sampling_is_float32() -> None:
    t = sample_logit_normal_t(4, device=torch.device("cpu"))
    assert t.dtype == torch.float32
    assert t.shape == (4,)


def test_train_step_fp32_masters_with_optional_bf16_autocast() -> None:
    """One micro-batch forward+backward+optimizer step on shipped model."""
    model = TextToLatentRFDiT(_tiny_cfg()).to(dtype=torch.float32)
    assert all(p.dtype == torch.float32 for p in model.parameters())

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    text_ids = torch.randint(0, 32, (2, 5))
    text_mask = torch.ones(2, 5, dtype=torch.bool)
    x0 = torch.randn(2, 6, 8)
    noise = torch.randn_like(x0)
    t = sample_logit_normal_t(2, device=torch.device("cpu"))
    x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * noise
    target = noise - x0

    model.train()
    optim.zero_grad(set_to_none=True)

    use_cuda_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    device = torch.device("cuda" if use_cuda_bf16 else "cpu")
    model = model.to(device)
    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device)
    x_t = x_t.to(device)
    t = t.to(device)
    target = target.to(device)

    if use_cuda_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(
                x_t=x_t,
                t=t,
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=None,
                ref_mask=None,
            )
            loss = torch.nn.functional.mse_loss(pred.float(), target.float())
    else:
        pred = model(
            x_t=x_t,
            t=t,
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=None,
            ref_mask=None,
        )
        loss = torch.nn.functional.mse_loss(pred, target)

    loss.backward()
    optim.step()

    assert torch.isfinite(loss.detach().float()).item()
    # Masters remain full precision on the mixed-precision path.
    assert all(p.dtype == torch.float32 for p in model.parameters())
