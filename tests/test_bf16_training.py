"""Training precision contract: FP32 masters + optional bf16 autocast (original path)."""

from __future__ import annotations

import torch

from core.config import ModelConfig, TrainConfig
from core.model import TextToLatentRFDiT
from core.rf import sample_logit_normal_t


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
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


def test_train_config_mixed_precision_defaults() -> None:
    cfg = TrainConfig()
    assert cfg.precision in {"fp32", "bf16"}
    assert not hasattr(cfg, "pure_bf16")
    assert not hasattr(cfg, "bf16_stochastic_round")


def test_rf_timestep_sampling_is_float32() -> None:
    t = sample_logit_normal_t(4, device=torch.device("cpu"))
    assert t.dtype == torch.float32
    assert t.shape == (4,)


def test_train_step_fp32_masters_with_optional_bf16_autocast() -> None:
    """One micro-batch forward+backward+optimizer step on the shipped model."""
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

    use_cuda_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    device = torch.device("cuda" if use_cuda_bf16 else "cpu")
    model = model.to(device)
    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device)
    x_t = x_t.to(device)
    t = t.to(device)
    target = target.to(device)

    model.train()
    optim.zero_grad(set_to_none=True)

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
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_train_step_explicit_fp32_precision() -> None:
    model = TextToLatentRFDiT(_tiny_cfg()).to(dtype=torch.float32)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2)
    text_ids = torch.randint(0, 32, (1, 4))
    text_mask = torch.ones(1, 4, dtype=torch.bool)
    x_t = torch.randn(1, 5, 8)
    t = torch.tensor([0.4])
    target = torch.randn(1, 5, 8)

    model.train()
    optim.zero_grad(set_to_none=True)
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
    assert torch.isfinite(loss.detach()).item()
    assert all(p.dtype == torch.float32 for p in model.parameters())
