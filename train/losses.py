from __future__ import annotations

import torch


def echo_style_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Echo/JAX-style token loss normalized by the valid-token ratio."""
    diff = (pred - target).square().mean(dim=-1)
    loss_weight = loss_mask.float()
    valid_weight = valid_mask.float()
    has_valid = (valid_weight.sum(dim=-1) > 0).float()[:, None]
    denominator = (loss_weight * valid_weight * has_valid).mean().clamp_min(1e-6)
    return (diff * loss_weight).mean() / denominator


def utterance_mean_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Average token MSE per utterance before averaging the batch."""
    diff = (pred - target).square().mean(dim=-1)
    weight = valid_mask.float()
    per_sample = (diff * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)
    return per_sample.mean()


def compute_rf_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Dispatch to a validated rectified-flow loss normalization."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "echo":
        return echo_style_masked_mse(
            pred,
            target,
            loss_mask=loss_mask,
            valid_mask=valid_mask,
        )
    if normalized_mode == "utterance_mean":
        return utterance_mean_masked_mse(pred, target, valid_mask=valid_mask)
    raise ValueError(
        f"Unsupported rf_loss_mode={normalized_mode!r}. Expected 'echo' or 'utterance_mean'."
    )
