from __future__ import annotations

import torch


def find_flattening_points(
    latents: torch.Tensor,
    target_value: float = 0.0,
    window_size: int = 20,
    std_threshold: float = 0.05,
    mean_threshold: float = 0.1,
) -> list[int]:
    """Find the first near-flat trailing window for every sequence in a batch.

    ``latents`` must have shape ``(B, T, D)``. All overlapping windows are
    evaluated in one tensor operation, avoiding one device synchronization per
    latent step on GPU accelerators.
    """
    if latents.ndim != 3:
        raise ValueError(f"Expected latent shape (B, T, D), got {tuple(latents.shape)}")

    batch_size, total_steps, latent_dim = map(int, latents.shape)
    if batch_size == 0:
        return []
    if total_steps <= 0 or window_size <= 0 or latent_dim <= 0:
        return [total_steps] * batch_size

    pad = torch.zeros(
        (batch_size, window_size, latent_dim),
        device=latents.device,
        dtype=latents.dtype,
    )
    padded = torch.cat((latents, pad), dim=1)
    # unfold produces one extra all-padding window; the historical heuristic
    # checks exactly T candidate starts, so keep only those windows.
    windows = padded.unfold(1, window_size, 1)[:, :total_steps]
    reduce_dims = (-2, -1)
    window_std = windows.std(dim=reduce_dims, unbiased=False)
    window_mean = windows.mean(dim=reduce_dims)
    matches = (window_std < std_threshold) & ((window_mean - target_value).abs() < mean_threshold)

    found = matches.any(dim=1)
    first = matches.to(dtype=torch.int64).argmax(dim=1)
    fallback = torch.full_like(first, total_steps)
    return torch.where(found, first, fallback).cpu().tolist()
