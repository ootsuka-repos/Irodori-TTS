"""Dependency-light tensor layout operations for codec latent sequences."""

from __future__ import annotations

import torch


def patchify_latent(latent: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert ``(B, T, D)`` latents to ``(B, T // patch, D * patch)``."""
    if patch_size <= 1:
        return latent
    batch_size, sequence_length, dimension = latent.shape
    usable = (sequence_length // patch_size) * patch_size
    return latent[:, :usable].reshape(
        batch_size,
        usable // patch_size,
        dimension * patch_size,
    )


def unpatchify_latent(
    patched: torch.Tensor,
    patch_size: int,
    latent_dim: int,
) -> torch.Tensor:
    """Convert patched latents back to ``(B, T * patch, D)``."""
    if patch_size <= 1:
        return patched
    return patched.reshape(
        patched.shape[0],
        patched.shape[1] * patch_size,
        latent_dim,
    )
