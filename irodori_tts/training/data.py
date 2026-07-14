from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import torch

from ..config import ModelConfig


def training_batch_keys(
    model_cfg: ModelConfig,
    *,
    duration_only: bool,
) -> tuple[str, ...]:
    """Return only the host-batch tensors consumed by the selected objective."""
    keys = ["text_ids", "text_mask", "num_frames"]
    if model_cfg.use_caption_condition:
        keys.extend(("caption_ids", "caption_mask", "has_caption"))
    if model_cfg.use_speaker_condition_resolved:
        keys.extend(("ref_latent_patched", "ref_latent_mask_patched", "has_speaker"))
    if not duration_only:
        keys.extend(("latent_patched", "latent_mask_patched", "latent_mask_valid_patched"))
    if model_cfg.use_duration_predictor:
        keys.append("duration_features")
    return tuple(keys)


def _move_selected_tensors(
    batch: Mapping[str, Any],
    *,
    keys: Sequence[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key in keys:
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Training batch value {key!r} must be a tensor, got {type(value)!r}")
        moved[key] = value.to(device=device, non_blocking=True)
    return moved


class _CudaPrefetchIterator(Iterator[dict[str, torch.Tensor]]):
    """Overlap the next pinned-memory H2D copy with the current model step."""

    def __init__(
        self,
        source: Iterable[Mapping[str, Any]],
        *,
        keys: Sequence[str],
        device: torch.device,
    ) -> None:
        self._source = iter(source)
        self._keys = tuple(keys)
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._next_batch: dict[str, torch.Tensor] | None = None
        self._preload()

    def __iter__(self) -> _CudaPrefetchIterator:
        return self

    def _preload(self) -> None:
        try:
            host_batch = next(self._source)
        except StopIteration:
            self._next_batch = None
            return
        with torch.cuda.stream(self._stream):
            self._next_batch = _move_selected_tensors(
                host_batch,
                keys=self._keys,
                device=self._device,
            )

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._next_batch is None:
            raise StopIteration
        current_stream = torch.cuda.current_stream(self._device)
        current_stream.wait_stream(self._stream)
        batch = self._next_batch
        self._preload()
        for value in batch.values():
            value.record_stream(current_stream)
        return batch


def iter_device_batches(
    source: Iterable[Mapping[str, Any]],
    *,
    keys: Sequence[str],
    device: torch.device,
    prefetch: bool,
) -> Iterator[dict[str, torch.Tensor]]:
    """Move selected tensors to the training device, optionally one batch ahead."""
    if prefetch and device.type == "cuda":
        yield from _CudaPrefetchIterator(source, keys=keys, device=device)
        return
    for batch in source:
        yield _move_selected_tensors(batch, keys=keys, device=device)
