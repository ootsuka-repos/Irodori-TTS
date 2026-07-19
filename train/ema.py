from __future__ import annotations

import torch


class ModelEMA:
    """
    Exponential moving average of model weights for RF/diffusion inference quality.

    The shadow copy tracks ``model.state_dict()`` keys so it can be loaded back
    into the same architecture directly. Floating tensors are kept in fp32 on
    ``device`` (CPU by default so a 16 GB GPU pays no VRAM cost); non-floating
    tensors are mirrored verbatim.

    For the common CUDA-model/CPU-shadow layout, updates stage all weights into
    pinned host buffers with one batched async device-to-host copy and then run
    a single fused ``_foreach_lerp_`` instead of a synchronous transfer per
    tensor, which keeps the EMA cost far below one optimizer step.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        decay: float,
        update_every: int = 1,
        device: str | torch.device | None = None,
    ) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"ema decay must be in (0, 1), got {decay}")
        if update_every < 1:
            raise ValueError(f"ema update_every must be >= 1, got {update_every}")
        self.decay = float(decay)
        self.update_every = int(update_every)
        # Calling update() every N optimizer steps with decay^N matches the
        # trajectory of a per-step EMA with the configured decay.
        self._effective_decay = self.decay**self.update_every
        target_device = torch.device(device) if device is not None else None
        self.shadow: dict[str, torch.Tensor] = {}
        for name, tensor in model.state_dict().items():
            value = tensor.detach().clone()
            if target_device is not None:
                value = value.to(device=target_device)
            if value.is_floating_point():
                value = value.float()
            self.shadow[name] = value
        self._float_names = [n for n, v in self.shadow.items() if v.is_floating_point()]
        self._other_names = [n for n, v in self.shadow.items() if not v.is_floating_point()]
        self._float_shadows = [self.shadow[n] for n in self._float_names]
        self._staging: list[torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self._effective_decay
        state = model.state_dict()
        if self._float_names:
            sources = [state[name].detach() for name in self._float_names]
            use_staging = (
                sources[0].device.type == "cuda"
                and self._float_shadows[0].device.type == "cpu"
            )
            if use_staging:
                if self._staging is None:
                    self._staging = [
                        torch.empty_like(shadow, pin_memory=True)
                        for shadow in self._float_shadows
                    ]
                for staged, value in zip(self._staging, sources, strict=True):
                    staged.copy_(value, non_blocking=True)
                torch.cuda.synchronize(sources[0].device)
                sources = self._staging
            else:
                sources = [
                    value.to(device=shadow.device, dtype=shadow.dtype)
                    for shadow, value in zip(self._float_shadows, sources, strict=True)
                ]
            torch._foreach_lerp_(self._float_shadows, sources, 1.0 - d)
        for name in self._other_names:
            shadow = self.shadow[name]
            shadow.copy_(state[name].detach().to(device=shadow.device))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.shadow)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> tuple[int, int]:
        """Copy matching keys; return (loaded, missing) counts for logging."""
        loaded = 0
        missing = 0
        for name, shadow in self.shadow.items():
            value = state.get(name)
            if value is None or tuple(value.shape) != tuple(shadow.shape):
                missing += 1
                continue
            shadow.copy_(value.to(device=shadow.device, dtype=shadow.dtype))
            loaded += 1
        return loaded, missing
