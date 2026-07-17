from __future__ import annotations

import torch


class ModelEMA:
    """
    Exponential moving average of model weights for RF/diffusion inference quality.

    The shadow copy tracks ``model.state_dict()`` keys so it can be loaded back
    into the same architecture directly. Floating tensors are kept in fp32 on
    ``device`` (CPU by default so a 16 GB GPU pays no VRAM cost); non-floating
    tensors are mirrored verbatim.
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

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self._effective_decay
        state = model.state_dict()
        for name, shadow in self.shadow.items():
            value = state[name].detach()
            if shadow.is_floating_point():
                shadow.mul_(d).add_(
                    value.to(device=shadow.device, dtype=shadow.dtype),
                    alpha=1.0 - d,
                )
            else:
                shadow.copy_(value.to(device=shadow.device))

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
