from __future__ import annotations

import math

import torch

from core.config import TrainConfig


class MuonWithAuxAdamW:
    """
    Simple single-process wrapper:
    - torch.optim.Muon for Muon-compatible parameters
    - AdamW for auxiliary parameters (embeddings, biases, output heads, etc.)
    """

    def __init__(self, muon_opt: torch.optim.Optimizer, aux_opt: torch.optim.Optimizer | None):
        self.muon_opt = muon_opt
        self.aux_opt = aux_opt

    @property
    def param_groups(self) -> list[dict]:
        # Must be a live view, not an __init__-time snapshot: torch's
        # Optimizer.load_state_dict replaces the inner optimizers' group dicts,
        # so a snapshot would leave the LR scheduler and the resume-time lr/wd
        # rebase mutating dead dicts while the real optimizers keep the
        # checkpoint-time values forever.
        groups = list(self.muon_opt.param_groups)
        if self.aux_opt is not None:
            groups.extend(self.aux_opt.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon_opt.zero_grad(set_to_none=set_to_none)
        if self.aux_opt is not None:
            self.aux_opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon_opt.step()
        if self.aux_opt is not None:
            self.aux_opt.step()

    def state_dict(self) -> dict:
        return {
            "muon": self.muon_opt.state_dict(),
            "aux": None if self.aux_opt is None else self.aux_opt.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "muon" not in state_dict:
            raise ValueError(
                "MuonWithAuxAdamW state_dict must contain 'muon' key (and optional 'aux' key)."
            )

        self.muon_opt.load_state_dict(state_dict["muon"])
        if self.aux_opt is not None and state_dict.get("aux") is not None:
            self.aux_opt.load_state_dict(state_dict["aux"])


class ScalarLRScheduler:
    """
    Scheduler that applies a scalar multiplier to all optimizer param-group LRs.
    Works with both torch optimizers and MuonWithAuxAdamW wrapper.
    """

    def __init__(self, optimizer, lr_lambda):
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_step = -1

    def step(self) -> None:
        self.last_step += 1
        scale = float(self.lr_lambda(self.last_step))
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups, strict=False):
            group["lr"] = base_lr * scale

    def state_dict(self) -> dict:
        return {
            "base_lrs": list(self.base_lrs),
            "last_step": int(self.last_step),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "base_lrs" in state_dict:
            loaded_base_lrs = [float(x) for x in state_dict["base_lrs"]]
            if len(loaded_base_lrs) == len(self.optimizer.param_groups):
                self.base_lrs = loaded_base_lrs
        if "last_step" in state_dict:
            self.last_step = int(state_dict["last_step"])


def _use_weight_decay(name: str, p: torch.nn.Parameter) -> bool:
    """
    Approximate Echo/JAX no-decay mask semantics for PyTorch modules.

    Flax side no-decay keys include:
      bias, scale, weight, gate, shift, freqs, phases, out_proj,
      adaln_rank_shift, adaln_rank_scale, adaln_rank_gate.

    In PyTorch:
    - keep all `.bias` no-decay,
    - map Flax `weight` mostly to normalization weights (`*norm*.weight`),
    - apply token-based exclusions for AdaLN / out-proj / freq-phase terms.
    """
    lname = name.lower()

    if lname.endswith(".bias"):
        return False

    # Flax `weight` token mostly corresponds to norm weights in this model family.
    if lname.endswith(".weight") and "norm" in lname:
        return False

    # Echo Flax wd_mask matches exact keys. In this PyTorch model the closest
    # equivalent is to restrict shift/scale/gate exclusions to AdaLN modules.
    if (".attention_adaln." in lname or ".mlp_adaln." in lname) and any(
        token in lname for token in ("shift", "scale", "gate", "adaln_rank_")
    ):
        return False

    if "out_proj" in lname:
        return False

    if "freqs" in lname or "phases" in lname:
        return False

    return True


def _partition_adamw_params(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _use_weight_decay(name, p):
            decay.append(p)
        else:
            no_decay.append(p)
    return decay, no_decay


def _partition_muon_params(
    model: torch.nn.Module,
) -> tuple[
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
]:
    """
    Split parameters into Muon-compatible tensors and aux-Adam tensors,
    each with decay/no-decay partitions.
    """
    muon_decay: list[torch.nn.Parameter] = []
    muon_no_decay: list[torch.nn.Parameter] = []
    aux_decay: list[torch.nn.Parameter] = []
    aux_no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Muon is intended for hidden matrix-like weights; torch.optim.Muon
        # hard-requires ndim == 2, so route anything else (vectors, but also
        # e.g. (1,1,dim) query tokens or conv kernels) to Adam instead of
        # crashing at construction. Keep embeddings/output heads/bias-like
        # params on Adam. Norm gains can be 2D here (per-head RMSNorm weights)
        # but are scales, not matrices, so orthogonalizing them would destroy
        # their meaning.
        lname = name.lower()
        is_muon_candidate = (
            p.ndim == 2
            and "embedding" not in lname
            and "norm" not in lname
            and not lname.endswith("out_proj.weight")
        )
        has_decay = _use_weight_decay(name, p)
        if is_muon_candidate:
            if has_decay:
                muon_decay.append(p)
            else:
                muon_no_decay.append(p)
        else:
            if has_decay:
                aux_decay.append(p)
            else:
                aux_no_decay.append(p)
    return muon_decay, muon_no_decay, aux_decay, aux_no_decay


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig):
    opt_name = cfg.optimizer.lower()
    if opt_name in {"adamw", "adamw8bit"}:
        decay, no_decay = _partition_adamw_params(model)
        param_groups = []
        # irodori_decay records each group's weight-decay intent so resume can
        # tell decay from no-decay groups after optimizer.load_state_dict pins
        # the live weight_decay values to the checkpoint's copies (see the
        # resume rebase in train.cli.train). It rounds-trips through state_dict.
        if decay:
            param_groups.append(
                {"params": decay, "weight_decay": cfg.weight_decay, "irodori_decay": True}
            )
        if no_decay:
            param_groups.append(
                {"params": no_decay, "weight_decay": 0.0, "irodori_decay": False}
            )
        if opt_name == "adamw8bit":
            try:
                from bitsandbytes.optim import AdamW8bit
            except ImportError as exc:
                raise RuntimeError(
                    "optimizer=adamw8bit requires bitsandbytes. Install the project dependencies."
                ) from exc
            optimizer_cls = AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW
        return optimizer_cls(
            param_groups if param_groups else model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=0.0,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
        )
    if opt_name == "muon":
        adjust_lr_fn = cfg.muon_adjust_lr_fn
        if adjust_lr_fn not in {"original", "match_rms_adamw"}:
            raise ValueError(
                "muon_adjust_lr_fn must be one of ['original', 'match_rms_adamw'], "
                f"got {adjust_lr_fn!r}"
            )
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "optimizer=muon requires torch.optim.Muon (available in newer PyTorch releases)."
            )
        muon_cls = torch.optim.Muon
        aux_cls = torch.optim.AdamW

        muon_decay, muon_no_decay, aux_decay, aux_no_decay = _partition_muon_params(model)
        muon_param_groups = []
        if muon_decay:
            muon_param_groups.append(
                {"params": muon_decay, "weight_decay": cfg.weight_decay, "irodori_decay": True}
            )
        if muon_no_decay:
            muon_param_groups.append(
                {"params": muon_no_decay, "weight_decay": 0.0, "irodori_decay": False}
            )
        if not muon_param_groups:
            raise ValueError("No Muon-compatible parameters found for optimizer=muon.")

        muon_opt = muon_cls(
            muon_param_groups,
            lr=cfg.learning_rate,
            weight_decay=0.0,
            momentum=cfg.muon_momentum,
            adjust_lr_fn=adjust_lr_fn,
        )
        aux_opt = None
        aux_param_groups = []
        if aux_decay:
            aux_param_groups.append(
                {"params": aux_decay, "weight_decay": cfg.weight_decay, "irodori_decay": True}
            )
        if aux_no_decay:
            aux_param_groups.append(
                {"params": aux_no_decay, "weight_decay": 0.0, "irodori_decay": False}
            )
        if aux_param_groups:
            aux_opt = aux_cls(
                aux_param_groups,
                lr=cfg.learning_rate,
                weight_decay=0.0,
                betas=(cfg.adam_beta1, cfg.adam_beta2),
                eps=cfg.adam_eps,
            )
        return MuonWithAuxAdamW(muon_opt=muon_opt, aux_opt=aux_opt)

    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def build_scheduler(
    optimizer,
    cfg: TrainConfig,
):
    sched_name = cfg.lr_scheduler.lower()
    if sched_name == "none":
        return None
    if sched_name not in {"cosine", "wsd"}:
        raise ValueError(f"Unsupported lr_scheduler: {cfg.lr_scheduler}")

    max_steps = max(1, int(cfg.max_steps))
    warmup_steps = max(0, int(cfg.warmup_steps))
    stable_steps = max(0, int(cfg.stable_steps))
    min_lr_scale = float(max(0.0, min(1.0, cfg.min_lr_scale)))

    def lr_lambda(step: int) -> float:
        s = int(step)
        if warmup_steps > 0 and s < warmup_steps:
            return float(s + 1) / float(warmup_steps)

        if sched_name == "cosine":
            denom = max(1, max_steps - warmup_steps)
            progress = min(1.0, max(0.0, float(s - warmup_steps) / float(denom)))
        else:
            if s < warmup_steps + stable_steps:
                return 1.0
            denom = max(1, max_steps - warmup_steps - stable_steps)
            progress = min(1.0, max(0.0, float(s - warmup_steps - stable_steps) / float(denom)))

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return ScalarLRScheduler(optimizer, lr_lambda=lr_lambda)


def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])
