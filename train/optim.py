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


# Fixed seed so every DDP rank draws identical rounding noise and parameters
# stay bit-identical across ranks (grads are already synced before step()).
_SR_SEED = 0x5EEDBF16


def _round_copy_(
    dst: torch.Tensor,
    src_f32: torch.Tensor,
    generators: dict[torch.device, torch.Generator],
) -> None:
    """
    Store full-precision results into ``dst``. For bf16 destinations use stochastic
    rounding: add uniform 16-bit noise to the full-precision bit pattern and truncate the
    low mantissa bits, which rounds to either bf16 neighbour with probability
    proportional to proximity (unbiased in expectation). Plain nearest rounding
    would silently drop any update smaller than half a bf16 ulp — at
    lr<=1e-4 that is the majority of Muon/AdamW updates and every weight-decay
    multiply.
    """
    if dst.dtype is not torch.bfloat16:
        dst.copy_(src_f32)
        return
    gen = generators.get(src_f32.device)
    if gen is None:
        gen = torch.Generator(device=src_f32.device)
        gen.manual_seed(_SR_SEED)
        generators[src_f32.device] = gen
    bits = src_f32.contiguous().view(torch.int32)
    noise = torch.randint(
        0, 1 << 16, bits.shape, device=bits.device, dtype=torch.int32, generator=gen
    )
    bits = (bits + noise) & -65536  # keep sign/exponent/high-mantissa (0xFFFF0000)
    dst.copy_(bits.view(torch.float))


def _zeropower_via_newtonschulz(
    grad: torch.Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    # Verbatim port of torch.optim._muon._zeropower_via_newtonschulz (2.10) so
    # MuonBF16SR matches torch.optim.Muon exactly; kept local because that
    # module is private API. NS deliberately runs in bf16.
    if ns_steps >= 100:
        raise ValueError("Number of steps must be less than 100 for computational efficiency")
    if grad.ndim != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


def _adjust_muon_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    # Port of torch.optim._muon._adjust_lr (2.10).
    A, B = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


class MuonBF16SR(torch.optim.Optimizer):
    """
    Muon with the exact update rule of torch.optim.Muon, but the momentum lerp
    and parameter update are computed in full-precision and stored back with stochastic
    rounding when the tensors are bf16 (pure_bf16 mode). Memory layout is
    unchanged: params/grads/momentum stay bf16, only per-tensor full-precision
    temporaries are allocated during step().

    Group defaults and per-param state ("momentum_buffer") are laid out
    identically to torch.optim.Muon, so checkpoints are interchangeable
    between the two implementations in both directions.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if momentum < 0.0:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        "Muon only supports 2D parameters whereas we found a parameter "
                        f"with size: {p.size()}"
                    )
        self._sr_generators: dict[torch.device, torch.Generator] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(grad, memory_format=torch.preserve_format)
                    state["momentum_buffer"] = buf
                buf_f32 = buf.float().lerp_(grad.float(), 1.0 - momentum)
                _round_copy_(buf, buf_f32, self._sr_generators)
                update = grad.float().lerp(buf_f32, momentum) if nesterov else buf_f32
                update = _zeropower_via_newtonschulz(
                    update,
                    group["ns_coefficients"],
                    int(group["ns_steps"]),
                    float(group["eps"]),
                )
                adjusted_lr = _adjust_muon_lr(lr, group["adjust_lr_fn"], p.shape)
                new_p = p.float().mul_(1.0 - lr * weight_decay).add_(
                    update.float(), alpha=-adjusted_lr
                )
                _round_copy_(p, new_p, self._sr_generators)
        return loss


class AdamWBF16(torch.optim.Optimizer):
    """AdamW whose parameters, gradients, moments, and tensor temporaries are bf16."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr must be non-negative, got {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            },
        )
        for group in self.param_groups:
            for param in group["params"]:
                if param.dtype is not torch.bfloat16:
                    raise ValueError(
                        "AdamWBF16 accepts only bf16 parameters, "
                        f"got {param.dtype} for shape={tuple(param.shape)}"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("AdamWBF16 does not support sparse gradients")
                if grad.dtype is not torch.bfloat16:
                    raise RuntimeError(
                        "AdamWBF16 requires bf16 gradients, "
                        f"got {grad.dtype} for shape={tuple(grad.shape)}"
                    )

                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                step = int(state["step"])
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2_sqrt = math.sqrt(1.0 - beta2**step)
                denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps)

                param.mul_(1.0 - lr * weight_decay)
                param.addcdiv_(exp_avg, denom, value=-(lr / bias_correction1))

        return loss


class AdamWBF16SR(torch.optim.AdamW):
    """
    AdamW (decoupled weight decay) computing the step in full-precision and storing
    params and both moments back with stochastic rounding when bf16. State
    layout ("step"/"exp_avg"/"exp_avg_sq") matches torch.optim.AdamW, so
    checkpoints load in either direction. Moments matter here too: the
    exp_avg_sq lerp contributes (1-beta2)=1e-3 relatively, below the bf16
    resolution, so without SR the second moment freezes.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sr_generators: dict[torch.device, torch.Generator] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get("amsgrad") or group.get("maximize"):
                raise RuntimeError(
                    "AdamWBF16SR supports only amsgrad=False, maximize=False."
                )
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                state["step"] += 1
                step_t = float(state["step"].item())
                exp_avg_f32 = state["exp_avg"].float().lerp_(grad, 1.0 - beta1)
                _round_copy_(state["exp_avg"], exp_avg_f32, self._sr_generators)
                exp_avg_sq_f32 = (
                    state["exp_avg_sq"].float().mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                )
                _round_copy_(state["exp_avg_sq"], exp_avg_sq_f32, self._sr_generators)
                bias_correction1 = 1.0 - beta1**step_t
                bias_correction2 = 1.0 - beta2**step_t
                denom = (exp_avg_sq_f32 / bias_correction2).sqrt_().add_(eps)
                new_p = p.float().mul_(1.0 - lr * weight_decay).addcdiv_(
                    exp_avg_f32, denom, value=-(lr / bias_correction1)
                )
                _round_copy_(p, new_p, self._sr_generators)
        return loss


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
    # pure_bf16 stores params/grads/moments in bf16; nearest rounding there
    # loses most updates (see _round_copy_), so switch to the full-precision-math +
    # stochastic-rounding implementations. Harmless when the model was left
    # full-precision (the SR store is dtype-gated per tensor).
    use_bf16_sr = bool(cfg.pure_bf16) and bool(cfg.bf16_stochastic_round)
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
            optimizer_cls = AdamWBF16SR if use_bf16_sr else torch.optim.AdamW
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
        if use_bf16_sr:
            muon_cls: type[torch.optim.Optimizer] = MuonBF16SR
            aux_cls: type[torch.optim.Optimizer] = AdamWBF16SR
        else:
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
