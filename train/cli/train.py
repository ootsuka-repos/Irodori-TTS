#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import webbrowser
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from core.config import (
    ModelConfig,
    TrainConfig,
    dump_configs,
    load_experiment_yaml,
    merge_dataclass_overrides,
)
from core.duration import set_duration_has_speaker_feature
from core.lora import (
    LORA_METADATA_NAME,
    LORA_TARGET_PRESETS,
    LORA_TRAIN_CONFIG_FIELDS,
    LORA_TRAINER_STATE_NAME,
    apply_lora,
    count_parameters,
    is_lora_adapter_dir,
    load_lora_adapter,
    train_config_uses_lora,
)
from core.model import (
    DURATION_ARCHITECTURES,
    DURATION_CAPTION_FUSIONS,
    DURATION_CAPTION_POOLINGS,
    DURATION_SPEAKER_FUSIONS,
    TextToLatentRFDiT,
)
from core.rf import (
    rf_interpolate,
    rf_velocity_target,
    sample_logit_normal_t,
    sample_stratified_logit_normal_t,
)
from core.speaker_inversion import (
    SPEAKER_EMBEDDING_KEY,
    SPEAKER_INVERSION_SAFETENSORS_SUFFIX,
    load_speaker_inversion_payload,
    save_speaker_inversion_checkpoint,
)
from core.tokenizer import PretrainedTextTokenizer
from train import (
    DURATION_CONDITION_GROUP_TOTAL_SIZE,
    compute_rf_loss,
    duration_condition_group_log_suffix,
    duration_condition_group_metrics,
    duration_condition_group_totals,
    duration_condition_group_wandb_metrics,
    iter_device_batches,
    training_batch_keys,
)
from train.dataset import LatentTextDataset, TTSCollator
from train.ema import ModelEMA
from train.optim import build_optimizer, build_scheduler, current_lr
from train.progress import TrainProgress
from train.sampler import LengthGroupedBatchSampler

WANDB_MODES = {"online", "offline", "disabled"}
TRAIN_MODES = {"rf", "duration_only"}
CHECKPOINT_STEP_RE = re.compile(
    rf"^checkpoint_(\d+)(?:\.pt|{re.escape(SPEAKER_INVERSION_SAFETENSORS_SUFFIX)})?$"
)
CHECKPOINT_BEST_VAL_LOSS_RE = re.compile(
    rf"^checkpoint_best_val_loss_(\d+)_(-?\d+(?:\.\d+)?)"
    rf"(?:\.pt|{re.escape(SPEAKER_INVERSION_SAFETENSORS_SUFFIX)})?$"
)
SAFETENSORS_CONFIG_META_KEY = "config_json"
SAFETENSORS_INFERENCE_CONFIG_KEYS = {"max_text_len", "max_caption_len", "fixed_target_latent_steps"}
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """Write via a temp file so a crash mid-save never corrupts the newest checkpoint."""
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    *,
    base_init: dict | None = None,
    epoch: int = 0,
    ema: ModelEMA | None = None,
) -> None:
    path = Path(path)
    if train_cfg.speaker_inversion_enabled:
        save_speaker_inversion_checkpoint(path, model=model)
        return

    if train_config_uses_lora(train_cfg):
        if path.exists():
            _safe_unlink(path)
        path.mkdir(parents=True, exist_ok=True)
        if not hasattr(model, "save_pretrained"):
            raise RuntimeError(
                "LoRA checkpoint saving requires a PEFT model with save_pretrained()."
            )
        model.save_pretrained(path)
        dump_configs(path / "config.json", model_cfg, train_cfg)
        (path / LORA_METADATA_NAME).write_text(
            json.dumps({"base_init": base_init}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _atomic_torch_save(
            {
                "step": step,
                "epoch": int(epoch),
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "model_config": asdict(model_cfg),
                "train_config": asdict(train_cfg),
                "base_init": base_init,
            },
            path / LORA_TRAINER_STATE_NAME,
        )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
    }
    if ema is not None:
        payload["ema_model"] = ema.state_dict()
    _atomic_torch_save(payload, path)


def _safe_unlink(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except FileNotFoundError:
        return


def list_periodic_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint_*"):
        match = CHECKPOINT_STEP_RE.match(path.name)
        if match is None:
            continue
        checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0], reverse=True)
    return checkpoints


def enforce_periodic_checkpoint_limit(output_dir: Path, keep_count: int) -> None:
    if keep_count <= 0:
        return
    checkpoints = list_periodic_checkpoints(output_dir)
    for _, stale_path in checkpoints[keep_count:]:
        _safe_unlink(stale_path)


def list_best_val_loss_checkpoints(output_dir: Path) -> list[tuple[float, int, Path]]:
    checkpoints: list[tuple[float, int, Path]] = []
    for path in output_dir.glob("checkpoint_best_val_loss_*"):
        match = CHECKPOINT_BEST_VAL_LOSS_RE.match(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        score = float(match.group(2))
        checkpoints.append((score, step, path))
    checkpoints.sort(key=lambda item: (item[0], item[1]))
    return checkpoints


def prune_best_val_loss_checkpoints(
    checkpoints: list[tuple[float, int, Path]],
    keep_best_n: int,
) -> list[tuple[float, int, Path]]:
    if keep_best_n <= 0:
        return checkpoints
    checkpoints = sorted(checkpoints, key=lambda item: (item[0], item[1]))
    while len(checkpoints) > keep_best_n:
        _, _, stale_path = checkpoints.pop()
        _safe_unlink(stale_path)
    return checkpoints


def maybe_save_best_val_loss_checkpoint(
    *,
    output_dir: Path,
    checkpoints: list[tuple[float, int, Path]],
    keep_best_n: int,
    val_loss: float,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    base_init: dict | None,
    epoch: int = 0,
    ema: ModelEMA | None = None,
) -> tuple[list[tuple[float, int, Path]], Path | None]:
    if keep_best_n <= 0:
        return checkpoints, None

    checkpoints = sorted(checkpoints, key=lambda item: (item[0], item[1]))
    if len(checkpoints) >= keep_best_n:
        worst_score = checkpoints[-1][0]
        if val_loss >= worst_score:
            return checkpoints, None

    kept: list[tuple[float, int, Path]] = []
    for score, saved_step, path in checkpoints:
        if saved_step == step:
            _safe_unlink(path)
            continue
        kept.append((score, saved_step, path))
    checkpoints = kept

    path = _best_checkpoint_path(output_dir, step=step, val_loss=val_loss, train_cfg=train_cfg)
    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        base_init=base_init,
        epoch=epoch,
        ema=ema,
    )
    checkpoints.append((float(val_loss), int(step), path))
    checkpoints = prune_best_val_loss_checkpoints(checkpoints, keep_best_n)
    return checkpoints, path


def cli_provided(argv: list[str], flag: str) -> bool:
    return any(x == flag or x.startswith(flag + "=") for x in argv)


def _periodic_checkpoint_path(output_dir: Path, step: int, train_cfg: TrainConfig) -> Path:
    if train_cfg.speaker_inversion_enabled:
        return output_dir / f"checkpoint_{step:07d}{SPEAKER_INVERSION_SAFETENSORS_SUFFIX}"
    if train_config_uses_lora(train_cfg):
        return output_dir / f"checkpoint_{step:07d}"
    return output_dir / f"checkpoint_{step:07d}.pt"


def _best_checkpoint_path(
    output_dir: Path, *, step: int, val_loss: float, train_cfg: TrainConfig
) -> Path:
    if train_cfg.speaker_inversion_enabled:
        return (
            output_dir / f"checkpoint_best_val_loss_{step:07d}_{val_loss:.6f}"
            f"{SPEAKER_INVERSION_SAFETENSORS_SUFFIX}"
        )
    if train_config_uses_lora(train_cfg):
        return output_dir / f"checkpoint_best_val_loss_{step:07d}_{val_loss:.6f}"
    return output_dir / f"checkpoint_best_val_loss_{step:07d}_{val_loss:.6f}.pt"


def _final_checkpoint_path(output_dir: Path, train_cfg: TrainConfig) -> Path:
    if train_cfg.speaker_inversion_enabled:
        return output_dir / f"checkpoint_final{SPEAKER_INVERSION_SAFETENSORS_SUFFIX}"
    if train_config_uses_lora(train_cfg):
        return output_dir / "checkpoint_final"
    return output_dir / "checkpoint_final.pt"


def run_epoch_test_inference(
    *,
    checkpoint_path: Path,
    manifest_path: str | Path,
    output_dir: Path,
    epoch: int,
    step: int,
    train_cfg: TrainConfig,
) -> tuple[bool, Path]:
    sample_dir = output_dir / "samples" / f"epoch_{epoch:06d}_step_{step:07d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_path = sample_dir / "inference.log"
    command = [
        sys.executable,
        "-m",
        "train.cli.sample_training_checkpoint",
        "--checkpoint",
        str(checkpoint_path.resolve()),
        "--manifest",
        str(Path(manifest_path).resolve()),
        "--output-dir",
        str(sample_dir.resolve()),
        "--num-steps",
        str(train_cfg.sample_num_steps),
        "--seed",
        str(train_cfg.sample_seed),
        "--reference-count",
        str(train_cfg.sample_reference_count),
        # Comparison samples should mirror the exported inference model, which
        # is converted from EMA weights whenever EMA is enabled.
        "--use-ema" if train_cfg.ema_enabled else "--no-use-ema",
    ]
    if train_cfg.sample_seconds is not None:
        command.extend(("--seconds", str(train_cfg.sample_seconds)))
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = str(train_cfg.sample_cuda_visible_devices)
    child_env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=child_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode == 0, sample_dir


def build_condition_tokenizer(
    *,
    repo_id: str,
    add_bos: bool,
    vocab_size: int,
    local_files_only: bool = False,
) -> PretrainedTextTokenizer:
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=repo_id,
        add_bos=bool(add_bos),
        local_files_only=local_files_only,
    )
    if tokenizer.vocab_size != vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size mismatch: expected {vocab_size} but tokenizer "
            f"({repo_id}) vocab_size={tokenizer.vocab_size}."
        )
    return tokenizer


def build_text_tokenizer(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> PretrainedTextTokenizer:
    return build_condition_tokenizer(
        repo_id=model_cfg.text_tokenizer_repo,
        add_bos=bool(model_cfg.text_add_bos),
        vocab_size=int(model_cfg.text_vocab_size),
        local_files_only=local_files_only,
    )


def build_caption_tokenizer(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> PretrainedTextTokenizer:
    return build_condition_tokenizer(
        repo_id=model_cfg.caption_tokenizer_repo_resolved,
        add_bos=model_cfg.caption_add_bos_resolved,
        vocab_size=model_cfg.caption_vocab_size_resolved,
        local_files_only=local_files_only,
    )


def validate_pretrained_backbone_dim(
    *,
    repo_id: str,
    expected_dim: int,
    local_files_only: bool = False,
) -> int:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for pretrained text embedding initialization. "
            "Install with `pip install transformers sentencepiece`."
        ) from exc

    text_cfg = AutoConfig.from_pretrained(
        repo_id,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    hidden_size = getattr(text_cfg, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(f"Could not read hidden_size from pretrained config: {repo_id}")
    hidden_size = int(hidden_size)
    if hidden_size != expected_dim:
        raise ValueError(
            f"Condition encoder dim mismatch: expected {expected_dim} but pretrained hidden_size={hidden_size} "
            f"for repo {repo_id}."
        )
    return hidden_size


def validate_text_backbone_dim(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> int:
    return validate_pretrained_backbone_dim(
        repo_id=model_cfg.text_tokenizer_repo,
        expected_dim=int(model_cfg.text_dim),
        local_files_only=local_files_only,
    )


def validate_caption_backbone_dim(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> int:
    return validate_pretrained_backbone_dim(
        repo_id=model_cfg.caption_tokenizer_repo_resolved,
        expected_dim=model_cfg.caption_dim_resolved,
        local_files_only=local_files_only,
    )


def initialize_embedding_from_pretrained(
    embedding: torch.nn.Embedding,
    *,
    repo_id: str,
    local_files_only: bool = False,
) -> None:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for pretrained text embedding initialization. "
            "Install with `pip install transformers sentencepiece`."
        ) from exc

    text_backbone = AutoModel.from_pretrained(
        repo_id,
        trust_remote_code=False,
        dtype=torch.float,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    pretrained_embedding = text_backbone.get_input_embeddings()
    if pretrained_embedding is None:
        raise ValueError(f"Pretrained model has no input embeddings: {repo_id}")
    src_weight = pretrained_embedding.weight.detach().to(device="cpu", dtype=torch.float)
    tgt_weight = embedding.weight
    src_vocab, src_dim = tuple(src_weight.shape)
    tgt_vocab, tgt_dim = tuple(tgt_weight.shape)
    if src_dim != tgt_dim:
        raise ValueError(
            f"Embedding hidden size mismatch: pretrained={src_dim} model={tgt_dim} for repo={repo_id}."
        )

    copy_rows = min(src_vocab, tgt_vocab)
    with torch.no_grad():
        tgt_weight[:copy_rows].copy_(
            src_weight[:copy_rows].to(device=tgt_weight.device, dtype=tgt_weight.dtype)
        )

    del text_backbone


def initialize_text_embedding_from_pretrained(
    model: TextToLatentRFDiT,
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> None:
    initialize_embedding_from_pretrained(
        model.text_encoder.text_embedding,
        repo_id=model_cfg.text_tokenizer_repo,
        local_files_only=local_files_only,
    )


def initialize_caption_embedding_from_pretrained(
    model: TextToLatentRFDiT,
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> None:
    if model.caption_encoder is None:
        raise RuntimeError(
            "Caption embedding initialization requested but caption encoder is absent."
        )
    initialize_embedding_from_pretrained(
        model.caption_encoder.text_embedding,
        repo_id=model_cfg.caption_tokenizer_repo_resolved,
        local_files_only=local_files_only,
    )


def _load_model_state_from_checkpoint(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict | None, dict | None]:
    if path.suffix.lower() == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file as load_safetensors_file

        checkpoint_model_cfg = None
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        config_json = metadata.get(SAFETENSORS_CONFIG_META_KEY)
        if config_json:
            parsed = json.loads(config_json)
            if isinstance(parsed, dict):
                checkpoint_model_cfg = {
                    key: value
                    for key, value in parsed.items()
                    if key not in SAFETENSORS_INFERENCE_CONFIG_KEYS
                }
        return load_safetensors_file(str(path), device="cpu"), checkpoint_model_cfg, None

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary, got {type(payload)!r}.")

    raw_model = payload.get("model")
    if raw_model is None and all(isinstance(v, torch.Tensor) for v in payload.values()):
        raw_model = payload
    if not isinstance(raw_model, dict):
        raise ValueError(f"Checkpoint does not contain a model state dictionary: {path}")

    checkpoint_model_cfg = payload.get("model_config")
    if checkpoint_model_cfg is not None and not isinstance(checkpoint_model_cfg, dict):
        raise ValueError(f"Checkpoint model_config must be a dictionary when present: {path}")
    checkpoint_train_cfg = payload.get("train_config")
    if checkpoint_train_cfg is not None and not isinstance(checkpoint_train_cfg, dict):
        raise ValueError(f"Checkpoint train_config must be a dictionary when present: {path}")
    return raw_model, checkpoint_model_cfg, checkpoint_train_cfg


def _check_model_config_compatibility(
    checkpoint_path: Path,
    checkpoint_model_cfg: dict | None,
    current_model_cfg: ModelConfig,
    *,
    require_caption_match: bool,
) -> None:
    if checkpoint_model_cfg is None:
        return

    checkpoint_cfg = merge_dataclass_overrides(
        ModelConfig(),
        checkpoint_model_cfg,
        section="checkpoint model_config",
    )

    comparisons: list[tuple[str, object, object]] = [
        ("latent_dim", checkpoint_cfg.latent_dim, current_model_cfg.latent_dim),
        (
            "latent_patch_size",
            checkpoint_cfg.latent_patch_size,
            current_model_cfg.latent_patch_size,
        ),
        ("model_dim", checkpoint_cfg.model_dim, current_model_cfg.model_dim),
        ("num_layers", checkpoint_cfg.num_layers, current_model_cfg.num_layers),
        ("num_heads", checkpoint_cfg.num_heads, current_model_cfg.num_heads),
        ("mlp_ratio", checkpoint_cfg.mlp_ratio, current_model_cfg.mlp_ratio),
        ("text_vocab_size", checkpoint_cfg.text_vocab_size, current_model_cfg.text_vocab_size),
        ("text_dim", checkpoint_cfg.text_dim, current_model_cfg.text_dim),
        ("text_layers", checkpoint_cfg.text_layers, current_model_cfg.text_layers),
        ("text_heads", checkpoint_cfg.text_heads, current_model_cfg.text_heads),
        (
            "text_mlp_ratio",
            checkpoint_cfg.text_mlp_ratio_resolved,
            current_model_cfg.text_mlp_ratio_resolved,
        ),
        ("adaln_rank", checkpoint_cfg.adaln_rank, current_model_cfg.adaln_rank),
    ]
    if (
        checkpoint_cfg.use_speaker_condition_resolved
        and current_model_cfg.use_speaker_condition_resolved
    ):
        comparisons.extend(
            [
                ("speaker_dim", checkpoint_cfg.speaker_dim, current_model_cfg.speaker_dim),
                ("speaker_layers", checkpoint_cfg.speaker_layers, current_model_cfg.speaker_layers),
                ("speaker_heads", checkpoint_cfg.speaker_heads, current_model_cfg.speaker_heads),
                (
                    "speaker_mlp_ratio",
                    checkpoint_cfg.speaker_mlp_ratio_resolved,
                    current_model_cfg.speaker_mlp_ratio_resolved,
                ),
                (
                    "speaker_patch_size",
                    checkpoint_cfg.speaker_patch_size,
                    current_model_cfg.speaker_patch_size,
                ),
            ]
        )
    if require_caption_match:
        comparisons.extend(
            [
                (
                    "use_caption_condition",
                    checkpoint_cfg.use_caption_condition,
                    current_model_cfg.use_caption_condition,
                ),
                (
                    "use_speaker_condition",
                    checkpoint_cfg.use_speaker_condition_resolved,
                    current_model_cfg.use_speaker_condition_resolved,
                ),
                (
                    "caption_vocab_size",
                    checkpoint_cfg.caption_vocab_size_resolved,
                    current_model_cfg.caption_vocab_size_resolved,
                ),
                (
                    "caption_tokenizer_repo",
                    checkpoint_cfg.caption_tokenizer_repo_resolved,
                    current_model_cfg.caption_tokenizer_repo_resolved,
                ),
                (
                    "caption_add_bos",
                    checkpoint_cfg.caption_add_bos_resolved,
                    current_model_cfg.caption_add_bos_resolved,
                ),
                (
                    "caption_dim",
                    checkpoint_cfg.caption_dim_resolved,
                    current_model_cfg.caption_dim_resolved,
                ),
                (
                    "caption_layers",
                    checkpoint_cfg.caption_layers_resolved,
                    current_model_cfg.caption_layers_resolved,
                ),
                (
                    "caption_heads",
                    checkpoint_cfg.caption_heads_resolved,
                    current_model_cfg.caption_heads_resolved,
                ),
                (
                    "caption_mlp_ratio",
                    checkpoint_cfg.caption_mlp_ratio_resolved,
                    current_model_cfg.caption_mlp_ratio_resolved,
                ),
            ]
        )

    for key, checkpoint_value, current_value in comparisons:
        if checkpoint_value != current_value:
            raise ValueError(
                f"Checkpoint/config mismatch for '{key}': checkpoint={checkpoint_value} "
                f"current={current_value} ({checkpoint_path})"
            )


def checkpoint_uses_caption_condition(
    checkpoint_model_cfg: dict | None,
    state_dict: dict[str, torch.Tensor],
) -> bool:
    if checkpoint_model_cfg is not None:
        checkpoint_cfg = merge_dataclass_overrides(
            ModelConfig(),
            checkpoint_model_cfg,
            section="checkpoint model_config",
        )
        if checkpoint_cfg.use_caption_condition:
            return True
    return any(
        key.startswith("caption_encoder.")
        or key.startswith("caption_norm.")
        or ".wk_caption." in key
        or ".wv_caption." in key
        for key in state_dict
    )


def checkpoint_uses_duration_predictor(
    checkpoint_model_cfg: dict | None,
    state_dict: dict[str, torch.Tensor],
) -> bool:
    if checkpoint_model_cfg is not None:
        checkpoint_cfg = merge_dataclass_overrides(
            ModelConfig(),
            checkpoint_model_cfg,
            section="checkpoint model_config",
        )
        if checkpoint_cfg.use_duration_predictor:
            return True
    return any(key.startswith("duration_predictor.") for key in state_dict)


def load_model_state_partially(
    model: TextToLatentRFDiT,
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str], list[str]]:
    model_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    skipped_extra: list[str] = []

    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            skipped_extra.append(key)
            continue
        if tuple(target.shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        filtered_state[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    if unexpected_keys:
        skipped_extra.extend(unexpected_keys)
    return missing_keys, skipped_shape, skipped_extra


def _canonical_parameter_key(key: str) -> str:
    prefix = "base_model.model."
    if key.startswith(prefix):
        return key[len(prefix) :]
    return key


def is_caption_only_parameter(key: str) -> bool:
    key = _canonical_parameter_key(key)
    return (
        key.startswith("caption_encoder.")
        or key.startswith("caption_norm.")
        or ".wk_caption." in key
        or ".wv_caption." in key
    )


def is_speaker_only_parameter(key: str) -> bool:
    key = _canonical_parameter_key(key)
    return (
        key.startswith("speaker_encoder.")
        or key.startswith("speaker_norm.")
        or ".wk_speaker." in key
        or ".wv_speaker." in key
    )


def is_duration_only_parameter(key: str) -> bool:
    key = _canonical_parameter_key(key)
    return key.startswith("duration_predictor.")


def clear_non_caption_grads(model: TextToLatentRFDiT) -> tuple[int, int]:
    caption_grad_params = 0
    cleared_grad_params = 0
    for key, param in model.named_parameters():
        if is_caption_only_parameter(key):
            if param.grad is not None:
                caption_grad_params += 1
            continue
        if param.grad is not None:
            cleared_grad_params += 1
        param.grad = None
    return caption_grad_params, cleared_grad_params


def freeze_for_duration_only(model: torch.nn.Module) -> tuple[int, int]:
    trainable_params = 0
    frozen_params = 0
    for key, param in model.named_parameters():
        if is_duration_only_parameter(key):
            param.requires_grad_(True)
            trainable_params += param.numel()
        else:
            param.requires_grad_(False)
            frozen_params += param.numel()
    return trainable_params, frozen_params


def freeze_for_speaker_inversion(model: torch.nn.Module) -> tuple[int, int]:
    trainable_params = 0
    frozen_params = 0
    for key, param in model.named_parameters():
        if _canonical_parameter_key(key).startswith("speaker_inversion."):
            param.requires_grad_(True)
            trainable_params += param.numel()
        else:
            param.requires_grad_(False)
            frozen_params += param.numel()
    return trainable_params, frozen_params


def validate_checkpoint_upgrade_partial_load(
    checkpoint_path: Path,
    missing_keys: list[str],
    skipped_shape: list[str],
    skipped_extra: list[str],
    *,
    allow_caption_missing: bool,
    allow_duration_missing: bool,
    allow_duration_extra: bool,
    allow_speaker_extra: bool,
) -> None:
    if skipped_shape:
        raise ValueError(
            "Checkpoint/config shape mismatch while upgrading checkpoint config: "
            f"{checkpoint_path} skipped_shape={skipped_shape[:8]}"
        )

    unexpected_extra = skipped_extra
    if allow_speaker_extra:
        unexpected_extra = [key for key in unexpected_extra if not is_speaker_only_parameter(key)]
    if allow_duration_extra:
        unexpected_extra = [key for key in unexpected_extra if not is_duration_only_parameter(key)]
    if unexpected_extra:
        raise ValueError(
            "Unexpected checkpoint keys while upgrading checkpoint config: "
            f"{checkpoint_path} skipped_extra={unexpected_extra[:8]}"
        )

    def _allowed_missing(key: str) -> bool:
        return (allow_caption_missing and is_caption_only_parameter(key)) or (
            allow_duration_missing and is_duration_only_parameter(key)
        )

    unexpected_missing = [key for key in missing_keys if not _allowed_missing(key)]
    if unexpected_missing:
        raise ValueError(
            "Partial init from checkpoint left unexpected parameters missing: "
            f"{checkpoint_path} missing={unexpected_missing[:8]}"
        )


def _load_checkpoint_payload(path: str | Path, *, map_location) -> dict:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        state_path = checkpoint_path / LORA_TRAINER_STATE_NAME
        payload = torch.load(state_path, map_location=map_location, weights_only=True)
    else:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary, got {type(payload)!r}.")
    return payload


def _normalize_checkpoint_path(path: str | Path) -> Path:
    return Path(os.path.abspath(str(Path(path).expanduser())))


def _lora_field_cli_explicit(field: str, args: argparse.Namespace, raw_argv: list[str]) -> bool:
    if field == "lora_enabled":
        return args.lora_enabled is not None
    flag = "--" + field.replace("_", "-")
    return cli_provided(raw_argv, flag)


def _restore_resume_lora_config(
    train_cfg: TrainConfig,
    *,
    resume_train_cfg: dict | None,
    args: argparse.Namespace,
    raw_argv: list[str],
    exp_cfg: dict,
) -> TrainConfig:
    if not isinstance(resume_train_cfg, dict):
        return train_cfg

    train_overrides = exp_cfg.get("train", {})
    if not isinstance(train_overrides, dict):
        train_overrides = {}

    updates: dict[str, object] = {}
    for field in LORA_TRAIN_CONFIG_FIELDS:
        if field not in resume_train_cfg:
            continue
        explicit = _lora_field_cli_explicit(field, args, raw_argv) or field in train_overrides
        current_value = getattr(train_cfg, field)
        resume_value = resume_train_cfg[field]
        if explicit:
            if current_value != resume_value:
                raise ValueError(
                    f"Resume checkpoint expects train.{field}={resume_value!r}, "
                    f"but current config requests {current_value!r}."
                )
            continue
        updates[field] = resume_value

    if updates:
        train_cfg = replace(train_cfg, **updates)
    return train_cfg


def _initialize_base_model_from_pretrained_embeddings(
    raw_model: torch.nn.Module,
    *,
    model_cfg: ModelConfig,
    distributed: bool,
    is_main_process: bool,
) -> None:
    if distributed:
        if is_main_process:
            print(
                f"Initializing text embedding from pretrained model: {model_cfg.text_tokenizer_repo}"
            )
            initialize_text_embedding_from_pretrained(
                raw_model,
                model_cfg,
                local_files_only=False,
            )
            if model_cfg.use_caption_condition:
                print(
                    "Initializing caption embedding from pretrained model: "
                    f"{model_cfg.caption_tokenizer_repo_resolved}"
                )
                initialize_caption_embedding_from_pretrained(
                    raw_model,
                    model_cfg,
                    local_files_only=False,
                )
        dist.barrier()
        if not is_main_process:
            initialize_text_embedding_from_pretrained(
                raw_model,
                model_cfg,
                local_files_only=True,
            )
            if model_cfg.use_caption_condition:
                initialize_caption_embedding_from_pretrained(
                    raw_model,
                    model_cfg,
                    local_files_only=True,
                )
        dist.barrier()
        return

    if is_main_process:
        print(f"Initializing text embedding from pretrained model: {model_cfg.text_tokenizer_repo}")
    initialize_text_embedding_from_pretrained(
        raw_model,
        model_cfg,
        local_files_only=False,
    )
    if model_cfg.use_caption_condition:
        if is_main_process:
            print(
                "Initializing caption embedding from pretrained model: "
                f"{model_cfg.caption_tokenizer_repo_resolved}"
            )
        initialize_caption_embedding_from_pretrained(
            raw_model,
            model_cfg,
            local_files_only=False,
        )


def _apply_base_initialization(
    raw_model: torch.nn.Module,
    *,
    model_cfg: ModelConfig,
    base_init: dict | None,
    distributed: bool,
    is_main_process: bool,
) -> None:
    mode = None if base_init is None else base_init.get("mode")
    if mode is None:
        _initialize_base_model_from_pretrained_embeddings(
            raw_model,
            model_cfg=model_cfg,
            distributed=distributed,
            is_main_process=is_main_process,
        )
        return

    if mode == "checkpoint":
        checkpoint_path = base_init.get("checkpoint_path")
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise ValueError("LoRA checkpoint metadata is missing base_init.checkpoint_path.")
        init_path = _normalize_checkpoint_path(checkpoint_path)
        init_state, init_model_cfg, _ = _load_model_state_from_checkpoint(init_path)
        checkpoint_has_caption = checkpoint_uses_caption_condition(init_model_cfg, init_state)
        current_has_caption = bool(model_cfg.use_caption_condition)
        checkpoint_has_duration = checkpoint_uses_duration_predictor(init_model_cfg, init_state)
        current_has_duration = bool(model_cfg.use_duration_predictor)
        drop_duration = checkpoint_has_duration and not current_has_duration
        if checkpoint_has_caption and not current_has_caption:
            raise ValueError(
                "Caption-conditioned checkpoint cannot initialize a caption-free config. "
                "Use a caption-enabled config for this checkpoint."
            )
        if drop_duration and not (current_has_caption and not checkpoint_has_caption):
            raise ValueError(
                "Duration-predictor checkpoint cannot initialize a duration-free config. "
                "Use a duration-enabled config for this checkpoint, or initialize a "
                "caption-enabled phase-1 VoiceDesign model from a caption-free base checkpoint."
            )

        require_caption_match = checkpoint_has_caption and current_has_caption
        _check_model_config_compatibility(
            init_path,
            init_model_cfg,
            model_cfg,
            require_caption_match=require_caption_match,
        )

        missing_keys: list[str] = []
        initialized_caption_embedding = False
        upgrade_caption = current_has_caption and not checkpoint_has_caption
        upgrade_duration = current_has_duration and not checkpoint_has_duration
        if upgrade_caption or upgrade_duration or drop_duration:
            missing_keys, skipped_shape, skipped_extra = load_model_state_partially(
                raw_model,
                init_state,
            )
            validate_checkpoint_upgrade_partial_load(
                init_path,
                missing_keys,
                skipped_shape,
                skipped_extra,
                allow_caption_missing=upgrade_caption,
                allow_duration_missing=upgrade_duration,
                allow_duration_extra=drop_duration,
                allow_speaker_extra=(
                    upgrade_caption and not model_cfg.use_speaker_condition_resolved
                ),
            )
        else:
            raw_model.load_state_dict(init_state, strict=True)

        if upgrade_caption:
            if distributed:
                if is_main_process:
                    print(
                        "Initializing caption embedding from pretrained model after caption-free checkpoint load: "
                        f"{model_cfg.caption_tokenizer_repo_resolved}"
                    )
                    initialize_caption_embedding_from_pretrained(
                        raw_model,
                        model_cfg,
                        local_files_only=False,
                    )
                dist.barrier()
                if not is_main_process:
                    initialize_caption_embedding_from_pretrained(
                        raw_model,
                        model_cfg,
                        local_files_only=True,
                    )
                dist.barrier()
            else:
                if is_main_process:
                    print(
                        "Initializing caption embedding from pretrained model after caption-free checkpoint load: "
                        f"{model_cfg.caption_tokenizer_repo_resolved}"
                    )
                initialize_caption_embedding_from_pretrained(
                    raw_model,
                    model_cfg,
                    local_files_only=False,
                )
            initialized_caption_embedding = True

        if is_main_process:
            print(f"Initialized model weights from: {init_path}")
            if missing_keys:
                print(f"Partial load missing keys: {len(missing_keys)}")
            if current_has_duration and not checkpoint_has_duration:
                print("Duration predictor was randomly initialized.")
            if initialized_caption_embedding:
                print("Caption embedding was initialized from its pretrained tokenizer backbone.")
        return

    raise ValueError(f"Unsupported base_init mode: {mode!r}")


def resolve_dist_env() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def setup_distributed(device_arg: str) -> tuple[int, int, int, bool, torch.device]:
    rank, world_size, local_rank = resolve_dist_env()
    distributed = world_size > 1
    if distributed:
        if not str(device_arg).startswith("cuda"):
            raise ValueError(
                f"WORLD_SIZE={world_size} detected, but --device={device_arg!r}. "
                "DDP multi-GPU training requires --device cuda."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("WORLD_SIZE>1 detected, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        backend = "nccl" if dist.is_nccl_available() else "gloo"
        init_method = os.environ.get("IRODORI_DIST_INIT_METHOD", "env://")
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device_arg)
    return rank, world_size, local_rank, distributed, device


def reduce_mean(value: torch.Tensor, world_size: int, distributed: bool) -> torch.Tensor:
    reduced = value.detach().clone()
    if not distributed:
        return reduced
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= float(world_size)
    return reduced


def reduce_sum(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    reduced = value.detach().clone()
    if distributed:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def split_train_valid_indices(
    *,
    num_samples: int,
    valid_ratio: float,
    seed: int,
    groups: list[str | None] | None = None,
) -> tuple[list[int], list[int]]:
    """
    Split sample indices into train/valid sets.

    When ``groups`` is provided (e.g. speaker IDs), whole groups are assigned to
    the validation set so the same voice never appears on both sides. Samples
    with a ``None`` group each count as their own singleton group.
    """
    if valid_ratio <= 0.0:
        return list(range(num_samples)), []
    if num_samples < 2:
        raise ValueError(
            f"Validation split requires at least 2 samples in manifest, got {num_samples}."
        )

    valid_count = int(num_samples * valid_ratio)
    valid_count = max(1, valid_count)
    if valid_count >= num_samples:
        valid_count = num_samples - 1

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    if groups is None:
        perm = torch.randperm(num_samples, generator=generator).tolist()
        valid_indices = sorted(perm[:valid_count])
        train_indices = sorted(perm[valid_count:])
    else:
        if len(groups) != num_samples:
            raise ValueError(
                f"groups length {len(groups)} does not match num_samples {num_samples}."
            )
        group_to_indices: dict[str, list[int]] = {}
        for index, group in enumerate(groups):
            key = f"__ungrouped_{index}" if group is None else str(group)
            group_to_indices.setdefault(key, []).append(index)
        group_keys = sorted(group_to_indices)
        if len(group_keys) < 2:
            raise ValueError(
                "Grouped validation split requires at least 2 distinct groups "
                f"(got {len(group_keys)}). Use valid_split_by='sample' instead."
            )
        order = torch.randperm(len(group_keys), generator=generator).tolist()
        valid_indices = []
        for position in order:
            candidate = group_to_indices[group_keys[position]]
            # Never consume every group: at least one must remain for training.
            if len(valid_indices) + len(candidate) > num_samples - 1:
                continue
            valid_indices.extend(candidate)
            if len(valid_indices) >= valid_count:
                break
        valid_index_set = set(valid_indices)
        valid_indices = sorted(valid_index_set)
        train_indices = [i for i in range(num_samples) if i not in valid_index_set]

    if not train_indices or not valid_indices:
        raise ValueError(
            "Failed to create non-empty train/valid split. "
            f"num_samples={num_samples} valid_ratio={valid_ratio}"
        )
    return train_indices, valid_indices


def run_validation(
    *,
    model,
    loader: DataLoader,
    train_cfg: TrainConfig,
    device: torch.device,
    use_bf16: bool,
    distributed: bool,
) -> dict[str, float]:
    was_training = model.training
    model_cfg = model.module.cfg if isinstance(model, DDP) else model.cfg
    duration_only = train_cfg.train_mode == "duration_only"
    model.eval()
    # Fixed noise/timestep RNG keeps validation losses comparable across
    # evaluations instead of adding sampling noise to best-checkpoint selection.
    valid_generator = torch.Generator(device=device)
    valid_generator.manual_seed(int(train_cfg.seed) + 0x5EED)
    totals = torch.zeros(
        6 + DURATION_CONDITION_GROUP_TOTAL_SIZE,
        device=device,
        dtype=torch.float64,
    )
    batch_keys = training_batch_keys(model_cfg, duration_only=duration_only)

    with torch.no_grad():
        for batch in iter_device_batches(
            loader,
            keys=batch_keys,
            device=device,
            prefetch=train_cfg.dataloader_prefetch_to_device,
        ):
            text_ids = batch["text_ids"]
            text_mask = batch["text_mask"]
            caption_ids = None
            caption_mask = None
            has_caption = None
            if model_cfg.use_caption_condition:
                caption_ids = batch["caption_ids"]
                caption_mask = batch["caption_mask"]
                has_caption = batch["has_caption"]
            num_frames = batch["num_frames"]
            duration_features = batch.get("duration_features")
            ref_latent = None
            ref_mask = None
            if model_cfg.use_speaker_condition_resolved:
                ref_latent = batch["ref_latent_patched"]
                ref_mask = batch["ref_latent_mask_patched"]
                has_speaker = batch["has_speaker"]
            else:
                has_speaker = None

            bsz = text_ids.shape[0]
            x0 = None
            x_mask = None
            x_mask_valid = None
            x_t = None
            t = None
            v_target = None
            if not duration_only:
                x0 = batch["latent_patched"]
                x_mask = batch["latent_mask_patched"]
                x_mask_valid = batch["latent_mask_valid_patched"]
                if train_cfg.timestep_stratified:
                    t = sample_stratified_logit_normal_t(
                        batch_size=bsz,
                        device=device,
                        mean=train_cfg.timestep_logit_mean,
                        std=train_cfg.timestep_logit_std,
                        t_min=train_cfg.timestep_min,
                        t_max=train_cfg.timestep_max,
                        generator=valid_generator,
                    )
                else:
                    t = sample_logit_normal_t(
                        batch_size=bsz,
                        device=device,
                        mean=train_cfg.timestep_logit_mean,
                        std=train_cfg.timestep_logit_std,
                        t_min=train_cfg.timestep_min,
                        t_max=train_cfg.timestep_max,
                        generator=valid_generator,
                    )
                noise = torch.randn(
                    x0.shape,
                    device=x0.device,
                    dtype=x0.dtype,
                    generator=valid_generator,
                )
                x_t = rf_interpolate(x0, noise, t)
                v_target = rf_velocity_target(x0, noise)

            if model_cfg.use_speaker_condition_resolved:
                if train_cfg.speaker_inversion_enabled:
                    # Speaker Inversion learns one embedding for this run, so validation
                    # should match training and treat every sample as speaker-conditioned.
                    use_speaker = torch.ones((bsz,), device=device, dtype=torch.bool)
                else:
                    use_speaker = has_speaker
                speaker_condition_dropout = ~use_speaker
                duration_has_speaker = use_speaker
                if model_cfg.use_duration_predictor:
                    if duration_features is None:
                        raise RuntimeError("Duration feature batch tensor is missing.")
                    duration_features = set_duration_has_speaker_feature(
                        duration_features,
                        duration_has_speaker,
                    )
            else:
                speaker_condition_dropout = None
                duration_has_speaker = None
            duration_has_caption = has_caption if model_cfg.use_caption_condition else None

            with (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            ):
                if duration_only:
                    duration_pred = model(
                        x_t=None,
                        t=None,
                        text_input_ids=text_ids,
                        text_mask=text_mask,
                        ref_latent=ref_latent,
                        ref_mask=ref_mask,
                        caption_input_ids=caption_ids,
                        caption_mask=caption_mask,
                        latent_mask=None,
                        duration_features=duration_features,
                        duration_has_speaker=duration_has_speaker,
                        duration_has_caption=duration_has_caption,
                        duration_only=True,
                    )
                    v_pred = None
                elif model_cfg.use_duration_predictor:
                    v_pred, duration_pred = model(
                        x_t=x_t,
                        t=t,
                        text_input_ids=text_ids,
                        text_mask=text_mask,
                        ref_latent=ref_latent,
                        ref_mask=ref_mask,
                        caption_input_ids=caption_ids,
                        caption_mask=caption_mask,
                        latent_mask=x_mask,
                        speaker_condition_dropout=speaker_condition_dropout,
                        duration_features=duration_features,
                        duration_has_speaker=duration_has_speaker,
                        duration_has_caption=duration_has_caption,
                    )
                else:
                    if model_cfg.use_speaker_condition_resolved:
                        ref_mask = ref_mask & use_speaker[:, None]
                        ref_latent = ref_latent * use_speaker[:, None, None].to(ref_latent.dtype)
                    v_pred = model(
                        x_t=x_t,
                        t=t,
                        text_input_ids=text_ids,
                        text_mask=text_mask,
                        ref_latent=ref_latent,
                        ref_mask=ref_mask,
                        caption_input_ids=caption_ids,
                        caption_mask=caption_mask,
                        latent_mask=x_mask,
                    )
                    duration_pred = None

            rf_loss = torch.zeros((), device=device, dtype=torch.float)
            if not duration_only:
                if v_pred is None or v_target is None or x_mask is None or x_mask_valid is None:
                    raise RuntimeError("RF validation tensors are missing.")
                v_pred = v_pred.float()
                rf_loss = compute_rf_loss(
                    pred=v_pred,
                    target=v_target.float(),
                    loss_mask=x_mask,
                    valid_mask=x_mask_valid,
                    mode=train_cfg.rf_loss_mode,
                )
            duration_loss = torch.zeros((), device=device, dtype=torch.float)
            duration_mae_frames = torch.zeros((), device=device, dtype=torch.float)
            if model_cfg.use_duration_predictor:
                if duration_pred is None:
                    raise RuntimeError(
                        "Duration predictor is enabled but duration_pred is missing."
                    )
                duration_target = torch.log1p(num_frames.float())
                duration_loss_per_sample = F.huber_loss(
                    duration_pred.float(),
                    duration_target,
                    delta=float(train_cfg.duration_huber_delta),
                    reduction="none",
                )
                duration_loss = duration_loss_per_sample.mean()
                pred_frames = torch.expm1(duration_pred.float()).clamp_min(0.0)
                duration_mae_frames = (pred_frames - num_frames.float()).abs().mean()
                if duration_only:
                    totals[6:] += duration_condition_group_totals(
                        duration_loss_per_sample=duration_loss_per_sample,
                        pred_frames=pred_frames,
                        target_frames=num_frames.float(),
                        has_speaker=has_speaker,
                        has_caption=has_caption,
                    )
            if duration_only:
                loss = duration_loss
            else:
                loss = rf_loss + (float(train_cfg.duration_loss_weight) * duration_loss)

            weight = float(bsz)
            totals[0] += loss.detach().double() * weight
            totals[1] += rf_loss.detach().double() * weight
            totals[2] += duration_loss.detach().double() * weight
            totals[3] += duration_mae_frames.detach().double() * weight
            totals[4] += num_frames.detach().double().sum()
            totals[5] += weight

    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    denom = max(float(totals[5].item()), 1.0)
    metrics = {
        "loss": float(totals[0].item() / denom),
        "rf_loss": float(totals[1].item() / denom),
        "duration_loss": float(totals[2].item() / denom),
        "duration_mae_frames": float(totals[3].item() / denom),
        "target_frames_mean": float(totals[4].item() / denom),
        "num_samples": float(totals[5].item()),
    }
    if duration_only:
        metrics.update(duration_condition_group_metrics(totals[6:]))
    if was_training:
        model.train()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Irodori-TTS.")
    parser.add_argument("--config", default=None, help="YAML config path (model/train overrides)")
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSONL manifest with text+latent_path (optional speaker_id for reference sampling).",
    )
    parser.add_argument("--output-dir", default="outputs/irodori_tts")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Compute precision for model forward pass. fp32 = full precision; bf16 = CUDA autocast.",
    )
    parser.add_argument(
        "--tf32",
        dest="allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable TF32 matmul/cuDNN kernels on CUDA for speed.",
    )
    parser.add_argument(
        "--compile-model",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable torch.compile for the training model.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable activation checkpointing on diffusion blocks to reduce memory.",
    )
    parser.add_argument(
        "--train-mode",
        choices=sorted(TRAIN_MODES),
        default=None,
        help="Training objective: rf runs DiT/RF training; duration_only trains only the duration predictor.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume full training state from a training checkpoint (.pt or LoRA checkpoint dir).",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Initialize model weights from a checkpoint (.pt or .safetensors) and start a new run "
            "with fresh optimizer / scheduler state."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Number of micro-batches to accumulate before optimizer.step(). "
            "1 disables accumulation."
        ),
    )
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=256,
        help="Maximum token length for text conditioning (right-truncated).",
    )
    parser.add_argument(
        "--max-caption-len",
        type=int,
        default=None,
        help="Maximum token length for caption conditioning (defaults to max_text_len).",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--length-bucketing",
        dest="length_bucketing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Group similar-length samples into the same batch to minimize padding "
            "compute (only effective with batch_size > 1)."
        ),
    )
    parser.add_argument(
        "--category-balancing",
        dest="category_balancing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Resample training data with replacement so every category "
            "(speech/mixed/aegi/chupa) is drawn with equal probability."
        ),
    )
    parser.add_argument(
        "--prefetch-to-device",
        dest="dataloader_prefetch_to_device",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overlap the next pinned-memory batch transfer with CUDA model compute.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adamw8bit", "muon"],
        default="muon",
        help="Optimizer implementation; adamw8bit uses bitsandbytes to reduce state memory.",
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "wsd"], default="none")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--caption-warmup",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "During the first caption_warmup_steps optimizer steps, update only caption-only parameters "
            "(caption encoder/norm and caption attention projections)."
        ),
    )
    parser.add_argument(
        "--caption-warmup-steps",
        type=int,
        default=0,
        help="Number of optimizer steps to run caption-only warmup for when caption_warmup is enabled.",
    )
    parser.add_argument("--stable-steps", type=int, default=0)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--latent-patch-size", type=int, default=1)
    parser.add_argument("--max-latent-steps", type=int, default=750)
    parser.add_argument(
        "--fixed-target-latent-steps",
        type=int,
        default=None,
        help=(
            "If set, always train on this fixed target latent length "
            "(short samples are right-padded with zeros, long samples are truncated)."
        ),
    )
    parser.add_argument(
        "--fixed-target-full-mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use full target mask for fixed-length training "
            "(Echo-style includes padded tail in loss; only effective with rf_loss_mode=echo)."
        ),
    )
    parser.add_argument(
        "--rf-loss-mode",
        choices=["echo", "utterance_mean"],
        default=None,
        help="RF loss normalization mode.",
    )
    parser.add_argument("--duration-loss-weight", type=float, default=None)
    parser.add_argument("--duration-speaker-dropout", type=float, default=None)
    parser.add_argument("--duration-caption-dropout", type=float, default=None)
    parser.add_argument("--duration-huber-delta", type=float, default=None)
    parser.add_argument(
        "--text-condition-dropout",
        type=float,
        default=0.1,
        help="Probability of dropping text conditioning during training.",
    )
    parser.add_argument(
        "--caption-condition-dropout",
        type=float,
        default=0.1,
        help="Probability of dropping caption conditioning during training.",
    )
    parser.add_argument(
        "--speaker-condition-dropout",
        type=float,
        default=0.1,
        help="Probability of dropping speaker/reference conditioning during training.",
    )
    speaker_inversion_group = parser.add_mutually_exclusive_group()
    speaker_inversion_group.add_argument(
        "--speaker-inversion",
        dest="speaker_inversion_enabled",
        action="store_true",
        help="Train only learned speaker inversion embedding tokens.",
    )
    speaker_inversion_group.add_argument(
        "--no-speaker-inversion",
        dest="speaker_inversion_enabled",
        action="store_false",
        help="Disable Speaker Inversion training.",
    )
    parser.set_defaults(speaker_inversion_enabled=None)
    parser.add_argument(
        "--speaker-inversion-tokens",
        type=int,
        default=None,
        help="Number of learned Speaker Inversion tokens.",
    )
    parser.add_argument(
        "--speaker-inversion-init-std",
        type=float,
        default=None,
        help="Stddev for random Speaker Inversion token initialization.",
    )
    parser.add_argument(
        "--speaker-inversion-init-embedding",
        default=None,
        help=("Optional existing Speaker Inversion .speaker.safetensors file to continue from."),
    )
    parser.add_argument(
        "--timestep-stratified",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use stratified logit-normal timestep sampling (Echo-style).",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--save-every-epochs", type=int, default=None)
    parser.add_argument("--checkpoint-latest-n", type=int, default=None)
    parser.add_argument(
        "--checkpoint-best-n",
        type=int,
        default=0,
        help=(
            "Keep up to N best validation-loss checkpoints in addition to latest. "
            "When validation is disabled, keeps latest N+1 periodic checkpoints. "
            "Set 0 to disable checkpoint-count limiting."
        ),
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.0,
        help=("Split ratio for validation set from the single manifest. 0 disables validation."),
    )
    parser.add_argument(
        "--valid-split-by",
        choices=["sample", "speaker"],
        default=None,
        help=(
            "Validation split unit: 'sample' picks random rows; 'speaker' holds out "
            "whole speaker_id groups to avoid same-voice train/valid leakage."
        ),
    )
    ema_cli_group = parser.add_mutually_exclusive_group()
    ema_cli_group.add_argument(
        "--ema",
        dest="ema_enabled",
        action="store_true",
        help="Maintain an EMA copy of the weights (saved as 'ema_model' in checkpoints).",
    )
    ema_cli_group.add_argument(
        "--no-ema",
        dest="ema_enabled",
        action="store_false",
        help="Disable EMA weight averaging.",
    )
    parser.set_defaults(ema_enabled=None)
    parser.add_argument("--ema-decay", type=float, default=None, help="Per-step EMA decay.")
    parser.add_argument(
        "--ema-update-every",
        type=int,
        default=None,
        help="Apply the EMA update every N optimizer steps (decay is compensated).",
    )
    parser.add_argument(
        "--ema-device",
        default=None,
        help="Device holding the EMA shadow weights (cpu avoids extra VRAM).",
    )
    parser.add_argument(
        "--valid-every",
        type=int,
        default=0,
        help=("Run validation every N training steps. Set <=0 to disable validation."),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable tqdm progress bar.",
    )
    parser.add_argument(
        "--progress-all",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show tqdm progress bars for all ranks in DDP mode (default: rank0 only).",
    )
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--wandb",
        dest="wandb_enabled",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    wandb_group.add_argument(
        "--no-wandb",
        dest="wandb_enabled",
        action="store_false",
        help="Disable Weights & Biases logging.",
    )
    parser.set_defaults(wandb_enabled=None)
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Weights & Biases entity/team name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=sorted(WANDB_MODES),
        default=None,
        help="Weights & Biases mode.",
    )
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument(
        "--lora",
        dest="lora_enabled",
        action="store_true",
        help="Enable PEFT LoRA fine-tuning.",
    )
    lora_group.add_argument(
        "--no-lora",
        dest="lora_enabled",
        action="store_false",
        help="Disable PEFT LoRA fine-tuning.",
    )
    parser.set_defaults(lora_enabled=None)
    parser.add_argument("--lora-r", type=int, default=None, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha scaling.")
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=None,
        help="LoRA dropout probability.",
    )
    parser.add_argument(
        "--lora-bias",
        choices=["none", "all", "lora_only"],
        default=None,
        help="Bias handling passed to PEFT LoRA.",
    )
    parser.add_argument(
        "--lora-target-modules",
        default=None,
        help=(
            "LoRA target preset, regex, or comma-separated module suffix list. "
            f"Presets: {', '.join(sorted(LORA_TARGET_PRESETS))}."
        ),
    )
    parser.add_argument(
        "--lora-modules-to-save",
        default=None,
        help=(
            "Comma-separated full modules to keep trainable and save inside the LoRA adapter. "
            "Use 'auto' to save duration_predictor for v3 duration models, or 'none' to disable."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    ddp_group = parser.add_mutually_exclusive_group()
    ddp_group.add_argument(
        "--ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_true",
        help=(
            "Enable DDP find_unused_parameters. Useful when conditional branches "
            "(e.g., speaker/text conditioning) may be fully masked in some steps."
        ),
    )
    ddp_group.add_argument(
        "--no-ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_false",
        help="Disable DDP find_unused_parameters.",
    )
    parser.set_defaults(ddp_find_unused_parameters=None)
    args = parser.parse_args()
    if args.resume is not None and Path(args.resume).suffix.lower() == ".safetensors":
        raise ValueError(
            "--resume expects a training checkpoint (.pt or LoRA checkpoint dir). "
            "Use --init-checkpoint for inference-only .safetensors weights."
        )

    rank, world_size, local_rank, distributed, device = setup_distributed(args.device)
    is_main_process = rank == 0

    raw_argv = sys.argv[1:]
    exp_cfg = load_experiment_yaml(args.config) if args.config else {}
    unknown_root = sorted(set(exp_cfg) - {"model", "train"})
    if unknown_root:
        raise ValueError(f"Unknown top-level config keys: {unknown_root}")
    if args.config and is_main_process:
        print(f"Loaded config: {args.config}")
    model_cfg = merge_dataclass_overrides(ModelConfig(), exp_cfg.get("model"), section="model")
    train_cfg = merge_dataclass_overrides(TrainConfig(), exp_cfg.get("train"), section="train")
    default_train_cfg = TrainConfig()

    train_cfg = replace(train_cfg, manifest_path=args.manifest)
    if train_cfg.output_dir == default_train_cfg.output_dir and not cli_provided(
        raw_argv, "--output-dir"
    ):
        train_cfg = replace(train_cfg, output_dir=args.output_dir)

    if cli_provided(raw_argv, "--output-dir"):
        train_cfg = replace(train_cfg, output_dir=args.output_dir)
    # Flags with non-None argparse defaults override the YAML config only when
    # explicitly provided on the command line.
    flag_overrides: tuple[tuple[str, str, object], ...] = (
        ("--precision", "precision", args.precision),
        ("--train-mode", "train_mode", args.train_mode),
        ("--batch-size", "batch_size", args.batch_size),
        ("--gradient-accumulation-steps", "gradient_accumulation_steps", args.gradient_accumulation_steps),
        ("--max-text-len", "max_text_len", args.max_text_len),
        ("--max-caption-len", "max_caption_len", args.max_caption_len),
        ("--num-workers", "num_workers", args.num_workers),
        ("--lr", "learning_rate", args.lr),
        ("--weight-decay", "weight_decay", args.weight_decay),
        ("--optimizer", "optimizer", args.optimizer),
        ("--adam-beta1", "adam_beta1", args.adam_beta1),
        ("--adam-beta2", "adam_beta2", args.adam_beta2),
        ("--adam-eps", "adam_eps", args.adam_eps),
        ("--muon-momentum", "muon_momentum", args.muon_momentum),
        ("--lr-scheduler", "lr_scheduler", args.lr_scheduler),
        ("--warmup-steps", "warmup_steps", args.warmup_steps),
        ("--caption-warmup-steps", "caption_warmup_steps", args.caption_warmup_steps),
        ("--stable-steps", "stable_steps", args.stable_steps),
        ("--min-lr-scale", "min_lr_scale", args.min_lr_scale),
        ("--max-steps", "max_steps", args.max_steps),
        ("--text-condition-dropout", "text_condition_dropout", args.text_condition_dropout),
        ("--caption-condition-dropout", "caption_condition_dropout", args.caption_condition_dropout),
        ("--speaker-condition-dropout", "speaker_condition_dropout", args.speaker_condition_dropout),
        ("--speaker-inversion-tokens", "speaker_inversion_tokens", args.speaker_inversion_tokens),
        ("--speaker-inversion-init-std", "speaker_inversion_init_std", args.speaker_inversion_init_std),
        ("--speaker-inversion-init-embedding", "speaker_inversion_init_embedding", args.speaker_inversion_init_embedding),
        ("--max-latent-steps", "max_latent_steps", args.max_latent_steps),
        ("--fixed-target-latent-steps", "fixed_target_latent_steps", args.fixed_target_latent_steps),
        ("--rf-loss-mode", "rf_loss_mode", args.rf_loss_mode),
        ("--duration-loss-weight", "duration_loss_weight", args.duration_loss_weight),
        ("--duration-speaker-dropout", "duration_speaker_dropout", args.duration_speaker_dropout),
        ("--duration-caption-dropout", "duration_caption_dropout", args.duration_caption_dropout),
        ("--duration-huber-delta", "duration_huber_delta", args.duration_huber_delta),
        ("--log-every", "log_every", args.log_every),
        ("--save-every", "save_every", args.save_every),
        ("--checkpoint-best-n", "checkpoint_best_n", args.checkpoint_best_n),
        ("--valid-ratio", "valid_ratio", args.valid_ratio),
        ("--valid-every", "valid_every", args.valid_every),
        ("--wandb-project", "wandb_project", args.wandb_project),
        ("--wandb-entity", "wandb_entity", args.wandb_entity),
        ("--wandb-run-name", "wandb_run_name", args.wandb_run_name),
        ("--wandb-mode", "wandb_mode", args.wandb_mode),
        ("--lora-r", "lora_r", args.lora_r),
        ("--lora-alpha", "lora_alpha", args.lora_alpha),
        ("--lora-dropout", "lora_dropout", args.lora_dropout),
        ("--lora-bias", "lora_bias", args.lora_bias),
        ("--lora-target-modules", "lora_target_modules", args.lora_target_modules),
        ("--lora-modules-to-save", "lora_modules_to_save", args.lora_modules_to_save),
        ("--seed", "seed", args.seed),
    )
    # None-default options (BooleanOptionalAction and friends) override when set.
    value_overrides: tuple[tuple[str, object], ...] = (
        ("allow_tf32", args.allow_tf32),
        ("compile_model", args.compile_model),
        ("gradient_checkpointing", args.gradient_checkpointing),
        ("length_bucketing", args.length_bucketing),
        ("category_balancing", args.category_balancing),
        ("dataloader_prefetch_to_device", args.dataloader_prefetch_to_device),
        ("caption_warmup", args.caption_warmup),
        ("speaker_inversion_enabled", args.speaker_inversion_enabled),
        ("timestep_stratified", args.timestep_stratified),
        ("fixed_target_full_mask", args.fixed_target_full_mask),
        ("save_every_epochs", args.save_every_epochs),
        ("checkpoint_latest_n", args.checkpoint_latest_n),
        ("valid_split_by", args.valid_split_by),
        ("progress", args.progress),
        ("progress_all_ranks", args.progress_all),
        ("wandb_enabled", args.wandb_enabled),
        ("ema_enabled", args.ema_enabled),
        ("ema_decay", args.ema_decay),
        ("ema_update_every", args.ema_update_every),
        ("ema_device", args.ema_device),
        ("lora_enabled", args.lora_enabled),
        ("ddp_find_unused_parameters", args.ddp_find_unused_parameters),
    )
    updates: dict[str, object] = {
        field: value for flag, field, value in flag_overrides if cli_provided(raw_argv, flag)
    }
    updates.update({field: value for field, value in value_overrides if value is not None})
    if updates:
        train_cfg = replace(train_cfg, **updates)

    resume_path = Path(args.resume).expanduser() if args.resume is not None else None
    resume_train_cfg = None
    resume_base_init = None
    resume_payload: dict | None = None
    if args.resume is not None:
        resume_payload = _load_checkpoint_payload(resume_path, map_location="cpu")
        raw_resume_train_cfg = resume_payload.get("train_config")
        if raw_resume_train_cfg is not None and not isinstance(raw_resume_train_cfg, dict):
            raise ValueError("Resume checkpoint train_config must be a dictionary when present.")
        resume_train_cfg = raw_resume_train_cfg
        raw_resume_base_init = resume_payload.get("base_init")
        if raw_resume_base_init is not None and not isinstance(raw_resume_base_init, dict):
            raise ValueError("Resume checkpoint base_init must be a dictionary when present.")
        resume_base_init = raw_resume_base_init
        train_cfg = _restore_resume_lora_config(
            train_cfg,
            resume_train_cfg=resume_train_cfg,
            args=args,
            raw_argv=raw_argv,
            exp_cfg=exp_cfg,
        )

    if cli_provided(raw_argv, "--latent-dim"):
        model_cfg = replace(model_cfg, latent_dim=args.latent_dim)
    if cli_provided(raw_argv, "--latent-patch-size"):
        model_cfg = replace(model_cfg, latent_patch_size=args.latent_patch_size)

    set_seed(train_cfg.seed + rank)
    if not (0.0 <= train_cfg.text_condition_dropout <= 1.0):
        raise ValueError(
            f"text_condition_dropout must be in [0, 1], got {train_cfg.text_condition_dropout}"
        )
    if train_cfg.max_text_len <= 0:
        raise ValueError(f"max_text_len must be > 0, got {train_cfg.max_text_len}")
    if str(train_cfg.train_mode).strip().lower() not in TRAIN_MODES:
        raise ValueError(
            f"train_mode must be one of {sorted(TRAIN_MODES)}, got {train_cfg.train_mode!r}"
        )
    train_cfg = replace(train_cfg, train_mode=str(train_cfg.train_mode).strip().lower())
    if train_cfg.max_caption_len is not None and train_cfg.max_caption_len <= 0:
        raise ValueError(f"max_caption_len must be > 0, got {train_cfg.max_caption_len}")
    if train_cfg.gradient_accumulation_steps <= 0:
        raise ValueError(
            f"gradient_accumulation_steps must be > 0, got {train_cfg.gradient_accumulation_steps}"
        )
    if not (0.0 <= train_cfg.speaker_condition_dropout <= 1.0):
        raise ValueError(
            "speaker_condition_dropout must be in [0, 1], "
            f"got {train_cfg.speaker_condition_dropout}"
        )
    if train_cfg.speaker_inversion_enabled:
        if not model_cfg.use_speaker_condition_resolved:
            raise ValueError(
                "speaker_inversion_enabled=True requires a speaker-conditioned model config."
            )
        if args.init_checkpoint is None:
            raise ValueError(
                "speaker_inversion_enabled=True requires --init-checkpoint so the frozen "
                "base TTS model is initialized from trained weights."
            )
        if args.resume is not None:
            raise ValueError(
                "speaker_inversion_enabled=True saves embedding-only checkpoints; "
                "--resume full trainer state is not supported. Use "
                "speaker_inversion_init_embedding to continue from a saved embedding."
            )
        if train_config_uses_lora(train_cfg):
            raise ValueError("speaker_inversion_enabled=True does not support LoRA training.")
        if train_cfg.train_mode != "rf":
            raise ValueError("speaker_inversion_enabled=True supports train_mode='rf' only.")
        if train_cfg.caption_warmup:
            raise ValueError("speaker_inversion_enabled=True does not support caption_warmup.")
        if train_cfg.speaker_inversion_tokens <= 0:
            raise ValueError(
                f"speaker_inversion_tokens must be > 0, got {train_cfg.speaker_inversion_tokens}"
            )
        if train_cfg.speaker_inversion_init_std < 0:
            raise ValueError(
                "speaker_inversion_init_std must be >= 0, "
                f"got {train_cfg.speaker_inversion_init_std}"
            )
        optimizer_explicit = cli_provided(raw_argv, "--optimizer") or (
            isinstance(exp_cfg.get("train"), dict) and "optimizer" in exp_cfg.get("train", {})
        )
        if str(train_cfg.optimizer).strip().lower() == "muon":
            if optimizer_explicit:
                raise ValueError(
                    "speaker_inversion_enabled=True supports optimizer='adamw'. "
                    "Muon has no compatible matrix parameter when only speaker tokens are trainable."
                )
            train_cfg = replace(train_cfg, optimizer="adamw")
    if not (0.0 <= train_cfg.caption_condition_dropout <= 1.0):
        raise ValueError(
            "caption_condition_dropout must be in [0, 1], "
            f"got {train_cfg.caption_condition_dropout}"
        )
    if train_cfg.fixed_target_latent_steps is not None and train_cfg.fixed_target_latent_steps <= 0:
        raise ValueError(
            "fixed_target_latent_steps must be > 0 when provided, "
            f"got {train_cfg.fixed_target_latent_steps}"
        )
    if train_cfg.fixed_target_full_mask and train_cfg.fixed_target_latent_steps is None:
        raise ValueError(
            "fixed_target_full_mask=True requires fixed_target_latent_steps to be set."
        )
    if str(train_cfg.rf_loss_mode).strip().lower() not in {"echo", "utterance_mean"}:
        raise ValueError(
            "rf_loss_mode must be one of ['echo', 'utterance_mean'], "
            f"got {train_cfg.rf_loss_mode!r}"
        )
    if (
        train_cfg.fixed_target_full_mask
        and str(train_cfg.rf_loss_mode).strip().lower() == "utterance_mean"
        and is_main_process
    ):
        print(
            "warning: fixed_target_full_mask=True has no effect with "
            "rf_loss_mode='utterance_mean' (this loss ignores the padded tail)."
        )
    if train_cfg.duration_loss_weight < 0:
        raise ValueError(f"duration_loss_weight must be >= 0, got {train_cfg.duration_loss_weight}")
    if not (0.0 <= train_cfg.duration_speaker_dropout <= 1.0):
        raise ValueError(
            f"duration_speaker_dropout must be in [0, 1], got {train_cfg.duration_speaker_dropout}"
        )
    if not (0.0 <= train_cfg.duration_caption_dropout <= 1.0):
        raise ValueError(
            f"duration_caption_dropout must be in [0, 1], got {train_cfg.duration_caption_dropout}"
        )
    if train_cfg.duration_huber_delta <= 0:
        raise ValueError(f"duration_huber_delta must be > 0, got {train_cfg.duration_huber_delta}")
    if train_cfg.train_mode == "duration_only" and not model_cfg.use_duration_predictor:
        raise ValueError("train_mode='duration_only' requires model.use_duration_predictor=True.")
    if train_cfg.train_mode == "duration_only" and train_config_uses_lora(train_cfg):
        raise ValueError("train_mode='duration_only' does not support LoRA training.")
    if train_cfg.train_mode == "duration_only" and train_cfg.caption_warmup:
        raise ValueError("train_mode='duration_only' does not support caption_warmup.")
    if (
        train_cfg.train_mode == "duration_only"
        and args.init_checkpoint is None
        and args.resume is None
    ):
        raise ValueError(
            "train_mode='duration_only' requires --init-checkpoint or --resume "
            "so the frozen text/speaker encoders are initialized from trained weights."
        )
    if model_cfg.use_duration_predictor:
        if model_cfg.duration_aux_dim <= 0:
            raise ValueError(f"duration_aux_dim must be > 0, got {model_cfg.duration_aux_dim}")
        if model_cfg.duration_hidden_dim <= 0:
            raise ValueError(
                f"duration_hidden_dim must be > 0, got {model_cfg.duration_hidden_dim}"
            )
        if model_cfg.duration_layers <= 0:
            raise ValueError(f"duration_layers must be > 0, got {model_cfg.duration_layers}")
        if not (0.0 <= model_cfg.duration_dropout <= 1.0):
            raise ValueError(
                f"duration_dropout must be in [0, 1], got {model_cfg.duration_dropout}"
            )
        if model_cfg.duration_attention_heads <= 0:
            raise ValueError(
                f"duration_attention_heads must be > 0, got {model_cfg.duration_attention_heads}"
            )
        if model_cfg.text_dim % model_cfg.duration_attention_heads != 0:
            raise ValueError(
                "text_dim must be divisible by duration_attention_heads: "
                f"text_dim={model_cfg.text_dim}, "
                f"duration_attention_heads={model_cfg.duration_attention_heads}"
            )
        duration_architecture = str(model_cfg.duration_architecture).strip().lower()
        if duration_architecture not in DURATION_ARCHITECTURES:
            raise ValueError(
                "duration_architecture must be one of "
                f"{sorted(DURATION_ARCHITECTURES)}, got {model_cfg.duration_architecture!r}"
            )
        if model_cfg.duration_token_init_frames <= 0:
            raise ValueError(
                "duration_token_init_frames must be > 0, "
                f"got {model_cfg.duration_token_init_frames}"
            )
        duration_speaker_fusion = str(model_cfg.duration_speaker_fusion).strip().lower()
        if duration_speaker_fusion not in DURATION_SPEAKER_FUSIONS:
            raise ValueError(
                "duration_speaker_fusion must be one of "
                f"{sorted(DURATION_SPEAKER_FUSIONS)}, got {model_cfg.duration_speaker_fusion!r}"
            )
        duration_caption_fusion = str(model_cfg.duration_caption_fusion).strip().lower()
        if duration_caption_fusion not in DURATION_CAPTION_FUSIONS:
            raise ValueError(
                "duration_caption_fusion must be one of "
                f"{sorted(DURATION_CAPTION_FUSIONS)}, got {model_cfg.duration_caption_fusion!r}"
            )
        duration_caption_pooling = str(model_cfg.duration_caption_pooling).strip().lower()
        if duration_caption_pooling not in DURATION_CAPTION_POOLINGS:
            raise ValueError(
                "duration_caption_pooling must be one of "
                f"{sorted(DURATION_CAPTION_POOLINGS)}, got {model_cfg.duration_caption_pooling!r}"
            )
        if (
            duration_architecture == "token_sum_adarn_zero_no_aux"
            and duration_speaker_fusion != "adarn_zero"
        ):
            raise ValueError(
                "duration_architecture='token_sum_adarn_zero_no_aux' requires "
                "duration_speaker_fusion='adarn_zero'."
            )
        if duration_architecture == "token_sum_dual_adarn_zero_no_aux":
            if duration_speaker_fusion != "adarn_zero":
                raise ValueError(
                    "duration_architecture='token_sum_dual_adarn_zero_no_aux' requires "
                    "duration_speaker_fusion='adarn_zero'."
                )
            if duration_caption_fusion != "adarn_zero":
                raise ValueError(
                    "duration_architecture='token_sum_dual_adarn_zero_no_aux' requires "
                    "duration_caption_fusion='adarn_zero'."
                )
            if not model_cfg.use_speaker_condition_resolved or not model_cfg.use_caption_condition:
                raise ValueError(
                    "duration_architecture='token_sum_dual_adarn_zero_no_aux' requires "
                    "both speaker and caption conditioning."
                )
        model_cfg = replace(
            model_cfg,
            duration_architecture=duration_architecture,
            duration_speaker_fusion=duration_speaker_fusion,
            duration_caption_fusion=duration_caption_fusion,
            duration_caption_pooling=duration_caption_pooling,
        )
    if train_cfg.caption_warmup_steps < 0:
        raise ValueError(f"caption_warmup_steps must be >= 0, got {train_cfg.caption_warmup_steps}")
    if train_cfg.dataloader_prefetch_factor <= 0:
        raise ValueError(
            f"dataloader_prefetch_factor must be > 0, got {train_cfg.dataloader_prefetch_factor}"
        )
    if train_cfg.length_bucket_mult <= 0:
        raise ValueError(
            f"length_bucket_mult must be > 0, got {train_cfg.length_bucket_mult}"
        )
    if not (0.0 <= train_cfg.valid_ratio < 1.0):
        raise ValueError(f"valid_ratio must be in [0, 1), got {train_cfg.valid_ratio}")
    if train_cfg.valid_every < 0:
        raise ValueError(f"valid_every must be >= 0, got {train_cfg.valid_every}")
    valid_split_by = str(train_cfg.valid_split_by).strip().lower()
    if valid_split_by not in {"sample", "speaker"}:
        raise ValueError(
            f"valid_split_by must be one of ['sample', 'speaker'], got {train_cfg.valid_split_by!r}"
        )
    train_cfg = replace(train_cfg, valid_split_by=valid_split_by)
    if train_cfg.ema_enabled:
        if not (0.0 < train_cfg.ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1), got {train_cfg.ema_decay}")
        if train_cfg.ema_update_every < 1:
            raise ValueError(f"ema_update_every must be >= 1, got {train_cfg.ema_update_every}")
        if train_config_uses_lora(train_cfg):
            raise ValueError("ema_enabled=True does not support LoRA training.")
        if train_cfg.speaker_inversion_enabled:
            raise ValueError("ema_enabled=True does not support Speaker Inversion training.")
    if train_cfg.save_every < 0:
        raise ValueError(f"save_every must be >= 0, got {train_cfg.save_every}")
    if train_cfg.save_every_epochs < 0:
        raise ValueError(
            f"save_every_epochs must be >= 0, got {train_cfg.save_every_epochs}"
        )
    if train_cfg.checkpoint_latest_n < 0:
        raise ValueError(
            f"checkpoint_latest_n must be >= 0, got {train_cfg.checkpoint_latest_n}"
        )
    if train_cfg.sample_every_epochs < 0:
        raise ValueError(
            f"sample_every_epochs must be >= 0, got {train_cfg.sample_every_epochs}"
        )
    if train_cfg.sample_num_steps <= 0:
        raise ValueError(f"sample_num_steps must be > 0, got {train_cfg.sample_num_steps}")
    if train_cfg.sample_seconds is not None and train_cfg.sample_seconds <= 0:
        raise ValueError(f"sample_seconds must be > 0, got {train_cfg.sample_seconds}")
    if train_cfg.sample_reference_count <= 0:
        raise ValueError(
            f"sample_reference_count must be > 0, got {train_cfg.sample_reference_count}"
        )
    if train_cfg.valid_ratio > 0.0 and train_cfg.valid_every <= 0:
        raise ValueError("valid_every must be > 0 when valid_ratio > 0.")
    if train_cfg.valid_ratio == 0.0 and train_cfg.valid_every > 0 and is_main_process:
        print("warning: valid_every is set but valid_ratio=0. Validation is disabled.")
    if train_cfg.checkpoint_best_n < 0:
        raise ValueError(f"checkpoint_best_n must be >= 0, got {train_cfg.checkpoint_best_n}")
    if train_cfg.wandb_mode not in WANDB_MODES:
        raise ValueError(
            f"wandb_mode must be one of {sorted(WANDB_MODES)}, got {train_cfg.wandb_mode!r}"
        )
    precision = str(train_cfg.precision).lower()
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"precision must be one of ['fp32', 'bf16'], got {train_cfg.precision!r}")
    if precision == "bf16":
        if device.type != "cuda":
            if is_main_process:
                print("warning: precision=bf16 requested on non-CUDA device. Falling back to fp32.")
            train_cfg = replace(train_cfg, precision="fp32")
        elif not torch.cuda.is_bf16_supported():
            if is_main_process:
                print("warning: CUDA bf16 is not supported on this GPU. Falling back to fp32.")
            train_cfg = replace(train_cfg, precision="fp32")
    use_bf16 = train_cfg.precision == "bf16"
    if device.type == "cuda":
        tf32_enabled = bool(train_cfg.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        torch.set_float32_matmul_precision("high" if tf32_enabled else "highest")
        if is_main_process:
            print(f"TF32 matmul/cuDNN: {'enabled' if tf32_enabled else 'disabled'}")
    elif train_cfg.allow_tf32 and is_main_process:
        print("warning: allow_tf32=True requested on non-CUDA device; ignoring.")

    output_dir = Path(train_cfg.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_configs(output_dir / "config.json", model_cfg, train_cfg)
        print(
            f"Compute precision={train_cfg.precision} "
            "(weights/optimizer states kept in full-precision)."
        )
    if distributed:
        dist.barrier()
    if is_main_process and distributed:
        print(f"DDP enabled: world_size={world_size} (local_rank={local_rank})")
    wandb_run = None
    if train_cfg.wandb_enabled and is_main_process:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled, but `wandb` is not installed. "
                "Install it with `pip install wandb`."
            ) from exc
        wandb_run = wandb.init(
            project=train_cfg.wandb_project,
            entity=train_cfg.wandb_entity,
            name=train_cfg.wandb_run_name,
            mode=train_cfg.wandb_mode,
            dir=str(output_dir),
            config={
                "model": asdict(model_cfg),
                "train": asdict(train_cfg),
                "script": "irodori-train",
            },
        )
        print(
            f"W&B enabled: project={train_cfg.wandb_project} mode={train_cfg.wandb_mode} run={wandb_run.name if wandb_run is not None else train_cfg.wandb_run_name}"
        )
    tensorboard_writer = None
    tensorboard_process = None
    if train_cfg.tensorboard_enabled and is_main_process:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard logging is enabled, but `tensorboard` is not installed. "
                "Run `uv sync --extra cu128` to install it."
            ) from exc
        tensorboard_dir = output_dir / train_cfg.tensorboard_log_dir
        tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_dir), purge_step=None)
        print(f"TensorBoard logging enabled: {tensorboard_dir}")
        if train_cfg.tensorboard_launch:
            tensorboard_url = f"http://localhost:{train_cfg.tensorboard_port}"
            tensorboard_command = [
                sys.executable,
                "-m",
                "tensorboard.main",
                "--logdir",
                str(tensorboard_dir),
                "--port",
                str(train_cfg.tensorboard_port),
            ]
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            tensorboard_process = subprocess.Popen(
                tensorboard_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            webbrowser.open(tensorboard_url)
            print(f"TensorBoard opened: {tensorboard_url}")

    if distributed:
        local_files_only = not is_main_process
        if is_main_process:
            tokenizer = build_text_tokenizer(model_cfg, local_files_only=False)
            text_hidden_size = validate_text_backbone_dim(model_cfg, local_files_only=False)
            caption_tokenizer = None
            caption_hidden_size = None
            if model_cfg.use_caption_condition:
                caption_tokenizer = build_caption_tokenizer(model_cfg, local_files_only=False)
                caption_hidden_size = validate_caption_backbone_dim(
                    model_cfg,
                    local_files_only=False,
                )
        dist.barrier()
        if not is_main_process:
            tokenizer = build_text_tokenizer(model_cfg, local_files_only=local_files_only)
            text_hidden_size = validate_text_backbone_dim(
                model_cfg,
                local_files_only=local_files_only,
            )
            caption_tokenizer = None
            caption_hidden_size = None
            if model_cfg.use_caption_condition:
                caption_tokenizer = build_caption_tokenizer(
                    model_cfg,
                    local_files_only=local_files_only,
                )
                caption_hidden_size = validate_caption_backbone_dim(
                    model_cfg,
                    local_files_only=local_files_only,
                )
        dist.barrier()
    else:
        tokenizer = build_text_tokenizer(model_cfg, local_files_only=False)
        text_hidden_size = validate_text_backbone_dim(model_cfg, local_files_only=False)
        caption_tokenizer = None
        caption_hidden_size = None
        if model_cfg.use_caption_condition:
            caption_tokenizer = build_caption_tokenizer(model_cfg, local_files_only=False)
            caption_hidden_size = validate_caption_backbone_dim(
                model_cfg,
                local_files_only=False,
            )
    if is_main_process:
        print(
            f"Text tokenizer={model_cfg.text_tokenizer_repo} vocab={tokenizer.vocab_size} add_bos={model_cfg.text_add_bos} padding_side=right "
            f"(pretrained hidden_size={text_hidden_size})."
        )
        if model_cfg.use_caption_condition and caption_tokenizer is not None:
            print(
                f"Caption tokenizer={model_cfg.caption_tokenizer_repo_resolved} vocab={caption_tokenizer.vocab_size} add_bos={model_cfg.caption_add_bos_resolved} padding_side=right "
                f"(pretrained hidden_size={caption_hidden_size})."
            )
    dataset_load_target_latent = train_cfg.train_mode != "duration_only"
    full_dataset = LatentTextDataset(
        manifest_path=train_cfg.manifest_path,
        latent_dim=model_cfg.latent_dim,
        max_latent_steps=train_cfg.max_latent_steps,
        enable_caption_condition=model_cfg.use_caption_condition,
        enable_speaker_condition=model_cfg.use_speaker_condition_resolved,
        show_manifest_progress=bool(train_cfg.progress and is_main_process),
        manifest_progress_desc="Index Manifest",
        load_target_latent=dataset_load_target_latent,
    )
    if is_main_process and full_dataset.style_filtered_count > 0:
        print(
            f"info: {full_dataset.style_filtered_count} manifest rows excluded from "
            "training because their (speaker, category) pool has no alternative "
            "clip for a style-matched reference (manifest untouched)."
        )
    if is_main_process and full_dataset.overlong_sample_count > 0:
        print(
            f"warning: {full_dataset.overlong_sample_count} manifest rows declare "
            f"num_frames > max_latent_steps={train_cfg.max_latent_steps}. Their latents are "
            "truncated at load time while the text stays full, creating text/audio mismatch. "
            "Consider filtering these rows during dataset preparation."
        )
    train_dataset = full_dataset
    valid_dataset = None
    if train_cfg.valid_ratio > 0.0:
        split_groups = None
        if train_cfg.valid_split_by == "speaker":
            split_groups = [
                full_dataset.manifest_index.speaker_ids[i]
                for i in full_dataset.sample_indices
            ]
        train_indices, valid_indices = split_train_valid_indices(
            num_samples=len(full_dataset),
            valid_ratio=train_cfg.valid_ratio,
            seed=train_cfg.seed,
            groups=split_groups,
        )
        # split indices are positions within the (possibly style-filtered)
        # full_dataset; map them back to manifest row indices for subsetting.
        train_indices = [full_dataset.sample_indices[i] for i in train_indices]
        valid_indices = [full_dataset.sample_indices[i] for i in valid_indices]
        train_dataset = LatentTextDataset(
            manifest_path=train_cfg.manifest_path,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=train_cfg.max_latent_steps,
            subset_indices=train_indices,
            enable_caption_condition=model_cfg.use_caption_condition,
            enable_speaker_condition=model_cfg.use_speaker_condition_resolved,
            manifest_index=full_dataset.manifest_index,
            load_target_latent=dataset_load_target_latent,
        )
        valid_dataset = LatentTextDataset(
            manifest_path=train_cfg.manifest_path,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=train_cfg.max_latent_steps,
            subset_indices=valid_indices,
            enable_caption_condition=model_cfg.use_caption_condition,
            enable_speaker_condition=model_cfg.use_speaker_condition_resolved,
            manifest_index=full_dataset.manifest_index,
            load_target_latent=dataset_load_target_latent,
        )
        if is_main_process:
            print(
                f"Validation split enabled: train={len(train_dataset)} valid={len(valid_dataset)} "
                f"(ratio={train_cfg.valid_ratio:.4f}, split_by={train_cfg.valid_split_by}, "
                f"valid_every={train_cfg.valid_every} steps)."
            )
    drop_last = len(train_dataset) >= train_cfg.batch_size
    if not drop_last and is_main_process:
        print(
            f"warning: dataset size ({len(train_dataset)}) is smaller than batch_size ({train_cfg.batch_size}). "
            "Using drop_last=False to avoid empty dataloader."
        )
    collator = TTSCollator(
        tokenizer=tokenizer,
        caption_tokenizer=caption_tokenizer,
        latent_dim=model_cfg.latent_dim,
        latent_patch_size=model_cfg.latent_patch_size,
        fixed_target_latent_steps=train_cfg.fixed_target_latent_steps,
        fixed_target_full_mask=train_cfg.fixed_target_full_mask,
        max_text_len=train_cfg.max_text_len,
        max_caption_len=(
            train_cfg.max_text_len
            if train_cfg.max_caption_len is None
            else train_cfg.max_caption_len
        ),
        include_target_latent=train_cfg.train_mode != "duration_only",
        include_reference_latent=model_cfg.use_speaker_condition_resolved,
        include_duration_features=model_cfg.use_duration_predictor,
    )
    if train_cfg.fixed_target_latent_steps is not None and is_main_process:
        print(
            f"Fixed target latent length enabled: steps={train_cfg.fixed_target_latent_steps} full_mask={train_cfg.fixed_target_full_mask}"
        )
    if not model_cfg.use_speaker_condition_resolved and is_main_process:
        print("Speaker conditioning disabled for this model config.")
    if train_cfg.caption_warmup and is_main_process:
        if not model_cfg.use_caption_condition:
            print(
                "warning: caption_warmup=True requested, but caption conditioning is disabled. Ignoring."
            )
        elif train_cfg.caption_warmup_steps <= 0:
            print(
                "warning: caption_warmup=True requested, but caption_warmup_steps <= 0. Ignoring."
            )
        else:
            print(
                "Caption warmup enabled: only caption-only parameters will update for the first "
                f"{train_cfg.caption_warmup_steps} optimizer steps."
            )
    if train_cfg.timestep_stratified and is_main_process:
        print("Using stratified logit-normal timestep sampling.")
    train_sampler = None
    train_batch_sampler = None
    use_length_bucketing = bool(train_cfg.length_bucketing) and train_cfg.batch_size > 1
    use_category_balancing = bool(train_cfg.category_balancing)
    sample_categories: list[str] | None = None
    if use_category_balancing:
        sample_categories, category_counts = train_dataset.sample_categories()
        if len(category_counts) <= 1:
            sample_categories = None
            use_category_balancing = False
            if is_main_process:
                print(
                    "info: category balancing disabled — manifest has a single "
                    f"category: {sorted(category_counts)}"
                )
        elif is_main_process:
            min_count = min(category_counts.values())
            print("Category-balanced sampling enabled (undersample to smallest category):")
            print(f"  {'category':<10}{'rows':>8}{'used/epoch':>12}")
            for name, count in sorted(category_counts.items()):
                print(f"  {name:<10}{count:>8}{min_count:>12}")
            print(
                f"  {'total':<10}{len(train_dataset):>8}"
                f"{min_count * len(category_counts):>12}"
            )
    if use_length_bucketing or use_category_balancing:
        train_batch_sampler = LengthGroupedBatchSampler(
            train_dataset.sample_lengths(),
            batch_size=train_cfg.batch_size,
            drop_last=drop_last,
            shuffle=True,
            seed=train_cfg.seed,
            bucket_mult=train_cfg.length_bucket_mult if use_length_bucketing else 1,
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
            groups=sample_categories,
        )
        if is_main_process and use_length_bucketing:
            print(
                "Length-bucketed batching enabled: "
                f"batch_size={train_cfg.batch_size} bucket_mult={train_cfg.length_bucket_mult}."
            )
    elif distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=drop_last,
        )
    dataloader_common_kwargs = {
        "num_workers": train_cfg.num_workers,
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collator,
    }
    if train_cfg.num_workers > 0:
        dataloader_common_kwargs["persistent_workers"] = bool(
            train_cfg.dataloader_persistent_workers
        )
        dataloader_common_kwargs["prefetch_factor"] = int(train_cfg.dataloader_prefetch_factor)
    elif train_cfg.dataloader_persistent_workers and is_main_process:
        print("warning: dataloader_persistent_workers=True is ignored because num_workers=0.")
    if train_batch_sampler is not None:
        loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            **dataloader_common_kwargs,
        )
    else:
        loader = DataLoader(
            dataset=train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=drop_last,
            **dataloader_common_kwargs,
        )
    if len(loader) == 0:
        raise ValueError("Dataloader yielded zero batches. Check manifest and batch_size settings.")
    valid_loader = None
    valid_sampler = None
    if valid_dataset is not None:
        if distributed:
            valid_sampler = DistributedSampler(
                valid_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            sampler=valid_sampler,
            drop_last=False,
            **dataloader_common_kwargs,
        )
        if len(valid_loader) == 0:
            raise ValueError(
                "Validation dataloader yielded zero batches. Decrease batch_size or valid_ratio."
            )

    has_validation = valid_loader is not None and train_cfg.valid_every > 0
    checkpoint_retention_enabled = train_cfg.checkpoint_best_n > 0
    periodic_checkpoint_keep = int(train_cfg.checkpoint_latest_n)
    if checkpoint_retention_enabled:
        if periodic_checkpoint_keep <= 0:
            periodic_checkpoint_keep = (
                1 if has_validation else int(train_cfg.checkpoint_best_n) + 1
            )
    best_val_checkpoints: list[tuple[float, int, Path]] = []
    if is_main_process:
        if checkpoint_retention_enabled and has_validation:
            best_val_checkpoints = list_best_val_loss_checkpoints(output_dir)
            best_val_checkpoints = prune_best_val_loss_checkpoints(
                best_val_checkpoints,
                train_cfg.checkpoint_best_n,
            )
        if checkpoint_retention_enabled and has_validation:
            print(
                "Checkpoint retention: "
                f"latest={periodic_checkpoint_keep} + "
                f"best_val_loss={train_cfg.checkpoint_best_n}."
            )
        elif checkpoint_retention_enabled:
            print(
                f"Checkpoint retention: validation disabled, keep latest {periodic_checkpoint_keep} periodic checkpoints."
            )

    if not (0.0 <= train_cfg.lora_dropout <= 1.0):
        raise ValueError(f"lora_dropout must be in [0, 1], got {train_cfg.lora_dropout}")
    if train_cfg.lora_r <= 0:
        raise ValueError(f"lora_r must be > 0, got {train_cfg.lora_r}")
    if train_cfg.lora_alpha <= 0:
        raise ValueError(f"lora_alpha must be > 0, got {train_cfg.lora_alpha}")

    if args.resume is not None:
        if train_config_uses_lora(train_cfg):
            if resume_path is None or not is_lora_adapter_dir(resume_path):
                raise ValueError("LoRA resume expects an adapter checkpoint directory.")
        elif resume_path is not None and resume_path.is_dir():
            raise ValueError(
                "Non-LoRA resume expects a .pt training checkpoint, not a checkpoint directory."
            )
        if args.init_checkpoint is not None and not train_config_uses_lora(train_cfg):
            raise ValueError(
                "--resume and --init-checkpoint can only be combined for LoRA adapter resumes."
            )

    if train_config_uses_lora(train_cfg) and args.resume is None and args.init_checkpoint is None:
        raise ValueError(
            "LoRA fine-tuning requires --init-checkpoint for the base model, "
            "or --resume from a LoRA adapter checkpoint directory."
        )

    raw_model: torch.nn.Module = TextToLatentRFDiT(model_cfg).to(device)
    lora_wrapped = False
    base_init: dict | None = None
    if args.resume is not None and train_config_uses_lora(train_cfg):
        base_init = resume_base_init
        if args.init_checkpoint is not None:
            override_init_path = _normalize_checkpoint_path(args.init_checkpoint)
            base_init = {"mode": "checkpoint", "checkpoint_path": str(override_init_path)}
        _apply_base_initialization(
            raw_model,
            model_cfg=model_cfg,
            base_init=base_init,
            distributed=distributed,
            is_main_process=is_main_process,
        )
        if resume_path is None or not is_lora_adapter_dir(resume_path):
            raise ValueError("LoRA resume expects an adapter checkpoint directory.")
        raw_model = load_lora_adapter(raw_model, resume_path, is_trainable=True)
        lora_wrapped = True
    elif args.resume is None and args.init_checkpoint is None:
        _apply_base_initialization(
            raw_model,
            model_cfg=model_cfg,
            base_init=None,
            distributed=distributed,
            is_main_process=is_main_process,
        )
        if train_config_uses_lora(train_cfg):
            raw_model = apply_lora(raw_model, train_cfg)
            lora_wrapped = True
    elif args.init_checkpoint is not None:
        init_checkpoint_path = _normalize_checkpoint_path(args.init_checkpoint)
        base_init = {"mode": "checkpoint", "checkpoint_path": str(init_checkpoint_path)}
        _apply_base_initialization(
            raw_model,
            model_cfg=model_cfg,
            base_init=base_init,
            distributed=distributed,
            is_main_process=is_main_process,
        )
        if train_config_uses_lora(train_cfg) and not lora_wrapped:
            raw_model = apply_lora(raw_model, train_cfg)
            lora_wrapped = True

    if train_config_uses_lora(train_cfg) and is_main_process:
        trainable_params, total_params = count_parameters(raw_model)
        print(
            "LoRA enabled: "
            f"r={train_cfg.lora_r} alpha={train_cfg.lora_alpha} "
            f"dropout={train_cfg.lora_dropout:.3f} "
            f"target_modules={train_cfg.lora_target_modules!r} "
            f"modules_to_save={train_cfg.lora_modules_to_save!r} "
            f"trainable={trainable_params:,}/{total_params:,}"
        )
    if train_cfg.speaker_inversion_enabled:
        init_embedding = None
        if train_cfg.speaker_inversion_init_embedding is not None:
            init_payload = load_speaker_inversion_payload(
                train_cfg.speaker_inversion_init_embedding,
            )
            init_embedding = init_payload[SPEAKER_EMBEDDING_KEY]
            if is_main_process:
                print(
                    "Loaded Speaker Inversion init embedding: "
                    f"{train_cfg.speaker_inversion_init_embedding}"
                )
        speaker_inversion = raw_model.enable_speaker_inversion(
            num_tokens=train_cfg.speaker_inversion_tokens,
            init_std=train_cfg.speaker_inversion_init_std,
            init_embedding=init_embedding,
        )
        speaker_inversion.to(device)
        if is_main_process:
            print(
                "Speaker Inversion parameters initialized: "
                f"embedding={tuple(speaker_inversion.embedding.shape)}."
            )
    if train_cfg.train_mode == "duration_only":
        trainable_duration_params, frozen_params = freeze_for_duration_only(raw_model)
        if trainable_duration_params == 0:
            raise RuntimeError(
                "No duration predictor parameters were found for duration_only mode."
            )
        if is_main_process:
            print(
                "Duration-only training enabled: "
                f"trainable={trainable_duration_params:,} frozen={frozen_params:,}."
            )
    if train_cfg.speaker_inversion_enabled:
        trainable_speaker_params, frozen_params = freeze_for_speaker_inversion(raw_model)
        if trainable_speaker_params == 0:
            raise RuntimeError("No Speaker Inversion parameters were found.")
        if is_main_process:
            print(
                "Speaker Inversion freeze applied: "
                f"trainable={trainable_speaker_params:,} frozen={frozen_params:,}."
            )
    if train_cfg.gradient_checkpointing:
        raw_model.set_gradient_checkpointing(True)
        if is_main_process:
            print("Gradient checkpointing enabled on diffusion blocks.")
    train_model = raw_model
    if train_cfg.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile_model=True requires torch.compile (PyTorch 2+).")
        if is_main_process:
            print("torch.compile enabled (dynamic=True).")
        train_model = torch.compile(raw_model, dynamic=True)
    ddp_find_unused_parameters = bool(train_cfg.ddp_find_unused_parameters)
    ddp_find_unused_parameters_explicit = args.ddp_find_unused_parameters is not None or (
        isinstance(exp_cfg.get("train"), dict)
        and "ddp_find_unused_parameters" in exp_cfg.get("train", {})
    )
    if distributed:
        # Auto-enable for common configs where conditional branches can be fully
        # masked in a step. Without this, DDP can hang after step 1 due to
        # unreduced gradients in ranks where a branch is entirely unused.
        if not ddp_find_unused_parameters and not ddp_find_unused_parameters_explicit:
            speaker_labeled_count = train_dataset.speaker_labeled_count
            has_partial_or_no_speaker_labels = speaker_labeled_count < len(train_dataset)
            caption_labeled_count = train_dataset.caption_labeled_count
            has_partial_or_no_caption_labels = (
                model_cfg.use_caption_condition and caption_labeled_count < len(train_dataset)
            )
            has_stochastic_cond_drop = (
                train_cfg.text_condition_dropout > 0.0
                or train_cfg.speaker_condition_dropout > 0.0
                or (model_cfg.use_caption_condition and train_cfg.caption_condition_dropout > 0.0)
            )
            if (
                has_partial_or_no_speaker_labels
                or has_partial_or_no_caption_labels
                or has_stochastic_cond_drop
            ):
                ddp_find_unused_parameters = True
                if is_main_process:
                    print(
                        "DDP find_unused_parameters auto-enabled "
                        "(conditional branches may be fully masked in some steps)."
                    )
        model = DDP(
            train_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused_parameters,
            broadcast_buffers=False,
        )
    elif train_cfg.data_parallel:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError(
                "data_parallel=True requires at least two visible CUDA devices."
            )
        model = torch.nn.DataParallel(
            train_model,
            device_ids=list(range(torch.cuda.device_count())),
            output_device=0,
        )
        if is_main_process:
            print(f"DataParallel enabled across {torch.cuda.device_count()} CUDA devices.")
    else:
        model = train_model
    optimizer = build_optimizer(raw_model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    if is_main_process:
        print(
            f"Optimizer={train_cfg.optimizer} Scheduler={train_cfg.lr_scheduler} lr={current_lr(optimizer):.3e}"
        )
        if train_config_uses_lora(train_cfg) and train_cfg.optimizer.lower() == "muon":
            print(
                "warning: optimizer=muon Newton-Schulz-orthogonalizes every 2D weight, "
                "which includes LoRA A/B factors; low-rank adapters are untested under "
                "Muon. Consider optimizer=adamw for LoRA runs."
            )
        if train_cfg.gradient_accumulation_steps > 1:
            print(
                f"Gradient accumulation enabled: steps={train_cfg.gradient_accumulation_steps} (effective global batch={train_cfg.batch_size * world_size * train_cfg.gradient_accumulation_steps})."
            )

    step = 0
    start_epoch = 0
    progress: TrainProgress | None = None
    ema: ModelEMA | None = None
    resume_ema_state = None
    if args.resume is not None:
        # Reuse the payload already loaded for config restore, and keep it on
        # CPU: load_state_dict moves tensors to the parameter device, so the
        # checkpoint copy never needs to occupy VRAM for the rest of the run.
        ckpt = (
            resume_payload
            if resume_payload is not None
            else _load_checkpoint_payload(resume_path, map_location="cpu")
        )
        resume_payload = None
        if not train_config_uses_lora(train_cfg):
            raw_model.load_state_dict(ckpt["model"])
        # Snapshot each group's weight-decay intent BEFORE load_state_dict pins the
        # live group hyperparameters to the checkpoint's copies. build_optimizer
        # tags groups with irodori_decay; the >0 fallback covers optimizer states
        # saved before that tag existed.
        resume_decay_flags = [
            bool(group.get("irodori_decay", float(group.get("weight_decay", 0.0)) > 0.0))
            for group in optimizer.param_groups
        ]
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            # A different optimizer choice (e.g. adamw8bit -> muon) makes the
            # stored state unloadable; warn and continue with fresh moments.
            if is_main_process:
                print(
                    "warning: optimizer state in the resume checkpoint is incompatible "
                    f"with optimizer={train_cfg.optimizer!r}; continuing with fresh "
                    f"optimizer state ({exc})."
                )
        step = int(ckpt["step"])
        start_epoch = int(ckpt.get("epoch", 0))
        resume_ema_state = ckpt.get("ema_model")
        resume_scheduler_state = ckpt.get("scheduler")
        del ckpt
        # Rebase every optimizer hyperparameter that has a config field from the
        # current yaml/CLI. load_state_dict otherwise pins them to the values the
        # checkpoint was saved with, so edits to lr / weight_decay / momentum /
        # betas / adjust_lr_fn between runs would be silently ignored. Adam moments
        # and step counts are preserved.
        new_base = float(train_cfg.learning_rate)
        new_wd = float(train_cfg.weight_decay)
        new_momentum = float(train_cfg.muon_momentum)
        new_betas = (float(train_cfg.adam_beta1), float(train_cfg.adam_beta2))
        new_eps = float(train_cfg.adam_eps)
        new_adjust_lr_fn = str(train_cfg.muon_adjust_lr_fn)
        for is_decay, group in zip(
            resume_decay_flags, optimizer.param_groups, strict=False
        ):
            # Re-tag so a group whose marker was dropped by loading a pre-tag
            # checkpoint carries irodori_decay in every checkpoint from here on.
            group["irodori_decay"] = is_decay
            group["weight_decay"] = new_wd if is_decay else 0.0
            if "momentum" in group:  # Muon group
                group["momentum"] = new_momentum
            if "adjust_lr_fn" in group:  # Muon group
                group["adjust_lr_fn"] = new_adjust_lr_fn
            if "betas" in group:  # AdamW / Muon-aux group (Muon groups have none)
                group["betas"] = new_betas
                group["eps"] = new_eps
        if scheduler is not None:
            scheduler_state = resume_scheduler_state
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            elif step > 0:
                scheduler.last_step = step
            scheduler.base_lrs = [new_base for _ in scheduler.base_lrs]
            scale = float(scheduler.lr_lambda(max(int(scheduler.last_step), 0)))
            for base_lr, group in zip(
                scheduler.base_lrs, optimizer.param_groups, strict=False
            ):
                group["lr"] = base_lr * scale
        else:
            for group in optimizer.param_groups:
                group["lr"] = new_base
        if is_main_process:
            print(
                f"Resumed from step={step} (epoch={start_epoch}) "
                f"lr={current_lr(optimizer):.3e} "
                f"(base={new_base:.3e} weight_decay={new_wd:g} "
                f"muon_momentum={new_momentum:g} adam_betas={new_betas} "
                f"adam_eps={new_eps:g} muon_adjust_lr_fn={new_adjust_lr_fn})"
            )
    if train_cfg.ema_enabled:
        ema = ModelEMA(
            raw_model,
            decay=train_cfg.ema_decay,
            update_every=train_cfg.ema_update_every,
            device=train_cfg.ema_device,
        )
        if isinstance(resume_ema_state, dict):
            ema_loaded, ema_missing = ema.load_state_dict(resume_ema_state)
            if is_main_process:
                print(f"Resumed EMA weights: loaded={ema_loaded} missing={ema_missing}")
        elif args.resume is not None and is_main_process:
            print("EMA state missing in resume checkpoint; EMA restarts from current weights.")
        if is_main_process:
            print(
                f"EMA enabled: decay={train_cfg.ema_decay} "
                f"update_every={train_cfg.ema_update_every} device={train_cfg.ema_device}"
            )
    resume_ema_state = None

    progress = TrainProgress(
        max_steps=train_cfg.max_steps,
        start_step=step,
        rank=rank,
        world_size=world_size,
        enabled=train_cfg.progress,
        show_all_ranks=train_cfg.progress_all_ranks,
        description="Train Duration" if train_cfg.train_mode == "duration_only" else "Train RF",
    )
    accum_steps = int(train_cfg.gradient_accumulation_steps)
    global_batch_size = train_cfg.batch_size * world_size * accum_steps
    duration_only = train_cfg.train_mode == "duration_only"
    batch_keys = training_batch_keys(raw_model.cfg, duration_only=duration_only)
    caption_warmup_active = bool(
        train_cfg.caption_warmup
        and model_cfg.use_caption_condition
        and train_cfg.caption_warmup_steps > 0
        and step < train_cfg.caption_warmup_steps
    )
    if caption_warmup_active and is_main_process:
        print(
            "Caption warmup active: non-caption gradients will be cleared for the first "
            f"{train_cfg.caption_warmup_steps} optimizer steps."
        )

    try:
        model.train()
        if scheduler is not None and step == 0:
            # Ensure the very first optimizer step uses warmup-scaled LR.
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        accum_micro_steps = 0
        accum_loss = torch.zeros((), device=device, dtype=torch.float)
        accum_rf_loss = torch.zeros((), device=device, dtype=torch.float)
        accum_duration_loss = torch.zeros((), device=device, dtype=torch.float)
        accum_duration_mae_frames = torch.zeros((), device=device, dtype=torch.float)
        accum_duration_group_totals = torch.zeros(
            DURATION_CONDITION_GROUP_TOTAL_SIZE,
            device=device,
            dtype=torch.float64,
        )
        nonfinite_step_count = 0
        last_grad_norm = 0.0
        epoch = start_epoch
        while step < train_cfg.max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            elif train_batch_sampler is not None:
                train_batch_sampler.set_epoch(epoch)
            epoch += 1
            device_batches = iter_device_batches(
                loader,
                keys=batch_keys,
                device=device,
                prefetch=train_cfg.dataloader_prefetch_to_device,
            )
            for epoch_step, batch in enumerate(device_batches, start=1):
                accum_micro_steps += 1
                text_ids = batch["text_ids"]
                text_mask = batch["text_mask"]
                caption_ids = None
                caption_mask = None
                has_caption = None
                if raw_model.cfg.use_caption_condition:
                    caption_ids = batch["caption_ids"]
                    caption_mask = batch["caption_mask"]
                    has_caption = batch["has_caption"]
                num_frames = batch["num_frames"]
                duration_features = batch.get("duration_features")
                ref_latent = None
                ref_mask = None
                if raw_model.cfg.use_speaker_condition_resolved:
                    ref_latent = batch["ref_latent_patched"]
                    ref_mask = batch["ref_latent_mask_patched"]
                    has_speaker = batch["has_speaker"]
                else:
                    has_speaker = None

                bsz = text_ids.shape[0]
                x_mask = None
                x_mask_valid = None
                x_t = None
                t = None
                v_target = None
                if not duration_only:
                    x0 = batch["latent_patched"]
                    x_mask = batch["latent_mask_patched"]
                    x_mask_valid = batch["latent_mask_valid_patched"]
                    if train_cfg.timestep_stratified:
                        t = sample_stratified_logit_normal_t(
                            batch_size=bsz,
                            device=device,
                            mean=train_cfg.timestep_logit_mean,
                            std=train_cfg.timestep_logit_std,
                            t_min=train_cfg.timestep_min,
                            t_max=train_cfg.timestep_max,
                        )
                    else:
                        t = sample_logit_normal_t(
                            batch_size=bsz,
                            device=device,
                            mean=train_cfg.timestep_logit_mean,
                            std=train_cfg.timestep_logit_std,
                            t_min=train_cfg.timestep_min,
                            t_max=train_cfg.timestep_max,
                        )
                    noise = torch.randn_like(x0)
                    x_t = rf_interpolate(x0, noise, t)
                    v_target = rf_velocity_target(x0, noise)

                text_cond_drop = torch.rand(bsz, device=device) < train_cfg.text_condition_dropout
                if not raw_model.cfg.use_duration_predictor:
                    # Applying the mask unconditionally avoids synchronizing on
                    # a device-side any() result every micro-batch.
                    text_mask = text_mask & (~text_cond_drop[:, None])
                caption_cond_drop = None
                caption_drop_for_model = None
                duration_has_caption = None
                if raw_model.cfg.use_caption_condition:
                    if has_caption is None or caption_mask is None:
                        raise RuntimeError(
                            "Caption conditioning is enabled but caption batch tensors are missing."
                        )
                    caption_cond_drop = (
                        torch.rand(bsz, device=device) < train_cfg.caption_condition_dropout
                    )
                    use_caption = has_caption & (~caption_cond_drop)
                    caption_drop_for_model = ~use_caption
                    if raw_model.cfg.use_duration_predictor:
                        duration_caption_drop = (
                            torch.rand(bsz, device=device) < train_cfg.duration_caption_dropout
                        )
                        duration_has_caption = has_caption & (~duration_caption_drop)
                    else:
                        caption_mask = caption_mask & use_caption[:, None]

                speaker_drop_for_model = None
                duration_has_speaker = None
                if raw_model.cfg.use_speaker_condition_resolved:
                    speaker_cond_drop = (
                        torch.rand(bsz, device=device) < train_cfg.speaker_condition_dropout
                    )
                    if train_cfg.speaker_inversion_enabled:
                        # Speaker Inversion learns one embedding for this run, so all samples are
                        # speaker-conditioned even when the manifest has no speaker_id.
                        use_speaker = ~speaker_cond_drop
                        speaker_drop_for_model = speaker_cond_drop
                    else:
                        use_speaker = has_speaker & (~speaker_cond_drop)
                        speaker_drop_for_model = ~use_speaker
                    if raw_model.cfg.use_duration_predictor:
                        duration_speaker_drop = (
                            torch.rand(bsz, device=device) < train_cfg.duration_speaker_dropout
                        )
                        if train_cfg.speaker_inversion_enabled:
                            duration_has_speaker = ~duration_speaker_drop
                        else:
                            duration_has_speaker = has_speaker & (~duration_speaker_drop)
                        if duration_features is None:
                            raise RuntimeError("Duration feature batch tensor is missing.")
                        duration_features = set_duration_has_speaker_feature(
                            duration_features,
                            duration_has_speaker,
                        )
                    if (
                        not raw_model.cfg.use_duration_predictor
                        and not train_cfg.speaker_inversion_enabled
                    ):
                        ref_mask = ref_mask & use_speaker[:, None]
                        ref_latent = ref_latent * use_speaker[:, None, None].to(ref_latent.dtype)

                should_step = (accum_micro_steps % accum_steps) == 0
                sync_context = model.no_sync() if distributed and not should_step else nullcontext()
                with sync_context:
                    with (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if use_bf16
                        else nullcontext()
                    ):
                        if duration_only:
                            duration_pred = model(
                                x_t=None,
                                t=None,
                                text_input_ids=text_ids,
                                text_mask=text_mask,
                                ref_latent=ref_latent,
                                ref_mask=ref_mask,
                                caption_input_ids=caption_ids,
                                caption_mask=caption_mask,
                                latent_mask=None,
                                duration_features=duration_features,
                                duration_has_speaker=duration_has_speaker,
                                duration_has_caption=duration_has_caption,
                                duration_only=True,
                            )
                            v_pred = None
                        elif raw_model.cfg.use_duration_predictor:
                            v_pred, duration_pred = model(
                                x_t=x_t,
                                t=t,
                                text_input_ids=text_ids,
                                text_mask=text_mask,
                                ref_latent=ref_latent,
                                ref_mask=ref_mask,
                                caption_input_ids=caption_ids,
                                caption_mask=caption_mask,
                                latent_mask=x_mask,
                                text_condition_dropout=text_cond_drop,
                                speaker_condition_dropout=speaker_drop_for_model,
                                caption_condition_dropout=caption_drop_for_model,
                                duration_features=duration_features,
                                duration_has_speaker=duration_has_speaker,
                                duration_has_caption=duration_has_caption,
                            )
                        else:
                            v_pred = model(
                                x_t=x_t,
                                t=t,
                                text_input_ids=text_ids,
                                text_mask=text_mask,
                                ref_latent=ref_latent,
                                ref_mask=ref_mask,
                                caption_input_ids=caption_ids,
                                caption_mask=caption_mask,
                                latent_mask=x_mask,
                                text_condition_dropout=None,
                                speaker_condition_dropout=speaker_drop_for_model
                                if train_cfg.speaker_inversion_enabled
                                else None,
                                caption_condition_dropout=None,
                            )
                            duration_pred = None

                    rf_loss = torch.zeros((), device=device, dtype=torch.float)
                    if not duration_only:
                        if (
                            v_pred is None
                            or v_target is None
                            or x_mask is None
                            or x_mask_valid is None
                        ):
                            raise RuntimeError("RF training tensors are missing.")
                        v_pred = v_pred.float()
                        rf_loss = compute_rf_loss(
                            pred=v_pred,
                            target=v_target.float(),
                            loss_mask=x_mask,
                            valid_mask=x_mask_valid,
                            mode=train_cfg.rf_loss_mode,
                        )
                    duration_loss = torch.zeros((), device=device, dtype=torch.float)
                    duration_mae_frames = torch.zeros((), device=device, dtype=torch.float)
                    duration_group_totals = torch.zeros(
                        DURATION_CONDITION_GROUP_TOTAL_SIZE,
                        device=device,
                        dtype=torch.float64,
                    )
                    if raw_model.cfg.use_duration_predictor:
                        if duration_pred is None:
                            raise RuntimeError(
                                "Duration predictor is enabled but duration_pred is missing."
                            )
                        duration_target = torch.log1p(num_frames.float())
                        duration_loss_per_sample = F.huber_loss(
                            duration_pred.float(),
                            duration_target,
                            delta=float(train_cfg.duration_huber_delta),
                            reduction="none",
                        )
                        duration_loss = duration_loss_per_sample.mean()
                        pred_frames = torch.expm1(duration_pred.float()).clamp_min(0.0)
                        duration_mae_frames = (pred_frames - num_frames.float()).abs().mean()
                        if duration_only:
                            duration_group_totals = duration_condition_group_totals(
                                duration_loss_per_sample=duration_loss_per_sample,
                                pred_frames=pred_frames,
                                target_frames=num_frames.float(),
                                has_speaker=has_speaker,
                                has_caption=has_caption,
                            )
                    if duration_only:
                        loss = duration_loss
                    else:
                        loss = rf_loss + (float(train_cfg.duration_loss_weight) * duration_loss)
                    (loss / float(accum_steps)).backward()
                    if caption_warmup_active:
                        clear_non_caption_grads(raw_model)

                accum_loss += loss.detach()
                accum_rf_loss += rf_loss.detach()
                accum_duration_loss += duration_loss.detach()
                accum_duration_mae_frames += duration_mae_frames.detach()
                accum_duration_group_totals += duration_group_totals
                if not should_step:
                    continue

                step_loss = accum_loss / float(accum_steps)
                step_rf_loss = accum_rf_loss / float(accum_steps)
                step_duration_loss = accum_duration_loss / float(accum_steps)
                step_duration_mae_frames = accum_duration_mae_frames / float(accum_steps)
                step_duration_group_totals = accum_duration_group_totals.clone()
                accum_loss.zero_()
                accum_rf_loss.zero_()
                accum_duration_loss.zero_()
                accum_duration_mae_frames.zero_()
                accum_duration_group_totals.zero_()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip_norm
                )
                grad_norm_value = float(grad_norm)
                grad_norm_finite = math.isfinite(grad_norm_value)
                if grad_norm_finite:
                    last_grad_norm = grad_norm_value
                    optimizer.step()
                else:
                    # Gradients are identical across DDP ranks, so every rank
                    # skips the same step and stays in sync.
                    nonfinite_step_count += 1
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                step += 1
                if not grad_norm_finite and is_main_process:
                    progress.write(
                        f"warning: non-finite grad norm at step={step}; "
                        f"optimizer step skipped (total skipped={nonfinite_step_count})."
                    )
                if ema is not None and grad_norm_finite and step % train_cfg.ema_update_every == 0:
                    ema.update(raw_model)
                progress.update(step)
                if caption_warmup_active and step >= train_cfg.caption_warmup_steps:
                    caption_warmup_active = False
                    if is_main_process:
                        progress.write("caption warmup complete; all parameters are now updating.")

                if step % train_cfg.log_every == 0:
                    loss_value = reduce_mean(step_loss, world_size, distributed).item()
                    rf_loss_value = reduce_mean(step_rf_loss, world_size, distributed).item()
                    duration_loss_value = reduce_mean(
                        step_duration_loss, world_size, distributed
                    ).item()
                    duration_mae_frames_value = reduce_mean(
                        step_duration_mae_frames, world_size, distributed
                    ).item()
                    duration_group_metrics: dict[str, float] = {}
                    if duration_only:
                        duration_group_totals = reduce_sum(
                            step_duration_group_totals,
                            distributed,
                        )
                        duration_group_metrics = duration_condition_group_metrics(
                            duration_group_totals
                        )
                    lr_value = current_lr(optimizer)
                    progress_metrics: dict[str, float] = {
                        "loss": loss_value,
                        "rf": rf_loss_value,
                        "lr": lr_value,
                        "gnorm": last_grad_norm,
                    }
                    if raw_model.cfg.use_duration_predictor:
                        progress_metrics["dur"] = duration_loss_value
                        progress_metrics["dur_mae"] = duration_mae_frames_value
                        if duration_only:
                            progress_metrics["dur_sp"] = duration_group_metrics[
                                "duration_loss_speaker"
                            ]
                            progress_metrics["dur_no_sp"] = duration_group_metrics[
                                "duration_loss_no_speaker"
                            ]
                            progress_metrics["dur_cap"] = duration_group_metrics[
                                "duration_loss_caption"
                            ]
                            progress_metrics["dur_no_cap"] = duration_group_metrics[
                                "duration_loss_no_caption"
                            ]
                    progress.log(
                        step=step,
                        epoch=epoch,
                        epoch_step=epoch_step,
                        epoch_total=len(loader),
                        metrics=progress_metrics,
                        global_batch_size=global_batch_size,
                    )
                    if is_main_process:
                        if raw_model.cfg.use_duration_predictor:
                            message = (
                                f"step={step} loss={loss_value:.6f} rf={rf_loss_value:.6f} "
                                f"dur={duration_loss_value:.6f} "
                                f"dur_mae={duration_mae_frames_value:.2f}"
                            )
                            if duration_only:
                                group_suffix = duration_condition_group_log_suffix(
                                    duration_group_metrics
                                )
                                if group_suffix:
                                    message += f" {group_suffix}"
                            progress.write(f"{message} lr={lr_value:.3e}")
                        else:
                            progress.write(
                                f"step={step} loss={loss_value:.6f} rf={rf_loss_value:.6f} "
                                f"lr={lr_value:.3e}"
                            )
                        if wandb_run is not None:
                            metrics = {
                                "train/loss": loss_value,
                                "train/rf_loss": rf_loss_value,
                                "train/lr": lr_value,
                                "train/grad_norm": last_grad_norm,
                                "train/nonfinite_steps": float(nonfinite_step_count),
                            }
                            if raw_model.cfg.use_duration_predictor:
                                metrics["train/duration_loss"] = duration_loss_value
                                metrics["train/duration_mae_frames"] = duration_mae_frames_value
                                if duration_only:
                                    metrics.update(
                                        duration_condition_group_wandb_metrics(
                                            "train",
                                            duration_group_metrics,
                                        )
                                    )
                            wandb_run.log(metrics, step=step)
                        if tensorboard_writer is not None:
                            tensorboard_writer.add_scalar("train/loss", loss_value, step)
                            tensorboard_writer.add_scalar("train/rf_loss", rf_loss_value, step)
                            tensorboard_writer.add_scalar("train/learning_rate", lr_value, step)
                            tensorboard_writer.add_scalar("train/epoch", epoch, step)
                            tensorboard_writer.add_scalar("train/grad_norm", last_grad_norm, step)
                            tensorboard_writer.add_scalar(
                                "train/nonfinite_steps", nonfinite_step_count, step
                            )
                            if raw_model.cfg.use_duration_predictor:
                                tensorboard_writer.add_scalar(
                                    "train/duration_loss", duration_loss_value, step
                                )
                                tensorboard_writer.add_scalar(
                                    "train/duration_mae_frames",
                                    duration_mae_frames_value,
                                    step,
                                )

                if (
                    train_cfg.save_every > 0
                    and step % train_cfg.save_every == 0
                    and is_main_process
                ):
                    save_checkpoint(
                        _periodic_checkpoint_path(output_dir, step, train_cfg),
                        raw_model,
                        optimizer,
                        scheduler,
                        step,
                        model_cfg,
                        train_cfg,
                        base_init=base_init,
                        epoch=epoch,
                        ema=ema,
                    )
                    enforce_periodic_checkpoint_limit(
                        output_dir=output_dir,
                        keep_count=periodic_checkpoint_keep,
                    )

                if (
                    valid_loader is not None
                    and train_cfg.valid_every > 0
                    and step % train_cfg.valid_every == 0
                ):
                    valid_metrics = run_validation(
                        model=model,
                        loader=valid_loader,
                        train_cfg=train_cfg,
                        device=device,
                        use_bf16=use_bf16,
                        distributed=distributed,
                    )
                    if is_main_process:
                        if raw_model.cfg.use_duration_predictor:
                            message = (
                                "valid step={} loss={:.6f} rf={:.6f} dur={:.6f} dur_mae={:.2f}"
                            ).format(
                                step,
                                valid_metrics["loss"],
                                valid_metrics["rf_loss"],
                                valid_metrics["duration_loss"],
                                valid_metrics["duration_mae_frames"],
                            )
                            if duration_only:
                                group_suffix = duration_condition_group_log_suffix(valid_metrics)
                                if group_suffix:
                                    message += f" {group_suffix}"
                            progress.write(
                                "{} (samples={:.0f})".format(
                                    message,
                                    valid_metrics["num_samples"],
                                )
                            )
                        else:
                            progress.write(
                                ("valid step={} loss={:.6f} rf={:.6f} (samples={:.0f})").format(
                                    step,
                                    valid_metrics["loss"],
                                    valid_metrics["rf_loss"],
                                    valid_metrics["num_samples"],
                                )
                            )
                        if wandb_run is not None:
                            metrics = {
                                "valid/loss": valid_metrics["loss"],
                                "valid/rf_loss": valid_metrics["rf_loss"],
                            }
                            if raw_model.cfg.use_duration_predictor:
                                metrics["valid/duration_loss"] = valid_metrics["duration_loss"]
                                metrics["valid/duration_mae_frames"] = valid_metrics[
                                    "duration_mae_frames"
                                ]
                                if duration_only:
                                    metrics.update(
                                        duration_condition_group_wandb_metrics(
                                            "valid",
                                            valid_metrics,
                                        )
                                    )
                            wandb_run.log(metrics, step=step)
                        if tensorboard_writer is not None:
                            tensorboard_writer.add_scalar(
                                "valid/loss", valid_metrics["loss"], step
                            )
                            tensorboard_writer.add_scalar(
                                "valid/rf_loss", valid_metrics["rf_loss"], step
                            )
                            if raw_model.cfg.use_duration_predictor:
                                tensorboard_writer.add_scalar(
                                    "valid/duration_loss", valid_metrics["duration_loss"], step
                                )
                                tensorboard_writer.add_scalar(
                                    "valid/duration_mae_frames",
                                    valid_metrics["duration_mae_frames"],
                                    step,
                                )
                        best_val_checkpoints, best_path = maybe_save_best_val_loss_checkpoint(
                            output_dir=output_dir,
                            checkpoints=best_val_checkpoints,
                            keep_best_n=train_cfg.checkpoint_best_n,
                            val_loss=float(valid_metrics["loss"]),
                            step=step,
                            model=raw_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            model_cfg=model_cfg,
                            train_cfg=train_cfg,
                            base_init=base_init,
                            epoch=epoch,
                            ema=ema,
                        )
                        if best_path is not None:
                            progress.write(
                                "saved best val checkpoint: {} (loss={:.6f})".format(
                                    best_path.name,
                                    float(valid_metrics["loss"]),
                                )
                            )

                if step >= train_cfg.max_steps:
                    break
            # A broken iteration may still own one prefetched GPU batch.
            # Closing the generator releases it before final validation/save.
            device_batches.close()
            epoch_checkpoint_path: Path | None = None
            if (
                train_cfg.save_every_epochs > 0
                and epoch % train_cfg.save_every_epochs == 0
                and step < train_cfg.max_steps
                and is_main_process
            ):
                epoch_checkpoint_path = _periodic_checkpoint_path(output_dir, step, train_cfg)
                save_checkpoint(
                    epoch_checkpoint_path,
                    raw_model,
                    optimizer,
                    scheduler,
                    step,
                    model_cfg,
                    train_cfg,
                    base_init=base_init,
                    epoch=epoch,
                    ema=ema,
                )
                enforce_periodic_checkpoint_limit(
                    output_dir=output_dir,
                    keep_count=periodic_checkpoint_keep,
                )
            if (
                train_cfg.sample_every_epochs > 0
                and epoch % train_cfg.sample_every_epochs == 0
                and step < train_cfg.max_steps
                and is_main_process
            ):
                # Sampling reads weights from a checkpoint file; on epochs
                # where no periodic checkpoint was saved, write a temporary
                # one and remove it after sampling.
                sample_checkpoint_path = epoch_checkpoint_path
                temp_sample_checkpoint: Path | None = None
                if sample_checkpoint_path is None:
                    temp_sample_checkpoint = (
                        output_dir / f"checkpoint_sample_tmp_{step:07d}.pt"
                    )
                    save_checkpoint(
                        temp_sample_checkpoint,
                        raw_model,
                        optimizer,
                        scheduler,
                        step,
                        model_cfg,
                        train_cfg,
                        base_init=base_init,
                        epoch=epoch,
                        ema=ema,
                    )
                    sample_checkpoint_path = temp_sample_checkpoint
                progress.write(
                    f"generating comparison samples for epoch={epoch} step={step}..."
                )
                if device.type == "cuda":
                    # The sampling subprocess may share this GPU; release
                    # cached blocks so it can allocate its model + codec.
                    torch.cuda.empty_cache()
                try:
                    sample_ok, sample_dir = run_epoch_test_inference(
                        checkpoint_path=sample_checkpoint_path,
                        manifest_path=train_cfg.manifest_path,
                        output_dir=output_dir,
                        epoch=epoch,
                        step=step,
                        train_cfg=train_cfg,
                    )
                finally:
                    if temp_sample_checkpoint is not None:
                        temp_sample_checkpoint.unlink(missing_ok=True)
                if sample_ok:
                    progress.write(f"saved comparison samples: {sample_dir}")
                else:
                    progress.write(
                        f"warning: comparison sample generation failed; see {sample_dir / 'inference.log'}"
                    )
            if distributed and train_cfg.sample_every_epochs > 0:
                dist.barrier()

        if (
            valid_loader is not None
            and train_cfg.valid_every > 0
            and step % train_cfg.valid_every != 0
        ):
            valid_metrics = run_validation(
                model=model,
                loader=valid_loader,
                train_cfg=train_cfg,
                device=device,
                use_bf16=use_bf16,
                distributed=distributed,
            )
            if is_main_process:
                if raw_model.cfg.use_duration_predictor:
                    message = (
                        "valid final step={} loss={:.6f} rf={:.6f} dur={:.6f} dur_mae={:.2f}"
                    ).format(
                        step,
                        valid_metrics["loss"],
                        valid_metrics["rf_loss"],
                        valid_metrics["duration_loss"],
                        valid_metrics["duration_mae_frames"],
                    )
                    if duration_only:
                        group_suffix = duration_condition_group_log_suffix(valid_metrics)
                        if group_suffix:
                            message += f" {group_suffix}"
                    progress.write(
                        "{} (samples={:.0f})".format(
                            message,
                            valid_metrics["num_samples"],
                        )
                    )
                else:
                    progress.write(
                        ("valid final step={} loss={:.6f} rf={:.6f} (samples={:.0f})").format(
                            step,
                            valid_metrics["loss"],
                            valid_metrics["rf_loss"],
                            valid_metrics["num_samples"],
                        )
                    )
                if wandb_run is not None:
                    metrics = {
                        "valid/loss": valid_metrics["loss"],
                        "valid/rf_loss": valid_metrics["rf_loss"],
                    }
                    if raw_model.cfg.use_duration_predictor:
                        metrics["valid/duration_loss"] = valid_metrics["duration_loss"]
                        metrics["valid/duration_mae_frames"] = valid_metrics["duration_mae_frames"]
                        if duration_only:
                            metrics.update(
                                duration_condition_group_wandb_metrics(
                                    "valid",
                                    valid_metrics,
                                )
                            )
                    wandb_run.log(metrics, step=step)
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("valid/loss", valid_metrics["loss"], step)
                    tensorboard_writer.add_scalar("valid/rf_loss", valid_metrics["rf_loss"], step)
                    if raw_model.cfg.use_duration_predictor:
                        tensorboard_writer.add_scalar(
                            "valid/duration_loss", valid_metrics["duration_loss"], step
                        )
                        tensorboard_writer.add_scalar(
                            "valid/duration_mae_frames",
                            valid_metrics["duration_mae_frames"],
                            step,
                        )
                best_val_checkpoints, best_path = maybe_save_best_val_loss_checkpoint(
                    output_dir=output_dir,
                    checkpoints=best_val_checkpoints,
                    keep_best_n=train_cfg.checkpoint_best_n,
                    val_loss=float(valid_metrics["loss"]),
                    step=step,
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    base_init=base_init,
                    epoch=epoch,
                    ema=ema,
                )
                if best_path is not None:
                    progress.write(
                        "saved best val checkpoint: {} (loss={:.6f})".format(
                            best_path.name,
                            float(valid_metrics["loss"]),
                        )
                    )

        if is_main_process:
            save_checkpoint(
                _final_checkpoint_path(output_dir, train_cfg),
                raw_model,
                optimizer,
                scheduler,
                step,
                model_cfg,
                train_cfg,
                base_init=base_init,
                epoch=epoch,
                ema=ema,
            )
            if wandb_run is not None:
                wandb_run.summary["train/final_step"] = step
            progress.write(f"Training finished at step={step}.")
    finally:
        if progress is not None:
            progress.close()
        if device.type == "cuda" and is_main_process:
            peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
            print(f"Peak CUDA memory allocated: {peak_gib:.2f} GiB")
        if wandb_run is not None:
            wandb_run.finish()
        if tensorboard_writer is not None:
            tensorboard_writer.flush()
            tensorboard_writer.close()
        if tensorboard_process is not None and tensorboard_process.poll() is None:
            tensorboard_process.terminate()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
