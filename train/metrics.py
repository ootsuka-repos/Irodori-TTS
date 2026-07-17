from __future__ import annotations

import torch

DURATION_CONDITION_GROUPS = (
    "speaker",
    "no_speaker",
    "caption",
    "no_caption",
    "speaker_caption",
    "speaker_no_caption",
    "no_speaker_caption",
    "no_speaker_no_caption",
)
DURATION_CONDITION_GROUP_TOTAL_SIZE = len(DURATION_CONDITION_GROUPS) * 3

_LOG_GROUPS = (
    ("sp", "speaker"),
    ("no_sp", "no_speaker"),
    ("cap", "caption"),
    ("no_cap", "no_caption"),
    ("sp_cap", "speaker_caption"),
    ("sp_no_cap", "speaker_no_caption"),
    ("no_sp_cap", "no_speaker_caption"),
    ("no_sp_no_cap", "no_speaker_no_caption"),
)


def duration_condition_group_totals(
    *,
    duration_loss_per_sample: torch.Tensor,
    pred_frames: torch.Tensor,
    target_frames: torch.Tensor,
    has_speaker: torch.Tensor | None,
    has_caption: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate all condition groups without host/device synchronization."""
    device = duration_loss_per_sample.device
    batch_size = int(duration_loss_per_sample.numel())
    absent = torch.zeros((batch_size,), device=device, dtype=torch.bool)
    speaker = has_speaker.to(device=device, dtype=torch.bool) if has_speaker is not None else None
    caption = has_caption.to(device=device, dtype=torch.bool) if has_caption is not None else None
    masks = (
        speaker if speaker is not None else absent,
        ~speaker if speaker is not None else absent,
        caption if caption is not None else absent,
        ~caption if caption is not None else absent,
        speaker & caption if speaker is not None and caption is not None else absent,
        speaker & (~caption) if speaker is not None and caption is not None else absent,
        (~speaker) & caption if speaker is not None and caption is not None else absent,
        (~speaker) & (~caption) if speaker is not None and caption is not None else absent,
    )
    weights = torch.stack(masks).to(dtype=torch.float64)
    loss = duration_loss_per_sample.detach().reshape(-1).double()
    mae = (pred_frames.float() - target_frames.float()).abs().detach().reshape(-1).double()
    totals = torch.stack(
        (
            (weights * loss.unsqueeze(0)).sum(dim=1),
            (weights * mae.unsqueeze(0)).sum(dim=1),
            weights.sum(dim=1),
        ),
        dim=1,
    )
    return totals.flatten()


def duration_condition_group_metrics(totals: torch.Tensor) -> dict[str, float]:
    """Convert aggregate tensors to metrics with one accelerator synchronization."""
    values = totals.detach().cpu().tolist()
    metrics: dict[str, float] = {}
    for group_index, group_name in enumerate(DURATION_CONDITION_GROUPS):
        offset = group_index * 3
        count = max(float(values[offset + 2]), 0.0)
        metrics[f"duration_loss_{group_name}"] = (
            float(values[offset] / count) if count > 0.0 else 0.0
        )
        metrics[f"duration_mae_frames_{group_name}"] = (
            float(values[offset + 1] / count) if count > 0.0 else 0.0
        )
        metrics[f"duration_samples_{group_name}"] = count
    return metrics


def duration_condition_group_log_suffix(metrics: dict[str, float]) -> str:
    chunks: list[str] = []
    for label, group in _LOG_GROUPS:
        count = metrics.get(f"duration_samples_{group}", 0.0)
        if count <= 0.0:
            continue
        chunks.append(
            "{}={:.6f} mae_{}={:.2f} n_{}={:.0f}".format(
                f"dur_{label}",
                metrics[f"duration_loss_{group}"],
                label,
                metrics[f"duration_mae_frames_{group}"],
                label,
                count,
            )
        )
    return " ".join(chunks)


def duration_condition_group_wandb_metrics(
    prefix: str,
    metrics: dict[str, float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for group_name in DURATION_CONDITION_GROUPS:
        for metric_name in (
            f"duration_loss_{group_name}",
            f"duration_mae_frames_{group_name}",
            f"duration_samples_{group_name}",
        ):
            out[f"{prefix}/{metric_name}"] = metrics[metric_name]
    return out
