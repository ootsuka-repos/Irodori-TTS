"""Deterministic acoustic segmentation for audio outside speech VAD regions.

The segmenter never decides whether audio is worth keeping.  Energy and spectral
features are used only to choose boundaries, and the returned primitives cover
the input exactly once from zero to its original duration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as torch_f
import torchaudio

ACOUSTIC_SEGMENTATION_VERSION = "logmel_pause_v1"


@dataclass(frozen=True)
class AcousticSegmentationConfig:
    """Parameters for natural acoustic boundary detection."""

    analysis_sample_rate: int = 16_000
    frame_length_ms: float = 25.0
    frame_hop_ms: float = 10.0
    n_fft: int = 512
    n_mels: int = 40
    f_min: float = 50.0
    f_max: float = 7_600.0
    rms_smoothing_ms: float = 70.0
    novelty_context_ms: float = 300.0
    local_envelope_seconds: float = 4.0
    pause_depth_db: float = 18.0
    pause_min_ms: float = 200.0
    preferred_min_seconds: float = 1.2
    max_seconds: float = 20.0
    peak_min_distance_ms: float = 500.0
    strong_peak_quantile: float = 0.99
    candidate_peak_quantile: float = 0.90
    minimum_novelty_score: float = 0.08
    valley_context_seconds: float = 1.0
    valley_candidate_quantile: float = 0.60

    def __post_init__(self) -> None:
        positive_values = {
            "analysis_sample_rate": self.analysis_sample_rate,
            "frame_length_ms": self.frame_length_ms,
            "frame_hop_ms": self.frame_hop_ms,
            "n_fft": self.n_fft,
            "n_mels": self.n_mels,
            "f_max": self.f_max,
            "rms_smoothing_ms": self.rms_smoothing_ms,
            "novelty_context_ms": self.novelty_context_ms,
            "local_envelope_seconds": self.local_envelope_seconds,
            "pause_depth_db": self.pause_depth_db,
            "pause_min_ms": self.pause_min_ms,
            "preferred_min_seconds": self.preferred_min_seconds,
            "max_seconds": self.max_seconds,
            "peak_min_distance_ms": self.peak_min_distance_ms,
            "valley_context_seconds": self.valley_context_seconds,
        }
        for name, value in positive_values.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.f_min) or self.f_min < 0:
            raise ValueError("f_min must be non-negative and finite")
        if self.f_min >= self.f_max:
            raise ValueError("f_min must be lower than f_max")
        if self.f_max > self.analysis_sample_rate / 2:
            raise ValueError("f_max cannot exceed the analysis Nyquist frequency")
        if self.n_fft < self.frame_samples:
            raise ValueError("n_fft cannot be shorter than frame_length_ms")
        if self.max_seconds < 2 * self.preferred_min_seconds:
            raise ValueError("max_seconds must be at least twice preferred_min_seconds")
        for name, value in {
            "strong_peak_quantile": self.strong_peak_quantile,
            "candidate_peak_quantile": self.candidate_peak_quantile,
            "valley_candidate_quantile": self.valley_candidate_quantile,
        }.items():
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.candidate_peak_quantile > self.strong_peak_quantile:
            raise ValueError("candidate_peak_quantile cannot exceed strong_peak_quantile")
        if not math.isfinite(self.minimum_novelty_score) or self.minimum_novelty_score < 0:
            raise ValueError("minimum_novelty_score must be non-negative and finite")

    @property
    def frame_samples(self) -> int:
        return max(1, round(self.analysis_sample_rate * self.frame_length_ms / 1_000))

    @property
    def hop_samples(self) -> int:
        return max(1, round(self.analysis_sample_rate * self.frame_hop_ms / 1_000))

    @property
    def hop_seconds(self) -> float:
        return self.hop_samples / self.analysis_sample_rate


@dataclass(frozen=True)
class AcousticFrameFeatures:
    """Frame-aligned features used to choose acoustic boundaries."""

    log_mel_db: torch.Tensor
    rms_dbfs: torch.Tensor
    spectral_flux: torch.Tensor
    mel_cosine_novelty: torch.Tensor
    rms_context_step_db: torch.Tensor
    hop_seconds: float

    @property
    def frames(self) -> int:
        return int(self.rms_dbfs.numel())


@dataclass(frozen=True)
class AcousticPrimitive:
    """One contiguous, relative interval from a naturally segmented waveform."""

    start: float
    end: float
    left_boundary_reason: str
    left_boundary_score: float
    right_boundary_reason: str
    right_boundary_score: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def boundary_reason(self) -> str:
        """Reason for this primitive's terminating boundary."""
        return self.right_boundary_reason

    @property
    def boundary_score(self) -> float:
        """Score for this primitive's terminating boundary."""
        return self.right_boundary_score


@dataclass(frozen=True)
class _Boundary:
    time: float
    frame_index: int | None
    reason: str
    score: float
    hard: bool = False


def _validate_waveform(waveform: torch.Tensor, sample_rate: int) -> None:
    if waveform.ndim != 1:
        raise ValueError("waveform must be a one-dimensional mono tensor")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not waveform.is_floating_point():
        raise TypeError("waveform must use a floating-point dtype")
    if waveform.numel() and not bool(torch.isfinite(waveform).all()):
        raise ValueError("waveform must contain only finite samples")


def _odd_frames(seconds: float, hop_seconds: float) -> int:
    frames = max(1, round(seconds / hop_seconds))
    return frames if frames % 2 else frames + 1


def _same_pool(values: torch.Tensor, kernel_size: int, *, mode: str) -> torch.Tensor:
    if values.numel() == 0 or kernel_size <= 1:
        return values.clone()
    kernel_size = min(kernel_size, 2 * int(values.numel()) - 1)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size <= 1:
        return values.clone()
    radius = kernel_size // 2
    padded = torch_f.pad(values[None, None], (radius, radius), mode="replicate")
    if mode == "mean":
        return torch_f.avg_pool1d(padded, kernel_size, stride=1).flatten()
    if mode == "max":
        return torch_f.max_pool1d(padded, kernel_size, stride=1).flatten()
    raise ValueError(f"Unsupported pool mode: {mode}")


def _align_frames(values: torch.Tensor, frames: int) -> torch.Tensor:
    if int(values.numel()) >= frames:
        return values[:frames]
    if values.numel() == 0:
        return torch.full((frames,), -120.0, dtype=torch.float, device=values.device)
    extension = values[-1].expand(frames - int(values.numel()))
    return torch.cat((values, extension))


def extract_acoustic_features(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    config: AcousticSegmentationConfig | None = None,
) -> AcousticFrameFeatures:
    """Calculate frame features without applying any keep/drop energy threshold."""
    config = config or AcousticSegmentationConfig()
    _validate_waveform(waveform, sample_rate)
    if waveform.numel() == 0:
        empty = torch.empty(0, dtype=torch.float, device=waveform.device)
        return AcousticFrameFeatures(
            log_mel_db=empty.reshape(config.n_mels, 0),
            rms_dbfs=empty,
            spectral_flux=empty,
            mel_cosine_novelty=empty,
            rms_context_step_db=empty,
            hop_seconds=config.hop_seconds,
        )

    analysis = waveform.detach().to(dtype=torch.float)
    if sample_rate != config.analysis_sample_rate:
        analysis = torchaudio.functional.resample(
            analysis,
            sample_rate,
            config.analysis_sample_rate,
        )
    minimum_samples = max(config.n_fft // 2 + 1, config.frame_samples)
    if analysis.numel() < minimum_samples:
        analysis = torch_f.pad(analysis, (0, minimum_samples - int(analysis.numel())))

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.analysis_sample_rate,
        n_fft=config.n_fft,
        win_length=config.frame_samples,
        hop_length=config.hop_samples,
        f_min=config.f_min,
        f_max=config.f_max,
        n_mels=config.n_mels,
        power=2.0,
        center=True,
        pad_mode="constant",
    ).to(device=analysis.device, dtype=analysis.dtype)
    mel_power = mel_transform(analysis).clamp_min(1e-12)
    frames = int(mel_power.shape[1])
    log_mel_db = 10.0 * torch.log10(mel_power)

    rms_padding = config.frame_samples // 2
    squared = torch_f.pad(analysis.square(), (rms_padding, rms_padding))
    rms_power = torch_f.avg_pool1d(
        squared[None, None],
        kernel_size=config.frame_samples,
        stride=config.hop_samples,
    ).flatten()
    rms_dbfs = 10.0 * torch.log10(rms_power.clamp_min(1e-12))
    rms_dbfs = _align_frames(rms_dbfs, frames)
    smoothing_frames = _odd_frames(config.rms_smoothing_ms / 1_000, config.hop_seconds)
    rms_dbfs = _same_pool(rms_dbfs, smoothing_frames, mode="mean")

    mel_distribution = mel_power / mel_power.sum(dim=0, keepdim=True).clamp_min(1e-12)
    spectral_flux = torch.zeros(frames, dtype=analysis.dtype, device=analysis.device)
    if frames > 1:
        positive_delta = torch.relu(mel_distribution[:, 1:] - mel_distribution[:, :-1])
        spectral_flux[1:] = positive_delta.square().sum(dim=0).sqrt()
    flux_smoothing = _odd_frames(0.05, config.hop_seconds)
    spectral_flux = _same_pool(spectral_flux, flux_smoothing, mode="mean")

    novelty = torch.zeros(frames, dtype=analysis.dtype, device=analysis.device)
    rms_step = torch.zeros_like(novelty)
    context_frames = max(1, round(config.novelty_context_ms / 1_000 / config.hop_seconds))
    if frames > 2 * context_frames:
        frame_index = torch.arange(context_frames, frames - context_frames, device=analysis.device)
        mel_prefix = torch_f.pad(torch.cumsum(mel_distribution, dim=1), (1, 0))
        left_mel = (
            mel_prefix[:, frame_index] - mel_prefix[:, frame_index - context_frames]
        ) / context_frames
        right_mel = (
            mel_prefix[:, frame_index + context_frames] - mel_prefix[:, frame_index]
        ) / context_frames
        novelty[frame_index] = 1.0 - torch_f.cosine_similarity(
            left_mel.transpose(0, 1),
            right_mel.transpose(0, 1),
            dim=1,
            eps=1e-8,
        )

        rms_prefix = torch_f.pad(torch.cumsum(rms_dbfs, dim=0), (1, 0))
        left_rms = (
            rms_prefix[frame_index] - rms_prefix[frame_index - context_frames]
        ) / context_frames
        right_rms = (
            rms_prefix[frame_index + context_frames] - rms_prefix[frame_index]
        ) / context_frames
        rms_step[frame_index] = (left_rms - right_rms).abs()

    return AcousticFrameFeatures(
        log_mel_db=log_mel_db,
        rms_dbfs=rms_dbfs,
        spectral_flux=spectral_flux,
        mel_cosine_novelty=novelty,
        rms_context_step_db=rms_step,
        hop_seconds=config.hop_seconds,
    )


def _local_peak_indices(
    values: torch.Tensor,
    *,
    radius: int,
    threshold: float,
    valid_start: int = 0,
    valid_end: int | None = None,
) -> list[int]:
    if values.numel() == 0:
        return []
    valid_end = int(values.numel()) if valid_end is None else valid_end
    local_max = _same_pool(values, 2 * radius + 1, mode="max")
    mask = (values >= local_max - 1e-8) & (values >= threshold)
    frame_indices = torch.where(mask)[0].detach().cpu().tolist()
    frame_indices = [index for index in frame_indices if valid_start <= index < valid_end]
    collapsed: list[int] = []
    for index in frame_indices:
        if not collapsed or index - collapsed[-1] > radius:
            collapsed.append(index)
            continue
        previous = collapsed[-1]
        if float(values[index]) > float(values[previous]) + 1e-9:
            collapsed[-1] = index
    return collapsed


def _pause_boundaries(
    features: AcousticFrameFeatures,
    config: AcousticSegmentationConfig,
) -> tuple[list[_Boundary], torch.Tensor]:
    envelope_frames = _odd_frames(config.local_envelope_seconds, features.hop_seconds)
    local_high = _same_pool(features.rms_dbfs, envelope_frames, mode="max")
    quiet_depth = local_high - features.rms_dbfs
    quiet_indices = torch.where(quiet_depth >= config.pause_depth_db)[0].detach().cpu().tolist()
    required_frames = max(1, math.ceil(config.pause_min_ms / 1_000 / features.hop_seconds))
    boundaries: list[_Boundary] = []
    if not quiet_indices:
        return boundaries, quiet_depth

    run_start = previous = quiet_indices[0]
    for next_index in [*quiet_indices[1:], None]:
        if next_index is not None and next_index == previous + 1:
            previous = next_index
            continue
        if previous - run_start + 1 >= required_frames:
            minimum_offset = int(torch.argmin(features.rms_dbfs[run_start : previous + 1]))
            frame_index = run_start + minimum_offset
            boundaries.append(
                _Boundary(
                    time=frame_index * features.hop_seconds,
                    frame_index=frame_index,
                    reason="pause_valley",
                    score=2.0 + float(quiet_depth[frame_index]) / config.pause_depth_db,
                )
            )
        if next_index is None:
            break
        run_start = previous = next_index
    return boundaries, quiet_depth


def _spectral_candidates(
    features: AcousticFrameFeatures,
    quiet_depth: torch.Tensor,
    config: AcousticSegmentationConfig,
) -> tuple[list[_Boundary], list[_Boundary]]:
    if features.frames == 0:
        return [], []
    flux_scale = torch.quantile(features.spectral_flux, 0.90).clamp_min(1e-8)
    combined = (
        features.mel_cosine_novelty
        + 0.12 * torch.clamp(features.rms_context_step_db / 12.0, 0.0, 1.0)
        + 0.08 * torch.clamp(features.spectral_flux / flux_scale, 0.0, 2.0)
    )
    context_frames = max(
        1,
        round(config.novelty_context_ms / 1_000 / features.hop_seconds),
    )
    peak_radius = max(1, round(config.peak_min_distance_ms / 1_000 / features.hop_seconds))
    valid_values = combined[context_frames : features.frames - context_frames]
    if valid_values.numel() == 0:
        return [], []
    candidate_threshold = max(
        config.minimum_novelty_score,
        float(torch.quantile(valid_values, config.candidate_peak_quantile)),
    )
    strong_threshold = max(
        config.minimum_novelty_score,
        float(torch.quantile(valid_values, config.strong_peak_quantile)),
    )
    indices = _local_peak_indices(
        combined,
        radius=peak_radius,
        threshold=candidate_threshold,
        valid_start=context_frames,
        valid_end=features.frames - context_frames,
    )
    candidates = [
        _Boundary(
            time=index * features.hop_seconds,
            frame_index=index,
            reason="spectral_change",
            score=0.5 + float(combined[index]) / max(candidate_threshold, 1e-8),
        )
        for index in indices
    ]
    strong = [
        boundary
        for boundary in candidates
        if float(combined[boundary.frame_index]) >= strong_threshold
    ]

    valley_mean_frames = _odd_frames(config.valley_context_seconds, features.hop_seconds)
    local_mean = _same_pool(features.rms_dbfs, valley_mean_frames, mode="mean")
    valley_strength = quiet_depth / config.pause_depth_db + torch.clamp(
        (local_mean - features.rms_dbfs) / 6.0,
        min=0.0,
    )
    valley_threshold = float(torch.quantile(valley_strength, config.valley_candidate_quantile))
    valley_indices = _local_peak_indices(
        valley_strength,
        radius=peak_radius,
        threshold=max(0.05, valley_threshold),
        valid_start=1,
        valid_end=max(1, features.frames - 1),
    )
    candidates.extend(
        _Boundary(
            time=index * features.hop_seconds,
            frame_index=index,
            reason="local_valley",
            score=0.4 + float(valley_strength[index]),
        )
        for index in valley_indices
    )
    return strong, candidates


def _deduplicate_boundaries(boundaries: list[_Boundary]) -> list[_Boundary]:
    by_frame: dict[int, _Boundary] = {}
    without_frames: list[_Boundary] = []
    for boundary in boundaries:
        if boundary.frame_index is None:
            without_frames.append(boundary)
            continue
        previous = by_frame.get(boundary.frame_index)
        if previous is None or boundary.score > previous.score:
            by_frame[boundary.frame_index] = boundary
    return sorted([*without_frames, *by_frame.values()], key=lambda item: item.time)


def _merge_short_fragments(
    boundaries: list[_Boundary],
    *,
    preferred_min_seconds: float,
) -> None:
    # Removing a boundary can only shorten the pair immediately to its left,
    # so resuming one step back is equivalent to rescanning from the start.
    left_index = 0
    while len(boundaries) > 2 and left_index < len(boundaries) - 1:
        right_index = left_index + 1
        if (
            boundaries[right_index].time - boundaries[left_index].time
            >= preferred_min_seconds - 1e-9
        ):
            left_index += 1
            continue
        left = boundaries[left_index]
        right = boundaries[right_index]
        if left.hard:
            remove_index = right_index
        elif right.hard:
            remove_index = left_index
        elif left.score <= right.score:
            remove_index = left_index
        else:
            remove_index = right_index
        boundaries.pop(remove_index)
        left_index = max(0, left_index - 1)


def _fallback_valley(
    left: _Boundary,
    right: _Boundary,
    features: AcousticFrameFeatures,
) -> _Boundary:
    span = right.time - left.time
    search_start = left.time + 0.25 * span
    search_end = right.time - 0.25 * span
    first_frame = max(1, math.ceil(search_start / features.hop_seconds))
    last_frame = min(features.frames - 2, math.floor(search_end / features.hop_seconds))
    if first_frame > last_frame:
        midpoint = (left.time + right.time) / 2
        frame_index = min(
            features.frames - 2,
            max(1, round(midpoint / features.hop_seconds)),
        )
    else:
        minimum_offset = int(torch.argmin(features.rms_dbfs[first_frame : last_frame + 1]))
        frame_index = first_frame + minimum_offset
    return _Boundary(
        time=frame_index * features.hop_seconds,
        frame_index=frame_index,
        reason="long_local_valley",
        score=0.1,
    )


def _split_long_fragments(
    boundaries: list[_Boundary],
    candidates: list[_Boundary],
    features: AcousticFrameFeatures,
    config: AcousticSegmentationConfig,
) -> None:
    candidate_pool = _deduplicate_boundaries(candidates)
    # Inserting a boundary only affects the pair being split, so the scan
    # never needs to restart from the beginning.
    left_index = 0
    while left_index < len(boundaries) - 1:
        right_index = left_index + 1
        if boundaries[right_index].time - boundaries[left_index].time <= config.max_seconds:
            left_index += 1
            continue

        left = boundaries[left_index]
        right = boundaries[right_index]
        choices: list[tuple[float, float, float, _Boundary]] = []
        midpoint = (left.time + right.time) / 2
        for candidate in candidate_pool:
            if candidate.time - left.time < config.preferred_min_seconds:
                continue
            if right.time - candidate.time < config.preferred_min_seconds:
                continue
            relative_position = (candidate.time - left.time) / (right.time - left.time)
            balance = 0.55 + 0.45 * math.sin(math.pi * relative_position)
            choices.append(
                (
                    candidate.score * balance,
                    -abs(candidate.time - midpoint),
                    -candidate.time,
                    candidate,
                )
            )
        if choices:
            chosen = max(choices, key=lambda item: item[:3])[3]
        else:
            chosen = _fallback_valley(left, right, features)
        boundaries.insert(right_index, chosen)


def segment_acoustic_primitives(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    config: AcousticSegmentationConfig | None = None,
) -> list[AcousticPrimitive]:
    """Split mono audio at natural acoustic boundaries without dropping audio.

    Times are relative to the input waveform.  The intervals are contiguous and
    span ``[0, waveform.numel() / sample_rate]`` exactly.
    """
    config = config or AcousticSegmentationConfig()
    _validate_waveform(waveform, sample_rate)
    if waveform.numel() == 0:
        return []
    duration = int(waveform.numel()) / sample_rate
    if duration <= config.preferred_min_seconds:
        return [
            AcousticPrimitive(
                start=0.0,
                end=duration,
                left_boundary_reason="gap_start",
                left_boundary_score=1.0,
                right_boundary_reason="gap_end",
                right_boundary_score=1.0,
            )
        ]

    features = extract_acoustic_features(waveform, sample_rate, config=config)
    pause_boundaries, quiet_depth = _pause_boundaries(features, config)
    strong_spectral, candidate_boundaries = _spectral_candidates(
        features,
        quiet_depth,
        config,
    )
    start = _Boundary(0.0, None, "gap_start", 1.0, hard=True)
    end = _Boundary(duration, None, "gap_end", 1.0, hard=True)
    primary = _deduplicate_boundaries([*pause_boundaries, *strong_spectral])
    boundaries = [
        start,
        *[
            boundary
            for boundary in primary
            if config.preferred_min_seconds
            <= boundary.time
            <= duration - config.preferred_min_seconds
        ],
        end,
    ]
    _merge_short_fragments(boundaries, preferred_min_seconds=config.preferred_min_seconds)
    _split_long_fragments(
        boundaries,
        [*pause_boundaries, *candidate_boundaries],
        features,
        config,
    )

    primitives: list[AcousticPrimitive] = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        primitives.append(
            AcousticPrimitive(
                start=left.time,
                end=right.time,
                left_boundary_reason=left.reason,
                left_boundary_score=left.score,
                right_boundary_reason=right.reason,
                right_boundary_score=right.score,
            )
        )
    return primitives
