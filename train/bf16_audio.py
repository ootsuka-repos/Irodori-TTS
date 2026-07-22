from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

BF16 = torch.bfloat16

_DFT_CACHE: dict[
    tuple[str, int | None, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_MEL_CACHE: dict[
    tuple[str, int | None, int, int, int, float, float | None],
    torch.Tensor,
] = {}


def _require_bf16(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype is not BF16:
        raise RuntimeError(f"{name} must be bf16, got {tensor.dtype}")


@torch.no_grad()
def _dft_kernels(
    device: torch.device,
    n_fft: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (device.type, device.index, n_fft)
    cached = _DFT_CACHE.get(key)
    if cached is not None:
        return cached

    # Generate constants accurately on the CPU once; only final bf16 tensors enter
    # the CUDA training path and remain cached there.
    sample = np.arange(n_fft, dtype=np.float64)
    frequency = np.arange(n_fft // 2 + 1, dtype=np.float64)
    phase = (2.0 * math.pi / n_fft) * frequency[:, None] * sample[None, :]
    real_values = np.ascontiguousarray(np.cos(phase).T)
    imag_values = np.ascontiguousarray(-np.sin(phase).T)
    window_values = np.hanning(n_fft + 1)[:-1].copy()
    real_kernel = torch.tensor(real_values, device=device, dtype=BF16)
    imag_kernel = torch.tensor(imag_values, device=device, dtype=BF16)
    window = torch.tensor(window_values, device=device, dtype=BF16)
    cached = (real_kernel, imag_kernel, window)
    _DFT_CACHE[key] = cached
    return cached


def bf16_stft_parts(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    match_stride: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex-free STFT returning bf16 real/imaginary tensors as (B, C, F, T)."""
    _require_bf16("waveform", waveform)
    if waveform.ndim != 3:
        raise ValueError(f"waveform must have shape (B,C,T), got {tuple(waveform.shape)}")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError(f"n_fft and hop_length must be positive, got {n_fft}, {hop_length}")

    audio = waveform
    if match_stride:
        length = int(audio.shape[-1])
        right_pad = math.ceil(length / hop_length) * hop_length - length
        stride_pad = (n_fft - hop_length) // 2
        audio = F.pad(
            audio,
            (stride_pad, stride_pad + right_pad),
            mode="reflect",
        )

    # Match torch.stft(center=True, pad_mode="reflect").
    audio = F.pad(audio, (n_fft // 2, n_fft // 2), mode="reflect")
    frames = audio.unfold(-1, n_fft, hop_length)
    real_kernel, imag_kernel, window = _dft_kernels(audio.device, n_fft)
    frames = frames * window
    real = frames @ real_kernel
    imag = frames @ imag_kernel

    if match_stride:
        real = real[..., 2:-2, :]
        imag = imag[..., 2:-2, :]

    real = real.transpose(-1, -2).contiguous()
    imag = imag.transpose(-1, -2).contiguous()
    _require_bf16("stft real", real)
    _require_bf16("stft imaginary", imag)
    return real, imag


def _bf16_magnitude(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    real, imag = bf16_stft_parts(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    magnitude = (real.square() + imag.square()).sqrt()
    _require_bf16("stft magnitude", magnitude)
    return magnitude


@torch.no_grad()
def _mel_basis(
    device: torch.device,
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float | None,
) -> torch.Tensor:
    key = (device.type, device.index, sample_rate, n_fft, n_mels, fmin, fmax)
    cached = _MEL_CACHE.get(key)
    if cached is not None:
        return cached

    from librosa.filters import mel as librosa_mel

    basis = librosa_mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    cached = torch.as_tensor(basis, device=device, dtype=BF16).contiguous()
    _MEL_CACHE[key] = cached
    return cached


def _audio_data(value: Any) -> torch.Tensor:
    tensor = value.audio_data if hasattr(value, "audio_data") else value
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected Tensor or AudioSignal-like value, got {type(value)!r}")
    _require_bf16("audio", tensor)
    return tensor


def _l1(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    loss = (left - right).abs().mean()
    _require_bf16("spectral loss", loss)
    return loss


class BF16MultiScaleSTFTLoss(nn.Module):
    def __init__(
        self,
        window_lengths: Sequence[int] = (2048, 512),
        *,
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        power: float = 2.0,
    ) -> None:
        super().__init__()
        self.window_lengths = tuple(int(value) for value in window_lengths)
        self.clamp_eps = float(clamp_eps)
        self.mag_weight = float(mag_weight)
        self.log_weight = float(log_weight)
        self.power = float(power)

    def forward(self, estimate: Any, reference: Any) -> torch.Tensor:
        estimate_audio = _audio_data(estimate)
        reference_audio = _audio_data(reference)
        loss = estimate_audio.new_zeros(())
        for window_length in self.window_lengths:
            hop_length = window_length // 4
            estimate_mag = _bf16_magnitude(
                estimate_audio,
                n_fft=window_length,
                hop_length=hop_length,
            )
            reference_mag = _bf16_magnitude(
                reference_audio,
                n_fft=window_length,
                hop_length=hop_length,
            )
            if self.log_weight:
                estimate_log = estimate_mag.clamp_min(self.clamp_eps).pow(self.power).log10()
                reference_log = reference_mag.clamp_min(self.clamp_eps).pow(self.power).log10()
                loss = loss + self.log_weight * _l1(estimate_log, reference_log)
            if self.mag_weight:
                loss = loss + self.mag_weight * _l1(estimate_mag, reference_mag)
        _require_bf16("multi-scale STFT loss", loss)
        return loss


class BF16MelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int,
        n_mels: Sequence[int],
        window_lengths: Sequence[int],
        mel_fmin: Sequence[float],
        mel_fmax: Sequence[float | None],
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        power: float = 1.0,
    ) -> None:
        super().__init__()
        lengths = {len(n_mels), len(window_lengths), len(mel_fmin), len(mel_fmax)}
        if len(lengths) != 1:
            raise ValueError("mel loss parameter lists must have identical lengths")
        self.sample_rate = int(sample_rate)
        self.n_mels = tuple(int(value) for value in n_mels)
        self.window_lengths = tuple(int(value) for value in window_lengths)
        self.mel_fmin = tuple(float(value) for value in mel_fmin)
        self.mel_fmax = tuple(None if value is None else float(value) for value in mel_fmax)
        self.clamp_eps = float(clamp_eps)
        self.mag_weight = float(mag_weight)
        self.log_weight = float(log_weight)
        self.power = float(power)

    def forward(self, estimate: Any, reference: Any) -> torch.Tensor:
        estimate_audio = _audio_data(estimate)
        reference_audio = _audio_data(reference)
        loss = estimate_audio.new_zeros(())
        settings = zip(
            self.n_mels,
            self.window_lengths,
            self.mel_fmin,
            self.mel_fmax,
            strict=True,
        )
        for n_mels, window_length, fmin, fmax in settings:
            hop_length = window_length // 4
            estimate_mag = _bf16_magnitude(
                estimate_audio,
                n_fft=window_length,
                hop_length=hop_length,
            )
            reference_mag = _bf16_magnitude(
                reference_audio,
                n_fft=window_length,
                hop_length=hop_length,
            )
            basis = _mel_basis(
                estimate_audio.device,
                sample_rate=self.sample_rate,
                n_fft=window_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
            )
            estimate_mel = (estimate_mag.transpose(-1, -2) @ basis.transpose(0, 1)).transpose(
                -1, -2
            )
            reference_mel = (reference_mag.transpose(-1, -2) @ basis.transpose(0, 1)).transpose(
                -1, -2
            )
            if self.log_weight:
                estimate_log = estimate_mel.clamp_min(self.clamp_eps).pow(self.power).log10()
                reference_log = reference_mel.clamp_min(self.clamp_eps).pow(self.power).log10()
                loss = loss + self.log_weight * _l1(estimate_log, reference_log)
            if self.mag_weight:
                loss = loss + self.mag_weight * _l1(estimate_mel, reference_mel)
        _require_bf16("multi-scale mel loss", loss)
        return loss


def configure_discriminator_bf16_stft(discriminator: nn.Module) -> int:
    """Replace DACVAE MRD complex STFT paths with the bf16 real-valued DFT path."""
    configured = 0
    for module in getattr(discriminator, "discriminators", ()):
        if not all(hasattr(module, name) for name in ("window_length", "hop_factor", "bands")):
            continue

        def spectrogram(
            waveform: torch.Tensor, *, target: nn.Module = module
        ) -> list[torch.Tensor]:
            _require_bf16("MRD waveform", waveform)
            window_length = int(target.window_length)
            real, imag = bf16_stft_parts(
                waveform,
                n_fft=window_length,
                hop_length=int(window_length * float(target.hop_factor)),
                match_stride=True,
            )
            batch, channels, _, _ = real.shape
            stacked = torch.stack((real, imag), dim=2)
            stacked = stacked.permute(0, 1, 2, 4, 3).reshape(
                batch * channels,
                2,
                real.shape[-1],
                real.shape[-2],
            )
            _require_bf16("MRD spectrogram", stacked)
            return [stacked[..., start:end] for start, end in target.bands]

        module.spectrogram = spectrogram
        configured += 1
    return configured
