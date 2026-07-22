from __future__ import annotations

import torch

from train.bf16_audio import (
    BF16MelSpectrogramLoss,
    BF16MultiScaleSTFTLoss,
    bf16_stft_parts,
    configure_discriminator_bf16_stft,
)
from train.cli.finetune_dacvae_decoder import normalize_batch
from train.optim import AdamWBF16


def test_bf16_dft_matches_stft_reference_and_backpropagates() -> None:
    waveform = torch.randn(2, 1, 64, dtype=torch.bfloat16, requires_grad=True)
    real, imag = bf16_stft_parts(waveform, n_fft=16, hop_length=4)

    assert real.shape == imag.shape == (2, 1, 9, 17)
    assert real.dtype == imag.dtype == torch.bfloat16

    window = torch.hann_window(16, periodic=True)
    reference = torch.stft(
        waveform.float().reshape(-1, 64),
        16,
        4,
        window=window,
        return_complex=True,
    )
    actual = torch.complex(real.float(), imag.float()).reshape_as(reference)
    assert torch.allclose(actual, reference, atol=0.08, rtol=0.08)

    target = waveform.detach() * torch.tensor(0.9, dtype=torch.bfloat16)
    loss = BF16MultiScaleSTFTLoss(window_lengths=(16, 8))(waveform, target)
    assert loss.dtype == torch.bfloat16
    loss.backward()
    assert waveform.grad is not None
    assert waveform.grad.dtype == torch.bfloat16


def test_bf16_mel_and_mrd_paths_stay_bf16() -> None:
    waveform = torch.randn(2, 1, 64, dtype=torch.bfloat16, requires_grad=True)
    target = waveform.detach() * torch.tensor(0.8, dtype=torch.bfloat16)
    mel_loss = BF16MelSpectrogramLoss(
        sample_rate=16_000,
        n_mels=(4,),
        window_lengths=(16,),
        mel_fmin=(0.0,),
        mel_fmax=(8_000.0,),
    )(waveform, target)
    assert mel_loss.dtype == torch.bfloat16
    mel_loss.backward()
    assert waveform.grad is not None
    assert waveform.grad.dtype == torch.bfloat16

    class FakeMRD(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.window_length = 16
            self.hop_factor = 0.25
            self.bands = [(0, 4), (4, 9)]

    class FakeDiscriminator(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.discriminators = torch.nn.ModuleList([FakeMRD()])

    discriminator = FakeDiscriminator()
    assert configure_discriminator_bf16_stft(discriminator) == 1
    bands = discriminator.discriminators[0].spectrogram(target)
    assert [band.shape for band in bands] == [(2, 2, 16, 4), (2, 2, 16, 5)]
    assert all(band.dtype == torch.bfloat16 for band in bands)


def test_adamw_bf16_keeps_all_tensor_state_in_bf16() -> None:
    parameter = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    optimizer = AdamWBF16([parameter], lr=1e-3, betas=(0.8, 0.99))

    parameter.square().mean().backward()
    optimizer.step()

    state = optimizer.state[parameter]
    assert parameter.grad is not None
    assert parameter.dtype == parameter.grad.dtype == torch.bfloat16
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.bfloat16
    assert isinstance(state["step"], int)


def test_dacvae_input_normalization_remains_bf16() -> None:
    waveform = torch.randn(2, 256, dtype=torch.bfloat16) * 0.01
    normalized = normalize_batch(waveform, target_db=-16.0)

    assert normalized.shape == (2, 1, 256)
    assert normalized.dtype == torch.bfloat16
