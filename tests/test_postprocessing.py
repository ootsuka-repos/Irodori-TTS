import torch

from irodori_tts.inference.postprocessing import (
    find_flattening_point,
    find_flattening_points,
)


def _reference_flattening_point(
    latent: torch.Tensor,
    *,
    target_value: float = 0.0,
    window_size: int = 20,
    std_threshold: float = 0.05,
    mean_threshold: float = 0.1,
) -> int:
    total_steps = int(latent.shape[0])
    if total_steps <= 0 or window_size <= 0:
        return total_steps
    padded = torch.cat(
        (latent, torch.zeros((window_size, latent.shape[1]), dtype=latent.dtype)),
        dim=0,
    )
    for index in range(padded.shape[0] - window_size):
        window = padded[index : index + window_size]
        if (
            window.std(unbiased=False) < std_threshold
            and (window.mean() - target_value).abs() < mean_threshold
        ):
            return index
    return total_steps


def test_batched_flattening_matches_reference() -> None:
    generator = torch.Generator().manual_seed(7)
    latents = torch.randn((4, 80, 12), generator=generator)
    latents[0, 31:] = 0.0
    latents[1, 57:] = 0.01
    latents[2] = 1.0

    expected = [_reference_flattening_point(latent, window_size=10) for latent in latents]
    assert find_flattening_points(latents, window_size=10) == expected


def test_single_wrapper_and_degenerate_inputs() -> None:
    latent = torch.cat((torch.ones((6, 3)), torch.zeros((8, 3))), dim=0)
    expected = _reference_flattening_point(latent, window_size=4)
    assert find_flattening_point(latent, window_size=4) == expected
    assert find_flattening_points(torch.empty((2, 0, 3))) == [0, 0]
    assert find_flattening_points(torch.empty((0, 4, 3))) == []
    assert find_flattening_point(torch.ones((4, 2)), window_size=0) == 4


def test_flattening_shape_validation() -> None:
    try:
        find_flattening_points(torch.zeros((4, 3)))
    except ValueError as exc:
        assert "(B, T, D)" in str(exc)
    else:
        raise AssertionError("Expected a shape validation error")
