"""Micro-benchmark for vectorized inference tail detection."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from functools import partial

import torch

from irodori_tts.inference.postprocessing import find_flattening_points


def _scalar_reference(
    latents: torch.Tensor,
    *,
    window_size: int,
) -> list[int]:
    results: list[int] = []
    for latent in latents:
        total_steps = int(latent.shape[0])
        padded = torch.cat(
            (
                latent,
                torch.zeros(
                    (window_size, latent.shape[1]),
                    device=latent.device,
                    dtype=latent.dtype,
                ),
            ),
            dim=0,
        )
        result = total_steps
        for index in range(total_steps):
            window = padded[index : index + window_size]
            if window.std(unbiased=False) < 0.05 and window.mean().abs() < 0.1:
                result = index
                break
        results.append(result)
    return results


def _measure(fn: Callable[[], object], *, repeats: int) -> float:
    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(
        (args.batch_size, args.steps, args.latent_dim),
        generator=generator,
    )
    latents[:, -100:] = 0.0
    scalar = partial(_scalar_reference, latents, window_size=args.window_size)
    vectorized = partial(find_flattening_points, latents, window_size=args.window_size)
    if scalar() != vectorized():
        raise RuntimeError("Vectorized result does not match scalar reference")

    scalar_seconds = _measure(scalar, repeats=args.repeats)
    vectorized_seconds = _measure(vectorized, repeats=args.repeats)
    print(f"scalar:     {scalar_seconds * 1000.0:.3f} ms")
    print(f"vectorized: {vectorized_seconds * 1000.0:.3f} ms")
    print(f"speedup:    {scalar_seconds / vectorized_seconds:.2f}x")


if __name__ == "__main__":
    main()
