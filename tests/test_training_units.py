from __future__ import annotations

import json

import pytest
import torch

from core.rf import sample_logit_normal_t, sample_stratified_logit_normal_t
from train.cli.train import split_train_valid_indices
from train.dataset import LatentTextDataset, _ManifestIndex
from train.ema import ModelEMA


def test_split_sample_mode_is_deterministic_and_disjoint():
    a = split_train_valid_indices(num_samples=100, valid_ratio=0.1, seed=7)
    b = split_train_valid_indices(num_samples=100, valid_ratio=0.1, seed=7)
    assert a == b
    train, valid = a
    assert len(valid) == 10
    assert sorted(train + valid) == list(range(100))


def test_split_speaker_mode_never_straddles_groups():
    groups = [f"spk{i % 5}" for i in range(50)]
    train, valid = split_train_valid_indices(
        num_samples=50, valid_ratio=0.2, seed=3, groups=groups
    )
    assert sorted(train + valid) == list(range(50))
    train_groups = {groups[i] for i in train}
    valid_groups = {groups[i] for i in valid}
    assert not (train_groups & valid_groups)
    assert valid_groups and train_groups


def test_split_speaker_mode_treats_none_as_singletons():
    groups = ["a"] * 4 + [None] * 4
    train, valid = split_train_valid_indices(
        num_samples=8, valid_ratio=0.25, seed=0, groups=groups
    )
    assert sorted(train + valid) == list(range(8))
    assert valid


def test_split_speaker_mode_requires_two_groups():
    with pytest.raises(ValueError):
        split_train_valid_indices(
            num_samples=4, valid_ratio=0.25, seed=0, groups=["only"] * 4
        )


def test_timestep_samplers_are_deterministic_with_generator():
    device = torch.device("cpu")
    for sampler in (sample_logit_normal_t, sample_stratified_logit_normal_t):
        g1 = torch.Generator(device=device).manual_seed(11)
        g2 = torch.Generator(device=device).manual_seed(11)
        t1 = sampler(batch_size=16, device=device, generator=g1)
        t2 = sampler(batch_size=16, device=device, generator=g2)
        assert torch.equal(t1, t2)


def test_model_ema_update_math_and_roundtrip():
    model = torch.nn.Linear(4, 4, bias=False)
    ema = ModelEMA(model, decay=0.9, update_every=2, device="cpu")
    initial = ema.shadow["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    effective = 0.9**2
    expected = initial * effective + model.weight.detach().float() * (1.0 - effective)
    assert torch.allclose(ema.shadow["weight"], expected)

    restored = ModelEMA(model, decay=0.9, update_every=2, device="cpu")
    loaded, missing = restored.load_state_dict(ema.state_dict())
    assert missing == 0 and loaded == len(ema.shadow)
    assert torch.allclose(restored.shadow["weight"], ema.shadow["weight"])


def _write_manifest(tmp_path, rows):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def test_manifest_index_records_num_frames_and_cache_roundtrip(tmp_path):
    rows = [
        {"text": "a", "latent_path": "x0.pt", "speaker_id": "s0", "num_frames": 10},
        {"text": "b", "latent_path": "x1.pt", "speaker_id": "s1"},
        {"text": "c", "latent_path": "x2.pt", "num_frames": 999},
    ]
    manifest = _write_manifest(tmp_path, rows)
    index = _ManifestIndex.build(manifest)
    assert index.num_frames == [10, -1, 999]
    cached = _ManifestIndex.build(manifest)
    assert cached.offsets == index.offsets
    assert cached.num_frames == index.num_frames
    assert cached.speaker_ids == index.speaker_ids


def test_dataset_overlong_count_and_latent_load_skip(tmp_path):
    latent = torch.randn(12, 8)
    torch.save(latent, tmp_path / "x0.pt")
    rows = [
        {"text": "a", "latent_path": "x0.pt", "num_frames": 12},
        # Latent file intentionally missing: the skip path must not touch it.
        {"text": "b", "latent_path": "missing.pt", "num_frames": 99},
    ]
    manifest = _write_manifest(tmp_path, rows)
    dataset = LatentTextDataset(
        manifest_path=manifest,
        latent_dim=8,
        max_latent_steps=50,
        enable_speaker_condition=False,
        load_target_latent=False,
    )
    assert dataset.overlong_sample_count == 1
    item = dataset[1]
    assert item["num_frames"] == 50  # capped at max_latent_steps
    assert item["latent"].shape == (0, 8)

    eager = LatentTextDataset(
        manifest_path=manifest,
        latent_dim=8,
        max_latent_steps=50,
        enable_speaker_condition=False,
        load_target_latent=True,
    )
    assert eager[0]["latent"].shape == (12, 8)
    with pytest.raises(FileNotFoundError):
        eager[1]


def test_dataset_getstate_drops_file_handle(tmp_path):
    torch.save(torch.randn(4, 8), tmp_path / "x0.pt")
    manifest = _write_manifest(
        tmp_path, [{"text": "a", "latent_path": "x0.pt", "num_frames": 4}]
    )
    dataset = LatentTextDataset(
        manifest_path=manifest,
        latent_dim=8,
        enable_speaker_condition=False,
    )
    dataset[0]
    assert dataset._manifest_fp is not None
    assert dataset.__getstate__()["_manifest_fp"] is None
