"""Tests for the dataset CLI fixes: sharding math, stage caching, latent naming."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import pytest

from dataset.cli import prepare_all
from dataset.cli.prepare_all import (
    PipelineRunner,
    _path_record,
)
from dataset.cli.prepare_all import (
    _parse_args as parse_full_pipeline_args,
)
from dataset.cli.prepare_manifest import (
    _count_rank_items,
    _count_rank_items_contiguous,
    _first_index_for_rank,
    _latent_filename,
)

# ---------------------------------------------------------------------------
# Rank sharding arithmetic (prepare_manifest)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("start", [0, 1, 5, 17])
@pytest.mark.parametrize("total", [0, 1, 7, 16, 33])
@pytest.mark.parametrize("world_size", [1, 2, 3, 8])
def test_stride_sharding_partitions_every_index_exactly_once(
    start: int, total: int, world_size: int
) -> None:
    """Mirror of the map-style stride path in _iter_rank_examples."""
    end = start + total
    combined: list[int] = []
    for rank in range(world_size):
        first = _first_index_for_rank(start, rank, world_size)
        indices = list(range(first, end, world_size))
        assert len(indices) == _count_rank_items(start, end, rank, world_size)
        combined.extend(indices)
    assert sorted(combined) == list(range(start, end))
    assert len(set(combined)) == len(combined)


@pytest.mark.parametrize("start", [0, 1, 5, 17])
@pytest.mark.parametrize("total", [0, 1, 7, 16, 33])
@pytest.mark.parametrize("world_size", [1, 2, 3, 8])
def test_contiguous_sharding_partitions_every_index_exactly_once(
    start: int, total: int, world_size: int
) -> None:
    """Mirror of the map-style contiguous path in _iter_rank_examples."""
    end = start + total
    combined: list[int] = []
    for rank in range(world_size):
        if end <= start:
            indices: list[int] = []
        else:
            per_rank = int(math.ceil(total / world_size))
            shard_start = start + (rank * per_rank)
            shard_end = min(end, shard_start + per_rank)
            indices = list(range(shard_start, shard_end)) if shard_end > shard_start else []
        assert len(indices) == _count_rank_items_contiguous(start, end, rank, world_size)
        combined.extend(indices)
    assert sorted(combined) == list(range(start, end))
    assert len(set(combined)) == len(combined)


# ---------------------------------------------------------------------------
# Deterministic latent naming (prepare_manifest)
# ---------------------------------------------------------------------------


def test_latent_filename_depends_only_on_sample_index() -> None:
    assert _latent_filename(7, rank=0, world_size=1) == "00000007.pt"
    assert _latent_filename(0, rank=0, world_size=1) == "00000000.pt"
    # Processing order must not influence names (no `written` counter anymore):
    names = [_latent_filename(idx, rank=0, world_size=1) for idx in (5, 3, 5)]
    assert names[0] == names[2] == "00000005.pt"
    assert names[1] == "00000003.pt"
    # Multi-rank runs keep a rank prefix because iterable-dataset shards
    # enumerate shard-local indices that may collide across ranks.
    assert _latent_filename(7, rank=1, world_size=4) == "rank01_00000007.pt"
    assert _latent_filename(7, rank=1, world_size=4) == _latent_filename(
        7, rank=1, world_size=4
    )


# ---------------------------------------------------------------------------
# Content-addressed fingerprints and stage caching (prepare_all)
# ---------------------------------------------------------------------------


def test_path_record_is_content_addressed_for_hashable_files(tmp_path: Path) -> None:
    target = tmp_path / "artifact.jsonl"
    target.write_text('{"x":1}\n', encoding="utf-8")
    first = _path_record(target)
    assert "sha256" in first
    assert "mtime_ns" not in first

    # A byte-identical rewrite (only mtime changes) must not change the record,
    # otherwise self-rewriting stages re-run forever.
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns + 5_000_000_000, stat.st_mtime_ns + 5_000_000_000))
    assert _path_record(target) == first

    target.write_text('{"x":2}\n', encoding="utf-8")
    assert _path_record(target) != first


def test_path_record_marks_missing_paths(tmp_path: Path) -> None:
    record = _path_record(tmp_path / "does-not-exist.jsonl")
    assert record.get("missing") is True


def _make_runner(tmp_path: Path) -> PipelineRunner:
    argv = [
        "--data-root",
        str(tmp_path / "data"),
        "--work-dir",
        str(tmp_path / "work"),
        "--nonverbal-dir",
        str(tmp_path / "nonverbal"),
        "--nonverbal-training-dir",
        str(tmp_path / "nonverbal_training"),
        "--manifest-dir",
        str(tmp_path / "manifests"),
        "--latent-root",
        str(tmp_path / "latents"),
    ]
    runner = PipelineRunner(parse_full_pipeline_args(argv))
    # GPU resolution is lazy; pin a fake device list so tests run on CPU hosts.
    runner._gpu_indices_cache = (0,)
    return runner


def test_run_stage_skips_when_fingerprint_and_outputs_match(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    output = tmp_path / "stage" / "artifact.txt"
    calls: list[int] = []

    def action() -> None:
        calls.append(1)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("ok", encoding="utf-8")

    runner.run_stage("speech", fingerprint="f1", outputs=(output,), action=action)
    runner.run_stage("speech", fingerprint="f1", outputs=(output,), action=action)
    assert len(calls) == 1  # cached complete

    runner.run_stage("speech", fingerprint="f2", outputs=(output,), action=action)
    assert len(calls) == 2  # fingerprint change re-runs

    output.unlink()
    runner.run_stage("speech", fingerprint="f2", outputs=(output,), action=action)
    assert len(calls) == 3  # missing output re-runs


def test_run_stage_records_failure_and_reraises(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)

    def boom() -> None:
        raise RuntimeError("stage exploded")

    with pytest.raises(RuntimeError, match="stage exploded"):
        runner.run_stage("speech", fingerprint="f1", outputs=(), action=boom)
    assert runner.state["stages"]["speech"]["status"] == "failed"

    runner.run_stage("speech", fingerprint="f1", outputs=(), action=lambda: None)
    assert runner.state["stages"]["speech"]["status"] == "complete"


# ---------------------------------------------------------------------------
# _latent_one: row-count contract, caching, encode-parameter keying
# ---------------------------------------------------------------------------


def _install_fake_latent_command(
    runner: PipelineRunner,
    monkeypatch: Any,
    commands: list[list[str]],
    state: dict[str, int],
) -> None:
    def fake_run_command(stage: str, command: Any) -> None:
        command = list(command)
        commands.append(command)
        output = Path(command[command.index("--output-manifest") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}\n" * state["rows"], encoding="utf-8")

    monkeypatch.setattr(runner, "run_command", fake_run_command)


def test_latent_one_rejects_row_count_mismatch_then_caches(
    tmp_path: Path, monkeypatch: Any
) -> None:
    runner = _make_runner(tmp_path)
    input_manifest = tmp_path / "input.jsonl"
    input_manifest.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
    commands: list[list[str]] = []
    state = {"rows": 1}
    _install_fake_latent_command(runner, monkeypatch, commands, state)

    with pytest.raises(RuntimeError, match="refusing publication"):
        runner._latent_one("speech", input_manifest)

    state["rows"] = 2
    published = runner._latent_one("speech", input_manifest)
    assert published.is_file()
    assert published.suffix == ".jsonl"

    call_count = len(commands)
    assert runner._latent_one("speech", input_manifest) == published
    assert len(commands) == call_count  # cache hit: no subprocess re-run

    # The encode parameters must be pinned on the command line.
    assert "--codec-repo" in commands[-1]
    assert "--normalize-db" in commands[-1]


def test_latent_one_key_tracks_input_content_and_encode_params(
    tmp_path: Path, monkeypatch: Any
) -> None:
    runner = _make_runner(tmp_path)
    input_manifest = tmp_path / "input.jsonl"
    input_manifest.write_text('{"a":1}\n', encoding="utf-8")
    commands: list[list[str]] = []
    state = {"rows": 1}
    _install_fake_latent_command(runner, monkeypatch, commands, state)

    first = runner._latent_one("speech", input_manifest)

    # Changing encode parameters must produce a new cache key (new output),
    # so stale latents are never reused for different codec settings.
    monkeypatch.setitem(prepare_all.LATENT_ENCODE_PARAMS, "normalize_db", -20.0)
    second = runner._latent_one("speech", input_manifest)
    assert second != first
    latest = commands[-1]
    assert latest[latest.index("--normalize-db") + 1] == "-20.0"

    # Changing input content must also produce a new key.
    input_manifest.write_text('{"a":2}\n', encoding="utf-8")
    third = runner._latent_one("speech", input_manifest)
    assert third not in {first, second}
