"""Pure-logic tests for the nonverbal data-prep fixes (no GPU / no network)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dataset._io_utils import append_jsonl
from dataset.cli.prepare_nonverbal_events import (
    _labeling_config_from_args,
)
from dataset.cli.prepare_nonverbal_events import (
    _parse_args as parse_events_args,
)
from dataset.cli.prepare_nonverbal_training import (
    _manifest_config_from_args,
)
from dataset.cli.prepare_nonverbal_training import (
    _parse_args as parse_training_args,
)
from dataset.nonverbal_clustering import (
    _prefetch_iter,
    _select_representatives,
    _source_key,
)
from dataset.nonverbal_event_pipeline import classify_primitives_resumable
from dataset.nonverbal_labeling import (
    NonverbalLabelingConfig,
    propagate_nonverbal_labels,
)
from dataset.nonverbal_training_manifest import (
    NonverbalTrainingManifestConfig,
    _confidence_reasons,
)

# ---------------------------------------------------------------------------
# source_key (#9)
# ---------------------------------------------------------------------------


def test_source_key_is_deterministic_and_order_independent() -> None:
    key = _source_key("uid-a")
    assert key == _source_key("uid-a")
    assert key != _source_key("uid-b")
    # No positional index: the key depends on the uid alone.
    assert key.startswith("s")
    assert len(key) == 17


# ---------------------------------------------------------------------------
# representative selection dedup (#5)
# ---------------------------------------------------------------------------


def test_representatives_never_repeat_when_sources_run_out() -> None:
    rows = [{"source_uid": "only-source"} for _ in range(3)]
    distances = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    selected = _select_representatives(
        np.asarray([0, 1, 2]),
        distances,
        rows,
        near_count=3,
        boundary_count=0,
    )
    indices = [index for index, _role in selected]
    assert len(indices) == 3
    assert len(set(indices)) == 3


def test_representatives_do_not_overlap_across_roles() -> None:
    rows = [{"source_uid": f"s{i}"} for i in range(4)]
    distances = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    selected = _select_representatives(
        np.asarray([0, 1, 2, 3]),
        distances,
        rows,
        near_count=2,
        boundary_count=2,
    )
    indices = [index for index, _role in selected]
    assert len(indices) == len(set(indices)) == 4
    roles = dict(selected)
    assert roles[0] == "near" and roles[3] == "boundary"


# ---------------------------------------------------------------------------
# confidence gates on merged events (#6)
# ---------------------------------------------------------------------------


def _merged_row(probabilities: dict[str, float] | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label_status": "merged_same_label",
        "primitive_label_statuses": {"classifier_seed": 1, "neighbor_supported": 1},
    }
    if probabilities is not None:
        row["prediction"] = {"probabilities": probabilities}
    return row


def test_confidence_gates_fire_on_merged_low_probability_events() -> None:
    config = NonverbalTrainingManifestConfig()
    reasons, evaluated = _confidence_reasons(
        _merged_row({"usual": 0.40, "aegi": 0.50, "chupa": 0.10}),
        "aegi",
        config,
    )
    assert evaluated is True
    assert "classifier_probability_low" in reasons
    assert "classifier_margin_low" in reasons


def test_confidence_gates_pass_confident_merged_events() -> None:
    config = NonverbalTrainingManifestConfig()
    reasons, evaluated = _confidence_reasons(
        _merged_row({"usual": 0.05, "aegi": 0.90, "chupa": 0.05}),
        "aegi",
        config,
    )
    assert evaluated is True
    assert reasons == []


def test_confidence_gates_skip_manual_provenance() -> None:
    config = NonverbalTrainingManifestConfig()
    row = _merged_row({"usual": 0.60, "aegi": 0.30, "chupa": 0.10})
    row["primitive_label_statuses"] = {"manual_override": 1, "classifier_seed": 1}
    reasons, evaluated = _confidence_reasons(row, "aegi", config)
    assert (reasons, evaluated) == ([], True)


def test_confidence_gates_count_rows_without_any_payload() -> None:
    config = NonverbalTrainingManifestConfig()
    reasons, evaluated = _confidence_reasons(_merged_row(None), "aegi", config)
    assert reasons == []
    assert evaluated is False


def test_confidence_gates_reject_untrusted_merged_provenance() -> None:
    config = NonverbalTrainingManifestConfig()
    row = _merged_row({"usual": 0.05, "aegi": 0.90, "chupa": 0.05})
    row["primitive_label_statuses"] = {"uncertain": 2}
    reasons, evaluated = _confidence_reasons(row, "aegi", config)
    assert reasons == ["merged_provenance_untrusted"]
    assert evaluated is True


# ---------------------------------------------------------------------------
# manual labels never act as classifier seeds (#7)
# ---------------------------------------------------------------------------

_LABELING = NonverbalLabelingConfig(
    seed_prob=0.80,
    seed_margin=0.50,
    k=3,
    min_cosine=0.90,
    neighbor_agreement=0.60,
    min_neighbors=1,
)


def _prediction_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": "a",
            "probabilities": {"usual": 0.02, "aegi": 0.96, "chupa": 0.02},
            "status": "ok",
        },
        {
            "id": "b",
            "probabilities": {"usual": 0.40, "aegi": 0.35, "chupa": 0.25},
            "status": "ok",
        },
    ]


_EMBEDDINGS = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)


def test_without_manual_label_the_classifier_seed_propagates() -> None:
    decisions = propagate_nonverbal_labels(_prediction_rows(), _EMBEDDINGS, config=_LABELING)
    assert decisions[0]["label_status"] == "classifier_seed"
    assert decisions[1]["final_label"] == "aegi"
    assert decisions[1]["label_status"] == "neighbor_supported"


def test_manual_correction_suppresses_classifier_seed_propagation() -> None:
    decisions = propagate_nonverbal_labels(
        _prediction_rows(),
        _EMBEDDINGS,
        manual_seeds={"a": "other"},
        config=_LABELING,
        propagate_manual_seeds=False,
    )
    assert decisions[0]["final_label"] == "other"
    assert decisions[0]["label_status"] == "manual_override"
    # The wrong classifier signal on row "a" must not label its neighbor.
    assert decisions[1]["final_label"] == "uncertain"


def test_manual_target_seed_propagates_only_when_allowed() -> None:
    rows = _prediction_rows()
    rows[0]["probabilities"] = {"usual": 0.90, "aegi": 0.05, "chupa": 0.05}
    withheld = propagate_nonverbal_labels(
        rows,
        _EMBEDDINGS,
        manual_seeds={"a": "aegi"},
        config=_LABELING,
        propagate_manual_seeds=False,
    )
    assert withheld[0]["final_label"] == "aegi"
    assert withheld[1]["final_label"] == "uncertain"

    spread = propagate_nonverbal_labels(
        rows,
        _EMBEDDINGS,
        manual_seeds={"a": "aegi"},
        config=_LABELING,
        propagate_manual_seeds=True,
    )
    assert spread[1]["final_label"] == "aegi"
    assert spread[1]["label_status"] == "neighbor_supported"


# ---------------------------------------------------------------------------
# classifier cache: invalidation, row fingerprints, last-write-wins (#2, #8)
# ---------------------------------------------------------------------------


class _StubClassifier:
    run_metadata = {"schema_version": 1, "classifier_type": "stub"}

    def __init__(self, aegi: float = 0.80) -> None:
        self.calls: list[list[Path]] = []
        self._aegi = aegi

    def predict(self, audio_paths: Any, *, batch_size: int) -> list[Any]:
        paths = list(audio_paths)
        self.calls.append(paths)
        probabilities = {"usual": 1.0 - self._aegi - 0.05, "aegi": self._aegi, "chupa": 0.05}
        return [SimpleNamespace(probabilities=dict(probabilities)) for _ in paths]


_ROWS = [
    {"id": "p1", "audio": "_work/primitive_clips/s/p1.flac"},
    {"id": "p2", "audio": "_work/primitive_clips/s/p2.flac"},
]


def _classify(tmp_path: Path, fingerprint: str, stub: _StubClassifier, *, force: bool = False):
    return classify_primitives_resumable(
        _ROWS,
        output_dir=tmp_path,
        feature_fingerprint=fingerprint,
        batch_size=8,
        classifier=stub,
        force=force,
    )


def test_classifier_cache_roundtrip_and_fingerprint_invalidation(tmp_path: Path) -> None:
    first = _StubClassifier()
    predictions, _metadata = _classify(tmp_path, "fp1", first)
    assert len(first.calls) == 1
    assert [row["id"] for row in predictions] == ["p1", "p2"]
    # Returned rows never leak the checkpoint-internal fingerprint field.
    assert all("classifier_fingerprint" not in row for row in predictions)
    final_path = tmp_path / "_work" / "classifier_predictions.jsonl"
    stored = [json.loads(line) for line in final_path.read_text("utf-8").splitlines()]
    fingerprints = {row["classifier_fingerprint"] for row in stored}
    assert len(fingerprints) == 1

    # Same fingerprint: fully cached, classifier untouched.
    second = _StubClassifier()
    cached, _ = _classify(tmp_path, "fp1", second)
    assert second.calls == []
    assert [row["id"] for row in cached] == ["p1", "p2"]

    # New feature fingerprint: stale final/partial removed and recomputed.
    third = _StubClassifier()
    _classify(tmp_path, "fp2", third)
    assert len(third.calls) == 1
    assert len(third.calls[0]) == len(_ROWS)


def test_stale_rows_with_foreign_fingerprint_are_discarded(tmp_path: Path) -> None:
    _classify(tmp_path, "fp1", _StubClassifier())
    partial_path = tmp_path / "_work" / "classifier_predictions.partial.jsonl"
    poisoned = {
        "id": "p1",
        "audio": "x",
        "top_label": "chupa",
        "classifier_fingerprint": "bogus-fingerprint",
    }
    append_jsonl(partial_path, [poisoned])

    stub = _StubClassifier()
    predictions, _ = _classify(tmp_path, "fp1", stub)
    assert stub.calls == []  # final snapshot still covers everything
    p1 = next(row for row in predictions if row["id"] == "p1")
    assert p1["top_label"] == "aegi"  # the poisoned row was ignored


def test_partial_checkpoint_merges_with_last_write_wins(tmp_path: Path) -> None:
    _classify(tmp_path, "fp1", _StubClassifier())
    work_dir = tmp_path / "_work"
    state = json.loads((work_dir / "classifier_state.json").read_text("utf-8"))
    fingerprint = state["classifier_fingerprint"]
    (work_dir / "classifier_predictions.jsonl").unlink()
    partial_path = work_dir / "classifier_predictions.partial.jsonl"
    base = {"audio": "x", "status": "ok", "classifier_fingerprint": fingerprint}
    append_jsonl(partial_path, [{**base, "id": "p1", "marker": "A"}])
    append_jsonl(partial_path, [{**base, "id": "p1", "marker": "B"}])

    stub = _StubClassifier()
    predictions, _ = _classify(tmp_path, "fp1", stub)
    assert len(stub.calls) == 1
    assert [str(path.name) for path in stub.calls[0]] == ["p2.flac"]
    p1 = next(row for row in predictions if row["id"] == "p1")
    assert p1["marker"] == "B"


# ---------------------------------------------------------------------------
# threshold resolution: dataclass is the single source of truth (#10, #11)
# ---------------------------------------------------------------------------


def test_labeling_dataclass_carries_canonical_defaults() -> None:
    config = NonverbalLabelingConfig()
    assert config.seed_prob == 0.80
    assert config.seed_margin == 0.50
    assert config.min_cosine == 0.92
    assert config.neighbor_agreement == 0.80
    assert config.min_neighbors == 3


def test_events_cli_overrides_only_supplied_flags() -> None:
    assert _labeling_config_from_args(parse_events_args([])) is None
    config = _labeling_config_from_args(parse_events_args(["--min-cosine", "0.5"]))
    assert config is not None
    assert config.min_cosine == 0.5
    assert config.seed_prob == 0.80  # untouched fields keep dataclass defaults


def test_training_cli_resolves_defaults_from_dataclass() -> None:
    config = _manifest_config_from_args(parse_training_args([]))
    assert config == NonverbalTrainingManifestConfig()
    assert config.maximum_seconds == 15.0
    assert config.neighbor_minimum_count == 3
    overridden = _manifest_config_from_args(parse_training_args(["--maximum-seconds", "12"]))
    assert overridden.maximum_seconds == 12.0
    assert overridden.minimum_seconds == 5.0


# ---------------------------------------------------------------------------
# prefetch iterator (#13)
# ---------------------------------------------------------------------------


def test_prefetch_iter_preserves_order_and_propagates_errors() -> None:
    assert list(_prefetch_iter(iter(range(100)), depth=2)) == list(range(100))

    def _boom():
        yield 1
        raise RuntimeError("decode failed")

    consumed: list[int] = []
    with pytest.raises(RuntimeError, match="decode failed"):
        for item in _prefetch_iter(_boom(), depth=1):
            consumed.append(item)
    assert consumed == [1]
