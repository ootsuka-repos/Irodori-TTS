#!/usr/bin/env python3
"""Finalize natural BEATs shards into labeled nonverbal training events."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dataset.nonverbal_event_pipeline import (
    EventMergeConfig,
    ManualSeedConfig,
    PipelineConfig,
    run_nonverbal_event_pipeline,
)
from dataset.nonverbal_labeling import NonverbalLabelingConfig
from dataset.nonverbal_report import write_nonverbal_report
from dataset.nonverbal_subclustering import SubclusteringConfig


def _write_default_report(
    stage: Path,
    rows: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
    summary: Mapping[str, Any],
) -> dict[str, object]:
    del embeddings, summary
    report = write_nonverbal_report(rows, stage, audio_root=stage)
    return {
        "dashboard": "dashboard.html",
        "summary_csv": "summary.csv",
        "event_count": report["event_count"],
        "class_counts": report["class_counts"],
        "cluster_group_count": report["dashboard"]["group_count"],
        "materialization": report["materialization"],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify naturally segmented outside-VAD BEATs shards, propagate labels "
            "within each source, conservatively rejoin matching events, and publish a dataset."
        )
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("dataset/data/grok_stt/nonverbal_events"),
        help="Root containing embeddings/shards/*.jsonl and matching *.npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/data/grok_stt/nonverbal_events"),
        help="Pipeline work root; the atomic result is published under dataset/.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("dataset/data"))
    parser.add_argument(
        "--manual-seeds",
        type=Path,
        default=Path("dataset/data/grok_stt/manual_nonverbal_seeds.jsonl"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=Path("dataset/data/grok_stt/_models/hf"),
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow pinned classifier files to be fetched when they are absent locally.",
    )
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument(
        "--final-clip-padding-seconds",
        type=float,
        default=0.35,
        help="Context retained before and after each finalized event (default: 0.35).",
    )
    parser.add_argument("--manual-overlap-ratio", type=float, default=0.50)
    parser.add_argument(
        "--propagation-scope",
        choices=("source", "global"),
        default="source",
        help="Source is the safe default; global propagation can cross works.",
    )
    # Labeling thresholds default to None: only explicitly given flags override
    # the canonical NonverbalLabelingConfig dataclass defaults.
    parser.add_argument("--seed-prob", type=float, default=None)
    parser.add_argument("--seed-margin", type=float, default=None)
    parser.add_argument("--neighbors", type=int, default=None)
    parser.add_argument("--min-cosine", type=float, default=None)
    parser.add_argument("--neighbor-agreement", type=float, default=None)
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=None,
        help="Require multiple agreeing seeds before assigning a non-seed.",
    )
    parser.add_argument("--other-prob", type=float, default=None)
    parser.add_argument("--other-margin", type=float, default=None)
    parser.add_argument("--merge-min-seconds", type=float, default=5.0)
    parser.add_argument("--merge-max-seconds", type=float, default=12.0)
    parser.add_argument("--strong-boundary-score", type=float, default=2.0)
    parser.add_argument("--subcluster-pca-components", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--keep-work-clips", action="store_true")
    parser.add_argument("--force-classifier", action="store_true")
    parser.add_argument("--force-finalize", action="store_true")
    return parser.parse_args(argv)


def _labeling_config_from_args(args: argparse.Namespace) -> NonverbalLabelingConfig | None:
    """Build a labeling config only from flags the user actually passed."""
    overrides = {
        "seed_prob": args.seed_prob,
        "seed_margin": args.seed_margin,
        "k": args.neighbors,
        "min_cosine": args.min_cosine,
        "neighbor_agreement": args.neighbor_agreement,
        "min_neighbors": args.min_neighbors,
        "other_prob": args.other_prob,
        "other_margin": args.other_margin,
    }
    provided = {key: value for key, value in overrides.items() if value is not None}
    return NonverbalLabelingConfig(**provided) if provided else None


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_nonverbal_event_pipeline(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        data_root=args.data_root,
        manual_seed_path=args.manual_seeds,
        device=args.device,
        hf_cache_dir=args.hf_cache_dir,
        local_files_only=not args.allow_model_download,
        manual_config=ManualSeedConfig(
            minimum_overlap_ratio=args.manual_overlap_ratio,
        ),
        labeling_config=_labeling_config_from_args(args),
        merge_config=EventMergeConfig(
            minimum_seconds=args.merge_min_seconds,
            max_seconds=args.merge_max_seconds,
            strong_boundary_score=args.strong_boundary_score,
        ),
        subclustering_config=SubclusteringConfig(
            pca_components=args.subcluster_pca_components,
            random_state=args.random_state,
        ),
        pipeline_config=PipelineConfig(
            classifier_batch_size=args.classifier_batch_size,
            propagation_scope=args.propagation_scope,
            propagate_manual_seeds=False,
            lock_confident_usual=True,
            reject_target_class_conflicts=True,
            keep_work_clips=args.keep_work_clips,
            force_classifier=args.force_classifier,
            force_finalize=args.force_finalize,
            final_clip_padding_seconds=args.final_clip_padding_seconds,
            # The default report hook below builds the classes/ tree itself,
            # so the pipeline must not materialize one that gets replaced.
            materialize_review_folders=False,
        ),
        visualization_hook=_write_default_report,
    )
    print(
        f"complete events={summary['event_count']} "
        f"published_primitives={summary['published_primitive_count']} "
        f"discarded_primitives={summary['discarded_primitive_count']} "
        f"joined={summary['published_primitives_joined']} "
        f"dataset={args.output_dir / 'dataset'}"
    )


if __name__ == "__main__":
    main()
