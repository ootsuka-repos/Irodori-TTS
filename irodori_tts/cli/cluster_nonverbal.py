#!/usr/bin/env python3
"""Cluster audio excluded by cached Silero VAD with GPU BEATs embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

from irodori_tts.data_prep.acoustic_segmentation import AcousticSegmentationConfig
from irodori_tts.data_prep.nonverbal_clustering import (
    ClusterConfig,
    FeatureConfig,
    run_nonverbal_clustering,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read original vad_regions from Grok STT caches, embed their exact complement with "
            "Microsoft BEATs, and export HDBSCAN/KMeans review clusters."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--raw-response-dir",
        type=Path,
        default=Path("data/grok_stt/raw_responses"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/grok_stt/nonverbal_clustering"),
    )
    parser.add_argument(
        "--beats-code-dir",
        type=Path,
        default=Path("data/grok_stt/_models/beats/code"),
    )
    parser.add_argument(
        "--beats-checkpoint",
        type=Path,
        default=Path("data/grok_stt/_models/beats/BEATs_iter3_plus_AS2M.pt"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--segmentation-mode",
        choices=("acoustic", "fixed"),
        default="acoustic",
        help="Use natural log-mel/pause boundaries, or the legacy fixed split.",
    )
    parser.add_argument("--event-min-seconds", type=float, default=1.2)
    parser.add_argument("--event-max-seconds", type=float, default=15.0)
    parser.add_argument(
        "--embedding-chunk-seconds",
        type=float,
        default=5.0,
        help=(
            "Internal BEATs forward-pass size. Longer natural events are duration-pooled, "
            "not truncated or exposed as multiple clips."
        ),
    )
    parser.add_argument(
        "--fixed-window-seconds",
        type=float,
        default=5.0,
        help="Legacy fixed segmentation size; used only with --segmentation-mode=fixed.",
    )
    parser.add_argument("--min-gap-seconds", type=float, default=0.08)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=30)
    parser.add_argument("--hdbscan-min-samples", type=int, default=10)
    parser.add_argument("--kmeans-clusters", type=int, default=96)
    parser.add_argument("--representatives-near", type=int, default=3)
    parser.add_argument("--representatives-boundary", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--source-shard-index", type=int, default=0)
    parser.add_argument("--source-shard-count", type=int, default=1)
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Only write resumable per-source BEATs shards; useful for one worker per GPU.",
    )
    parser.add_argument(
        "--export-all-clips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export every candidate once and hard-link it into cluster member folders.",
    )
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.event_min_seconds <= 0:
        raise ValueError("--event-min-seconds must be positive")
    if args.event_max_seconds < 2 * args.event_min_seconds:
        raise ValueError("--event-max-seconds must be at least twice --event-min-seconds")
    if args.embedding_chunk_seconds <= 0:
        raise ValueError("--embedding-chunk-seconds must be positive")
    if args.fixed_window_seconds <= 0:
        raise ValueError("--fixed-window-seconds must be positive")
    if args.min_gap_seconds < 0:
        raise ValueError("--min-gap-seconds cannot be negative")
    if args.pca_components <= 0:
        raise ValueError("--pca-components must be positive")
    if args.hdbscan_min_cluster_size < 2:
        raise ValueError("--hdbscan-min-cluster-size must be at least 2")
    if args.hdbscan_min_samples <= 0:
        raise ValueError("--hdbscan-min-samples must be positive")
    if args.kmeans_clusters <= 0:
        raise ValueError("--kmeans-clusters must be positive")
    if args.representatives_near < 0 or args.representatives_boundary < 0:
        raise ValueError("Representative counts cannot be negative")
    if args.max_sources is not None and args.max_sources <= 0:
        raise ValueError("--max-sources must be positive")
    if args.source_shard_count <= 0:
        raise ValueError("--source-shard-count must be positive")
    if not 0 <= args.source_shard_index < args.source_shard_count:
        raise ValueError("--source-shard-index must be in [0, --source-shard-count)")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    summary = run_nonverbal_clustering(
        data_root=args.data_root.expanduser().resolve(),
        raw_response_dir=args.raw_response_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        beats_code_dir=args.beats_code_dir.expanduser().resolve(),
        checkpoint_path=args.beats_checkpoint.expanduser().resolve(),
        device=args.device,
        batch_size=args.batch_size,
        feature_config=FeatureConfig(
            embedding_chunk_seconds=args.embedding_chunk_seconds,
            min_gap_seconds=args.min_gap_seconds,
            segmentation_mode=args.segmentation_mode,
            fixed_window_seconds=args.fixed_window_seconds,
            acoustic=AcousticSegmentationConfig(
                preferred_min_seconds=args.event_min_seconds,
                max_seconds=args.event_max_seconds,
            ),
        ),
        cluster_config=ClusterConfig(
            pca_components=args.pca_components,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
            hdbscan_min_samples=args.hdbscan_min_samples,
            kmeans_clusters=args.kmeans_clusters,
            representatives_near=args.representatives_near,
            representatives_boundary=args.representatives_boundary,
            random_state=args.random_state,
        ),
        max_sources=args.max_sources,
        force_features=args.force_features,
        source_shard_index=args.source_shard_index,
        source_shard_count=args.source_shard_count,
        features_only=args.features_only,
        export_all_clips=args.export_all_clips,
    )
    if args.features_only:
        print(
            f"features complete shard={args.source_shard_index}/{args.source_shard_count} "
            f"candidates={summary['candidate_windows']}"
        )
        return
    print(
        f"complete candidates={summary['candidate_windows']} "
        f"hdbscan_clusters={summary['hdbscan_clusters']} "
        f"hdbscan_noise_ratio={summary['hdbscan_noise_ratio']:.3f} "
        f"kmeans_clusters={summary['kmeans_clusters']}"
    )


if __name__ == "__main__":
    main()
