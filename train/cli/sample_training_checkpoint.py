"""Generate fixed comparison WAVs from an epoch checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from inference.runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    save_wav,
)

# (name, ref category, text) — the reference clip is picked from the same
# category so sampling matches the style-matched conditioning used in training.
SAMPLES = (
    ("normal", "speech", "先生、今日も一日お疲れ様でした。ゆっくり休んでくださいね。"),
    ("normal2", "speech", "ねえ、こっち来て？今日は二人きりだから、いっぱい甘えていいんだよ。"),
    ("aegi", "aegi", "あっ、あっ、んっ……だめ、気持ちいい……！"),
    ("aegi2", "aegi", "んんっ……はぁ、はぁっ……そこ、だめぇ……っ、あぁんっ……！"),
    ("aegi3", "aegi", "ひゃうっ……！あっ、待って、んっ、イっちゃう、イっちゃうからぁ……！"),
    ("aegi4", "aegi", "ふぁ……あぁっ……奥、当たってる……んっ、はぁっ、あっ、あっ……！"),
    ("chupa", "chupa", "ちゅっ、ちゅるっ、ちゅぱっ……んっ、ちゅぅ……。"),
    ("chupa2", "chupa", "んちゅ……ちゅぷっ、じゅるっ……はむっ……んんっ、ちゅぱぁ……。"),
    ("chupa3", "chupa", "れろっ……れろれろっ……ちゅうぅ……んはぁ……じゅぽっ、じゅぽっ……。"),
    ("chupa4", "chupa", "じゅるるっ……んくっ、んんー……ちゅぱっ、ちゅっ、れろぉ……ぷはぁ……。"),
)

_FALLBACK_CATEGORY = "speech"


def _fixed_references(
    manifest_path: Path, count: int
) -> list[tuple[str, dict[str, tuple[Path, int]]]]:
    """Return up to `count` works as (speaker_id, {category: (latent, row_index)}).

    Deterministic: works appear in manifest order; each (work, category) slot is
    filled by the first usable row. A work qualifies once its speech slot is
    filled; missing categories fall back to speech at synthesis time.
    """
    works: dict[str, dict[str, tuple[Path, int]]] = {}
    work_speakers: dict[str, str] = {}
    work_order: list[str] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            speaker_id = str(row.get("speaker_id", ""))
            work_match = re.search(r"RJ\d+", speaker_id, flags=re.IGNORECASE)
            if work_match is None:
                work_match = re.search(
                    r"RJ\d+", str(row["latent_path"]), flags=re.IGNORECASE
                )
            work_id = work_match.group(0).upper() if work_match is not None else speaker_id
            if not speaker_id or not work_id:
                continue
            category = str(row.get("category", "") or "")
            if category not in {c for _, c, _ in SAMPLES} | {_FALLBACK_CATEGORY}:
                continue
            slots = works.setdefault(work_id, {})
            if category in slots:
                continue
            latent_path = Path(str(row["latent_path"])).expanduser()
            if not latent_path.is_absolute():
                latent_path = (manifest_path.parent / latent_path).resolve()
            if not latent_path.is_file():
                continue
            slots[category] = (latent_path, index)
            work_speakers.setdefault(work_id, speaker_id)
            if work_id not in work_order and _FALLBACK_CATEGORY in slots:
                work_order.append(work_id)

    references = [
        (work_speakers[work_id], works[work_id]) for work_id in work_order[:count]
    ]
    if len(references) < count:
        raise ValueError(
            f"Manifest has only {len(references)} distinct usable references; "
            f"requested {count}: {manifest_path}"
        )
    return references


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=24)
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Manual duration; omit to use the checkpoint duration predictor.",
    )
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--reference-count", type=int, default=3)
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Sample from the checkpoint's EMA weights (matches the exported "
            "inference model). Falls back to raw weights when absent."
        ),
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    references = _fixed_references(manifest, int(args.reference_count))

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=str(checkpoint),
            model_device="cuda",
            model_precision="bf16",
            codec_device="cuda",
            codec_precision="bf16",
            use_ema=bool(args.use_ema),
        )
    )
    generated: list[dict[str, object]] = []
    try:
        for ref_number, (speaker_id, slots) in enumerate(references, start=1):
            for name, category, text in SAMPLES:
                ref_category = category if category in slots else _FALLBACK_CATEGORY
                ref_latent, ref_index = slots[ref_category]
                result = runtime.synthesize(
                    SamplingRequest(
                        text=text,
                        ref_latent=str(ref_latent),
                        seconds=None if args.seconds is None else float(args.seconds),
                        trim_tail=False,
                        num_steps=int(args.num_steps),
                        seed=int(args.seed),
                    ),
                    log_fn=print,
                )
                wav_path = save_wav(
                    output_dir / f"ref{ref_number:02d}_{name}.wav",
                    result.audio,
                    result.sample_rate,
                )
                generated.append(
                    {
                        "reference": ref_number,
                        "reference_latent": ref_latent.as_posix(),
                        "reference_manifest_index": ref_index,
                        "reference_speaker_id": speaker_id,
                        "reference_category": ref_category,
                        "name": name,
                        "text": text,
                        "wav": wav_path.name,
                        "seed": result.used_seed,
                        "sample_rate": result.sample_rate,
                    }
                )
    finally:
        runtime.unload()

    metadata = {
        "checkpoint": checkpoint.as_posix(),
        "reference_count": len(references),
        "num_steps": int(args.num_steps),
        "seconds": None if args.seconds is None else float(args.seconds),
        "use_ema": bool(args.use_ema),
        "samples": generated,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
