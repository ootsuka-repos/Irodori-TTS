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

SAMPLES = (
    ("normal", "先生、今日も一日お疲れ様でした。ゆっくり休んでくださいね。"),
    ("aegi", "あっ、あっ、んっ……だめ、気持ちいい……！"),
    ("chupa", "ちゅっ、ちゅるっ、ちゅぱっ……んっ、ちゅぅ……。"),
)


def _fixed_references(
    manifest_path: Path, count: int
) -> list[tuple[Path, int, str]]:
    references: list[tuple[Path, int, str]] = []
    seen_works: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if "latents/nonverbal/" in str(row["latent_path"]).replace("\\", "/"):
                continue
            speaker_id = str(row.get("speaker_id", ""))
            work_match = re.search(r"RJ\d+", speaker_id, flags=re.IGNORECASE)
            if work_match is None:
                work_match = re.search(
                    r"RJ\d+", str(row["latent_path"]), flags=re.IGNORECASE
                )
            work_id = work_match.group(0).upper() if work_match is not None else speaker_id
            if not speaker_id or not work_id or work_id in seen_works:
                continue
            latent_raw = str(row["latent_path"])
            latent_path = Path(latent_raw).expanduser()
            if not latent_path.is_absolute():
                latent_path = (manifest_path.parent / latent_path).resolve()
            if latent_path.is_file():
                references.append((latent_path, index, speaker_id))
                seen_works.add(work_id)
                if len(references) >= count:
                    return references
    raise ValueError(
        f"Manifest has only {len(references)} distinct usable references; requested {count}: "
        f"{manifest_path}"
    )


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
        )
    )
    generated: list[dict[str, object]] = []
    try:
        for ref_number, (ref_latent, ref_index, speaker_id) in enumerate(references, start=1):
            for name, text in SAMPLES:
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
        "samples": generated,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
