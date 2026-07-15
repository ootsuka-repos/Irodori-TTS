"""Conservatively correct cached Grok STT text with surrounding context.

This intentionally edits manifests only; clip boundaries/audio are produced by
``prepare_grok_stt.py``.  Raw model suggestions and accepted changes are kept
under the dataset directory for auditability.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Reviewed exceptions where the language model either over-reconstructed audio
# it could not hear, or where a smaller correction is clearly preferable.
REJECTED_SUGGESTION_TEXTS = {
    "ギャンブルをするときはワンちゃんじゃなくてお客様じゃないですか座りしてないで席についてください",
    "書は主人の絶対的権威のもと下僕がその修正と方式を賛成することは目的とし両者間の義務と権利を明確にするものである第一章契約の目的木は主人のありとあらゆる命令に従い例族し",
    "活魚ですよね年下の女の子の靴裏にキスなんて",
    "けがうまくいけばそれでよかったし違ってるようなら餌で釣り上げてもう一度任せばいい一度でも人にかわれちゃったら、もう抜け出せないよね。",
    "夢見のものになれて偉いね",
    "見間違えちゃったハイコ数個入ったねどういうこと私がリフルシャッフルを誤った？",
    "お手っかげそんなワンちゃんみたいなことはクソ私がやらせようと思ってたのにこんな屈辱",
}
MANUAL_TEXT_OVERRIDES = {
    "絶対無縁と見てましたよねそういうのはダメですよ": "絶対胸元見てましたよねそういうのはダメですよ",
    "うんじゃあごぼしちゃだめでしょ": "うん、じゃあ、こぼしちゃダメでしょ",
    "これだからさかりのついたおっすわ": "これだから盛りのついたオスは",
    "まあどうしてもって言うなら勝負してあげますよ一回負けたくらいで向きになっちゃって": "まあどうしてもって言うなら勝負してあげますよ一回負けたくらいでムキになっちゃって",
}


def _json_from_output(output: str) -> list[dict[str, Any]]:
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", output, re.DOTALL)
    payload = fenced.group(1) if fenced else output[output.find("[") : output.rfind("]") + 1]
    value = json.loads(payload)
    if not isinstance(value, list):
        raise ValueError("Grok correction response is not a JSON array")
    return value


def _compact(text: str) -> str:
    return re.sub(r"[\s、。！？!?…・,.]", "", text)


def _safe_change(before: str, after: str) -> bool:
    if not after.strip() or len(after) > max(12, int(len(before) * 1.25)):
        return False
    return difflib.SequenceMatcher(None, _compact(before), _compact(after)).ratio() >= 0.82


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/grok_stt"))
    parser.add_argument("--model", default="grok-4.5")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    all_path = dataset_dir / "all.jsonl"
    rows = [json.loads(line) for line in all_path.read_text(encoding="utf-8").splitlines()]
    previous_edits_path = dataset_dir / "text_edits.jsonl"
    if previous_edits_path.is_file():
        originals = {
            item["id"]: item["before"]
            for item in (
                json.loads(line)
                for line in previous_edits_path.read_text(encoding="utf-8").splitlines()
            )
        }
        for row in rows:
            if row["id"] in originals:
                row["text"] = originals[row["id"]]
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["source_uid"])].append((index, row))

    cache_dir = dataset_dir / "text_correction_batches"
    cache_dir.mkdir(exist_ok=True)
    batches: list[tuple[str, int, list[tuple[int, dict[str, Any]]]]] = []
    for source_uid, entries in grouped.items():
        for offset in range(0, len(entries), args.batch_size):
            batches.append((source_uid, offset, entries[offset : offset + args.batch_size]))

    def review_batch(
        source_uid: str, offset: int, entries: list[tuple[int, dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        cache_path = cache_dir / f"{source_uid}_{offset:05d}.json"
        if cache_path.is_file():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        transcript = "\n".join(f"{index}: {row['text']}" for index, row in entries)
        prompt = f"""あなたは日本語音声認識結果の保守的な校正者です。
同じ音声トラックの発話を時系列で示します。前後文と繰り返し登場する語彙を使い、
明白な同音誤変換・脱字・単語途中の誤分割だけを直してください。
句読点は必要な場合のみ補ってください。言い換え、創作、口調の標準化は禁止です。
確信できない候補は出力しないでください。変更が必要な行だけを
JSON配列 [{{"index":整数,"corrected_text":"...","reason":"..."}}] で返してください。
コードフェンス以外の説明は不要です。

source={source_uid}
{transcript}
"""
        result = subprocess.run(
            [
                "grok", "-p", prompt, "--model", args.model,
                "--output-format", "plain", "--max-turns", "1", "--no-subagents",
                "--disable-web-search", "--no-memory", "--permission-mode", "dontAsk",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
        )
        items = _json_from_output(result.stdout)
        for item in items:
            item["source_uid"] = source_uid
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        temporary.replace(cache_path)
        return items

    suggestions: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(review_batch, source_uid, offset, entries): (source_uid, offset)
            for source_uid, offset, entries in batches
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            source_uid, offset = futures[future]
            try:
                items = future.result()
            except (json.JSONDecodeError, subprocess.SubprocessError, ValueError) as exc:
                print(
                    f"skipped={source_uid} offset={offset} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            suggestions.extend(items)
            print(
                f"reviewed={completed}/{len(batches)} source={source_uid} "
                f"offset={offset} suggestions={len(items)}",
                flush=True,
            )

    accepted: list[dict[str, Any]] = []
    for item in suggestions:
        index = item.get("index")
        after = item.get("corrected_text")
        if not isinstance(index, int) or not 0 <= index < len(rows) or not isinstance(after, str):
            continue
        before = str(rows[index]["text"])
        if before in REJECTED_SUGGESTION_TEXTS:
            continue
        after = MANUAL_TEXT_OVERRIDES.get(before, after)
        if before == after or not _safe_change(before, after):
            continue
        change = {
            "index": index,
            "id": rows[index]["id"],
            "before": before,
            "after": after,
            "reason": str(item.get("reason", "")),
        }
        rows[index]["text"] = after
        accepted.append(change)

    _write_jsonl(dataset_dir / "text_correction_suggestions.jsonl", suggestions)
    _write_jsonl(dataset_dir / "text_edits.jsonl", accepted)
    _write_jsonl(all_path, rows)
    _write_jsonl(dataset_dir / "train.jsonl", [row for row in rows if row["status"] == "train"])
    _write_jsonl(dataset_dir / "review.jsonl", [row for row in rows if row["status"] == "review"])
    print(f"accepted={len(accepted)} total={len(rows)}")


if __name__ == "__main__":
    main()
