"""Context-aware transcript correction through installed coding-agent CLIs."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset._io_utils import atomic_write_json as _atomic_write_json
from dataset._textnorm import compact_for_similarity, normalize_transcript

CORRECTION_PROMPT_VERSION = "ja-asr-context-classify-v3"

# Closed set the LLM assigns to every target segment. "mixed" covers e.g.
# フェラしながらセリフ; "other" is non-vocal noise / unclassifiable content.
CONTENT_CATEGORIES = ("speech", "aegi", "chupa", "mixed", "other")

# Rows without a usable start time must not interleave with real timestamps;
# they are parked (stably, by row index) after every timed segment.
_TIMELINE_MISSING_START_OFFSET = 1e12
CORRECTION_CACHE_SCHEMA_VERSION = 1
CORRECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "corrected_text": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["unchanged", "corrected", "uncertain"],
                    },
                    "category": {
                        "type": "string",
                        "enum": ["speech", "aegi", "chupa", "mixed", "other"],
                    },
                },
                "required": ["id", "corrected_text", "status", "category"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}


class CorrectionAgentError(RuntimeError):
    """Raised when a CLI cannot return a valid structured correction."""


@dataclass(frozen=True)
class CorrectionConfig:
    """Batch/context and conservative acceptance settings."""

    agent_priority: tuple[str, ...] = ("codex", "claude", "grok")
    target_batch_size: int = 16
    context_segments: int = 10
    workers: int = 12
    timeout_seconds: float = 300.0
    attempts_per_agent: int = 2
    minimum_similarity: float = 0.60
    minimum_length_ratio: float = 0.45
    maximum_length_ratio: float = 1.65
    # Raw (punctuation-inclusive) length bounds; the compact ratio above
    # cannot see punctuation-only insertions or deletions.
    minimum_raw_length_ratio: float = 0.40
    maximum_raw_length_ratio: float = 2.00
    codex_model: str | None = None
    claude_model: str | None = None
    grok_model: str | None = None

    def __post_init__(self) -> None:
        supported = {"codex", "claude", "grok"}
        normalized = tuple(str(item).strip().lower() for item in self.agent_priority)
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("agent_priority must contain unique agent names")
        if any(item not in supported for item in normalized):
            raise ValueError(f"agent_priority must be selected from {sorted(supported)}")
        object.__setattr__(self, "agent_priority", normalized)
        if self.target_batch_size <= 0:
            raise ValueError("target_batch_size must be positive")
        if self.context_segments < 0:
            raise ValueError("context_segments cannot be negative")
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if self.attempts_per_agent <= 0:
            raise ValueError("attempts_per_agent must be positive")
        if not 0.0 <= self.minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be in [0, 1]")
        if not 0.0 < self.minimum_length_ratio <= 1.0:
            raise ValueError("minimum_length_ratio must be in (0, 1]")
        if self.maximum_length_ratio < 1.0:
            raise ValueError("maximum_length_ratio must be at least 1")
        if not 0.0 < self.minimum_raw_length_ratio <= 1.0:
            raise ValueError("minimum_raw_length_ratio must be in (0, 1]")
        if self.maximum_raw_length_ratio < 1.0:
            raise ValueError("maximum_raw_length_ratio must be at least 1")


@dataclass(frozen=True)
class _TimelineItem:
    kind: str
    row_index: int
    row_id: str
    source_uid: str
    start: float
    end: float
    backend: str
    raw_text: str


@dataclass(frozen=True)
class _Batch:
    source_uid: str
    ordinal: int
    timeline: tuple[_TimelineItem, ...]
    target_ids: tuple[str, ...]


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _source_uid(row: Mapping[str, Any]) -> str:
    source_uid = str(row.get("source_uid", "") or "").strip()
    if source_uid:
        return source_uid
    source_audio = str(row.get("source_audio", "") or "").strip().replace("\\", "/")
    if source_audio:
        return f"path:{source_audio.casefold()}"
    raise ValueError(f"row {row.get('id')!r} has neither source_uid nor source_audio")


def _finite_time(value: Any, *, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if math.isfinite(parsed) else fallback


def _build_timeline(
    speech_rows: Sequence[Mapping[str, Any]],
    nonverbal_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[_TimelineItem]]]:
    speech = [dict(row) for row in speech_rows]
    nonverbal = [dict(row) for row in nonverbal_rows]
    grouped: dict[str, list[_TimelineItem]] = defaultdict(list)
    seen_ids: set[str] = set()

    for kind, rows in (("speech", speech), ("nonverbal", nonverbal)):
        for index, row in enumerate(rows):
            row_id = str(row.get("id", "") or "").strip()
            if not row_id:
                raise ValueError(f"{kind} row {index} is missing id")
            if row_id in seen_ids:
                raise ValueError(f"duplicate timeline id: {row_id}")
            seen_ids.add(row_id)
            if kind == "speech":
                raw = row.get("asr_text_raw", row.get("text", ""))
                backend = str(row.get("asr_backend", "grok-stt") or "grok-stt")
            else:
                raw = row.get("transcript_text_raw", row.get("transcript_text", ""))
                backend = str(row.get("transcript_backend", "anime-whisper") or "anime-whisper")
            text = normalize_transcript(str(raw or ""))
            if not text:
                continue
            start = _finite_time(
                row.get("start"),
                fallback=_TIMELINE_MISSING_START_OFFSET + float(index),
            )
            end = _finite_time(row.get("end"), fallback=start)
            source = _source_uid(row)
            grouped[source].append(
                _TimelineItem(
                    kind=kind,
                    row_index=index,
                    row_id=row_id,
                    source_uid=source,
                    start=start,
                    end=end,
                    backend=backend,
                    raw_text=text,
                )
            )

    for values in grouped.values():
        values.sort(key=lambda item: (item.start, item.end, item.kind, item.row_id))
    return speech, nonverbal, dict(grouped)


def _make_batches(
    grouped: Mapping[str, Sequence[_TimelineItem]], config: CorrectionConfig
) -> list[_Batch]:
    batches: list[_Batch] = []
    for source_uid in sorted(grouped, key=str.casefold):
        timeline = list(grouped[source_uid])
        for ordinal, offset in enumerate(range(0, len(timeline), config.target_batch_size)):
            targets = timeline[offset : offset + config.target_batch_size]
            left = max(0, offset - config.context_segments)
            right = min(
                len(timeline),
                offset + len(targets) + config.context_segments,
            )
            batches.append(
                _Batch(
                    source_uid=source_uid,
                    ordinal=ordinal,
                    timeline=tuple(timeline[left:right]),
                    target_ids=tuple(item.row_id for item in targets),
                )
            )
    return batches


def _prompt_payload(batch: _Batch) -> dict[str, Any]:
    targets = set(batch.target_ids)
    return {
        "source": batch.source_uid,
        "segments": [
            {
                "id": item.row_id,
                "start": round(item.start, 3),
                "end": round(item.end, 3),
                "backend": item.backend,
                "role": "target" if item.row_id in targets else "context_only",
                "text": item.raw_text,
            }
            for item in batch.timeline
        ],
        "target_ids": list(batch.target_ids),
    }


def _build_prompt(batch: _Batch) -> tuple[str, dict[str, Any]]:
    payload = _prompt_payload(batch)
    prompt = f"""あなたは日本語音声認識結果の保守的な校正者兼分類者です。
同一音源の区間が時刻順に並んでいます。role=context_only は前後文脈として読むだけで、
role=target の区間だけを校正・分類してください。

校正の規則:
- 誤変換、脱字、明白な同音語、文脈上明白な句読点だけを直す。
- 喘ぎ、息、フィラー、反復、擬音、くだけた口調、性的表現を削除・婉曲化しない。
- 文体の美化、要約、言い換え、説明追加、未知内容の推測は禁止。
- 聞き取りを確定できない場合は原文を保持して status=uncertain とする。
- target_ids の全IDを1回ずつ返し、context_only のIDは返さない。
- 変更不要なら原文をそのまま corrected_text に入れて status=unchanged とする。

分類の規則 (category を必ず1つ付ける):
- speech: 通常のセリフ・語り。多少の間投詞や息を含んでもセリフが主体。
- aegi: 喘ぎ声が主体。あんっ/はぁっ等の反復で、意味のある文がほぼない。
- chupa: フェラ・キス等の口唇音が主体。ちゅぱ/じゅる/れろ等の擬音が支配的。
- mixed: セリフと喘ぎ/口唇音が同程度に混在 (例: フェラしながらセリフ)。
- other: 上記のどれでもない (無意味な断片、ノイズ、判定不能)。
- テキストと前後文脈から判定する。分類は校正statusと独立に必ず返す。

- 出力は指定JSON Schemaだけに従う。

input:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
"""
    return prompt, payload


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    text = str(value or "").strip()
    if not text:
        raise CorrectionAgentError("agent returned empty output")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced is None:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise CorrectionAgentError("agent output does not contain a JSON object") from None
            candidate = text[start : end + 1]
        else:
            candidate = fenced.group(1)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise CorrectionAgentError(f"invalid agent JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise CorrectionAgentError("agent output is not a JSON object")
    return parsed


def _validate_result(payload: Mapping[str, Any], target_ids: Sequence[str]) -> list[dict[str, str]]:
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise CorrectionAgentError("agent result has no segments array")
    expected = set(target_ids)
    actual: set[str] = set()
    result: list[dict[str, str]] = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise CorrectionAgentError(f"agent segment {index} is not an object")
        row_id = str(raw.get("id", "") or "")
        corrected = raw.get("corrected_text")
        status = str(raw.get("status", "") or "")
        category = str(raw.get("category", "") or "")
        if row_id not in expected:
            raise CorrectionAgentError(f"agent returned unexpected id: {row_id!r}")
        if row_id in actual:
            raise CorrectionAgentError(f"agent returned duplicate id: {row_id}")
        if not isinstance(corrected, str):
            raise CorrectionAgentError(f"corrected_text for {row_id} is not a string")
        if status not in {"unchanged", "corrected", "uncertain"}:
            raise CorrectionAgentError(f"invalid correction status for {row_id}: {status!r}")
        if category not in CONTENT_CATEGORIES:
            raise CorrectionAgentError(f"invalid category for {row_id}: {category!r}")
        actual.add(row_id)
        result.append(
            {
                "id": row_id,
                "corrected_text": normalize_transcript(corrected),
                "status": status,
                "category": category,
            }
        )
    if actual != expected:
        missing = sorted(expected - actual)
        raise CorrectionAgentError(f"agent omitted target ids: {missing[:10]}")
    order = {row_id: index for index, row_id in enumerate(target_ids)}
    result.sort(key=lambda row: order[row["id"]])
    return result


def _run_subprocess(
    command: Sequence[str],
    *,
    prompt: str,
    cwd: Path,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            list(command),
            input=prompt,
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CorrectionAgentError(f"{command[0]} invocation failed: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")[-1200:]
        raise CorrectionAgentError(
            f"{command[0]} exited with {result.returncode}: {detail or 'no diagnostic'}"
        )
    return result


def _run_codex(
    prompt: str,
    *,
    workspace: Path,
    schema_path: Path,
    result_path: Path,
    model: str | None,
    timeout: float,
) -> dict[str, Any]:
    executable = shutil.which("codex")
    if executable is None:
        raise CorrectionAgentError("codex CLI is not installed")
    result_path.unlink(missing_ok=True)
    command = [
        executable,
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--color",
        "never",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(result_path),
    ]
    if model:
        command.extend(["--model", model])
    command.append("-")
    process = _run_subprocess(command, prompt=prompt, cwd=workspace, timeout=timeout)
    if result_path.is_file():
        return _parse_json_object(result_path.read_text(encoding="utf-8"))
    return _parse_json_object(process.stdout)


def _run_claude(
    prompt: str,
    *,
    workspace: Path,
    model: str | None,
    timeout: float,
) -> dict[str, Any]:
    executable = shutil.which("claude")
    if executable is None:
        raise CorrectionAgentError("claude CLI is not installed")
    command = [
        executable,
        "--print",
        "--output-format",
        "json",
        "--json-schema",
        _canonical_json(CORRECTION_SCHEMA),
        "--no-session-persistence",
        "--safe-mode",
        "--permission-mode",
        "dontAsk",
        "--tools",
        "",
    ]
    if model:
        command.extend(["--model", model])
    process = _run_subprocess(command, prompt=prompt, cwd=workspace, timeout=timeout)
    envelope = _parse_json_object(process.stdout)
    structured = envelope.get("structured_output")
    if isinstance(structured, Mapping):
        return dict(structured)
    result = envelope.get("result")
    if isinstance(result, (str, Mapping)):
        return _parse_json_object(result)
    if "segments" in envelope:
        return envelope
    raise CorrectionAgentError("claude output has no structured result")


def _run_grok(
    prompt: str,
    *,
    workspace: Path,
    prompt_path: Path,
    model: str | None,
    timeout: float,
) -> dict[str, Any]:
    executable = shutil.which("grok")
    if executable is None:
        raise CorrectionAgentError("grok CLI is not installed")
    prompt_path.write_text(prompt, encoding="utf-8")
    command = [
        executable,
        "--prompt-file",
        str(prompt_path),
        "--json-schema",
        _canonical_json(CORRECTION_SCHEMA),
        "--max-turns",
        "1",
        "--no-subagents",
        "--disable-web-search",
        "--no-memory",
        "--permission-mode",
        "dontAsk",
        "--no-auto-update",
        "--verbatim",
    ]
    if model:
        command.extend(["--model", model])
    process = _run_subprocess(command, prompt="", cwd=workspace, timeout=timeout)
    envelope = _parse_json_object(process.stdout)
    structured = envelope.get("structured_output")
    if isinstance(structured, Mapping):
        return dict(structured)
    text = envelope.get("text")
    if isinstance(text, (str, Mapping)):
        return _parse_json_object(text)
    if "segments" in envelope:
        return envelope
    raise CorrectionAgentError("grok output has no structured result")


def _agent_model(config: CorrectionConfig, agent: str) -> str | None:
    return {
        "codex": config.codex_model,
        "claude": config.claude_model,
        "grok": config.grok_model,
    }[agent]


def _call_agents(
    prompt: str,
    *,
    target_ids: Sequence[str],
    config: CorrectionConfig,
    workspace: Path,
    schema_path: Path,
    temporary_prefix: Path,
) -> tuple[list[dict[str, str]], str, str | None, list[dict[str, str]]]:
    failures: list[dict[str, str]] = []
    for agent in config.agent_priority:
        model = _agent_model(config, agent)
        for attempt in range(1, config.attempts_per_agent + 1):
            try:
                if agent == "codex":
                    payload = _run_codex(
                        prompt,
                        workspace=workspace,
                        schema_path=schema_path,
                        result_path=temporary_prefix.with_suffix(".codex.json"),
                        model=model,
                        timeout=config.timeout_seconds,
                    )
                elif agent == "claude":
                    payload = _run_claude(
                        prompt,
                        workspace=workspace,
                        model=model,
                        timeout=config.timeout_seconds,
                    )
                else:
                    payload = _run_grok(
                        prompt,
                        workspace=workspace,
                        prompt_path=temporary_prefix.with_suffix(".prompt.txt"),
                        model=model,
                        timeout=config.timeout_seconds,
                    )
                return _validate_result(payload, target_ids), agent, model, failures
            except CorrectionAgentError as exc:
                failures.append(
                    {
                        "agent": agent,
                        "attempt": str(attempt),
                        "error": str(exc),
                    }
                )
    detail = "; ".join(f"{item['agent']}: {item['error']}" for item in failures[-3:])
    raise CorrectionAgentError(f"all correction agents failed: {detail}")


def _cache_path(cache_dir: Path, batch: _Batch) -> Path:
    source_digest = hashlib.sha1(batch.source_uid.encode("utf-8")).hexdigest()[:12]
    target_digest = hashlib.sha1("\n".join(batch.target_ids).encode("utf-8")).hexdigest()[:12]
    return cache_dir / source_digest / f"batch_{batch.ordinal:05d}_{target_digest}.json"


def _load_cache(path: Path, input_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("schema_version") != CORRECTION_CACHE_SCHEMA_VERSION
        or payload.get("prompt_version") != CORRECTION_PROMPT_VERSION
        or payload.get("input_hash") != input_hash
    ):
        return None
    return payload


def _safe_change(before: str, after: str, config: CorrectionConfig) -> tuple[bool, float, float]:
    compact_before = compact_for_similarity(before)
    compact_after = compact_for_similarity(after)
    if not compact_after:
        return False, 0.0, 0.0
    if not compact_before:
        return False, 0.0, math.inf
    similarity = difflib.SequenceMatcher(None, compact_before, compact_after).ratio()
    ratio = len(compact_after) / len(compact_before)
    # The compact metrics ignore punctuation entirely, so also bound the raw
    # length; unlimited punctuation-only inflation or deletion would shift the
    # pause-mark distribution the TTS model learns from.
    raw_ratio = len(after) / len(before) if before else math.inf
    safe = (
        similarity >= config.minimum_similarity
        and config.minimum_length_ratio <= ratio <= config.maximum_length_ratio
        and config.minimum_raw_length_ratio <= raw_ratio <= config.maximum_raw_length_ratio
    )
    return safe, similarity, ratio


def correct_transcript_rows(
    speech_rows: Sequence[Mapping[str, Any]],
    nonverbal_rows: Sequence[Mapping[str, Any]],
    *,
    cache_dir: str | Path,
    config: CorrectionConfig | None = None,
    force: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Correct both ASR streams in shared source timelines with cached CLI calls."""
    settings = config or CorrectionConfig()
    speech, nonverbal, grouped = _build_timeline(speech_rows, nonverbal_rows)
    batches = _make_batches(grouped, settings)
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    workspace = cache_root / "_agent_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    schema_path = cache_root / "correction.schema.json"
    _atomic_write_json(schema_path, CORRECTION_SCHEMA)

    by_id = {item.row_id: item for timeline in grouped.values() for item in timeline}
    suggestions: dict[str, dict[str, Any]] = {}
    batch_audit: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def review(batch: _Batch) -> tuple[list[dict[str, str]], dict[str, Any]]:
        prompt, input_payload = _build_prompt(batch)
        input_hash = _hash(
            {
                "prompt_version": CORRECTION_PROMPT_VERSION,
                "input": input_payload,
                "schema": CORRECTION_SCHEMA,
            }
        )
        path = _cache_path(cache_root, batch)
        cached = None if force else _load_cache(path, input_hash)
        if cached is not None:
            try:
                result = _validate_result(cached, batch.target_ids)
            except CorrectionAgentError:
                # A hash-matching but structurally invalid cache must not
                # permanently fail the batch; fall through to the agents.
                cached = None
            else:
                audit = {
                    "source_uid": batch.source_uid,
                    "ordinal": batch.ordinal,
                    "target_count": len(batch.target_ids),
                    "context_count": len(batch.timeline) - len(batch.target_ids),
                    "input_hash": input_hash,
                    "agent": cached.get("agent"),
                    "model": cached.get("model"),
                    "cached": True,
                    "agent_failures": cached.get("agent_failures", []),
                }
                return result, audit

        prefix = path.with_suffix("")
        path.parent.mkdir(parents=True, exist_ok=True)
        result, agent, model, agent_failures = _call_agents(
            prompt,
            target_ids=batch.target_ids,
            config=settings,
            workspace=workspace,
            schema_path=schema_path,
            temporary_prefix=prefix,
        )
        payload = {
            "schema_version": CORRECTION_CACHE_SCHEMA_VERSION,
            "prompt_version": CORRECTION_PROMPT_VERSION,
            "input_hash": input_hash,
            "source_uid": batch.source_uid,
            "target_ids": list(batch.target_ids),
            "agent": agent,
            "model": model,
            "agent_failures": agent_failures,
            "corrected_at": datetime.now(timezone.utc).isoformat(),
            "segments": result,
        }
        _atomic_write_json(path, payload)
        audit = {
            "source_uid": batch.source_uid,
            "ordinal": batch.ordinal,
            "target_count": len(batch.target_ids),
            "context_count": len(batch.timeline) - len(batch.target_ids),
            "input_hash": input_hash,
            "agent": agent,
            "model": model,
            "cached": False,
            "agent_failures": agent_failures,
        }
        return result, audit

    with ThreadPoolExecutor(max_workers=min(settings.workers, max(1, len(batches)))) as executor:
        future_to_batch = {executor.submit(review, batch): batch for batch in batches}
        for completed, future in enumerate(as_completed(future_to_batch), 1):
            batch = future_to_batch[future]
            try:
                result, audit = future.result()
            except Exception as exc:
                failures.append(
                    {
                        "source_uid": batch.source_uid,
                        "ordinal": batch.ordinal,
                        "target_ids": list(batch.target_ids),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"correction failed source={batch.source_uid} batch={batch.ordinal} error={exc}",
                    flush=True,
                )
                continue
            batch_audit.append(audit)
            for row in result:
                suggestions[row["id"]] = {**row, **audit}
            print(
                f"correction batches={completed}/{len(batches)} agent={audit['agent']} "
                f"cached={str(audit['cached']).lower()} source={batch.source_uid}",
                flush=True,
            )

    accepted = 0
    unchanged = 0
    uncertain = 0
    rejected = 0
    missing = 0
    change_rows: list[dict[str, Any]] = []
    for row_id, item in by_id.items():
        suggestion = suggestions.get(row_id)
        before = item.raw_text
        after = before
        outcome = "missing"
        similarity = 1.0
        length_ratio = 1.0
        if suggestion is None:
            missing += 1
        else:
            requested = normalize_transcript(suggestion["corrected_text"])
            status = suggestion["status"]
            if status == "uncertain":
                outcome = "uncertain"
                uncertain += 1
            elif status == "unchanged" or requested == before:
                outcome = "unchanged"
                unchanged += 1
            else:
                safe, similarity, length_ratio = _safe_change(before, requested, settings)
                if safe:
                    after = requested
                    outcome = "accepted"
                    accepted += 1
                else:
                    outcome = "rejected_unsafe"
                    rejected += 1
        target = speech[item.row_index] if item.kind == "speech" else nonverbal[item.row_index]
        llm_category = suggestion.get("category") if suggestion else None
        metadata = {
            "prompt_version": CORRECTION_PROMPT_VERSION,
            "status": outcome,
            "agent_status": suggestion.get("status") if suggestion else None,
            "agent": suggestion.get("agent") if suggestion else None,
            "model": suggestion.get("model") if suggestion else None,
            "input_hash": suggestion.get("input_hash") if suggestion else None,
            "similarity": round(similarity, 6),
            "length_ratio": round(length_ratio, 6),
            "category": llm_category,
        }
        if item.kind == "speech":
            target["asr_text_raw"] = before
            target["asr_backend"] = item.backend
            target["text"] = after
            target["text_correction"] = metadata
            # The LLM verdict is the published label; VAD said "speech" but
            # moans/lip noise leak through, so record what the content really is.
            target["category"] = llm_category or "speech"
            if llm_category == "other":
                # "other" means noise/unclassifiable — never trainable.
                review_reasons = [str(r) for r in target.get("review_reasons", []) or []]
                if "llm_category_other" not in review_reasons:
                    review_reasons.append("llm_category_other")
                target["review_reasons"] = review_reasons
                target["status"] = "review"
        else:
            target["transcript_text_raw"] = before
            target["transcript_text"] = after
            target["transcript_correction"] = metadata
            if llm_category:
                target["llm_category"] = llm_category
        if outcome in {"accepted", "rejected_unsafe", "uncertain"}:
            change_rows.append(
                {
                    "id": row_id,
                    "kind": item.kind,
                    "source_uid": item.source_uid,
                    "start": item.start,
                    "end": item.end,
                    "before": before,
                    "after": after,
                    "suggested": suggestion.get("corrected_text") if suggestion else None,
                    **metadata,
                }
            )

    summary = {
        "speech_rows": len(speech),
        "nonverbal_rows": len(nonverbal),
        "timeline_segments": len(by_id),
        "batches": len(batches),
        "batches_completed": len(batch_audit),
        "batches_failed": len(failures),
        "accepted": accepted,
        "unchanged": unchanged,
        "uncertain": uncertain,
        "rejected_unsafe": rejected,
        "missing": missing,
        "config": asdict(settings),
        "batch_audit": sorted(
            batch_audit,
            key=lambda row: (str(row["source_uid"]), int(row["ordinal"])),
        ),
        "failures": failures,
        "changes": change_rows,
    }
    return speech, nonverbal, summary
