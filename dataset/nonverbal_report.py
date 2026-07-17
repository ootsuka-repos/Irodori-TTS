"""Materialize and summarize finalized ``aegi``/``chupa`` training events.

The report deliberately operates on already-cut audio clips.  It does not
decode or crop source tracks, so producing the review folders stays cheap even
for tens of thousands of events.  Every class member is a hard link when the
filesystem supports it and a byte-for-byte copy otherwise.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset._io_utils import atomic_write_bytes, json_default

REPORT_SCHEMA_VERSION = 2
EVENT_CLASSES = ("aegi", "chupa")
CLUSTERED_CLASSES = frozenset(EVENT_CLASSES)

_WINDOWS_FORBIDDEN = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
)
_DURATION_EDGES = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 60.0)
_PROBABILITY_EDGES = tuple(index / 10 for index in range(11))


@dataclass(frozen=True)
class NonverbalReportConfig:
    """Controls the bounded HTML payload and representative selection."""

    representatives_per_cluster: int = 3
    html_cluster_limit: int = 500

    def __post_init__(self) -> None:
        if self.representatives_per_cluster < 1:
            raise ValueError("representatives_per_cluster must be positive")
        if self.html_cluster_limit < 1:
            raise ValueError("html_cluster_limit must be positive")


@dataclass(frozen=True)
class _Event:
    row: Mapping[str, Any]
    event_id: str
    label: str
    cluster: str | None
    audio_path: Path
    extension: str
    duration: float
    probability: float | None
    source_identity: str


# Shared fsync-and-retry writers from _io_utils; the local wrappers only add
# BOM handling for the CSV and bind the project JSON encoder.
_json_default = json_default


def _atomic_write_text(path: Path, text: str, *, bom: bool = False) -> None:
    encoding = "utf-8-sig" if bom else "utf-8"
    atomic_write_bytes(path, text.encode(encoding))


def _atomic_write_json(path: Path, payload: Any) -> None:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
        default=_json_default,
    )
    _atomic_write_text(path, text + "\n")


def _safe_component(value: Any, *, field: str) -> str:
    component = str(value).strip()
    if not component:
        raise ValueError(f"{field} must not be empty")
    if component in {".", ".."} or component.endswith((".", " ")):
        raise ValueError(f"Unsafe {field}: {component!r}")
    if any(character in _WINDOWS_FORBIDDEN or ord(character) < 32 for character in component):
        raise ValueError(f"Unsafe {field}: {component!r}")
    if component.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        raise ValueError(f"Unsafe {field}: {component!r}")
    if len(component) > 180:
        raise ValueError(f"{field} is too long")
    return component


def _event_label(row: Mapping[str, Any]) -> str:
    value = row.get("final_label", row.get("event_class", "uncertain"))
    label = str(value or "uncertain").strip().lower()
    if label not in EVENT_CLASSES:
        raise ValueError(f"Unsupported final_label: {label!r}")
    return label


def _event_cluster(row: Mapping[str, Any], label: str) -> str | None:
    if label not in CLUSTERED_CLASSES:
        return None
    value = row.get("class_cluster")
    if value is None or not str(value).strip():
        value = row.get("event_cluster")
    if value is None or not str(value).strip():
        value = row.get("fallback_cluster")
    if value is None or not str(value).strip():
        value = "unclustered"
    return _safe_component(value, field="class_cluster")


def _event_duration(row: Mapping[str, Any]) -> float:
    value = row.get("duration")
    if value is None:
        try:
            value = float(row["end"]) - float(row["start"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Each row needs duration, or numeric start/end") from exc
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid duration: {value!r}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"duration must be positive and finite: {value!r}")
    return duration


def _probability_mapping(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = row.get("probabilities")
    if isinstance(value, Mapping):
        return value
    for field in ("prediction", "evidence"):
        nested = row.get(field)
        if isinstance(nested, Mapping) and isinstance(nested.get("probabilities"), Mapping):
            return nested["probabilities"]
    return None


def _event_probability(row: Mapping[str, Any], label: str) -> float | None:
    probabilities = _probability_mapping(row)
    value: Any = probabilities.get(label) if probabilities is not None else None
    if value is None:
        for field in ("final_probability", "final_confidence", "confidence"):
            if row.get(field) is not None:
                value = row[field]
                break
    if value is None:
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return None
    return probability


def _resolve_audio_path(row: Mapping[str, Any], audio_root: Path) -> Path:
    raw_path: Any = None
    for field in ("audio", "clip_audio", "source"):
        if row.get(field) is not None and str(row[field]).strip():
            raw_path = row[field]
            break
    if raw_path is None:
        raise ValueError("Each row needs an already-cut audio, clip_audio, or source path")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = audio_root / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"Event audio does not exist: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"Event audio is not a file: {resolved}")
    return resolved


def _source_identity(row: Mapping[str, Any], audio_path: Path) -> str:
    for field in ("source_uid", "source_key", "source_audio"):
        if row.get(field) is not None and str(row[field]).strip():
            return str(row[field]).strip()
    return audio_path.as_posix()


def _normalize_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    audio_root: Path,
) -> tuple[list[_Event], int]:
    events: list[_Event] = []
    identities: dict[str, tuple[Any, ...]] = {}
    duplicate_count = 0
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Row {row_index} is not an object")
        event_id = _safe_component(row.get("id", ""), field="id")
        label = _event_label(row)
        cluster = _event_cluster(row, label)
        duration = _event_duration(row)
        audio_path = _resolve_audio_path(row, audio_root)
        probability = _event_probability(row, label)
        extension = audio_path.suffix.lower()
        if not extension or len(extension) > 12 or any(char in _WINDOWS_FORBIDDEN for char in extension):
            extension = ".audio"
        identity = (
            label,
            cluster,
            os.path.normcase(str(audio_path)),
            round(duration, 9),
            row.get("start"),
            row.get("end"),
        )
        previous = identities.get(event_id)
        if previous is not None:
            if previous != identity:
                raise ValueError(f"Duplicate id has conflicting rows: {event_id!r}")
            duplicate_count += 1
            continue
        identities[event_id] = identity
        events.append(
            _Event(
                row=row,
                event_id=event_id,
                label=label,
                cluster=cluster,
                audio_path=audio_path,
                extension=extension,
                duration=duration,
                probability=probability,
                source_identity=_source_identity(row, audio_path),
            )
        )
    events.sort(key=lambda event: (EVENT_CLASSES.index(event.label), event.cluster or "", event.event_id))
    return events, duplicate_count


def _relative_member_path(event: _Event) -> Path:
    filename = f"{event.event_id}{event.extension}"
    if event.label in CLUSTERED_CLASSES:
        assert event.cluster is not None
        return Path(event.label) / event.cluster / filename
    return Path(event.label) / filename


def _materialize_classes(events: Sequence[_Event], output_dir: Path) -> tuple[int, int]:
    classes_root = output_dir / "classes"
    if classes_root.is_symlink():
        raise ValueError(f"Refusing to replace symlinked classes directory: {classes_root}")
    stage = Path(tempfile.mkdtemp(dir=output_dir, prefix=".classes.", suffix=".tmp"))
    hardlinks = 0
    copies = 0
    try:
        for label in EVENT_CLASSES:
            (stage / label).mkdir(parents=True, exist_ok=True)
        playlists: dict[Path, list[str]] = defaultdict(lambda: ["#EXTM3U"])
        for event in events:
            relative = _relative_member_path(event)
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(event.audio_path, destination)
                hardlinks += 1
            except OSError:
                shutil.copy2(event.audio_path, destination)
                copies += 1
            playlists[destination.parent].extend(
                [
                    f"#EXTINF:{event.duration:.6f},{event.event_id}",
                    destination.name,
                ]
            )
        for directory, lines in playlists.items():
            (directory / "members.m3u8").write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
                newline="\n",
            )

        backup = output_dir / f".classes.{uuid.uuid4().hex}.old"
        moved_existing = False
        committed = False
        try:
            if classes_root.exists():
                if not classes_root.is_dir():
                    raise ValueError(f"classes exists but is not a directory: {classes_root}")
                os.replace(classes_root, backup)
                moved_existing = True
            os.replace(stage, classes_root)
            committed = True
        except BaseException:
            if moved_existing and backup.exists() and not classes_root.exists():
                os.replace(backup, classes_root)
            raise
        finally:
            # If restoring the previous tree itself failed, keep its backup for
            # recovery instead of turning a replace error into data loss.
            if committed and backup.exists():
                shutil.rmtree(backup)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return hardlinks, copies


def _quantile(sorted_values: Sequence[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": _quantile(ordered, 0.10),
        "median": _quantile(ordered, 0.50),
        "p90": _quantile(ordered, 0.90),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _fixed_histogram(values: Sequence[float], edges: Sequence[float]) -> dict[str, Any]:
    counts = [0] * len(edges)
    for value in values:
        index = len(edges) - 1
        for candidate in range(len(edges) - 1):
            if value < edges[candidate + 1]:
                index = candidate
                break
        counts[index] += 1
    labels = [
        f"{edges[index]:g}–{edges[index + 1]:g}"
        for index in range(len(edges) - 1)
    ] + [f"≥{edges[-1]:g}"]
    return {"labels": labels, "counts": counts}


def _distribution(events: Sequence[_Event]) -> dict[str, Any]:
    probabilities = [event.probability for event in events if event.probability is not None]
    return {
        "duration_seconds": {
            "statistics": _numeric_summary([event.duration for event in events]),
            "histogram": _fixed_histogram(
                [event.duration for event in events],
                _DURATION_EDGES,
            ),
        },
        "final_label_probability": {
            "statistics": _numeric_summary(probabilities),
            "histogram": _fixed_histogram(probabilities, _PROBABILITY_EDGES),
        },
    }


def _distance(event: _Event) -> float:
    value = event.row.get("distance")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.inf
    return result if math.isfinite(result) else math.inf


def _select_representatives(events: Sequence[_Event], count: int) -> list[_Event]:
    ordered = sorted(
        events,
        key=lambda event: (
            _distance(event),
            -(event.probability if event.probability is not None else -1.0),
            event.event_id,
        ),
    )
    selected: list[_Event] = []
    selected_ids: set[str] = set()
    used_sources: set[str] = set()
    for require_new_source in (True, False):
        for event in ordered:
            if len(selected) >= count:
                return selected
            if event.event_id in selected_ids:
                continue
            if require_new_source and event.source_identity in used_sources:
                continue
            selected.append(event)
            selected_ids.add(event.event_id)
            used_sources.add(event.source_identity)
    return selected


def _representative_row(event: _Event) -> dict[str, Any]:
    return {
        "id": event.event_id,
        "duration": event.duration,
        "probability": event.probability,
        "source_uid": str(event.row.get("source_uid", event.row.get("source_key", ""))),
        "audio": (Path("classes") / _relative_member_path(event)).as_posix(),
    }


def _group_summary(
    label: str,
    cluster: str | None,
    events: Sequence[_Event],
    *,
    representatives_per_cluster: int,
) -> dict[str, Any]:
    return {
        "label": label,
        "cluster": cluster,
        "event_count": len(events),
        "total_seconds": sum(event.duration for event in events),
        "distribution": _distribution(events),
        "representatives": [
            _representative_row(event)
            for event in _select_representatives(events, representatives_per_cluster)
        ],
    }


def _build_summary(
    events: Sequence[_Event],
    *,
    duplicate_count: int,
    hardlinks: int,
    copies: int,
    config: NonverbalReportConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_label: dict[str, list[_Event]] = defaultdict(list)
    by_group: dict[tuple[str, str | None], list[_Event]] = defaultdict(list)
    for event in events:
        by_label[event.label].append(event)
        by_group[(event.label, event.cluster)].append(event)

    flat_groups: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    for label in EVENT_CLASSES:
        label_events = by_label[label]
        clusters = [
            _group_summary(
                group_label,
                cluster,
                group_events,
                representatives_per_cluster=config.representatives_per_cluster,
            )
            for (group_label, cluster), group_events in sorted(
                by_group.items(),
                key=lambda item: (item[0][0], item[0][1] or ""),
            )
            if group_label == label
        ]
        flat_groups.extend(clusters)
        classes.append(
            {
                "label": label,
                "event_count": len(label_events),
                "total_seconds": sum(event.duration for event in label_events),
                "cluster_count": len(clusters) if label in CLUSTERED_CLASSES else 0,
                "distribution": _distribution(label_events),
                "clusters": clusters,
            }
        )

    dashboard_groups = sorted(
        flat_groups,
        key=lambda group: (-int(group["event_count"]), str(group["label"]), group["cluster"] or ""),
    )[: config.html_cluster_limit]
    summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "event_count": len(events),
        "duplicate_rows_ignored": duplicate_count,
        "total_seconds": sum(event.duration for event in events),
        "class_counts": {label: len(by_label[label]) for label in EVENT_CLASSES},
        "class_seconds": {
            label: sum(event.duration for event in by_label[label]) for label in EVENT_CLASSES
        },
        "distribution": _distribution(events),
        "classes": classes,
        "materialization": {
            "hardlinks_created": hardlinks,
            "fallback_copies_created": copies,
        },
        "dashboard": {
            "group_count": len(flat_groups),
            "included_group_count": len(dashboard_groups),
            "omitted_group_count": len(flat_groups) - len(dashboard_groups),
            "representatives_per_cluster": config.representatives_per_cluster,
        },
        "files": {
            "classes": "classes/",
            "summary_json": "summary.json",
            "summary_csv": "summary.csv",
            "dashboard_html": "dashboard.html",
        },
    }
    return summary, dashboard_groups


def _summary_csv(summary: Mapping[str, Any]) -> str:
    fields = (
        "label",
        "cluster",
        "event_count",
        "total_seconds",
        "mean_duration_seconds",
        "median_duration_seconds",
        "probability_count",
        "mean_probability",
        "median_probability",
        "representative_ids",
        "representative_audio",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for class_row in summary["classes"]:
        groups = class_row["clusters"]
        if not groups:
            groups = [
                {
                    "label": class_row["label"],
                    "cluster": None,
                    "event_count": class_row["event_count"],
                    "total_seconds": class_row["total_seconds"],
                    "distribution": class_row["distribution"],
                    "representatives": [],
                }
            ]
        for group in groups:
            duration = group["distribution"]["duration_seconds"]["statistics"]
            probability = group["distribution"]["final_label_probability"]["statistics"]
            representatives = group["representatives"]
            writer.writerow(
                {
                    "label": group["label"],
                    "cluster": group["cluster"] or "",
                    "event_count": group["event_count"],
                    "total_seconds": f"{float(group['total_seconds']):.6f}",
                    "mean_duration_seconds": _csv_number(duration["mean"]),
                    "median_duration_seconds": _csv_number(duration["median"]),
                    "probability_count": probability["count"],
                    "mean_probability": _csv_number(probability["mean"]),
                    "median_probability": _csv_number(probability["median"]),
                    "representative_ids": "|".join(row["id"] for row in representatives),
                    "representative_audio": "|".join(row["audio"] for row in representatives),
                }
            )
    return buffer.getvalue()


def _csv_number(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"


def _safe_json_for_html(payload: Any) -> str:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_json_default)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _dashboard_html(summary: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "overview": {
            "event_count": summary["event_count"],
            "total_seconds": summary["total_seconds"],
            "class_counts": summary["class_counts"],
            "class_seconds": summary["class_seconds"],
            "distribution": summary["distribution"],
            "dashboard": summary["dashboard"],
        },
        "groups": list(groups),
    }
    data = _safe_json_for_html(payload)
    template = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>非言語音声データセット</title>
<style>
:root{color-scheme:dark;background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif}*{box-sizing:border-box}
body{max-width:1500px;margin:auto;padding:20px}h1{font-size:22px;margin:0 0 8px}.muted{color:#9da7b3}.stats,.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}.pill,.card,.chart{background:#161b22;border:1px solid #30363d;border-radius:9px;padding:10px}.pill{min-width:135px}.pill b{display:block;font-size:19px}
.toolbar{position:sticky;top:0;background:#0d1117e8;padding:8px 0;z-index:2}input,select{background:#161b22;color:#e6edf3;border:1px solid #484f58;border-radius:7px;padding:8px}.charts{display:grid;grid-template-columns:1fr 1fr;gap:10px}.bars{height:105px;display:flex;align-items:end;gap:3px}.bar{flex:1;background:#58a6ff;min-height:1px;border-radius:3px 3px 0 0}.labels{font-size:10px;color:#8b949e;display:flex;justify-content:space-between}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:10px}.card h2{font-size:16px;margin:0 0 5px}.meta{font-size:13px;color:#aab4c0}.audio-row{border-top:1px solid #30363d;padding-top:8px;margin-top:8px;font-size:12px}audio{display:block;width:100%;height:34px;margin-top:4px}.aegi{border-top:3px solid #d2a8ff}.chupa{border-top:3px solid #ff7b72}
@media(max-width:700px){.charts{grid-template-columns:1fr}}
</style></head><body>
<h1>非言語音声データセット</h1><div id="note" class="muted"></div><div id="stats" class="stats"></div>
<div class="charts"><div class="chart"><b>長さ（秒）</b><div id="duration"></div></div><div class="chart"><b>分類確率</b><div id="probability"></div></div></div>
<div class="toolbar"><select id="class-filter"><option value="">全クラス</option><option>aegi</option><option>chupa</option></select><input id="search" type="search" placeholder="クラスタを検索"><span id="visible" class="muted"></span></div><div id="grid" class="grid"></div>
<script id="report-data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('report-data').textContent),grid=document.getElementById('grid');
const fmt=x=>Number(x).toLocaleString(undefined,{maximumFractionDigits:1}), esc=x=>document.createTextNode(String(x));
function pill(name,count,seconds){let e=document.createElement('div');e.className='pill';let b=document.createElement('b');b.append(esc(fmt(count)));e.append(b,esc(`${name} · ${fmt(seconds)}秒`));return e}
document.getElementById('stats').append(pill('全部',data.overview.event_count,data.overview.total_seconds));for(const c of ['aegi','chupa'])document.getElementById('stats').append(pill(c,data.overview.class_counts[c],data.overview.class_seconds[c]));
const dash=data.overview.dashboard;document.getElementById('note').textContent=`代表音のみ掲載: ${dash.included_group_count}/${dash.group_count} グループ（省略 ${dash.omitted_group_count}）`;
function chart(id,h){let root=document.getElementById(id),bars=document.createElement('div');bars.className='bars';let max=Math.max(1,...h.counts);h.counts.forEach((n,i)=>{let b=document.createElement('div');b.className='bar';b.style.height=(100*n/max)+'%';b.title=`${h.labels[i]}: ${n}`;bars.append(b)});let labels=document.createElement('div');labels.className='labels';labels.append(esc(h.labels[0]),esc(h.labels[h.labels.length-1]));root.append(bars,labels)}
chart('duration',data.overview.distribution.duration_seconds.histogram);chart('probability',data.overview.distribution.final_label_probability.histogram);
function render(){let cls=document.getElementById('class-filter').value,q=document.getElementById('search').value.trim().toLowerCase(),shown=0;grid.replaceChildren();for(const g of data.groups){let key=(g.cluster||g.label).toLowerCase();if((cls&&g.label!==cls)||(q&&!key.includes(q)))continue;shown++;let card=document.createElement('section');card.className='card '+g.label;let h=document.createElement('h2');h.append(esc(`${g.label} / ${g.cluster||'all'}`));let m=document.createElement('div');m.className='meta';m.append(esc(`${fmt(g.event_count)}件 · ${fmt(g.total_seconds)}秒`));card.append(h,m);for(const r of g.representatives){let row=document.createElement('div');row.className='audio-row';row.append(esc(`${r.id} · ${fmt(r.duration)}秒${r.probability===null?'':` · p=${r.probability.toFixed(3)}`}`));let audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=r.audio;row.append(audio);card.append(row)}grid.append(card)}document.getElementById('visible').textContent=`${shown} グループ`}
document.getElementById('class-filter').addEventListener('change',render);document.getElementById('search').addEventListener('input',render);render();
</script></body></html>"""
    return template.replace("__DATA__", data)


def write_nonverbal_report(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    audio_root: Path | None = None,
    config: NonverbalReportConfig | None = None,
) -> dict[str, Any]:
    """Write target-only class folders, compact summaries, and a dashboard.

    ``audio`` (or ``clip_audio``/``source``) must point to an already-cut event
    clip.  Relative audio paths are resolved below ``audio_root``; when it is
    omitted they are resolved below ``output_dir``.  Only finalized ``aegi`` and
    ``chupa`` rows are accepted.  Exact duplicate IDs are
    collapsed, while conflicting duplicate IDs fail before any output changes.
    """
    config = config or NonverbalReportConfig()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(audio_root).expanduser().resolve() if audio_root is not None else output_dir
    events, duplicate_count = _normalize_events(rows, audio_root=root)

    hardlinks, copies = _materialize_classes(events, output_dir)
    summary, dashboard_groups = _build_summary(
        events,
        duplicate_count=duplicate_count,
        hardlinks=hardlinks,
        copies=copies,
        config=config,
    )
    _atomic_write_text(output_dir / "summary.csv", _summary_csv(summary), bom=True)
    _atomic_write_text(output_dir / "dashboard.html", _dashboard_html(summary, dashboard_groups))
    # This is the completion marker and is intentionally committed last.
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary
