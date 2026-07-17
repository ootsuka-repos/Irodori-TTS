"""Shared atomic-write and audio-file helpers for the data preparation pipeline.

Every writer combines the strongest guarantees previously scattered across the
data_prep modules: fsync-before-replace (crash safety) and bounded retries on
``PermissionError`` (Windows readers such as indexers/antivirus can briefly deny
``os.replace`` on otherwise local files).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_REPLACE_ATTEMPTS = 8
_REPLACE_BACKOFF_SECONDS = 0.025


def replace_with_retry(temporary: Path, path: Path) -> None:
    """``os.replace`` with retries for transient Windows sharing violations."""
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_SECONDS * (attempt + 1))


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def json_default(value: Any) -> Any:
    """Serialize Path and numpy scalars; raise for anything else."""
    if isinstance(value, Path):
        return value.as_posix()
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_write_json(path: Path, payload: Any, *, default: Any = None) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=default) + "\n",
    )


def atomic_write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    default: Any = None,
) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"), default=default)
            + "\n"
            for row in rows
        ),
    )


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, default: Any = None) -> None:
    """Append rows durably; readers must apply last-write-wins per logical key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"), default=default) + "\n"
        for row in rows
    )
    if not text:
        return
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # utf-8-sig tolerates BOM-prefixed files produced by Windows editors.
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_save_npy(path: Path, array: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_flac(path: Path, samples: Any, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.flac")
    try:
        sf.write(temporary, samples, sample_rate, format="FLAC", subtype="PCM_16")
        replace_with_retry(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def valid_audio_file(path: Path) -> bool:
    """True when the file exists and decodes to at least one frame."""
    import soundfile as sf

    if not path.is_file() or path.stat().st_size <= 44:
        return False
    try:
        return sf.info(path).frames > 0
    except (RuntimeError, TypeError):
        return False
