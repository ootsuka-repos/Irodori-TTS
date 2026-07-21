#!/usr/bin/env python3
"""Backward-compatible shim: this stage was renamed to ``prepare_transcribe``.

The stage historically called ``anime_whisper`` actually runs faster-whisper
(TransWithAI/whisper-ja-1.5B-ct2) by default, so it was renamed to
``transcribe`` for clarity. This module remains as a thin alias so any pinned
command or in-flight subprocess invoking ``dataset.cli.prepare_anime_whisper``
keeps working. Prefer ``dataset.cli.prepare_transcribe``.
"""

from __future__ import annotations

from dataset.cli.prepare_transcribe import main

if __name__ == "__main__":
    main()
