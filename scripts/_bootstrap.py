"""Shared bootstrap for running repository scripts without installation."""

import sys
from importlib import import_module
from pathlib import Path


def run(module_name: str) -> None:
    """Import ``module_name`` from the project root and call its ``main``."""
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    entrypoint = import_module(module_name).main
    entrypoint()
