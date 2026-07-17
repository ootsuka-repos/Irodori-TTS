"""Shared model, config, codec and sampling primitives used by all layers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import ModelConfig, TrainConfig
    from .model import TextToLatentRFDiT
    from .tokenizer import PretrainedTextTokenizer

__all__ = [
    "LORA_TARGET_PRESETS",
    "ModelConfig",
    "PretrainedTextTokenizer",
    "TextToLatentRFDiT",
    "TrainConfig",
]

_LAZY_EXPORTS = {
    "LORA_TARGET_PRESETS": (".lora", "LORA_TARGET_PRESETS"),
    "ModelConfig": (".config", "ModelConfig"),
    "PretrainedTextTokenizer": (".tokenizer", "PretrainedTextTokenizer"),
    "TextToLatentRFDiT": (".model", "TextToLatentRFDiT"),
    "TrainConfig": (".config", "TrainConfig"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
