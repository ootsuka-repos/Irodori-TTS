"""Compatibility facade for the reorganized inference package.

New code should import from :mod:`irodori_tts.inference.runtime`.
"""

from .inference.postprocessing import find_flattening_point, find_flattening_points
from .inference.runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    SamplingResult,
    _load_checkpoint_for_inference,
    clear_cached_runtime,
    default_runtime_device,
    get_cached_runtime,
    list_available_runtime_devices,
    list_available_runtime_precisions,
    resolve_cfg_scales,
    resolve_runtime_device,
    resolve_runtime_dtype,
    save_wav,
)

__all__ = [
    "InferenceRuntime",
    "RuntimeKey",
    "SamplingRequest",
    "SamplingResult",
    "_load_checkpoint_for_inference",
    "clear_cached_runtime",
    "default_runtime_device",
    "find_flattening_point",
    "find_flattening_points",
    "get_cached_runtime",
    "list_available_runtime_devices",
    "list_available_runtime_precisions",
    "resolve_cfg_scales",
    "resolve_runtime_device",
    "resolve_runtime_dtype",
    "save_wav",
]
