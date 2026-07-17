"""Local weak labeling with litagin's Japanese Ero Voice Classifier.

The original Hugging Face Space classifies a whole audio clip into three closed-set
labels: ``usual``, ``aegi``, and ``chupa``.  This module reproduces that model locally;
audio is read from disk and is never sent to the Space or another external API.

The classifier is intentionally exposed as a weak labeler.  It has no unknown class,
so its confidence must not be interpreted as an out-of-distribution score.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ERO_LABELS = ("usual", "aegi", "chupa")
ERO_LABEL_TO_ID = {label: index for index, label in enumerate(ERO_LABELS)}

SPACE_REPO_ID = "litagin/Japanese-Ero-Voice-Classifier"
SPACE_REPO_TYPE = "space"
SPACE_REVISION = "174f89c6201668ad12402971ffdf74cb3e07ef97"
SPACE_CONFIG_FILENAME = "ckpt/config.json"
# Digest of the LF-normalized blob served by hf_hub_download at SPACE_REVISION.
SPACE_CONFIG_SHA256 = "f481cf3602e580d3810a390f983b52c6839af503d918bc434acfaad192e60a3f"
SPACE_CHECKPOINT_FILENAME = "ckpt/model_final.pth"
SPACE_CHECKPOINT_SHA256 = "67ffab6e224d9c7f9acbeab40892cfda200a88c9dc2ee2714621bc90eed7a4d5"

WESPEAKER_REPO_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
WESPEAKER_REVISION = "837717ddb9ff5507820346191109dc79c958d614"
WESPEAKER_CONFIG_FILENAME = "config.yaml"
WESPEAKER_CONFIG_SHA256 = "6ff718cff3c5d7a4493537ab7f4780cad7e3d32453f59099b4076aefa07a9974"
WESPEAKER_CHECKPOINT_FILENAME = "pytorch_model.bin"
WESPEAKER_CHECKPOINT_SHA256 = "366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56"
WESPEAKER_FEATURE_DIM = 256
WESPEAKER_SAMPLE_RATE = 16_000

_COMPATIBILITY_LOCK = threading.RLock()
_MISSING = object()


class EroVoiceClassifierError(RuntimeError):
    """Base error raised by the local weak classifier."""


class EroVoiceArtifactIntegrityError(EroVoiceClassifierError):
    """Raised when a pinned model artifact does not match its expected digest."""


class EroVoiceDependencyError(EroVoiceClassifierError):
    """Raised when the optional pyannote runtime is unavailable."""


@dataclass(frozen=True)
class EroVoiceArtifacts:
    """Verified local paths for both parts of the classifier."""

    space_config: Path
    space_checkpoint: Path
    wespeaker_config: Path
    wespeaker_checkpoint: Path


@dataclass(frozen=True)
class EroVoicePrediction:
    """Per-file closed-set probabilities returned in the original label order.

    A failed clip is returned as a marker with ``error`` set; its probability
    accessors raise so downstream consumers cannot mistake it for a real
    prediction (batch callers with per-row error handling convert it into an
    error row instead of losing the whole batch).
    """

    audio: str
    usual: float
    aegi: float
    chupa: float
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    @property
    def probabilities(self) -> dict[str, float]:
        if self.error is not None:
            raise EroVoiceClassifierError(f"prediction failed for {self.audio}: {self.error}")
        return {"usual": self.usual, "aegi": self.aegi, "chupa": self.chupa}

    @property
    def label(self) -> str:
        probabilities = self.probabilities
        return max(ERO_LABELS, key=probabilities.__getitem__)

    @property
    def confidence(self) -> float:
        return self.probabilities[self.label]

    def to_dict(self) -> dict[str, Any]:
        if self.error is not None:
            return {"audio": self.audio, "error": self.error}
        return {
            "audio": self.audio,
            "probabilities": self.probabilities,
            "label": self.label,
            "confidence": self.confidence,
        }


def logits_to_probabilities(
    logits: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
) -> list[dict[str, float]]:
    """Convert a batch of three logits into numerically stable probabilities."""
    if isinstance(logits, torch.Tensor):
        values = logits.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape[1] != len(ERO_LABELS):
        raise ValueError(f"logits must have shape [batch, {len(ERO_LABELS)}]")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite")

    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    return [
        {label: float(row[index]) for index, label in enumerate(ERO_LABELS)}
        for row in probabilities
    ]


class JapaneseEroVoiceMLP(nn.Module):
    """Exact MLP architecture used by the pinned Hugging Face Space."""

    def __init__(
        self,
        label2id: Mapping[str, int],
        feature_dim: int = WESPEAKER_FEATURE_DIM,
        hidden_dim: int = 256,
        dropout_rate: float = 0.5,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.label2id = dict(label2id)
        self.fc1 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Mish(),
                    nn.Dropout(dropout_rate),
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.fc_last = nn.Linear(hidden_dim, len(self.label2id))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(features)
        for layer in self.hidden_layers:
            hidden = layer(hidden)
        return self.fc_last(hidden)


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_hf_download(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    filename: str,
    expected_sha256: str,
    cache_dir: Path | None,
    local_files_only: bool,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - a core project dependency
        raise EroVoiceDependencyError("huggingface-hub is required to fetch model files") from exc

    arguments: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "filename": filename,
        "local_files_only": local_files_only,
    }
    if cache_dir is not None:
        arguments["cache_dir"] = str(cache_dir)
    path = Path(hf_hub_download(**arguments))
    actual = sha256_file(path)
    if actual != expected_sha256 and not local_files_only:
        arguments["force_download"] = True
        path = Path(hf_hub_download(**arguments))
        actual = sha256_file(path)
    if actual != expected_sha256:
        raise EroVoiceArtifactIntegrityError(
            f"SHA-256 mismatch for {repo_id}@{revision}/{filename}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return path


def fetch_ero_voice_artifacts(
    *,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> EroVoiceArtifacts:
    """Download only pinned model files and verify every file before loading it."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
    space_config = _verified_hf_download(
        repo_id=SPACE_REPO_ID,
        repo_type=SPACE_REPO_TYPE,
        revision=SPACE_REVISION,
        filename=SPACE_CONFIG_FILENAME,
        expected_sha256=SPACE_CONFIG_SHA256,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    space_checkpoint = _verified_hf_download(
        repo_id=SPACE_REPO_ID,
        repo_type=SPACE_REPO_TYPE,
        revision=SPACE_REVISION,
        filename=SPACE_CHECKPOINT_FILENAME,
        expected_sha256=SPACE_CHECKPOINT_SHA256,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    wespeaker_config = _verified_hf_download(
        repo_id=WESPEAKER_REPO_ID,
        repo_type="model",
        revision=WESPEAKER_REVISION,
        filename=WESPEAKER_CONFIG_FILENAME,
        expected_sha256=WESPEAKER_CONFIG_SHA256,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    wespeaker_checkpoint = _verified_hf_download(
        repo_id=WESPEAKER_REPO_ID,
        repo_type="model",
        revision=WESPEAKER_REVISION,
        filename=WESPEAKER_CHECKPOINT_FILENAME,
        expected_sha256=WESPEAKER_CHECKPOINT_SHA256,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return EroVoiceArtifacts(
        space_config=space_config,
        space_checkpoint=space_checkpoint,
        wespeaker_config=wespeaker_config,
        wespeaker_checkpoint=wespeaker_checkpoint,
    )


@contextmanager
def _pyannote_import_compatibility() -> Any:
    """Expose removed aliases only while importing legacy pyannote.audio 3.x."""
    with _COMPATIBILITY_LOCK:
        numpy_nan = getattr(np, "NaN", _MISSING)
        if numpy_nan is _MISSING:
            np.NaN = np.nan

        try:
            torchaudio = importlib.import_module("torchaudio")
        except ImportError as exc:  # pragma: no cover - a core project dependency
            if numpy_nan is _MISSING:
                delattr(np, "NaN")
            raise EroVoiceDependencyError("torchaudio is required by pyannote.audio") from exc

        audio_backend = getattr(torchaudio, "set_audio_backend", _MISSING)
        if audio_backend is _MISSING:

            def set_audio_backend_compatibility(_backend: str) -> None:
                return None

            torchaudio.set_audio_backend = set_audio_backend_compatibility

        try:
            yield
        finally:
            if audio_backend is _MISSING:
                delattr(torchaudio, "set_audio_backend")
            if numpy_nan is _MISSING:
                delattr(np, "NaN")


def _import_pyannote_runtime() -> tuple[Any, Any, list[type[Any]], str]:
    try:
        with _pyannote_import_compatibility():
            pyannote_audio = importlib.import_module("pyannote.audio")
            task_module = importlib.import_module("pyannote.audio.core.task")
    except ModuleNotFoundError as exc:
        if exc.name == "pyannote" or (exc.name and exc.name.startswith("pyannote.")):
            raise EroVoiceDependencyError(
                "Japanese Ero Voice Classifier requires optional dependency "
                "'pyannote.audio' (tested with 3.1.1). Install it in the runtime "
                "environment before enabling this weak labeler."
            ) from exc
        raise

    from torch.torch_version import TorchVersion

    safe_globals = [
        TorchVersion,
        task_module.Specifications,
        task_module.Problem,
        task_module.Resolution,
    ]
    return (
        pyannote_audio.Model,
        pyannote_audio.Inference,
        safe_globals,
        str(getattr(pyannote_audio, "__version__", "unknown")),
    )


def _load_space_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_config = payload.get("model")
    if not isinstance(model_config, dict):
        raise EroVoiceClassifierError("Space config does not contain a model object")
    label2id = model_config.get("label2id")
    if label2id != ERO_LABEL_TO_ID:
        raise EroVoiceClassifierError(
            f"Unexpected Space labels: expected {ERO_LABEL_TO_ID}, got {label2id}"
        )
    return model_config


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise EroVoiceClassifierError("CUDA was requested but is not available")
    return resolved


class JapaneseEroVoiceClassifier:
    """Local whole-file classifier suitable for weak cluster annotations."""

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        cache_dir: Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.device = _resolve_device(device)
        model_class, inference_class, safe_globals, pyannote_version = _import_pyannote_runtime()
        self.artifacts = fetch_ero_voice_artifacts(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        model_config = _load_space_config(self.artifacts.space_config)
        self.head = JapaneseEroVoiceMLP(**model_config)
        state_dict = torch.load(
            self.artifacts.space_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state_dict, Mapping):
            raise EroVoiceClassifierError("Space checkpoint is not a state dictionary")
        self.head.load_state_dict(state_dict, strict=True)
        self.head.to(self.device).eval()

        # The pinned official checkpoint contains four known non-tensor metadata
        # types. safe_globals keeps Torch 2.10's weights-only loader enabled and
        # avoids changing torch.load process-wide.
        with torch.serialization.safe_globals(safe_globals):
            embedding_model = model_class.from_pretrained(
                str(self.artifacts.wespeaker_checkpoint),
                map_location="cpu",
            )
        if embedding_model is None:
            raise EroVoiceClassifierError("Failed to load the pinned WeSpeaker checkpoint")
        embedding_model.eval()
        self.embedding_inference = inference_class(embedding_model, window="whole")
        self.embedding_inference.to(self.device)
        self.pyannote_version = pyannote_version

    @property
    def run_metadata(self) -> dict[str, Any]:
        """Reproducibility and privacy metadata for a completed labeling run."""
        try:
            torchaudio_version = importlib.import_module("torchaudio").__version__
        except (ImportError, AttributeError):  # pragma: no cover - defensive metadata
            torchaudio_version = "unknown"
        return {
            "schema_version": 1,
            "role": "weak_closed_set_cluster_labeler",
            "execution": "local",
            "audio_uploaded": False,
            "labels": list(ERO_LABELS),
            "classifier": {
                "repo_id": SPACE_REPO_ID,
                "repo_type": SPACE_REPO_TYPE,
                "revision": SPACE_REVISION,
                "config": SPACE_CONFIG_FILENAME,
                "config_sha256": SPACE_CONFIG_SHA256,
                "checkpoint": SPACE_CHECKPOINT_FILENAME,
                "checkpoint_sha256": SPACE_CHECKPOINT_SHA256,
            },
            "embedding": {
                "repo_id": WESPEAKER_REPO_ID,
                "repo_type": "model",
                "revision": WESPEAKER_REVISION,
                "config": WESPEAKER_CONFIG_FILENAME,
                "config_sha256": WESPEAKER_CONFIG_SHA256,
                "checkpoint": WESPEAKER_CHECKPOINT_FILENAME,
                "checkpoint_sha256": WESPEAKER_CHECKPOINT_SHA256,
                "feature_dim": WESPEAKER_FEATURE_DIM,
                "sample_rate": WESPEAKER_SAMPLE_RATE,
                "window": "whole",
            },
            "runtime": {
                "device": str(self.device),
                "torch": torch.__version__,
                "torchaudio": torchaudio_version,
                "numpy": np.__version__,
                "pyannote_audio": self.pyannote_version,
            },
            "integrity_verified": True,
        }

    def _embedding_from_file(self, audio_path: Path) -> np.ndarray:
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - a core project dependency
            raise EroVoiceDependencyError("soundfile is required to read local audio") from exc

        samples, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=True,
        )
        if samples.shape[0] == 0:
            raise EroVoiceClassifierError(f"Audio file is empty: {audio_path}")
        mono = np.mean(samples, axis=1, dtype=np.float32)
        waveform = torch.from_numpy(np.ascontiguousarray(mono[np.newaxis, :]))
        with torch.inference_mode():
            embedding = self.embedding_inference(
                {"waveform": waveform, "sample_rate": int(sample_rate)}
            )
        values = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if values.shape != (WESPEAKER_FEATURE_DIM,):
            raise EroVoiceClassifierError(
                f"Expected a {WESPEAKER_FEATURE_DIM}-D embedding for {audio_path}, "
                f"got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise EroVoiceClassifierError(f"Embedding contains non-finite values: {audio_path}")
        return values

    def predict(
        self,
        audio_paths: Sequence[str | Path],
        *,
        batch_size: int = 64,
    ) -> list[EroVoicePrediction]:
        """Return local probabilities for files in input order.

        WeSpeaker whole-file embeddings are extracted one clip at a time because the
        source clips have variable length.  The small classifier head is evaluated in
        GPU batches.  A clip whose embedding fails is isolated as an error-marked
        prediction instead of aborting the whole batch.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        paths = [Path(path) for path in audio_paths]
        predictions: list[EroVoicePrediction] = []
        for offset in range(0, len(paths), batch_size):
            chunk = paths[offset : offset + batch_size]
            embeddings: list[np.ndarray | None] = []
            errors: list[str | None] = []
            for path in chunk:
                try:
                    embeddings.append(self._embedding_from_file(path))
                    errors.append(None)
                except Exception as exc:  # one unreadable clip must not lose the batch
                    embeddings.append(None)
                    errors.append(f"{type(exc).__name__}: {exc}")
            valid_indices = [index for index, item in enumerate(embeddings) if item is not None]
            rows_by_index: dict[int, dict[str, float]] = {}
            if valid_indices:
                features = np.stack([embeddings[index] for index in valid_indices])
                feature_tensor = torch.from_numpy(features).to(self.device)
                with torch.inference_mode():
                    logits = self.head(feature_tensor)
                rows = logits_to_probabilities(logits)
                for index, row in zip(valid_indices, rows, strict=True):
                    rows_by_index[index] = row
            for index, path in enumerate(chunk):
                error = errors[index]
                if error is not None:
                    predictions.append(
                        EroVoicePrediction(
                            audio=str(path),
                            usual=0.0,
                            aegi=0.0,
                            chupa=0.0,
                            error=error,
                        )
                    )
                    continue
                row = rows_by_index[index]
                predictions.append(
                    EroVoicePrediction(
                        audio=str(path),
                        usual=row["usual"],
                        aegi=row["aegi"],
                        chupa=row["chupa"],
                    )
                )
        return predictions

    def predict_probabilities(
        self,
        audio_paths: Sequence[str | Path],
        *,
        batch_size: int = 64,
    ) -> list[dict[str, float]]:
        """Convenience wrapper returning only ``usual/aegi/chupa`` mappings.

        Raises :class:`EroVoiceClassifierError` when any clip failed; use
        :meth:`predict` to receive error-marked predictions instead.
        """
        return [
            prediction.probabilities
            for prediction in self.predict(audio_paths, batch_size=batch_size)
        ]
