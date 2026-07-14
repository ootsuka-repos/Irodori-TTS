import torch

from irodori_tts.config import ModelConfig
from irodori_tts.dataset import TTSCollator
from irodori_tts.tokenizer import ByteTokenizer
from irodori_tts.training.data import iter_device_batches, training_batch_keys
from irodori_tts.training.metrics import (
    duration_condition_group_metrics,
    duration_condition_group_totals,
)


def _sample(length: int, *, speaker: bool) -> dict:
    latent = torch.arange(length * 2, dtype=torch.float32).reshape(length, 2)
    return {
        "text": "sample",
        "caption": "",
        "latent": latent,
        "ref_latent": latent,
        "num_frames": length,
        "has_speaker": speaker,
        "has_caption": False,
    }


def test_compact_collator_skips_unused_latent_work() -> None:
    collator = TTSCollator(
        tokenizer=ByteTokenizer.for_vocab_size(512),
        caption_tokenizer=None,
        latent_dim=2,
        latent_patch_size=1,
        include_target_latent=False,
        include_reference_latent=False,
        include_duration_features=False,
        return_unpatched_latents=False,
    )
    batch = collator([_sample(3, speaker=False), _sample(5, speaker=True)])
    assert set(batch) == {"text_ids", "text_mask", "num_frames", "has_speaker"}
    assert batch["num_frames"].tolist() == [3, 5]


def test_compact_collator_keeps_only_patched_training_latents() -> None:
    collator = TTSCollator(
        tokenizer=ByteTokenizer.for_vocab_size(512),
        caption_tokenizer=None,
        latent_dim=2,
        latent_patch_size=1,
        include_duration_features=False,
        return_unpatched_latents=False,
    )
    batch = collator([_sample(3, speaker=True), _sample(5, speaker=True)])
    assert "latent_patched" in batch
    assert "ref_latent_patched" in batch
    assert "latent" not in batch
    assert "ref_latent" not in batch
    assert "latent_padding_mask_patched" not in batch


def test_training_batch_selection_and_cpu_transfer() -> None:
    config = ModelConfig(use_duration_predictor=True, use_caption_condition=False)
    keys = training_batch_keys(config, duration_only=True)
    assert "latent_patched" not in keys
    assert "ref_latent_patched" in keys
    assert "duration_features" in keys

    source = [{key: torch.tensor([index]) for index, key in enumerate(keys)}]
    batches = list(
        iter_device_batches(
            source,
            keys=keys,
            device=torch.device("cpu"),
            prefetch=True,
        )
    )
    assert tuple(batches[0]) == keys


def test_duration_group_metrics_are_vectorized_and_exact() -> None:
    totals = duration_condition_group_totals(
        duration_loss_per_sample=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        pred_frames=torch.tensor([11.0, 18.0, 35.0, 35.0]),
        target_frames=torch.tensor([10.0, 20.0, 30.0, 40.0]),
        has_speaker=torch.tensor([True, True, False, False]),
        has_caption=torch.tensor([True, False, True, False]),
    )
    metrics = duration_condition_group_metrics(totals)
    assert metrics["duration_samples_speaker"] == 2.0
    assert metrics["duration_loss_speaker"] == 1.5
    assert metrics["duration_mae_frames_speaker"] == 1.5
    assert metrics["duration_loss_no_speaker"] == 3.5
    assert metrics["duration_samples_speaker_caption"] == 1.0
    assert metrics["duration_loss_no_speaker_no_caption"] == 4.0
