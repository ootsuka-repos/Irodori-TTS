import torch

from irodori_tts.tokenizer import PretrainedTextTokenizer


class _FakeTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token_id = 2
    eos_token = "</s>"
    bos_token_id = 1
    padding_side = "left"

    def __init__(self, *, is_fast: bool) -> None:
        self.is_fast = is_fast

    def __len__(self) -> int:
        return 256

    @staticmethod
    def _tokens(text: str) -> list[int]:
        return [3 + (ord(char) % 200) for char in text]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return self._tokens(text)

    def __call__(self, texts: list[str], **kwargs) -> dict[str, torch.Tensor]:
        values = [self._tokens(text) for text in texts]
        max_length = kwargs.get("max_length")
        if kwargs["truncation"]:
            values = [value[:max_length] for value in values]
        width = (
            max_length
            if kwargs["padding"] == "max_length"
            else max(
                (len(value) for value in values),
                default=0,
            )
        )
        ids = torch.full((len(values), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(values), width), dtype=torch.long)
        for index, value in enumerate(values):
            length = min(len(value), width)
            ids[index, :length] = torch.tensor(value[:length])
            mask[index, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _encode_pair(
    texts: list[str],
    *,
    add_bos: bool,
    max_length: int | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    slow = PretrainedTextTokenizer(_FakeTokenizer(is_fast=False), add_bos=add_bos)
    fast = PretrainedTextTokenizer(_FakeTokenizer(is_fast=True), add_bos=add_bos)
    return (
        slow.batch_encode(texts, max_length=max_length),
        fast.batch_encode(texts, max_length=max_length),
    )


def test_fast_batch_encoding_matches_python_fallback() -> None:
    texts = ["abc", "", "日本語", "longer text"]
    for add_bos in (False, True):
        for max_length in (None, 1, 4, 12):
            slow, fast = _encode_pair(
                texts,
                add_bos=add_bos,
                max_length=max_length,
            )
            assert torch.equal(slow[0], fast[0])
            assert torch.equal(slow[1], fast[1])
