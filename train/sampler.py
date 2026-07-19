from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class LengthGroupedBatchSampler(Sampler[list[int]]):
    """
    Yield index batches whose samples have similar lengths to minimize padding.

    Each epoch the index order is reshuffled, sharded evenly across replicas,
    cut into windows of ``batch_size * bucket_mult`` samples, and every window
    is sorted by length before being split into batches. The batch order is
    then shuffled again so sequence length does not correlate with training
    step. All replicas yield the same number of batches per epoch.
    """

    def __init__(
        self,
        lengths: list[int],
        *,
        batch_size: int,
        drop_last: bool,
        shuffle: bool = True,
        seed: int = 0,
        bucket_mult: int = 64,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if not lengths:
            raise ValueError("lengths must contain at least one sample length.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if bucket_mult <= 0:
            raise ValueError(f"bucket_mult must be > 0, got {bucket_mult}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be > 0, got {num_replicas}")
        if not (0 <= rank < num_replicas):
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_mult = int(bucket_mult)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _shard_size(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.num_replicas
        return (n + self.num_replicas - 1) // self.num_replicas

    def __len__(self) -> int:
        per_replica = self._shard_size()
        if self.drop_last:
            return per_replica // self.batch_size
        return (per_replica + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        n = len(self.lengths)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            order = torch.randperm(n, generator=generator).tolist()
        else:
            order = list(range(n))

        if self.num_replicas > 1:
            per_replica = self._shard_size()
            if self.drop_last:
                order = order[: per_replica * self.num_replicas]
            else:
                order = order + order[: per_replica * self.num_replicas - n]
            order = order[self.rank :: self.num_replicas]

        window = self.batch_size * self.bucket_mult
        batches: list[list[int]] = []
        for start in range(0, len(order), window):
            chunk = order[start : start + window]
            chunk.sort(key=self.lengths.__getitem__)
            for batch_start in range(0, len(chunk), self.batch_size):
                batch = chunk[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle and len(batches) > 1:
            batch_order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in batch_order]
        yield from batches
