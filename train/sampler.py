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
        groups: list[str] | None = None,
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
        # Category-balanced undersampling: each epoch every group contributes
        # min(group sizes) samples drawn without replacement, so the epoch is
        # num_groups * min_size and all groups are equally represented.
        self.group_to_indices: dict[str, list[int]] | None = None
        if groups is not None:
            if len(groups) != len(self.lengths):
                raise ValueError(
                    f"groups length {len(groups)} != lengths length {len(self.lengths)}"
                )
            if not shuffle:
                raise ValueError("group-balanced sampling requires shuffle=True")
            group_to_indices: dict[str, list[int]] = {}
            for index, group in enumerate(groups):
                group_to_indices.setdefault(str(group), []).append(index)
            self.group_to_indices = group_to_indices
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _epoch_size(self) -> int:
        if self.group_to_indices is not None:
            min_size = min(len(v) for v in self.group_to_indices.values())
            return min_size * len(self.group_to_indices)
        return len(self.lengths)

    def _shard_size(self) -> int:
        n = self._epoch_size()
        if self.drop_last:
            return n // self.num_replicas
        return (n + self.num_replicas - 1) // self.num_replicas

    def __len__(self) -> int:
        per_replica = self._shard_size()
        if self.drop_last:
            return per_replica // self.batch_size
        return (per_replica + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.group_to_indices is not None:
            # Undersample every group to the smallest group's size, without
            # replacement, with a fresh random subset each epoch.
            min_size = min(len(v) for v in self.group_to_indices.values())
            order = []
            for _, indices in sorted(self.group_to_indices.items()):
                perm = torch.randperm(len(indices), generator=generator).tolist()
                order.extend(indices[i] for i in perm[:min_size])
            shuffle_order = torch.randperm(len(order), generator=generator).tolist()
            order = [order[i] for i in shuffle_order]
        elif self.shuffle:
            order = torch.randperm(len(self.lengths), generator=generator).tolist()
        else:
            order = list(range(len(self.lengths)))
        n = len(order)

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
