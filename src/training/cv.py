"""The temporal cross-validation by race: expanding window, no shuffle, no gap, no seed."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    number: int  # 1 to n_folds
    train_races: list[tuple[int, int]]
    valid_races: list[tuple[int, int]]


def temporal_folds(races: list[tuple[int, int]], n_folds: int) -> list[Fold]:
    """The races in `(year, round)` order, cut into `n_folds + 1` consecutive blocks (sizes differ by at most one,
    the larger ones first). Fold k trains on blocks 1 to k and validates on block k + 1, so the first block is only
    ever training data. Fewer than `n_folds + 1` races raises `ValueError`.
    """
    ordered = sorted(set(races))
    if len(ordered) < n_folds + 1:
        raise ValueError(f"{n_folds} folds need at least {n_folds + 1} races, got {len(ordered)}")
    blocks = [[ordered[i] for i in idx] for idx in np.array_split(np.arange(len(ordered)), n_folds + 1)]
    return [
        Fold(
            number=k,
            train_races=[race for block in blocks[:k] for race in block],
            valid_races=blocks[k],
        )
        for k in range(1, n_folds + 1)
    ]
