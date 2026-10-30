"""The temporal cross-validation by race: expanding window, no shuffle, no gap, no seed."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone

from training.data import TrainSet
from training.metrics import METRICS, score


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


@dataclass(frozen=True)
class CvResult:
    folds: list[dict[str, float]]  # METRICS of each fold, in order
    mean: dict[str, float]
    std: dict[str, float]  # np.std of the fold values (ddof 0)
    oof: pd.DataFrame  # entry_id, year, round, driver, fold, y_true, y_pred; validation rows only


def cross_validate(estimator, train: TrainSet, n_folds: int) -> CvResult:
    """Fits a clone of `estimator` on each fold-train and scores the fold-validation. Nothing is shuffled, and the
    metrics are pooled over the rows of a fold before the mean and standard deviation over the folds.
    """
    race_keys = pd.MultiIndex.from_frame(train.meta[["year", "round"]])
    races = [(int(y), int(r)) for y, r in race_keys.unique()]
    folds, oof = [], []
    for fold in temporal_folds(races, n_folds):
        in_train = race_keys.isin(fold.train_races)
        in_valid = race_keys.isin(fold.valid_races)
        model = clone(estimator).fit(train.X[in_train], train.y[in_train])
        predicted = model.predict(train.X[in_valid])
        valid_meta = train.meta[in_valid]
        folds.append(score(train.y[in_valid], predicted, valid_meta))
        oof.append(valid_meta.assign(fold=fold.number, y_true=train.y[in_valid].to_numpy(), y_pred=predicted))
    return CvResult(
        folds=folds,
        mean={m: float(np.mean([f[m] for f in folds])) for m in METRICS},
        std={m: float(np.std([f[m] for f in folds])) for m in METRICS},
        oof=pd.concat(oof, ignore_index=True)[["entry_id", "year", "round", "driver", "fold", "y_true", "y_pred"]],
    )
