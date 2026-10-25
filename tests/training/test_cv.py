import random

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from training.cv import cross_validate, temporal_folds
from training.data import TrainSet


def races_of(n: int) -> list[tuple[int, int]]:
    """n races over consecutive seasons of 22 rounds: (2023, 1) ... in calendar order."""
    return [(2023 + i // 22, i % 22 + 1) for i in range(n)]


def test_56_races_and_5_folds_give_the_blocks_of_the_spec():
    folds = temporal_folds(races_of(56), 5)
    assert [len(f.train_races) for f in folds] == [10, 20, 29, 38, 47]
    assert [len(f.valid_races) for f in folds] == [10, 9, 9, 9, 9]
    assert [f.number for f in folds] == [1, 2, 3, 4, 5]


def test_all_training_races_come_before_the_validation_races():
    for fold in temporal_folds(races_of(56), 5):
        assert max(fold.train_races) < min(fold.valid_races)


def test_validation_blocks_are_the_last_five_and_do_not_overlap():
    races = races_of(56)
    folds = temporal_folds(races, 5)
    validated = [race for f in folds for race in f.valid_races]
    assert len(validated) == len(set(validated)) == 46
    assert validated == races[10:]


def test_training_expands_by_one_block_and_includes_the_previous_validation():
    folds = temporal_folds(races_of(56), 5)
    for previous, current in zip(folds, folds[1:], strict=False):
        assert current.train_races == previous.train_races + previous.valid_races


def test_input_order_does_not_matter():
    races = races_of(56)
    shuffled = races[:]
    random.Random(0).shuffle(shuffled)
    assert temporal_folds(shuffled, 5) == temporal_folds(races, 5)


def test_exactly_n_folds_plus_one_races_gives_blocks_of_one():
    folds = temporal_folds(races_of(6), 5)
    assert [len(f.train_races) for f in folds] == [1, 2, 3, 4, 5]
    assert all(len(f.valid_races) == 1 for f in folds)


def test_too_few_races_raises():
    with pytest.raises(ValueError, match="at least 6 races"):
        temporal_folds(races_of(5), 5)


# ---- cross_validate ----


def train_set(n_races: int, y_of) -> TrainSet:
    """Two drivers per race; the feature `x` is the row number and `y_of(race, driver_index, x)` the target."""
    rows = []
    for r in range(n_races):
        for d, driver in enumerate(("AAA", "BBB")):
            rows.append({"entry_id": len(rows), "year": 2024, "round": r + 1, "driver": driver, "d": d, "r": r})
    meta = pd.DataFrame(rows)
    x = np.arange(len(meta), dtype="float64")
    y = pd.Series([y_of(row.r, row.d, xi) for row, xi in zip(meta.itertuples(), x, strict=True)], dtype="float64")
    return TrainSet(X=pd.DataFrame({"x": x}), y=y, meta=meta[["entry_id", "year", "round", "driver"]])


def test_exact_line_gives_zero_error_in_every_fold():
    train = train_set(12, lambda r, d, x: 3 * x - 2)
    result = cross_validate(LinearRegression(), train, 5)
    assert len(result.folds) == 5
    assert all(f["mae"] == pytest.approx(0, abs=1e-9) for f in result.folds)
    assert result.mean["mae"] == pytest.approx(0, abs=1e-9)


def test_mean_and_std_of_the_folds_match_a_hand_computation():
    # driver A scores r and driver B r + 1; the median floor of the fold-train misses by 1, 1.5, 2, 2.5 and 3
    train = train_set(6, lambda r, d, x: r + d)
    result = cross_validate(DummyRegressor(strategy="median"), train, 5)
    assert [f["mae"] for f in result.folds] == pytest.approx([1.0, 1.5, 2.0, 2.5, 3.0])
    assert [f["mean_error"] for f in result.folds] == pytest.approx([-1.0, -1.5, -2.0, -2.5, -3.0])
    assert result.mean["mae"] == pytest.approx(2.0)
    assert result.std["mae"] == pytest.approx(np.sqrt(0.5))
    assert [f["pairwise_accuracy"] for f in result.folds] == [0.5] * 5


def test_the_estimator_passed_in_is_cloned_not_fitted():
    estimator = LinearRegression()
    cross_validate(estimator, train_set(12, lambda r, d, x: x), 5)
    assert not hasattr(estimator, "coef_")


def test_oof_has_one_row_per_validation_row_and_none_from_the_first_block():
    train = train_set(12, lambda r, d, x: 2 * x)
    oof = cross_validate(LinearRegression(), train, 5).oof
    assert list(oof.columns) == ["entry_id", "year", "round", "driver", "fold", "y_true", "y_pred"]
    assert oof["entry_id"].is_unique
    assert len(oof) == 20  # the last 5 blocks of 2 races, 2 drivers each
    assert oof["round"].min() == 3  # the first block (rounds 1 and 2) is never validated
    assert oof.groupby("fold")["round"].agg(["min", "max"]).values.tolist() == [
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
    ]
    assert oof["y_pred"].to_numpy() == pytest.approx(oof["y_true"].to_numpy())


class Spy(BaseEstimator, RegressorMixin):
    """Remembers, in a module-level log, the rows each clone saw in fit and in predict."""

    log: list = []

    def fit(self, X, y):
        Spy.log.append(("fit", set(X["x"])))
        return self

    def predict(self, X):
        Spy.log.append(("predict", set(X["x"])))
        return np.zeros(len(X))


def test_no_row_is_on_both_sides_of_a_fold():
    Spy.log = []
    train = train_set(12, lambda r, d, x: x)
    cross_validate(Spy(), train, 5)
    fits, predicts = Spy.log[0::2], Spy.log[1::2]
    assert len(fits) == len(predicts) == 5
    for (_, seen), (_, predicted) in zip(fits, predicts, strict=True):
        assert seen.isdisjoint(predicted)
        assert max(seen) < min(predicted)
