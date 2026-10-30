import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from training.baseline import ALGORITHMS, FLOOR, QUALIFYING, median_floor, qualifying_baseline
from training.cv import cross_validate
from training.data import TrainSet


def synthetic(n_races: int = 12, slope: float = 1.5, intercept: float = -0.25) -> TrainSet:
    """Four drivers per race; the target is an exact line of `quali_delta_pct`, and `other` is noise to ignore."""
    rng = np.random.default_rng(0)
    rows = [{"year": 2024, "round": r + 1, "driver": f"D{d}"} for r in range(n_races) for d in range(4)]
    meta = pd.DataFrame(rows).assign(entry_id=lambda m: range(len(m)))[["entry_id", "year", "round", "driver"]]
    quali = rng.uniform(0, 3, len(meta))
    X = pd.DataFrame({"quali_delta_pct": quali, "other": rng.normal(size=len(meta))})
    return TrainSet(X=X, y=pd.Series(slope * quali + intercept), meta=meta)


def test_baseline_recovers_the_slope_and_intercept():
    train = synthetic()
    model = qualifying_baseline().fit(train.X, train.y)
    linear = model.named_steps["linear"]
    assert linear.coef_ == pytest.approx([1.5])
    assert linear.intercept_ == pytest.approx(-0.25)


def test_baseline_uses_only_quali_delta_pct():
    train = synthetic()
    model = qualifying_baseline().fit(train.X, train.y)
    assert model.named_steps["linear"].n_features_in_ == 1
    changed = train.X.assign(other=999.0)
    assert model.predict(changed) == pytest.approx(model.predict(train.X))


def test_floor_predicts_the_median_of_the_train_target():
    floor = median_floor().fit(pd.DataFrame({"x": [0, 0, 0]}), pd.Series([1.0, 2.0, 10.0]))
    assert floor.predict(pd.DataFrame({"x": [5, 6]})).tolist() == [2.0, 2.0]


def test_floor_scores_chance_pairwise_and_no_positive_r2():
    result = cross_validate(median_floor(), synthetic(), 5)
    assert [f["pairwise_accuracy"] for f in result.folds] == [0.5] * 5
    assert all(f["r2"] <= 0 for f in result.folds)


def test_both_references_go_through_cross_validate():
    train = synthetic()
    baseline = cross_validate(qualifying_baseline(), train, 5)
    floor = cross_validate(median_floor(), train, 5)
    assert baseline.mean["mae"] == pytest.approx(0, abs=1e-9)
    assert baseline.mean["pairwise_accuracy"] == 1.0
    assert baseline.mean["mae"] < floor.mean["mae"]
    clone(qualifying_baseline())
    clone(median_floor())


def test_algorithm_tags():
    assert ALGORITHMS == {QUALIFYING: "baseline", FLOOR: "dummy"}
