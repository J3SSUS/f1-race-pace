import numpy as np
import pandas as pd
import pytest
from test_explain_champion import synthetic_X

from features.catalog import load_catalog
from training.data import TrainSet
from training.explain import Champion, calibration_line, permutation_importance_mae
from training.pipelines import build_pipeline

FEATURES = [f.name for f in load_catalog().features]
USED = next(f.name for f in load_catalog().features if f.dtype != "category")  # y depends on this one only
N = 200


@pytest.fixture(scope="module")
def train() -> TrainSet:
    X = synthetic_X(N)
    meta = pd.DataFrame({"entry_id": np.arange(N), "year": 2024, "round": np.repeat(np.arange(N // 4), 4), "driver": 1})
    return TrainSet(X=X, y=2.0 * X[USED], meta=meta)


@pytest.fixture(scope="module")
def champion(train) -> Champion:
    model = build_pipeline("ridge", {"alpha": 1e-6}, "median", 42).fit(train.X, train.y)
    return Champion(1, "1", "run", "sampling", "ridge", model)


def test_a_feature_the_model_ignores_has_an_importance_near_zero_and_a_used_one_a_positive_one(champion, train):
    result = permutation_importance_mae(champion, train).set_index("feature")
    ignored = result.drop(index=USED)
    assert ignored["mae_increase_mean"].abs().max() < 1e-4
    # permuting 2x with x ~ N(0, 1): MAE = 2 * E|x - x'| = 2 * 1.13, about 2.3
    assert result.loc[USED, "mae_increase_mean"] > 0.5


def test_the_same_seed_gives_the_same_result_and_another_seed_does_not(champion, train):
    first = permutation_importance_mae(champion, train, seed=7)
    again = permutation_importance_mae(champion, train, seed=7)
    other = permutation_importance_mae(champion, train, seed=8)
    pd.testing.assert_frame_equal(first, again)
    assert not first.equals(other)


def test_the_seed_defaults_to_the_one_of_the_config(champion, train):
    pd.testing.assert_frame_equal(
        permutation_importance_mae(champion, train), permutation_importance_mae(champion, train, seed=42)
    )


def test_the_result_has_the_15_original_columns_sorted_largest_first(champion, train):
    result = permutation_importance_mae(champion, train)
    assert list(result.columns) == ["feature", "mae_increase_mean", "mae_increase_std"]
    assert sorted(result["feature"]) == sorted(FEATURES)  # circuit_type is one row, not three
    assert result["mae_increase_mean"].is_monotonic_decreasing
    assert result["feature"].iloc[0] == USED
    assert (result["mae_increase_std"] >= 0).all()


def test_the_repeats_are_the_ones_asked_for(champion, train):
    assert permutation_importance_mae(champion, train, n_repeats=3)["mae_increase_std"].notna().all()


def oof(y_true, y_pred) -> pd.DataFrame:
    n = len(y_true)
    return pd.DataFrame({"year": 2024, "round": np.repeat(np.arange(n // 4), 4), "y_true": y_true, "y_pred": y_pred})


def test_perfect_predictions_give_slope_1_and_intercept_0():
    actual = np.random.default_rng(0).normal(size=40)
    line = calibration_line(oof(actual, actual))
    assert line["slope"] == pytest.approx(1.0)
    assert line["intercept"] == pytest.approx(0.0, abs=1e-12)
    assert (line["n_rows"], line["n_races"]) == (40, 10)


def test_predictions_built_as_a_plus_b_times_actual_give_the_known_line():
    actual = np.random.default_rng(1).normal(size=40)
    a, b = 0.3, 0.5  # y_pred = a + b * y_true  =>  y_true = -a / b + (1 / b) * y_pred
    line = calibration_line(oof(actual, a + b * actual))
    assert line["slope"] == pytest.approx(1 / b)
    assert line["intercept"] == pytest.approx(-a / b)


def test_an_oof_without_rows_raises():
    with pytest.raises(ValueError, match="no rows"):
        calibration_line(oof([], []))


def test_predictions_that_do_not_vary_raise():
    with pytest.raises(ValueError, match="do not vary"):
        calibration_line(oof(np.arange(8.0), np.ones(8)))
