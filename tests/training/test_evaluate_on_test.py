import numpy as np
import pandas as pd
import pytest

import training.evaluation as evaluation
from training.baseline import BASELINE_FEATURE, median_floor, qualifying_baseline
from training.data import TrainSet
from training.metrics import score

# 3 races of 4 drivers, hand-made
RACE = np.repeat([1, 2, 3], 4)
Y_TRUE = np.array([0.0, 1.0, 2.0, 3.0, 0.5, 1.5, 1.0, 2.5, -1.0, 0.0, 1.0, 2.0])
Y_PRED = np.array([0.5, 1.0, 1.0, 3.5, 0.5, 1.0, 1.5, 2.0, -1.0, 0.5, 0.0, 2.0])


class Fixed:
    """A model that returns a fixed vector."""

    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, X):
        assert len(X) == len(self.prediction)
        return self.prediction


def make_set(y=Y_TRUE, quali=None) -> TrainSet:
    n = len(y)
    X = pd.DataFrame({BASELINE_FEATURE: np.arange(n, dtype="float64") if quali is None else quali})
    meta = pd.DataFrame({"entry_id": np.arange(n), "year": 2024, "round": RACE[:n], "driver": np.arange(n)})
    return TrainSet(X=X, y=pd.Series(y, dtype="float64"), meta=meta)


def evaluate(prediction=Y_PRED, **kwargs):
    return evaluation.evaluate_on_test(Fixed(prediction), make_set(), **kwargs)


def test_metrics_equal_score_of_spec_07_and_are_by_hand_right():
    result = evaluate()
    expected = score(Y_TRUE, Y_PRED, pd.DataFrame({"year": 2024, "round": RACE}))
    assert result.metrics == {k: expected[k] for k in ("mae", "rmse", "r2", "pairwise_accuracy")}
    # errors: .5 0 1 .5 | 0 .5 .5 .5 | 0 .5 1 0  -> sum 5.0 over 12 rows
    assert result.metrics["mae"] == pytest.approx(5.0 / 12)
    assert result.n_rows == 12 and result.n_races == 3


def test_interval_contains_the_point_value():
    result = evaluate()
    for low, high in result.intervals.values():
        assert low <= high
    low, high = result.intervals["mae"]
    assert low <= result.metrics["mae"] <= high


def test_interval_is_reproducible_with_the_seed():
    # 3 races have only 27 possible resamples, so any seed covers them all: use 12 races for the seed to matter
    rng = np.random.default_rng(0)
    n = 48
    y = rng.normal(0, 1, n)
    meta = pd.DataFrame({"entry_id": np.arange(n), "year": 2024, "round": np.repeat(np.arange(12), 4), "driver": 1})
    test = TrainSet(X=pd.DataFrame({BASELINE_FEATURE: np.zeros(n)}), y=pd.Series(y), meta=meta)
    model = Fixed(y + rng.normal(0, 0.5, n))
    first = evaluation.evaluate_on_test(model, test, seed=3)
    assert evaluation.evaluate_on_test(model, test, seed=3).intervals == first.intervals
    assert evaluation.evaluate_on_test(model, test, seed=4).intervals != first.intervals


def test_mean_error_and_quantiles_use_the_absolute_error():
    result = evaluate()
    abs_error = np.abs(Y_PRED - Y_TRUE)
    assert result.mean_error == pytest.approx(np.mean(Y_PRED - Y_TRUE))
    assert result.error_quantiles == {
        "median": pytest.approx(np.median(abs_error)),
        "p90": pytest.approx(np.percentile(abs_error, 90)),
        "p95": pytest.approx(np.percentile(abs_error, 95)),
        "max": 1.0,
    }
    assert set(result.metrics) == {"mae", "rmse", "r2", "pairwise_accuracy"}


def test_a_resample_that_repeats_a_race_keeps_the_races_apart():
    # a perfect model has pairwise accuracy 1 in every resample, even when a race is drawn twice
    result = evaluate(prediction=Y_TRUE.copy())
    assert result.intervals["pairwise_accuracy"] == (1.0, 1.0)
    assert result.intervals["mae"] == (0.0, 0.0)


def test_the_name_is_carried():
    assert evaluate(name="ridge").name == "ridge"


def test_the_baseline_with_a_nan_feature_raises_before_predicting():
    quali = np.arange(12, dtype="float64")
    quali[5] = np.nan
    train = make_set(quali=np.arange(12, dtype="float64"))
    baseline = qualifying_baseline().fit(train.X, train.y)
    with pytest.raises(ValueError, match="quali_delta_pct"):
        evaluation.evaluate_on_test(baseline, make_set(quali=quali))


def test_the_baseline_with_complete_features_is_evaluated():
    quali = np.arange(12, dtype="float64")
    train = make_set(y=0.1 * quali + 1, quali=quali)  # y is exactly linear in the feature
    baseline = qualifying_baseline().fit(train.X, train.y)
    result = evaluation.evaluate_on_test(baseline, train, name="baseline")
    assert result.metrics["mae"] == pytest.approx(0.0, abs=1e-9)


def test_a_model_that_is_not_the_baseline_is_not_checked_for_nan():
    # other models carry their own imputer; the pre-check is only for the baseline's pipeline
    floor = median_floor().fit(make_set().X, make_set().y)
    quali = np.arange(12, dtype="float64")
    quali[0] = np.nan
    assert evaluation.evaluate_on_test(floor, make_set(quali=quali)).n_rows == 12


def test_test_metric_rows_are_four_test_overall_rows():
    rows = evaluation.test_metric_rows(7, evaluate().metrics)
    assert [r["metric_name"] for r in rows] == ["mae", "rmse", "r2", "pairwise_acc"]
    assert {r["split"] for r in rows} == {"test"}
    assert {r["segment_type"] for r in rows} == {"overall"}
    assert all(r["race_id"] is None and r["segment_value"] is None and r["model_version_id"] == 7 for r in rows)
