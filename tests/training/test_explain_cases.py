import numpy as np
import pandas as pd
import pytest
from fake_supabase import FakeSupabase
from test_explain_champion import synthetic_X, synthetic_y
from test_pipelines import fixed_params

from features.catalog import load_catalog
from training.explain import Champion, build_cases, shap_by_feature
from training.pipelines import build_pipeline

FEATURES = [f.name for f in load_catalog().features]
PREDICTED = next(f.name for f in load_catalog().features if f.dtype != "category")  # the stub predicts this column
BASE = 0.2


class Stub:
    """A model whose prediction is the value of one feature, so the test sets every prediction by hand."""

    def predict(self, X):
        return X[PREDICTED].to_numpy()


def champion(model=None) -> Champion:
    return Champion(1, "1", "run", "sampling", "ridge", model or Stub())


def frames(predictions: list[float], races=None):
    """X, meta and a SHAP frame that adds up, for rows with the given predictions and `(year, round, split)`."""
    n = len(predictions)
    races = races or [(2025, 5, "test")] * n
    X = synthetic_X(n)
    X[PREDICTED] = predictions
    meta = pd.DataFrame(
        {
            "entry_id": 100 + np.arange(n),
            "year": [r[0] for r in races],
            "round": [r[1] for r in races],
            "driver": [f"D{i}" for i in range(n)],
            "split": [r[2] for r in races],
        }
    )
    shap = pd.DataFrame(0.0, index=X.index, columns=FEATURES)
    shap[PREDICTED] = np.array(predictions) - BASE  # base + sum == prediction
    return X, meta, shap


class Spy:
    """A stand-in for `read_actuals` that records its calls and returns the same actual for every entry."""

    def __init__(self, actual=0.5, per_entry=None):
        self.calls, self.actual, self.per_entry = [], actual, per_entry or {}

    def __call__(self, client, entry_ids, version="v1"):
        self.calls.append(list(entry_ids))
        return {e: self.per_entry.get(e, self.actual) for e in entry_ids}


def cases_of(predictions, races=None, spy=None):
    X, meta, shap = frames(predictions, races)
    spy = spy or Spy()
    return build_cases(champion(), X, meta, shap, BASE, client=None, reader=spy), spy


def test_the_fastest_is_the_lowest_prediction_and_the_slowest_the_highest_in_that_order():
    cases, _ = cases_of([0.3, -0.8, 0.1, 0.9, -0.2])
    assert [c.label for c in cases] == ["fastest_predicted", "slowest_predicted"]
    assert [c.entry_id for c in cases] == [101, 103]
    assert [c.prediction for c in cases] == [-0.8, 0.9]
    assert [c.driver for c in cases] == ["D1", "D3"]


def test_a_tie_goes_to_the_lowest_entry_id():
    cases, _ = cases_of([0.5, -0.8, 0.9, -0.8, 0.9])
    assert [c.entry_id for c in cases] == [101, 102]  # entries 101 and 103 tie low, 102 and 104 tie high


def test_only_the_most_recent_test_race_counts_whatever_the_row_order():
    races = [(2025, 5, "test"), (2025, 6, "test"), (2024, 20, "test"), (2025, 6, "test"), (2025, 7, "train")]
    cases, _ = cases_of([-9.0, 0.1, -5.0, 0.4, -7.0], races)
    assert [c.entry_id for c in cases] == [101, 103]  # the 2025 round 6 race; train and older races are ignored
    assert {(c.year, c.round) for c in cases} == {(2025, 6)}


def test_the_most_recent_race_is_found_by_year_before_round():
    races = [(2024, 24, "test"), (2024, 24, "test"), (2025, 1, "test"), (2025, 1, "test")]
    cases, _ = cases_of([-1.0, 1.0, -0.5, 0.5], races)
    assert {(c.year, c.round) for c in cases} == {(2025, 1)}


def test_the_actuals_are_read_once_after_the_choice_with_exactly_the_two_entries():
    spy = Spy()
    cases_of([0.3, -0.8, 0.1, 0.9, -0.2], spy=spy)
    assert spy.calls == [[101, 103]]


def test_the_actual_values_never_change_which_cases_are_chosen():
    spy = Spy(per_entry={100: -99.0, 102: 99.0, 104: 99.0})  # extreme actuals for drivers that were not chosen
    cases, _ = cases_of([0.3, -0.8, 0.1, 0.9, -0.2], spy=spy)
    assert [c.entry_id for c in cases] == [101, 103]


def test_a_failure_before_the_choice_never_reads_a_target():
    spy = Spy()
    X, meta, shap = frames([0.5])  # a single valid row
    with pytest.raises(ValueError):
        build_cases(champion(), X, meta, shap, BASE, client=None, reader=spy)
    assert spy.calls == []


def test_error_is_prediction_minus_actual():
    cases, _ = cases_of([0.3, -0.8, 0.1, 0.9, -0.2], spy=Spy(per_entry={101: -0.5, 103: 1.5}))
    assert [(c.prediction, c.actual, c.error) for c in cases] == [(-0.8, -0.5, pytest.approx(-0.3)), (0.9, 1.5, -0.6)]


def test_the_contributions_add_up_to_the_prediction_minus_the_base_value():
    cases, _ = cases_of([0.3, -0.8, 0.1, 0.9, -0.2])
    for case in cases:
        assert list(case.contributions) == FEATURES
        assert case.base_value == BASE
        assert sum(case.contributions.values()) == pytest.approx(case.prediction - BASE)


def test_contributions_that_do_not_add_up_raise():
    X, meta, shap = frames([0.3, -0.8, 0.1])
    shap[PREDICTED] += 1.0
    with pytest.raises(ValueError, match="do not add up"):
        build_cases(champion(), X, meta, shap, BASE, client=None, reader=Spy())


def test_a_test_race_with_a_single_valid_row_raises_and_names_the_race():
    races = [(2025, 5, "test"), (2025, 6, "test"), (2025, 6, "test"), (2025, 6, "test")]
    X, meta, shap = frames([0.1, 0.2, 0.3, 0.4], races)
    only = meta.assign(split=["test", "test", "train", "train"])  # round 6 keeps one test row
    with pytest.raises(ValueError, match=r"2025 round 6.*1 valid row"):
        build_cases(champion(), X, only, shap, BASE, client=None, reader=Spy())


def test_the_same_prediction_for_every_driver_raises():
    X, meta, shap = frames([0.4, 0.4, 0.4])
    with pytest.raises(ValueError, match="same prediction"):
        build_cases(champion(), X, meta, shap, BASE, client=None, reader=Spy())


def test_it_reads_the_actuals_from_ground_truth_by_default():
    X, meta, shap = frames([0.3, -0.8, 0.1, 0.9])
    client = FakeSupabase(
        {"ground_truth": [{"entry_id": 100 + i, "actual_pace_delta_pct": float(i)} for i in range(4)]}
    )
    cases = build_cases(champion(), X, meta, shap, BASE, client)
    assert [(c.entry_id, c.actual) for c in cases] == [(101, 1.0), (103, 3.0)]
    assert client.reads == [("ground_truth", 0, 2)]


@pytest.mark.parametrize("name", ["ridge", "random_forest"])
def test_with_a_real_model_the_contributions_add_up_to_its_prediction(name):
    train = synthetic_X(60)
    model = build_pipeline(name, fixed_params(name), "median", 42).fit(train, synthetic_y(train))
    real = Champion(1, "1", "run", "sampling", {"ridge": "ridge", "random_forest": "rf"}[name], model)
    X = synthetic_X(12, seed=5)
    meta = pd.DataFrame(
        {"entry_id": np.arange(12), "year": 2025, "round": 3, "driver": [f"D{i}" for i in range(12)], "split": "test"}
    )
    values, base = shap_by_feature(real, X, train)
    cases = build_cases(real, X, meta, values, base, client=None, reader=Spy())
    predictions = model.predict(X)
    assert [c.prediction for c in cases] == pytest.approx([predictions.min(), predictions.max()])
    for case in cases:
        assert base + sum(case.contributions.values()) == pytest.approx(case.prediction, abs=1e-5)
