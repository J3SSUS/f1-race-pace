import numpy as np
import pandas as pd
import pytest
from test_explain_champion import synthetic_X, synthetic_y
from test_pipelines import fixed_params

from features.catalog import load_catalog
from live.prediction import Models, PredictionError, predict_entries
from training.baseline import qualifying_baseline
from training.explain import Champion
from training.pipelines import build_pipeline

N = 8
DRIVERS = [f"D{i:02d}" for i in range(N)]


def champion_of(model) -> Champion:
    return Champion(4, "4", "run", "sampling", "ridge", model)


@pytest.fixture(scope="module")
def fitted():
    X = synthetic_X(40)
    y = synthetic_y(X)
    return (
        build_pipeline("ridge", fixed_params("ridge"), "median", 42).fit(X, y),
        qualifying_baseline().fit(X[["quali_delta_pct"]], y),
    )


@pytest.fixture
def models(fitted) -> Models:
    return Models(champion_of(fitted[0]), fitted[1])


@pytest.fixture
def frame() -> pd.DataFrame:
    """What `race_features` returns: identification columns, then the features in catalog order."""
    X = synthetic_X(N, seed=7)
    ident = pd.DataFrame(
        {"year": 2026, "round": 17, "circuit": "Synthetic", "driver": DRIVERS, "team": ["T1", "T2"] * (N // 2)}
    )
    return pd.concat([ident, X[load_catalog().names]], axis=1)


class Recorder:
    """A stand-in model that records the columns it is given and returns a fixed prediction."""

    def __init__(self, value=0.5):
        self.value, self.columns, self.rows = value, [], []

    def predict(self, X):
        self.columns.append(list(X.columns))
        self.rows.append(len(X))
        return np.full(len(X), self.value)


def test_one_row_per_entry_with_what_each_model_predicts(models, frame):
    result = predict_entries(models, frame)
    names = load_catalog().names
    assert list(result.columns) == ["driver", "predicted_pace_delta_pct", "baseline_pace_delta_pct"]
    assert result["driver"].tolist() == DRIVERS
    np.testing.assert_allclose(result["predicted_pace_delta_pct"], models.champion.model.predict(frame[names]))
    np.testing.assert_allclose(result["baseline_pace_delta_pct"], models.baseline.predict(frame[["quali_delta_pct"]]))
    assert result["predicted_pace_delta_pct"].notna().all()


def test_the_champion_gets_the_catalog_features_in_catalog_order_and_nothing_else(frame):
    champion, baseline = Recorder(), Recorder()
    shuffled = frame[["driver", "team", *reversed(load_catalog().names), "year", "round", "circuit"]]
    predict_entries(Models(champion_of(champion), baseline), shuffled)
    assert champion.columns == [load_catalog().names]
    assert baseline.columns == [["quali_delta_pct"]]


def test_the_baseline_is_nan_where_quali_is_missing_and_the_champion_still_predicts(models, frame):
    frame.loc[[1, 5], "quali_delta_pct"] = np.nan
    result = predict_entries(models, frame)
    assert result.loc[[1, 5], "baseline_pace_delta_pct"].isna().all()
    assert result.drop(index=[1, 5])["baseline_pace_delta_pct"].notna().all()
    assert result["predicted_pace_delta_pct"].notna().all()


def test_the_baseline_only_sees_the_rows_with_a_qualifying_delta(frame):
    baseline = Recorder()
    frame.loc[[0, 3], "quali_delta_pct"] = np.nan
    predict_entries(Models(champion_of(Recorder()), baseline), frame)
    assert baseline.rows == [N - 2]


def test_no_qualifying_delta_at_all_never_calls_the_baseline(frame):
    baseline = Recorder()
    frame["quali_delta_pct"] = np.nan
    result = predict_entries(Models(champion_of(Recorder()), baseline), frame)
    assert baseline.rows == [] and result["baseline_pace_delta_pct"].isna().all()
    assert result["predicted_pace_delta_pct"].tolist() == [0.5] * N


def test_a_nan_champion_prediction_raises_naming_the_driver(frame):
    class Partial(Recorder):
        def predict(self, X):
            out = np.full(len(X), 0.1)
            out[2] = np.nan
            return out

    with pytest.raises(PredictionError, match=r"predicted NaN for D02"):
        predict_entries(Models(champion_of(Partial()), Recorder()), frame)


def test_a_frame_with_a_non_default_index_keeps_the_rows_aligned(models, frame):
    shifted = frame.set_index(pd.Index(range(100, 100 + N)))
    shifted.loc[102, "quali_delta_pct"] = np.nan
    result = predict_entries(models, shifted)
    assert result["driver"].tolist() == DRIVERS and result.index.tolist() == list(range(N))
    assert result["baseline_pace_delta_pct"].isna().tolist() == [i == 2 for i in range(N)]


def test_it_does_not_change_the_frame(models, frame):
    before = frame.copy()
    predict_entries(models, frame)
    pd.testing.assert_frame_equal(frame, before)
