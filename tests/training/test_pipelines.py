import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from common.config import load_config
from features.catalog import load_catalog
from training.pipelines import ALGORITHMS, LINEAR, build_pipeline

CIRCUITS = ["street", "permanent", "hybrid"]


def fixed_params(model: str) -> dict:
    """The fixed params of the repo config, plus small values for the ranges so the fits are quick."""
    small = {
        "ridge": {"alpha": 1.0},
        "elasticnet": {"alpha": 0.1, "l1_ratio": 0.5},
        "random_forest": {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 2},
        "xgboost": {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.3, "subsample": 0.8},
        "lightgbm": {"n_estimators": 10, "num_leaves": 8, "learning_rate": 0.3, "min_child_samples": 5},
        "catboost": {"iterations": 10, "depth": 3, "learning_rate": 0.3},
    }[model]
    fixed = {k: v.value for k, v in load_config().enabled_models()[model].items() if hasattr(v, "value")}
    return {**small, **fixed}


def table(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """The 15 features of the catalog with NaN, nullable Int64 columns and the three circuit types."""
    rng = np.random.default_rng(seed)
    cols = {}
    for f in load_catalog().features:
        if f.dtype == "category":
            cols[f.name] = pd.Series(rng.choice(CIRCUITS, n), dtype="object")
        elif f.dtype == "float":
            cols[f.name] = pd.Series(rng.normal(size=n))
        else:
            cols[f.name] = pd.Series(rng.integers(0, 2, n))
    X = pd.DataFrame(cols)
    X.loc[::7, "fp_tyre_deg"] = np.nan
    X["grid_position"] = pd.array(rng.integers(1, 21, n), dtype="Int64")
    X.loc[::5, "grid_position"] = pd.NA
    X["rain_flag"] = pd.array(rng.integers(0, 2, n), dtype="Int64")
    X.loc[::9, "rain_flag"] = pd.NA
    y = pd.Series(X["quali_delta_pct"].to_numpy() * 0.5 + rng.normal(scale=0.1, size=n))
    return X, y


@pytest.mark.parametrize("model", list(ALGORITHMS))
def test_every_model_fits_and_predicts_with_nan(model):
    X, y = table()
    pred = build_pipeline(model, fixed_params(model), "median", 42).fit(X, y).predict(X)
    assert pred.shape == (len(X),)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("strategy", ["median", "mean"])
def test_imputer_uses_the_given_strategy(strategy):
    X, y = table()
    pipe = build_pipeline("ridge", {"alpha": 1.0}, strategy, 42).fit(X, y)
    imputer = pipe.named_steps["prep"].named_transformers_["numeric"].named_steps["impute"]
    assert imputer.strategy == strategy
    numeric = [f.name for f in load_catalog().features if f.dtype != "category"]
    col = numeric.index("fp_tyre_deg")
    expected = getattr(np, "nanmedian" if strategy == "median" else "nanmean")(X["fp_tyre_deg"].to_numpy())
    assert imputer.statistics_[col] == pytest.approx(expected)


def test_unknown_imputation_is_rejected():
    with pytest.raises(ValueError, match="imputation"):
        build_pipeline("ridge", {}, "mode", 42)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        build_pipeline("svm", {}, "median", 42)


@pytest.mark.parametrize("model", list(ALGORITHMS))
def test_only_linear_models_scale(model):
    pipe = build_pipeline(model, fixed_params(model), "median", 42)
    numeric = (
        pipe.named_steps["prep"].named_transformers["numeric"] if False else pipe.named_steps["prep"].transformers[0][1]
    )
    scales = any(isinstance(s, StandardScaler) for _, s in numeric.steps)
    assert scales == (model in LINEAR)


def test_all_nan_column_is_kept_and_filled_with_zero():
    X, y = table()
    X["fp_tyre_deg"] = np.nan
    pipe = build_pipeline("random_forest", fixed_params("random_forest"), "median", 42).fit(X, y)
    out = pipe.named_steps["prep"].transform(X)
    assert out.shape[1] == 14 + 3
    assert np.isfinite(out).all()
    col = [c for c in range(14) if np.all(out[:, c] == 0)]
    assert len(col) == 1


def test_unseen_circuit_type_does_not_fail():
    X, y = table()
    pipe = build_pipeline("ridge", {"alpha": 1.0}, "median", 42).fit(
        X[X["circuit_type"] != "hybrid"], y[X["circuit_type"] != "hybrid"]
    )
    pred = pipe.predict(X)
    assert np.isfinite(pred).all()
    assert pipe.named_steps["prep"].transform(X).shape[1] == 14 + 3


def test_loss_params_reach_each_estimator():
    losses = {
        "random_forest": ("criterion", "absolute_error"),
        "xgboost": ("objective", "reg:absoluteerror"),
        "lightgbm": ("objective", "l1"),
        "catboost": ("loss_function", "MAE"),
    }
    for model, (name, value) in losses.items():
        est = build_pipeline(model, fixed_params(model), "median", 42).named_steps["model"]
        assert est.get_params()[name] == value


@pytest.mark.parametrize("model", list(ALGORITHMS))
def test_same_seed_same_predictions(model):
    X, y = table()
    a = build_pipeline(model, fixed_params(model), "median", 7).fit(X, y).predict(X)
    b = build_pipeline(model, fixed_params(model), "median", 7).fit(X, y).predict(X)
    assert a == pytest.approx(b)


def test_catboost_writes_no_catboost_info(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y = table()
    build_pipeline("catboost", fixed_params("catboost"), "median", 42).fit(X, y)
    assert not (tmp_path / "catboost_info").exists()
