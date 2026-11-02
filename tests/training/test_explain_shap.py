import numpy as np
import pandas as pd
import pytest
from test_explain_champion import synthetic_X, synthetic_y
from test_pipelines import fixed_params

from features.catalog import load_catalog
from training.explain import Champion, shap_by_feature
from training.pipelines import ALGORITHMS, build_pipeline

FEATURES = [f.name for f in load_catalog().features]
TOLERANCE = 1e-5  # XGBoost keeps its trees in float32


def fitted_champion(name: str, X: pd.DataFrame) -> Champion:
    model = build_pipeline(name, fixed_params(name), "median", 42).fit(X, synthetic_y(X))
    return Champion(1, "1", "run", "sampling", ALGORITHMS[name], model)


@pytest.fixture(scope="module")
def X() -> pd.DataFrame:
    return synthetic_X(60)


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_each_row_gets_one_value_for_each_of_the_15_features(X, name):
    champion = fitted_champion(name, X)
    values, _ = shap_by_feature(champion, X, X)
    assert list(values.columns) == FEATURES
    assert len(FEATURES) == 15
    assert values.index.equals(X.index)
    assert not any(c.startswith("circuit_type_") for c in values.columns)  # the one-hot columns come out as one


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_base_value_plus_the_values_is_the_prediction(X, name):
    champion = fitted_champion(name, X)
    values, base = shap_by_feature(champion, X, X)
    np.testing.assert_allclose(base + values.sum(axis=1), champion.model.predict(X), atol=TOLERANCE)


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_a_row_with_a_missing_feature_still_adds_up(X, name):
    champion = fitted_champion(name, X)
    rows = synthetic_X(10, seed=3)
    numeric = [f.name for f in load_catalog().features if f.dtype != "category"]
    rows.loc[rows.index[:5], numeric[0]] = np.nan
    rows.loc[rows.index[3:7], numeric[2]] = np.nan
    values, base = shap_by_feature(champion, rows, X)
    np.testing.assert_allclose(base + values.sum(axis=1), champion.model.predict(rows), atol=TOLERANCE)


@pytest.mark.parametrize("name", ["ridge", "elasticnet"])
def test_linear_values_are_coef_times_x_minus_mean_by_hand(X, name):
    champion = fitted_champion(name, X)
    values, base = shap_by_feature(champion, X, X)
    estimator = champion.model.named_steps["model"]
    scaler = champion.model.named_steps["prep"].named_transformers_["numeric"].named_steps["scale"]
    numeric = [f.name for f in load_catalog().features if f.dtype != "category"]
    coef = dict(zip(numeric, estimator.coef_[: len(numeric)], strict=True))
    for feature in numeric:  # the numeric block is standardized: coef * (x - mean) / scale, mean of the background
        i = numeric.index(feature)
        expected = coef[feature] * (X[feature] - scaler.mean_[i]) / scaler.scale_[i]
        np.testing.assert_allclose(values[feature], expected, atol=1e-10)

    categories = list(load_catalog().features[FEATURES.index("circuit_type")].allowed)
    onehot = pd.get_dummies(X["circuit_type"]).reindex(columns=categories, fill_value=False).astype("float64")
    expected = (onehot - onehot.mean()).to_numpy() @ estimator.coef_[len(numeric) :]  # three columns summed into one
    np.testing.assert_allclose(values["circuit_type"], expected, atol=1e-10)
    assert base == pytest.approx(estimator.intercept_ + estimator.coef_ @ _background_mean(champion, X))


def _background_mean(champion: Champion, background: pd.DataFrame) -> np.ndarray:
    prep = champion.model.named_steps["prep"]
    return np.asarray(prep.transform(background), dtype="float64").mean(axis=0)


def test_a_linear_explanation_uses_the_background_given(X):
    champion = fitted_champion("ridge", X)
    other = synthetic_X(60, seed=9)
    _, base_train = shap_by_feature(champion, X, X)
    _, base_other = shap_by_feature(champion, X, other)
    assert base_train != pytest.approx(base_other)


def test_an_output_column_that_cannot_be_mapped_raises_instead_of_being_dropped(X):
    champion = fitted_champion("ridge", X)
    prep = champion.model.named_steps["prep"]
    width = max(s.stop for s in prep.output_indices_.values())
    prep.output_indices_ = {**prep.output_indices_, "extra": slice(width, width + 1)}
    with pytest.raises(ValueError, match="cannot be mapped"):
        shap_by_feature(champion, X, X)


def test_an_algorithm_without_an_explainer_raises(X):
    champion = fitted_champion("ridge", X)
    with pytest.raises(ValueError, match="baseline"):
        shap_by_feature(Champion(1, "1", "run", "sampling", "baseline", champion.model), X, X)
