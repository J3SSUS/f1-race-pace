"""The two references every model is measured against.

The qualifying baseline is the contract's: linear regression of the target on `quali_delta_pct`. The floor predicts
the median of the train target (median, because the main metric is MAE). Neither is a model version.
"""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

QUALIFYING = "baseline"  # algorithm tag "baseline"
FLOOR = "floor"  # algorithm tag "dummy"
ALGORITHMS = {QUALIFYING: "baseline", FLOOR: "dummy"}
BASELINE_FEATURE = "quali_delta_pct"


def qualifying_baseline() -> Pipeline:
    """Linear regression on `quali_delta_pct` alone: any other column is dropped before the fit."""
    select = ColumnTransformer([("quali", "passthrough", [BASELINE_FEATURE])], remainder="drop")
    return Pipeline([("select", select), ("linear", LinearRegression())])


def median_floor() -> DummyRegressor:
    return DummyRegressor(strategy="median")
