"""One pipeline per model: imputation, encoding, scaling (linear models only) and the estimator.

Imputation lives inside the pipeline so it is fitted on the train part of each fold only. The columns come from the
feature catalog (`config/features.yml`): `category` features are one-hot encoded with the categories of the catalog,
every other feature is numeric.
"""

from typing import Any

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from common.config import IMPUTATIONS
from features.catalog import load_catalog

LINEAR = ("ridge", "elasticnet")  # these two also get StandardScaler
ALGORITHMS = {  # config key -> model_versions.algorithm
    "ridge": "ridge",
    "elasticnet": "elasticnet",
    "random_forest": "rf",
    "xgboost": "xgb",
    "lightgbm": "lgbm",
    "catboost": "catboost",
}


def _estimator(model: str, params: dict[str, Any], seed: int):
    if model == "ridge":
        return Ridge(**params, random_state=seed)
    if model == "elasticnet":
        return ElasticNet(**params, random_state=seed)
    if model == "random_forest":
        return RandomForestRegressor(**params, random_state=seed)
    if model == "xgboost":
        return XGBRegressor(**params, random_state=seed)
    if model == "lightgbm":
        return LGBMRegressor(**{"verbose": -1, **params}, random_state=seed)
    if model == "catboost":
        return CatBoostRegressor(**params, random_seed=seed, allow_writing_files=False)
    raise ValueError(f"unknown model '{model}'; allowed: {sorted(ALGORITHMS)}")


def build_pipeline(model: str, params: dict[str, Any], imputation: str, seed: int) -> Pipeline:
    """Preprocessing plus the estimator `model` with `params` (the imputation is not one of them)."""
    if imputation not in IMPUTATIONS:
        raise ValueError(f"unknown imputation '{imputation}'; allowed: {sorted(IMPUTATIONS)}")
    estimator = _estimator(model, params, seed)

    features = load_catalog().features
    numeric = [f.name for f in features if f.dtype != "category"]
    steps = [
        # nullable Int64 columns hold pd.NA, which the imputer cannot read: cast to float (NaN) first
        ("cast", FunctionTransformer(lambda X: X.astype("float64"))),
        # a feature that is all NaN in the train part of a fold is kept (filled with 0), so the shape never changes
        ("impute", SimpleImputer(strategy=imputation, keep_empty_features=True)),
    ]
    if model in LINEAR:
        steps.append(("scale", StandardScaler()))
    parts = [("numeric", Pipeline(steps), numeric)]
    for f in features:
        if f.dtype == "category":
            parts.append((f.name, OneHotEncoder(categories=[list(f.allowed)], handle_unknown="ignore"), [f.name]))
    return Pipeline([("prep", ColumnTransformer(parts, remainder="drop")), ("model", estimator)])
