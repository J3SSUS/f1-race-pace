"""Explain the champion of `f1-race-pace`: SHAP by feature, permutation importance, diagnostics and two local cases.

The explainer is picked by the champion's algorithm (`EXPLAINERS`): `LinearExplainer` for Ridge and ElasticNet,
`TreeExplainer` for the four tree models. The champion is read from the `champion` alias and loaded through
`runs:/<mlflow_run_id>/model`, because the registry cannot load `models:/f1-race-pace@champion` (see DECISIONS.md).
Explaining never promotes: it sets no alias and writes no `champion_history`, `metrics` or `predictions`.
"""

from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from common.supabase_io import get_client
from features.catalog import load_catalog
from features.store import read_table
from training.registry import REGISTERED_MODEL
from training.sampling import SAMPLING_TAG, resolve_tracking_uri

ALIAS = "champion"
EXPLAINERS = {
    "ridge": "linear",
    "elasticnet": "linear",
    "rf": "tree",
    "xgb": "tree",
    "lgbm": "tree",
    "catboost": "tree",
}
# model_versions.algorithm -> SHAP explainer; "baseline" and "dummy" are not explained
N_REPEATS = 30  # permutation importance; the seed is `optuna.seed` of the config
CHARTS_DIR = Path("site/charts")  # relative to the repository root


@dataclass(frozen=True)
class Champion:
    model_version_id: int  # model_versions.model_version_id
    mlflow_version: str  # version of f1-race-pace holding the `champion` alias
    mlflow_run_id: str  # the model run (tag of the version)
    sampling_run_id: str
    algorithm: str  # a key of EXPLAINERS
    model: Pipeline  # loaded from runs:/<mlflow_run_id>/model


def explainer_kind(algorithm: str) -> str:
    """`"linear"` or `"tree"` for a `model_versions.algorithm`; `ValueError` if the algorithm is not explained."""
    if algorithm in EXPLAINERS:
        return EXPLAINERS[algorithm]
    if algorithm in ("baseline", "dummy"):
        raise ValueError(f"the champion's algorithm is '{algorithm}', which is not explained")
    raise ValueError(f"algorithm '{algorithm}' has no explainer; allowed: {sorted(EXPLAINERS)}")


def load_champion(tracking_uri: str | None = None, client=None) -> Champion:
    """The version holding the `champion` alias, with its `model_versions` row and its model.

    Everything that can be wrong is checked before the model is loaded: no alias, a version without its
    `mlflow_run_id` or `sampling_run_id` tag, no `model_versions` row, or an algorithm that is not explained. The
    model is loaded through `runs:/<mlflow_run_id>/model`.
    """
    uri = resolve_tracking_uri(tracking_uri)
    ml = MlflowClient(uri)
    try:
        version = ml.get_model_version_by_alias(REGISTERED_MODEL, ALIAS)
    except MlflowException as exc:
        raise ValueError(f"{REGISTERED_MODEL} has no '{ALIAS}' alias: run scripts/promote.py first") from exc
    run_id = version.tags.get("mlflow_run_id", "")
    sampling_run_id = version.tags.get(SAMPLING_TAG, "")
    if not run_id or not sampling_run_id:
        raise ValueError(f"version {version.version} of {REGISTERED_MODEL} has no mlflow_run_id or sampling_run_id tag")

    client = client if client is not None else get_client()
    rows = read_table(client, "model_versions", "model_version_id,mlflow_run_id,algorithm")
    row = next((r for r in rows if r["mlflow_run_id"] == run_id), None)
    if row is None:
        raise ValueError(f"the champion (run {run_id}) has no model_versions row: run scripts/train.py")
    explainer_kind(row["algorithm"])  # raises before any model is loaded

    mlflow.set_tracking_uri(uri)
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    return Champion(row["model_version_id"], str(version.version), run_id, sampling_run_id, row["algorithm"], model)


def _feature_of_each_column(prep) -> list[str]:
    """The original feature behind each output column of the fitted `ColumnTransformer`, in output order.

    `output_indices_` gives each transformer's slice: the numeric block has one output column per input column, and
    a one-hot block (named after its feature) has one per category. Anything that does not fit raises instead of
    being dropped.
    """
    owner: dict[int, str] = {}
    for name, _, columns in prep.transformers_:
        if name == "remainder":
            continue
        block = prep.output_indices_[name]
        indices = list(range(block.start, block.stop))
        if name == "numeric":
            if len(indices) != len(columns):
                raise ValueError(f"the numeric block has {len(indices)} output columns for {len(columns)} features")
            owner.update(zip(indices, columns, strict=True))
        else:
            owner.update({i: name for i in indices})
    width = sum(s.stop - s.start for s in prep.output_indices_.values())
    missing = [i for i in range(width) if i not in owner]
    if missing:
        raise ValueError(f"output columns {missing} of the preprocessing cannot be mapped to a feature")
    return [owner[i] for i in range(width)]


def _dense(matrix) -> np.ndarray:
    return np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix, dtype="float64")


def shap_by_feature(champion: Champion, X: pd.DataFrame, background: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """`(rows x 15, base_value)`: the SHAP value of each original feature for each row of `X`.

    The pipeline's own preprocessing runs first and the final estimator is explained on the preprocessed matrix; the
    columns of the one-hot `circuit_type` are summed into one value. `LinearExplainer` uses `background` (the valid
    rows of train); `TreeExplainer` uses `tree_path_dependent`, whose reference is the data the trees saw. Values and
    base value are in the scale of the target, and `base_value + row sum` is the pipeline's prediction.
    """
    kind = explainer_kind(champion.algorithm)
    prep, estimator = champion.model.named_steps["prep"], champion.model.named_steps["model"]
    owners = _feature_of_each_column(prep)
    matrix = _dense(prep.transform(X))
    if kind == "linear":
        explainer = shap.LinearExplainer(estimator, _dense(prep.transform(background)))
    else:
        explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
    values = np.asarray(explainer.shap_values(matrix))
    base_value = float(np.ravel(explainer.expected_value)[0])

    names = [f.name for f in load_catalog().features]
    if set(owners) != set(names):
        raise ValueError(
            f"the preprocessing does not cover the catalog's features: extra {sorted(set(owners) - set(names))}, "
            f"missing {sorted(set(names) - set(owners))}"
        )
    columns = {name: values[:, [i for i, o in enumerate(owners) if o == name]].sum(axis=1) for name in names}
    return pd.DataFrame(columns, index=X.index), base_value
