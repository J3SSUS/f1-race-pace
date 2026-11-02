"""Explain the champion of `f1-race-pace`: SHAP by feature, permutation importance, diagnostics and two local cases.

The explainer is picked by the champion's algorithm (`EXPLAINERS`): `LinearExplainer` for Ridge and ElasticNet,
`TreeExplainer` for the four tree models. The champion is read from the `champion` alias and loaded through
`runs:/<mlflow_run_id>/model`, because the registry cannot load `models:/f1-race-pace@champion` (see DECISIONS.md).
Explaining never promotes: it sets no alias and writes no `champion_history`, `metrics` or `predictions`.
"""

import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from common.config import TrainingConfig, load_config
from common.supabase_io import get_client, upsert_rows
from features.catalog import load_catalog
from features.store import StoreError, read_table, read_wide
from training import explain_charts as charts
from training.data import NOT_FEATURES, TARGET, TrainSet, load_train
from training.registry import REGISTERED_MODEL
from training.sampling import SAMPLING_TAG, read_split, resolve_tracking_uri

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
class Case:
    label: str  # "fastest_predicted" | "slowest_predicted"
    entry_id: int
    year: int
    round: int
    driver: str
    prediction: float  # race_pace_delta_pct predicted
    actual: float  # race_pace_delta_pct of the race (read after choosing the case)
    error: float  # prediction - actual
    base_value: float  # the explainer's base value
    contributions: dict[str, float]  # feature -> SHAP; base_value + sum(contributions) == prediction


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
        reference = _dense(prep.transform(background))
        # the default masker keeps a random 100 rows: ask for every row of the background
        explainer = shap.LinearExplainer(estimator, shap.maskers.Independent(reference, max_samples=len(reference)))
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


def permutation_importance_mae(
    champion: Champion, train: TrainSet, *, n_repeats: int = N_REPEATS, seed: int | None = None
) -> pd.DataFrame:
    """How much the MAE of the champion rises when each of the 15 original features is permuted, on the train rows.

    The permutation is done on the pipeline's input columns (not the encoded ones) over the valid rows of train, with
    the champion as it was fitted. The seed defaults to `optuna.seed` of the config. Columns `feature`,
    `mae_increase_mean` and `mae_increase_std`, largest mean first (a tie keeps the catalog order).
    """
    seed = load_config().optuna.seed if seed is None else seed
    result = permutation_importance(
        champion.model,
        train.X,
        train.y,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=seed,
    )
    frame = pd.DataFrame(
        {
            "feature": list(train.X.columns),
            "mae_increase_mean": result.importances_mean,
            "mae_increase_std": result.importances_std,
        }
    )
    return frame.sort_values("mae_increase_mean", ascending=False, kind="stable").reset_index(drop=True)


def calibration_line(oof: pd.DataFrame) -> dict[str, float]:
    """The Mincer-Zarnowitz line of the out-of-fold predictions: `y_true = intercept + slope * y_pred`.

    A calibrated model has slope 1 and intercept 0. Returns `slope`, `intercept`, `n_rows` and `n_races` (distinct
    `(year, round)`). An empty OOF, or predictions that do not vary, raise `ValueError`.
    """
    if oof.empty:
        raise ValueError("the out-of-fold predictions have no rows")
    y_true = oof["y_true"].to_numpy(dtype="float64")
    y_pred = oof["y_pred"].to_numpy(dtype="float64")
    if np.ptp(y_pred) == 0:
        raise ValueError("the predictions do not vary, so the calibration line is undefined")
    slope, intercept = np.polyfit(y_pred, y_true, 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "n_rows": len(oof),
        "n_races": int(oof[["year", "round"]].drop_duplicates().shape[0]),
    }


def load_explain_rows(
    run_id: str,
    client=None,
    *,
    version: str = "v1",
    tracking_uri: str | None = None,
    reader: Callable = read_wide,
    splitter: Callable = read_split,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(X, meta)` of the valid rows of the train and test races of the sampling run, WITHOUT the target.

    `X` has the 15 features in catalog order. `meta` has `entry_id`, `year`, `round`, `driver` and `split`
    (`"train"` or `"test"`), with the same index as `X`, in the order of `read_wide`. The split is read first, so an
    unknown run stops before Supabase is touched. The target and the other ground-truth columns are dropped as soon as
    the rows are chosen: no return value holds them. Both sides of the split must have at least one valid row.
    """
    labels = splitter(run_id, tracking_uri)[["year", "round", "split"]]
    wide = reader(client if client is not None else get_client(), version)

    rows = wide.merge(labels, on=["year", "round"], how="inner")  # keeps the order of `wide`; drops other races
    rows = rows[rows["is_valid_row"].to_numpy()].reset_index(drop=True)
    for side in ("train", "test"):
        if not (rows["split"] == side).any():
            raise ValueError(f"The {side} races of run '{run_id}' have no valid row in feature set {version}")

    features = [c for c in rows.columns if c not in NOT_FEATURES | {"split"}]
    return rows[features], rows[["entry_id", "year", "round", "driver", "split"]]


def read_actuals(client, entry_ids: list[int], version: str = "v1") -> dict[int, float]:
    """`race_pace_delta_pct` of the entries asked for and of no other: the one read of a test target in SPEC 11.

    It asks Supabase for those entries only (not the whole table). The caller chooses the entries from the
    predictions first; the value is only displayed, never used in a metric, a choice or a tuning. `version` is
    accepted like in the other readers; the target is the same for every feature set. An entry that does not exist,
    or has no actual value, raises `ValueError`.
    """
    wanted = sorted(set(entry_ids))
    if not wanted:
        return {}
    try:
        rows = (
            client.table("ground_truth")
            .select("entry_id,actual_pace_delta_pct")
            .in_("entry_id", wanted)
            .range(0, len(wanted) - 1)
            .execute()
            .data
        )
    except Exception as exc:
        raise StoreError(f"read from ground_truth failed: {type(exc).__name__}") from None
    found = {r["entry_id"]: r["actual_pace_delta_pct"] for r in rows}
    missing = [e for e in wanted if found.get(e) is None]
    if missing:
        raise ValueError(f"no {TARGET} for entries {missing}: they do not exist in ground_truth or have no value")
    return {e: float(found[e]) for e in wanted}


def shap_rows(
    shap_values: pd.DataFrame, meta: pd.DataFrame, feature_ids: dict[str, int], model_version_id: int
) -> list[dict]:
    """One `shap_values` row per entry and feature, in the order of the rows and then of the columns.

    `shap_values` and `meta` share the same index (see `load_explain_rows` and `shap_by_feature`). Every column of
    `shap_values` must be in `feature_ids` and every value must be finite (the column is NOT NULL): otherwise it
    raises, so nothing is written.
    """
    missing = [name for name in shap_values.columns if name not in feature_ids]
    if missing:
        raise ValueError(f"features {missing} are not in the features table: load the feature set first")
    values = shap_values.to_numpy("float64")
    if not np.isfinite(values).all():
        raise ValueError("shap_values has a value that is not finite")
    entries = meta.loc[shap_values.index, "entry_id"].astype(int).tolist()
    ids = [feature_ids[name] for name in shap_values.columns]
    return [
        {"entry_id": entry, "model_version_id": model_version_id, "feature_id": feature, "shap_value": float(value)}
        for entry, row in zip(entries, values, strict=True)
        for feature, value in zip(ids, row, strict=True)
    ]


def upsert_shap_values(client, champion: Champion, meta: pd.DataFrame, shap_values: pd.DataFrame) -> int:
    """Upsert the SHAP values of the champion on `entry_id,model_version_id,feature_id`; returns the rows written.

    The `feature_id`s are read from `features` and the rows are built before the first write, so a missing feature
    raises with nothing stored. Running it again with the same values leaves the same rows. It touches no other table.
    """
    feature_ids = {r["name"]: r["feature_id"] for r in read_table(client, "features", "feature_id,name")}
    rows = shap_rows(shap_values, meta, feature_ids, champion.model_version_id)
    return len(upsert_rows(client, "shap_values", rows, "entry_id,model_version_id,feature_id"))


ADDITIVITY_TOLERANCE = 1e-4  # a case whose contributions miss its prediction by more than this is a bug, not a case


def build_cases(
    champion: Champion,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    shap_values: pd.DataFrame,
    base_value: float,
    client,
    *,
    version: str = "v1",
    reader: Callable = read_actuals,
) -> list[Case]:
    """The two local cases of the most recent test race: the driver predicted fastest and the one predicted slowest.

    `X`, `meta` (from `load_explain_rows`) and `shap_values` (from `shap_by_feature`) share the same index. The
    most recent test race is the largest `(year, round)` among the `test` rows; the lowest prediction is the fastest
    (`race_pace_delta_pct` is the gap to the field) and the highest the slowest; a tie goes to the lowest `entry_id`.
    Both are chosen from the predictions alone and only then `reader` is called, once, with exactly those two
    entries. The actual values are only attached to the cases: they never take part in the choice. A race with fewer
    than two valid rows, or with identical predictions for all of them, raises.
    """
    test = meta[meta["split"] == "test"]
    last = test.sort_values(["year", "round"]).iloc[-1]
    race = test[(test["year"] == last["year"]) & (test["round"] == last["round"])]
    label = f"{int(last['year'])} round {int(last['round'])}"
    if len(race) < 2:
        raise ValueError(f"the most recent test race ({label}) has {len(race)} valid row, two cases need at least two")

    rows = race.assign(prediction=champion.model.predict(X.loc[race.index]))
    if rows["prediction"].nunique() == 1:
        raise ValueError(f"the most recent test race ({label}) has the same prediction for every driver")
    fastest = rows.sort_values(["prediction", "entry_id"]).iloc[0]
    slowest = rows.assign(neg=-rows["prediction"]).sort_values(["neg", "entry_id"]).iloc[0]
    chosen = [("fastest_predicted", fastest), ("slowest_predicted", slowest)]

    actuals = reader(client, [int(row["entry_id"]) for _, row in chosen], version)  # the first read of a target
    cases = []
    for name, row in chosen:
        contributions = {feature: float(value) for feature, value in shap_values.loc[row.name].items()}
        prediction = float(row["prediction"])
        if abs(base_value + sum(contributions.values()) - prediction) > ADDITIVITY_TOLERANCE:
            raise ValueError(f"the contributions of entry {int(row['entry_id'])} do not add up to its prediction")
        actual = actuals[int(row["entry_id"])]
        cases.append(
            Case(
                label=name,
                entry_id=int(row["entry_id"]),
                year=int(row["year"]),
                round=int(row["round"]),
                driver=str(row["driver"]),
                prediction=prediction,
                actual=actual,
                error=prediction - actual,
                base_value=base_value,
                contributions=contributions,
            )
        )
    return cases


EXPLAIN = "explain"  # run name of the step


@dataclass(frozen=True)
class Explanation:
    shap: pd.DataFrame  # entry_id + the 15 features; valid rows of train and test
    base_value: float
    importance: pd.DataFrame  # feature, mae_increase_mean, mae_increase_std; largest first
    coefficients: dict[str, float] | None  # linear champions only, by preprocessed column
    calibration: dict[str, float]  # Mincer-Zarnowitz on the OOF: slope, intercept, n_rows, n_races
    cases: list[Case]  # two, in the order fastest, slowest


def linear_coefficients(champion: Champion) -> dict[str, float] | None:
    """The coefficient of each preprocessed column of a linear champion; `None` for a tree champion.

    The numeric columns are standardized by the pipeline and keep their feature name; the one-hot columns of a
    categorical feature are named `<feature>_<category>` and kept apart.
    """
    if explainer_kind(champion.algorithm) != "linear":
        return None
    prep, estimator = champion.model.named_steps["prep"], champion.model.named_steps["model"]
    owners = _feature_of_each_column(prep)
    names = []
    for index, owner in enumerate(owners):
        if owner in prep.named_transformers_ and owner != "numeric":
            transformer = prep.named_transformers_[owner]
            encoder = transformer[-1] if isinstance(transformer, Pipeline) else transformer
            start = prep.output_indices_[owner].start
            names.append(f"{owner}_{encoder.categories_[0][index - start]}")
        else:
            names.append(owner)
    coefficients = np.ravel(estimator.coef_)
    if len(coefficients) != len(names):
        raise ValueError(f"the model has {len(coefficients)} coefficients for {len(names)} preprocessed columns")
    return dict(zip(names, (float(c) for c in coefficients), strict=True))


def load_champion_oof(champion: Champion, tracking_uri: str | None = None) -> pd.DataFrame:
    """The `oof_predictions.parquet` of the champion's own training run (nothing is refitted)."""
    client = MlflowClient(resolve_tracking_uri(tracking_uri))
    try:
        path = client.download_artifacts(champion.mlflow_run_id, "oof_predictions.parquet")
    except MlflowException as exc:
        raise ValueError(f"the champion's run {champion.mlflow_run_id} has no oof_predictions.parquet") from exc
    return pd.read_parquet(path)


def _draw_all(explanation: Explanation, X: pd.DataFrame, oof: pd.DataFrame, version: int, folder: Path) -> list[Path]:
    """Draw every `.svg` chart into `folder` and return the paths (no coefficients chart for a tree champion)."""
    shap_only = explanation.shap.drop(columns="entry_id")
    fastest, slowest = explanation.cases
    drawn = [
        charts.coefficients_chart(explanation.coefficients, version, folder / "coefficients.svg"),
        charts.permutation_chart(explanation.importance, version, folder / "permutation_importance.svg"),
        charts.shap_summary_chart(shap_only, X, version, folder / "shap_summary.svg"),
        charts.shap_bar_chart(shap_only, version, folder / "shap_bar.svg"),
        charts.waterfall_chart(fastest, version, folder / "case_fastest.svg"),
        charts.waterfall_chart(slowest, version, folder / "case_slowest.svg"),
        charts.predicted_vs_actual_chart(oof, explanation.calibration, version, folder / "predicted_vs_actual.svg"),
        charts.residuals_chart(oof, version, folder / "residuals.svg"),
    ]
    return [path for path in drawn if path is not None]


def _explain_json(champion: Champion, explanation: Explanation, explainer: str, run_id: str, now: datetime) -> dict:
    shap_only = explanation.shap.drop(columns="entry_id")
    return {
        "run_id": run_id,
        "sampling_run_id": champion.sampling_run_id,
        "model_version_id": champion.model_version_id,
        "mlflow_version": champion.mlflow_version,
        "algorithm": champion.algorithm,
        "explainer": explainer,
        "generated_at": now.isoformat(),
        "base_value": explanation.base_value,
        "importance": explanation.importance.to_dict("records"),
        "mean_abs_shap": {name: float(v) for name, v in shap_only.abs().mean().items()},
        "coefficients": explanation.coefficients,
        "calibration": explanation.calibration,
        "cases": [asdict(case) for case in explanation.cases],
    }


def run_explain(
    config: TrainingConfig | None = None,
    client=None,
    tracking_uri: str | None = None,
    charts_dir: Path = CHARTS_DIR,
    *,
    version: str = "v1",
    now: datetime | None = None,
    rows_loader: Callable = load_explain_rows,
    train_loader: Callable = load_train,
    oof_loader: Callable = load_champion_oof,
    actuals_reader: Callable = read_actuals,
) -> dict:
    """The `explain` step: explain the champion and publish the result.

    Everything that can fail without writing runs first (champion, rows, OOF, SHAP, importance, calibration, cases).
    Then an `explain` root run opens in the sampling run's experiment and, inside it, the SHAP rows are upserted into
    Supabase, the charts are drawn into a temporary folder and only then copied to `charts_dir` with `explain.json`
    and logged as artifacts. A Supabase failure leaves `charts_dir` untouched and the run FAILED. It can run any
    number of times (upsert plus overwritten files) and never promotes: no alias, no `champion_history`, `metrics`
    or `predictions`. Returns `{"run_id", "champion", "explanation", "files", "shap_rows"}`.
    """
    config = config or load_config()
    uri = resolve_tracking_uri(tracking_uri)
    client = client if client is not None else get_client()
    now = now or datetime.now(UTC)
    seed = config.optuna.seed

    champion = load_champion(uri, client)
    explainer = explainer_kind(champion.algorithm)
    X, meta = rows_loader(champion.sampling_run_id, client, version=version, tracking_uri=uri)
    train = train_loader(champion.sampling_run_id, client, version=version, tracking_uri=uri)
    oof = oof_loader(champion, uri)

    background = X[(meta["split"] == "train").to_numpy()]
    values, base_value = shap_by_feature(champion, X, background)
    importance = permutation_importance_mae(champion, train, seed=seed)
    calibration = calibration_line(oof)
    cases = build_cases(champion, X, meta, values, base_value, client, version=version, reader=actuals_reader)
    explanation = Explanation(
        shap=values.assign(entry_id=meta["entry_id"].astype(int))[["entry_id", *values.columns]],
        base_value=base_value,
        importance=importance,
        coefficients=linear_coefficients(champion),
        calibration=calibration,
        cases=cases,
    )

    ml = MlflowClient(uri)
    experiment_id = ml.get_run(champion.sampling_run_id).info.experiment_id
    mlflow.set_tracking_uri(uri)
    tags = {
        "step": EXPLAIN,
        SAMPLING_TAG: champion.sampling_run_id,
        "model_version_id": str(champion.model_version_id),
        "algorithm": champion.algorithm,
    }
    with mlflow.start_run(run_name=EXPLAIN, experiment_id=experiment_id, tags=tags) as run:
        n_rows = upsert_shap_values(client, champion, meta, values)  # Supabase first: a failure writes no file

        with tempfile.TemporaryDirectory() as folder:
            drawn = _draw_all(explanation, X, oof, champion.model_version_id, Path(folder))
            report = Path(folder) / "explain.json"
            report.write_text(
                json.dumps(_explain_json(champion, explanation, explainer, run.info.run_id, now), indent=2)
            )
            charts_dir = Path(charts_dir)
            charts_dir.mkdir(parents=True, exist_ok=True)
            files = [Path(shutil.copy2(path, charts_dir / path.name)) for path in [*drawn, report]]

        mlflow.log_params(
            {"explainer": explainer, "n_repeats": N_REPEATS, "seed": seed, "background_rows": len(background)}
        )
        metrics = {"calibration_slope": calibration["slope"], "calibration_intercept": calibration["intercept"]}
        for row in importance.itertuples():
            metrics[f"perm_{row.feature}"] = float(row.mae_increase_mean)
        for name, value in values.abs().mean().items():
            metrics[f"mean_abs_shap_{name}"] = float(value)
        for case in cases:
            metrics[f"case_{case.label}_prediction"] = case.prediction
            metrics[f"case_{case.label}_actual"] = case.actual
        mlflow.log_metrics(metrics)
        for path in files:
            mlflow.log_artifact(str(path))
    return {
        "run_id": run.info.run_id,
        "champion": champion,
        "explanation": explanation,
        "files": files,
        "shap_rows": n_rows,
    }
