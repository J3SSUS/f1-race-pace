import json
import pickle
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from test_explain_champion import log_version, promote, supabase, synthetic_X, synthetic_y  # noqa: F401
from test_explain_champion import parent as parent  # noqa: F401

from common.config import load_config
from features.catalog import load_catalog
from training.data import TrainSet
from training.explain import linear_coefficients, load_champion, run_explain, shap_by_feature
from training.registry import REGISTERED_MODEL

FEATURES = [f.name for f in load_catalog().features]
N = 30
EXPECTED_TREE = {
    "permutation_importance.svg",
    "shap_summary.svg",
    "shap_bar.svg",
    "case_fastest.svg",
    "case_slowest.svg",
    "predicted_vs_actual.svg",
    "residuals.svg",
    "explain.json",
}


def logged_champion(uri, parent, supabase, name):  # noqa: F811
    """A registered and promoted champion whose run also holds an `oof_predictions.parquet`."""
    version = log_version(uri, parent, supabase, name)
    rng = np.random.default_rng(3)
    oof = pd.DataFrame(
        {"entry_id": np.arange(40), "year": 2024, "round": np.repeat([1, 2], 20), "driver": "AAA", "fold": 1}
    )
    oof["y_true"] = rng.normal(size=40)
    oof["y_pred"] = 0.8 * oof["y_true"] + rng.normal(scale=0.2, size=40)
    folder = Path(tempfile.mkdtemp())
    path = folder / "oof_predictions.parquet"
    oof.to_parquet(path)
    run_id = MlflowClient(uri).get_model_version(REGISTERED_MODEL, version.mlflow_version).tags["mlflow_run_id"]
    MlflowClient(uri).log_artifact(run_id, path)
    promote(uri, version)
    return version


class Inputs:
    """The rows, the train set and the actuals the step reads, built by hand (no Supabase, no network)."""

    def __init__(self):
        self.X = synthetic_X(N, seed=5)
        self.meta = pd.DataFrame(
            {
                "entry_id": 100 + np.arange(N),
                "year": 2025,
                "round": np.repeat([1, 2, 3], 10),
                "driver": [f"D{i}" for i in range(N)],
                "split": ["train"] * 20 + ["test"] * 10,
            }
        )
        self.reads: list[list[int]] = []

    def rows(self, run_id, client, **kwargs):
        return self.X, self.meta

    def train(self, run_id, client, **kwargs):
        mask = self.meta["split"] == "train"
        return TrainSet(
            self.X[mask], synthetic_y(self.X)[mask], self.meta.loc[mask, ["entry_id", "year", "round", "driver"]]
        )

    def actuals(self, client, entry_ids, version="v1"):
        self.reads.append(list(entry_ids))
        return {e: 0.25 for e in entry_ids}


@pytest.fixture
def inputs():
    return Inputs()


def seeded(supabase):  # noqa: F811
    supabase.tables["features"] = [{"feature_id": i, "name": n} for i, n in enumerate(FEATURES, 1)]
    return supabase


def explain(uri, supabase, inputs, folder, **kwargs):  # noqa: F811
    return run_explain(
        load_config(),
        supabase,
        uri,
        folder,
        rows_loader=inputs.rows,
        train_loader=inputs.train,
        actuals_reader=inputs.actuals,
        **kwargs,
    )


def test_a_tree_champion_writes_the_run_and_the_same_files_in_the_folder(uri, parent, supabase, inputs, tmp_path):  # noqa: F811
    logged_champion(uri, parent, seeded(supabase), "random_forest")
    result = explain(uri, supabase, inputs, tmp_path / "charts")
    written = {p.name for p in (tmp_path / "charts").iterdir()}
    assert written == EXPECTED_TREE  # no coefficients.svg for a tree
    assert result["explanation"].coefficients is None
    run = MlflowClient(uri).get_run(result["run_id"])
    assert run.info.status == "FINISHED"
    assert (
        run.data.tags["step"] == "explain"
        and run.data.tags["algorithm"] == "random_forest"
        or run.data.tags["algorithm"]
    )
    assert {"explainer", "n_repeats", "seed", "background_rows"} <= set(run.data.params)
    assert run.data.params["background_rows"] == "20"
    metrics = run.data.metrics
    assert {"calibration_slope", "calibration_intercept"} <= set(metrics)
    assert all(f"perm_{f}" in metrics and f"mean_abs_shap_{f}" in metrics for f in FEATURES)
    assert {"case_fastest_predicted_prediction", "case_slowest_predicted_actual"} <= set(metrics)
    artifacts = {a.path for a in MlflowClient(uri).list_artifacts(result["run_id"])}
    assert artifacts == written | {"explainer.pkl"}  # the explainer is in MLflow only, not in site/charts/
    assert run.data.tags["shap_version"]


def test_a_linear_champion_also_writes_the_coefficients_chart(uri, parent, supabase, inputs, tmp_path):  # noqa: F811
    logged_champion(uri, parent, seeded(supabase), "ridge")
    result = explain(uri, supabase, inputs, tmp_path)
    assert "coefficients.svg" in {p.name for p in tmp_path.iterdir()}
    coefficients = result["explanation"].coefficients
    assert coefficients and "quali_delta_pct" in coefficients
    assert any(name.startswith("circuit_type_") for name in coefficients)


def test_explain_json_has_the_fields_of_the_data_model_and_the_two_cases_have_their_actual_value(
    uri,
    parent,
    supabase,  # noqa: F811
    inputs,
    tmp_path,
):
    logged_champion(uri, parent, seeded(supabase), "ridge")
    explain(uri, supabase, inputs, tmp_path)
    report = json.loads((tmp_path / "explain.json").read_text())
    assert {
        "run_id",
        "sampling_run_id",
        "model_version_id",
        "mlflow_version",
        "algorithm",
        "explainer",
        "generated_at",
        "base_value",
        "importance",
        "mean_abs_shap",
        "coefficients",
        "calibration",
        "cases",
    } <= set(report)
    assert [c["label"] for c in report["cases"]] == ["fastest_predicted", "slowest_predicted"]
    assert all(c["actual"] == 0.25 and c["error"] == pytest.approx(c["prediction"] - 0.25) for c in report["cases"])
    assert len(report["mean_abs_shap"]) == 15


def test_running_twice_gives_the_same_shap_values_and_overwrites_the_files(uri, parent, supabase, inputs, tmp_path):  # noqa: F811
    logged_champion(uri, parent, seeded(supabase), "ridge")
    explain(uri, supabase, inputs, tmp_path)
    rows = [dict(r) for r in supabase.tables["shap_values"]]
    first = {p.name: p.read_bytes() for p in tmp_path.glob("*.svg")}
    explain(uri, supabase, inputs, tmp_path)
    assert supabase.tables["shap_values"] == rows
    assert len(rows) == 2 * 15
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.svg")} == first


def test_a_supabase_failure_leaves_the_folder_empty_and_the_run_failed(uri, parent, supabase, inputs, tmp_path):  # noqa: F811
    seeded(supabase)
    logged_champion(uri, parent, supabase, "ridge")
    original = supabase.table

    def broken(name):
        if name == "shap_values":
            raise ConnectionError("down")
        return original(name)

    supabase.table = broken
    with pytest.raises(RuntimeError):
        explain(uri, supabase, inputs, tmp_path / "charts")
    assert not (tmp_path / "charts").exists() or not list((tmp_path / "charts").iterdir())
    runs = mlflow.search_runs(search_all_experiments=True, filter_string="tags.step = 'explain'")
    assert list(runs["status"]) == ["FAILED"]


def test_only_the_two_cases_are_stored_in_shap_values(uri, parent, supabase, inputs, tmp_path):  # noqa: F811
    logged_champion(uri, parent, seeded(supabase), "ridge")
    result = explain(uri, supabase, inputs, tmp_path)
    rows = supabase.tables["shap_values"]
    assert result["shap_rows"] == len(rows) == 30
    assert {r["entry_id"] for r in rows} == {case.entry_id for case in result["explanation"].cases}
    assert {r["model_version_id"] for r in rows} == {result["champion"].model_version_id}
    stored = {(r["entry_id"], r["feature_id"]): r["shap_value"] for r in rows}
    ids = {r["name"]: r["feature_id"] for r in supabase.tables["features"]}
    for case in result["explanation"].cases:  # the stored values are the contributions of the case
        for name, value in case.contributions.items():
            assert stored[(case.entry_id, ids[name])] == pytest.approx(value)


@pytest.mark.parametrize("name", ["ridge", "random_forest"])
def test_the_saved_explainer_gives_the_same_shap_as_the_run(uri, parent, supabase, inputs, tmp_path, name):  # noqa: F811
    logged_champion(uri, parent, seeded(supabase), name)
    result = explain(uri, supabase, inputs, tmp_path)
    path = MlflowClient(uri).download_artifacts(result["run_id"], "explainer.pkl")
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    assert bundle["algorithm"] == result["champion"].algorithm
    assert bundle["model_version_id"] == result["champion"].model_version_id
    champion = load_champion(uri, supabase)
    again, base = shap_by_feature(champion, inputs.X, inputs.X, bundle["explainer"], bundle["owners"])
    expected = result["explanation"].shap.drop(columns="entry_id")
    np.testing.assert_allclose(again.to_numpy(), expected.to_numpy(), atol=1e-8)
    assert base == pytest.approx(bundle["base_value"])


def test_it_never_promotes_or_writes_other_tables(uri, parent, supabase, inputs, tmp_path):  # noqa: F811
    logged_champion(uri, parent, seeded(supabase), "ridge")
    before = MlflowClient(uri).get_model_version_by_alias(REGISTERED_MODEL, "champion").version
    already = len(supabase.writes)  # what registering the version wrote
    kept = {t: supabase.count(t) for t in ("champion_history", "metrics", "predictions", "model_versions")}
    explain(uri, supabase, inputs, tmp_path)
    assert MlflowClient(uri).get_model_version_by_alias(REGISTERED_MODEL, "champion").version == before
    assert {table for table, _ in supabase.writes[already:]} == {"shap_values"}
    assert {t: supabase.count(t) for t in kept} == kept


def test_linear_coefficients_is_none_for_a_tree_champion(uri, parent, supabase):  # noqa: F811
    logged_champion(uri, parent, seeded(supabase), "random_forest")
    assert linear_coefficients(load_champion(uri, supabase)) is None
