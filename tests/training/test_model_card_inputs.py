import mlflow
import pytest
from fake_supabase import FakeSupabase
from mlflow import MlflowClient
from test_explain_champion import NOW  # noqa: F401  (the parent fixture below needs the same clock)
from test_manifest import config, make_parts

from training.model_card import BOOKKEEPING_KEYS, CardInputs, load_card_inputs
from training.registry import REGISTERED_MODEL
from training.sampling import register_run

PARAMS = {
    "alpha": "3.2",
    "imputation": "median",
    "n_trials": "50",
    "seed": "42",
    "n_folds": "5",
    "train_races": "46",
    "train_rows": "851",
    "features": "grid_position,quali_delta_pct",
    "metric_main": "mae",
}
EXPLAIN_JSON = {"run_id": "x", "model_version_id": 7, "importance": [], "cases": []}
SELECTION = {"candidate": "ridge", "reference": "ridge", "table": []}
TRAINED = {"feature_set_id": 5, "trained_through_race_id": 102}
REFERENCE = {
    "feature_sets": [{"feature_set_id": 5, "version": "v1"}, {"feature_set_id": 6, "version": "v2"}],
    "races": [{"race_id": 102, "year": 2024, "round": 2}, {"race_id": 103, "year": 2024, "round": 3}],
}
METRICS = {"mae": 0.4, "rmse": 0.5, "r2": 0.1, "pairwise_accuracy": 0.6}
TEST_METRICS = {
    name: {"metrics": METRICS, "intervals": {m: [0.1, 0.9] for m in METRICS}, "n_rows": 10, "n_races": 2}
    for name in ("ridge", "baseline")
}


@pytest.fixture
def parent(uri, tmp_path) -> str:
    return register_run(make_parts(), config(), [], split_dir=tmp_path / "splits", tracking_uri=uri, now=NOW)


def experiment_of(uri, parent) -> str:
    return MlflowClient(uri).get_run(parent).info.experiment_id


def log_run(uri, parent, name, tags, params=None, artifacts=None, fail=False) -> str:
    """A root run in the sampling run's experiment, with params and JSON artifacts, no model."""
    mlflow.set_tracking_uri(uri)
    with mlflow.start_run(run_name=name, experiment_id=experiment_of(uri, parent), tags=tags) as run:
        mlflow.log_params(params or {})
        for file, content in (artifacts or {}).items():
            mlflow.log_dict(content, file)
        if fail:
            mlflow.end_run(status="FAILED")
    return run.info.run_id


def register(uri, parent, run_id, algorithm="ridge", alias=True) -> str:
    """Register `run_id` as a version of f1-race-pace by hand (no model is needed to read the card's inputs)."""
    ml = MlflowClient(uri)
    ml.create_registered_model(REGISTERED_MODEL)
    tags = {"algorithm": algorithm, "mlflow_run_id": run_id, "sampling_run_id": parent}
    version = ml.create_model_version(REGISTERED_MODEL, "file:///nowhere", tags=tags)
    if alias:
        ml.set_registered_model_alias(REGISTERED_MODEL, "champion", str(version.version))
    return str(version.version)


def explain_run(uri, parent, version_id=7, fail=False, content=None) -> str:
    tags = {"step": "explain", "sampling_run_id": parent, "model_version_id": str(version_id)}
    return log_run(uri, parent, "explain", tags, artifacts={"explain.json": content or EXPLAIN_JSON}, fail=fail)


def evaluate_run(uri, parent, test_metrics=None) -> str:
    tags = {"step": "evaluate", "sampling_run_id": parent}
    artifacts = {"selection.json": SELECTION, "test_metrics.json": test_metrics or TEST_METRICS}
    return log_run(uri, parent, "evaluate", tags, artifacts=artifacts)


@pytest.fixture
def world(uri, parent):
    """A champion (model version 7, ridge) with everything the card reads."""
    training = log_run(
        uri, parent, "ridge", {"step": "training", "algorithm": "ridge", "sampling_run_id": parent}, PARAMS
    )
    mlflow_version = register(uri, parent, training)
    supabase = FakeSupabase(
        {
            **REFERENCE,
            "model_versions": [
                {"model_version_id": 7, "mlflow_run_id": training, "algorithm": "ridge", **TRAINED},
                {"model_version_id": 3, "mlflow_run_id": "other", "algorithm": "baseline", **TRAINED},
            ],
            "champion_history": [
                {
                    "model_version_id": 3,
                    "promoted_at": "2026-10-01T00:00:00+00:00",
                    "demoted_at": "2026-10-10T00:00:00+00:00",
                },
                {"model_version_id": 7, "promoted_at": "2026-10-10T00:00:00+00:00", "demoted_at": None},
            ],
        }
    )
    return {
        "uri": uri,
        "parent": parent,
        "training": training,
        "mlflow_version": mlflow_version,
        "supabase": supabase,
        "explain": explain_run(uri, parent),
        "evaluate": evaluate_run(uri, parent),
    }


def load(world, **kwargs) -> CardInputs:
    return load_card_inputs(config(), world["supabase"], world["uri"], **kwargs)


def test_every_field_comes_from_the_right_run(world):
    inputs = load(world)
    assert inputs.model_version_id == 7
    assert inputs.mlflow_version == world["mlflow_version"]
    assert inputs.algorithm == "ridge"
    assert inputs.feature_set_version == "v1"
    assert inputs.trained_through == {"year": 2024, "round": 2}
    assert inputs.sampling_run_id == world["parent"]
    assert inputs.explain_run_id == world["explain"]
    assert inputs.evaluate_run_id == world["evaluate"]
    assert inputs.promoted_at == "2026-10-10T00:00:00+00:00"
    assert inputs.explain == EXPLAIN_JSON
    assert inputs.selection == SELECTION
    assert inputs.test_metrics == TEST_METRICS
    assert inputs.cv_folds == 5
    assert inputs.data_range == {"first_year": 2023, "last_year": 2025, "cutoff": None}
    assert inputs.split == {
        "test_fraction": 0.2,
        "n_train_races": 2,
        "n_test_races": 1,
        "first_train_race": {"year": 2024, "round": 1},
        "last_train_race": {"year": 2024, "round": 2},
        "first_test_race": {"year": 2024, "round": 3},
        "last_test_race": {"year": 2024, "round": 3},
    }


def test_hyperparameters_leave_out_the_bookkeeping_and_the_tuning_budget_is_apart(world):
    inputs = load(world)
    assert inputs.hyperparameters == {"alpha": "3.2", "imputation": "median"}
    assert not set(inputs.hyperparameters) & set(BOOKKEEPING_KEYS)
    assert inputs.tuning == {"n_trials": "50", "seed": "42", "n_folds": "5"}


def test_a_training_run_without_hyperparameters_raises_and_names_the_run(uri, parent):
    bare = {k: v for k, v in PARAMS.items() if k in BOOKKEEPING_KEYS}
    training = log_run(
        uri, parent, "ridge", {"step": "training", "algorithm": "ridge", "sampling_run_id": parent}, bare
    )
    register(uri, parent, training)
    explain_run(uri, parent)
    evaluate_run(uri, parent)
    supabase = FakeSupabase(
        {
            **REFERENCE,
            "model_versions": [{"model_version_id": 7, "mlflow_run_id": training, "algorithm": "ridge", **TRAINED}],
            "champion_history": [],
        }
    )
    with pytest.raises(ValueError, match=training):
        load_card_inputs(config(), supabase, uri)


def test_a_champion_with_no_history_row_has_no_promoted_at(world):
    world["supabase"].tables["champion_history"].clear()
    assert load(world).promoted_at is None


def test_no_alias_raises(uri, parent):
    training = log_run(
        uri, parent, "ridge", {"step": "training", "algorithm": "ridge", "sampling_run_id": parent}, PARAMS
    )
    register(uri, parent, training, alias=False)
    with pytest.raises(ValueError, match="no 'champion' alias"):
        load_card_inputs(config(), FakeSupabase({"model_versions": [], "champion_history": []}), uri)


def test_a_champion_with_no_model_versions_row_raises(world):
    world["supabase"].tables["model_versions"].clear()
    with pytest.raises(ValueError, match="model_versions"):
        load(world)


def test_a_missing_feature_set_or_training_race_raises_and_names_it(world):
    world["supabase"].tables["feature_sets"].clear()
    with pytest.raises(ValueError, match="feature set 5"):
        load(world)
    world["supabase"].tables["feature_sets"].append({"feature_set_id": 5, "version": "v1"})
    world["supabase"].tables["races"].clear()
    with pytest.raises(ValueError, match="race 102"):
        load(world)


def test_no_explain_run_for_the_champions_version_raises_and_names_it(world, uri, parent):
    other = explain_run(uri, parent, version_id=99)  # a run of another version is never used
    client = MlflowClient(uri)
    client.delete_run(world["explain"])
    assert client.get_run(other).info.status == "FINISHED"
    with pytest.raises(ValueError, match="model version 7"):
        load(world)


def test_the_latest_finished_explain_run_wins_and_a_failed_one_is_ignored(world, uri, parent):
    newer = explain_run(uri, parent, content={**EXPLAIN_JSON, "base_value": 1.5})
    explain_run(uri, parent, fail=True)  # the latest of all, but FAILED
    inputs = load(world)
    assert inputs.explain_run_id == newer
    assert inputs.explain["base_value"] == 1.5


def test_an_explain_run_of_another_version_is_never_used(world, uri, parent):
    explain_run(uri, parent, version_id=99, content={**EXPLAIN_JSON, "model_version_id": 99})  # the newest run
    inputs = load(world)
    assert inputs.explain_run_id == world["explain"]
    assert inputs.explain["model_version_id"] == 7


def test_no_evaluate_run_raises(world):
    MlflowClient(world["uri"]).delete_run(world["evaluate"])
    with pytest.raises(ValueError, match="no finished evaluate run"):
        load(world)


def test_an_evaluate_run_without_an_artifact_raises_and_names_it(world, uri, parent):
    MlflowClient(uri).delete_run(world["evaluate"])
    log_run(
        uri,
        parent,
        "evaluate",
        {"step": "evaluate", "sampling_run_id": parent},
        artifacts={"selection.json": SELECTION},
    )
    with pytest.raises(ValueError, match="test_metrics.json"):
        load(world)


def test_a_champion_missing_from_test_metrics_raises_and_never_shows_the_candidate(world, uri, parent):
    MlflowClient(uri).delete_run(world["evaluate"])
    evaluate_run(uri, parent, test_metrics={k: v for k, v in TEST_METRICS.items() if k != "ridge"})
    with pytest.raises(ValueError, match="champion 'ridge'"):
        load(world)


def test_the_champion_is_looked_up_by_its_config_name(uri, parent):
    """`model_versions.algorithm` is `rf` but `test_metrics.json` is keyed by the config name `random_forest`."""
    training = log_run(
        uri, parent, "random_forest", {"step": "training", "algorithm": "rf", "sampling_run_id": parent}, PARAMS
    )
    register(uri, parent, training, algorithm="rf")
    explain_run(uri, parent)
    evaluate_run(
        uri, parent, test_metrics={"random_forest": TEST_METRICS["ridge"], "baseline": TEST_METRICS["baseline"]}
    )
    supabase = FakeSupabase(
        {
            **REFERENCE,
            "model_versions": [{"model_version_id": 7, "mlflow_run_id": training, "algorithm": "rf", **TRAINED}],
            "champion_history": [],
        }
    )
    assert load_card_inputs(config(), supabase, uri).algorithm == "rf"


def test_nothing_is_written_to_supabase_or_mlflow(world):
    tables = {name: [dict(r) for r in rows] for name, rows in world["supabase"].tables.items()}
    ml = MlflowClient(world["uri"])
    runs = len(ml.search_runs([e.experiment_id for e in ml.search_experiments()]))
    alias = ml.get_model_version_by_alias(REGISTERED_MODEL, "champion").version
    load(world)
    assert world["supabase"].tables == tables
    assert len(ml.search_runs([e.experiment_id for e in ml.search_experiments()])) == runs
    assert ml.get_model_version_by_alias(REGISTERED_MODEL, "champion").version == alias
