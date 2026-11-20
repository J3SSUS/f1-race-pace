import mlflow
import mlflow.sklearn
import pytest
from fake_supabase import FakeSupabase
from mlflow import MlflowClient
from test_register_version import Flaky, mlflow_versions, parent, supabase  # noqa: F401
from test_tuning import train_set

from training.baseline import qualifying_baseline
from training.registry import REFIT_STEP, Version, register_refit
from training.sampling import SAMPLING_TAG


def refit_run(uri, parent, *, step=REFIT_STEP, algorithm="baseline", sampling=True, status="FINISHED", model=True):  # noqa: F811
    """A run like the one `scripts/refit.py` leaves: tags, and a fitted pipeline logged as `model`."""
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(uri)
    tags = {"step": step, "algorithm": algorithm, "refit_of": "source-run"}
    if sampling:
        tags[SAMPLING_TAG] = parent
    with mlflow.start_run(
        run_name="refit-baseline", experiment_id=client.get_run(parent).info.experiment_id, tags=tags
    ) as run:
        if model:
            data = train_set()
            mlflow.sklearn.log_model(qualifying_baseline().fit(data.X, data.y), name="model")
    if status != "FINISHED":
        client.set_terminated(run.info.run_id, status)
    return run.info.run_id


def test_a_refit_is_registered_through_the_last_race_of_the_split_with_no_metrics(uri, parent, supabase):  # noqa: F811
    run_id = refit_run(uri, parent)
    version = register_refit("baseline", run_id, supabase, uri)
    assert version == Version("baseline", "baseline", run_id, "1", version.model_version_id)
    assert supabase.tables["model_versions"] == [
        {
            "model_version_id": version.model_version_id,
            "mlflow_run_id": run_id,
            "algorithm": "baseline",
            "feature_set_id": 5,
            "trained_through_race_id": 102,  # round 3 is the test race: the last race of the whole split
            "is_shadow": False,
        }
    ]
    assert supabase.tables.get("metrics", []) == []
    assert len(mlflow_versions(uri)) == 1


def test_the_version_has_no_alias_and_the_champion_history_is_untouched(uri, parent, supabase):  # noqa: F811
    register_refit("baseline", refit_run(uri, parent), supabase, uri)
    (version,) = mlflow_versions(uri)
    assert not version.aliases
    assert supabase.tables.get("champion_history", []) == []


def test_registering_twice_changes_nothing(uri, parent, supabase):  # noqa: F811
    run_id = refit_run(uri, parent)
    first = register_refit("baseline", run_id, supabase, uri)
    second = register_refit("baseline", run_id, supabase, uri)
    assert first == second
    assert len(supabase.tables["model_versions"]) == 1
    assert len(mlflow_versions(uri)) == 1


def test_a_supabase_failure_keeps_the_mlflow_version_and_a_rerun_completes_the_row(uri, parent, supabase):  # noqa: F811
    run_id = refit_run(uri, parent)
    flaky = Flaky(supabase, "model_versions")
    with pytest.raises(RuntimeError, match="model_versions"):
        register_refit("baseline", run_id, flaky, uri)
    assert len(mlflow_versions(uri)) == 1
    flaky.broken = False
    register_refit("baseline", run_id, flaky, uri)
    assert len(mlflow_versions(uri)) == 1
    assert len(supabase.tables["model_versions"]) == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"status": "FAILED"}, "FAILED"),
        ({"step": "training"}, "not a refit run"),
        ({"algorithm": "svm"}, "algorithm 'svm'"),
        ({"algorithm": ""}, "algorithm ''"),
        ({"sampling": False}, "no sampling_run_id"),
        ({"model": False}, "no model to register"),
    ],
)
def test_a_bad_run_raises_before_anything_is_written(uri, parent, supabase, kwargs, message):  # noqa: F811
    run_id = refit_run(uri, parent, **kwargs)
    with pytest.raises(ValueError, match=message):
        register_refit("baseline", run_id, supabase, uri)
    assert supabase.tables.get("model_versions", []) == []
    assert mlflow_versions(uri) == []


def test_an_unknown_run_raises(uri, supabase):  # noqa: F811
    with pytest.raises(ValueError, match="No MLflow run"):
        register_refit("baseline", "nope", supabase, uri)


def test_a_missing_feature_set_or_race_stops_before_the_mlflow_version(uri, parent):  # noqa: F811
    run_id = refit_run(uri, parent)
    from features.store import StoreError

    with pytest.raises(StoreError, match="feature set"):
        register_refit("baseline", run_id, FakeSupabase({"feature_sets": [], "races": []}), uri)
    assert mlflow_versions(uri) == []
