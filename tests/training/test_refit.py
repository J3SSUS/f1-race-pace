import mlflow.sklearn
import numpy as np
import pytest
from mlflow import MlflowClient
from test_data import FLAGS, TEST, TRAIN, split_of, wide_of
from test_manifest import config, make_parts
from test_register_version import NOW, Flaky, mlflow_versions, supabase  # noqa: F401
from test_tuning import train_set, tune

from features.catalog import load_catalog
from training.baseline import qualifying_baseline
from training.refit import RefitError, RefitResult, check_gate, load_all_races, run_refit
from training.registry import REGISTERED_MODEL
from training.runs import run_candidate
from training.sampling import SAMPLING_TAG, register_run


class Loader:
    """The rows a refit fits on (more races than the train the models were tuned on) and how many times it was asked."""

    def __init__(self, n_races: int = 20):
        self.data, self.calls = train_set(n_races=n_races), 0

    def __call__(self, run_id, client, tracking_uri=None):
        self.calls += 1
        return self.data


@pytest.fixture
def parent(uri, tmp_path) -> str:
    """A sampling run made with a cutoff (2024 round 3)."""
    return register_run(
        make_parts(), config(cutoff={"year": 2024, "round": 3}), [], split_dir=tmp_path / "s", tracking_uri=uri, now=NOW
    )


@pytest.fixture
def no_cutoff_parent(uri, tmp_path) -> str:
    return register_run(make_parts(), config(), [], split_dir=tmp_path / "n", tracking_uri=uri, now=NOW)


def make_evaluate(uri, parent, *, candidate="ridge", test_mae=0.4, baseline_mae=0.5, status="FINISHED", metrics=True):
    client = MlflowClient(uri)
    tags = {"step": "evaluate", SAMPLING_TAG: parent}
    if candidate:
        tags["candidate"] = candidate
    run = client.create_run(client.get_run(parent).info.experiment_id, tags=tags, run_name="evaluate")
    if metrics:
        client.log_metric(run.info.run_id, "test_mae", test_mae)
        client.log_metric(run.info.run_id, "baseline_test_mae", baseline_mae)
    client.set_terminated(run.info.run_id, status)
    return run.info.run_id


@pytest.fixture
def trained(uri, parent):
    """The runs a training leaves: the tuned ridge and the baseline, tagged with the sampling run."""
    return {
        "ridge": tune(uri, parent, "ridge", trials=2).run_id,
        "baseline": run_candidate("baseline", "baseline", qualifying_baseline(), train_set(), parent, 5, uri),
    }


@pytest.fixture
def ready(uri, parent, trained):
    make_evaluate(uri, parent)
    return parent


def all_runs(uri):
    client = MlflowClient(uri)
    return client.search_runs([e.experiment_id for e in client.search_experiments()])


def nothing_was_written(uri, supabase, runs_before):  # noqa: F811
    assert len(all_runs(uri)) == runs_before
    assert mlflow_versions(uri) == []
    assert supabase.tables.get("model_versions", []) == []


# --- the refit ---


def test_the_candidate_and_the_baseline_are_refit_and_registered(uri, ready, supabase):  # noqa: F811
    loader = Loader()
    results = run_refit(ready, supabase, uri, loader)
    assert [r.name for r in results] == ["ridge", "baseline"]
    assert all(isinstance(r, RefitResult) and not r.reused for r in results)
    assert loader.calls == 1
    rows = supabase.tables["model_versions"]
    assert [(r["algorithm"], r["trained_through_race_id"], r["is_shadow"]) for r in rows] == [
        ("ridge", 102, False),  # 2024 round 3, the cutoff race and the last of the split
        ("baseline", 102, False),
    ]
    assert [r.version.model_version_id for r in results] == [r["model_version_id"] for r in rows]
    assert len(mlflow_versions(uri)) == 2
    assert supabase.tables.get("metrics", []) == []


def test_each_refit_run_carries_its_tags_and_params(uri, ready, trained, supabase):  # noqa: F811
    results = run_refit(ready, supabase, uri, Loader())
    ml = MlflowClient(uri)
    for result in results:
        run = ml.get_run(result.version.mlflow_run_id)
        assert run.info.run_name == f"refit-{result.name}"
        assert run.data.tags["step"] == "refit"
        assert run.data.tags["algorithm"] == result.version.algorithm
        assert run.data.tags[SAMPLING_TAG] == ready
        assert run.data.tags["refit_of"] == trained[result.name]
        assert run.data.params["refit_rows"] == str(result.rows) == "100"
        assert run.data.params["refit_races"] == str(result.races) == "20"
        assert run.data.params["cutoff"] == "2024-3"
        assert run.data.params["features"] == ",".join(load_catalog().names)


def test_the_refit_pipeline_keeps_the_hyperparameters_but_is_fitted_again(uri, ready, trained, supabase):  # noqa: F811
    results = run_refit(ready, supabase, uri, Loader())
    source = mlflow.sklearn.load_model(f"runs:/{trained['ridge']}/model")
    refit = mlflow.sklearn.load_model(f"runs:/{results[0].version.mlflow_run_id}/model")
    assert refit[-1].get_params() == source[-1].get_params()
    assert refit[0].get_params().keys() == source[0].get_params().keys()
    assert not np.allclose(refit[-1].coef_, source[-1].coef_)  # fitted on other rows


def test_the_refit_never_promotes(uri, ready, supabase):  # noqa: F811
    run_refit(ready, supabase, uri, Loader())
    assert all(not v.aliases for v in mlflow_versions(uri))
    assert supabase.tables.get("champion_history", []) == []


def test_the_sources_are_not_changed(uri, ready, trained, supabase):  # noqa: F811
    ml = MlflowClient(uri)
    before = {name: ml.get_run(run_id).data for name, run_id in trained.items()}
    run_refit(ready, supabase, uri, Loader())
    for name, run_id in trained.items():
        after = ml.get_run(run_id).data
        assert (after.params, after.tags, after.metrics) == (
            before[name].params,
            before[name].tags,
            before[name].metrics,
        )


# --- running again ---


def test_running_again_reuses_the_runs_and_changes_no_count(uri, ready, supabase):  # noqa: F811
    first = run_refit(ready, supabase, uri, Loader())
    runs = len(all_runs(uri))
    loader = Loader()
    second = run_refit(ready, supabase, uri, loader)
    assert loader.calls == 0  # nothing was loaded or fitted
    assert [r.reused for r in second] == [True, True]
    assert [r.version for r in second] == [r.version for r in first]
    assert [(r.rows, r.races) for r in second] == [(100, 20), (100, 20)]
    assert len(all_runs(uri)) == runs
    assert len(mlflow_versions(uri)) == 2
    assert len(supabase.tables["model_versions"]) == 2


def test_a_failure_in_the_second_registration_is_completed_by_a_rerun(uri, ready, supabase):  # noqa: F811
    flaky = Flaky(supabase, "model_versions")
    with pytest.raises(RuntimeError, match="model_versions"):
        run_refit(ready, flaky, uri, Loader())
    flaky.broken = False
    results = run_refit(ready, flaky, uri, Loader())
    assert [r.name for r in results] == ["ridge", "baseline"]
    assert len(mlflow_versions(uri)) == 2
    assert len(supabase.tables["model_versions"]) == 2
    refits = [r for r in all_runs(uri) if r.data.tags.get("step") == "refit"]
    assert len(refits) == 2  # the first attempt's run was reused, not repeated


# --- the gate: nothing is written ---


def stops(uri, supabase, sampling_run_id, message, loader=None):  # noqa: F811
    runs_before = len(all_runs(uri))
    loader = loader or Loader()
    with pytest.raises(RefitError, match=message):
        run_refit(sampling_run_id, supabase, uri, loader)
    assert loader.calls == 0
    nothing_was_written(uri, supabase, runs_before)


def test_a_sampling_run_with_no_cutoff_is_never_refit(uri, no_cutoff_parent, supabase):  # noqa: F811
    make_evaluate(uri, no_cutoff_parent)
    stops(uri, supabase, no_cutoff_parent, "has no cutoff")


def test_an_unknown_sampling_run_stops(uri, supabase):  # noqa: F811
    stops(uri, supabase, "nope", "unknown sampling run")


def test_no_evaluate_run_stops(uri, parent, trained, supabase):  # noqa: F811
    stops(uri, supabase, parent, "no finished evaluate run")


def test_a_failed_evaluate_run_does_not_count(uri, parent, trained, supabase):  # noqa: F811
    make_evaluate(uri, parent, status="FAILED")
    stops(uri, supabase, parent, "no finished evaluate run")


def test_an_evaluate_run_with_no_candidate_stops(uri, parent, trained, supabase):  # noqa: F811
    make_evaluate(uri, parent, candidate="")
    stops(uri, supabase, parent, "has no candidate")


def test_an_evaluate_run_with_no_test_mae_stops(uri, parent, trained, supabase):  # noqa: F811
    make_evaluate(uri, parent, metrics=False)
    stops(uri, supabase, parent, "no test_mae metric")


@pytest.mark.parametrize("test_mae, baseline_mae", [(0.5, 0.5), (0.6, 0.5)])
def test_a_candidate_that_does_not_beat_the_baseline_on_the_test_stops(
    uri,
    parent,
    trained,
    supabase,  # noqa: F811
    test_mae,
    baseline_mae,
):
    make_evaluate(uri, parent, test_mae=test_mae, baseline_mae=baseline_mae)
    stops(uri, supabase, parent, rf"the gate: the test MAE of ridge \({test_mae:.3f}\) is not lower")


def test_a_candidate_without_a_finished_run_stops(uri, parent, trained, supabase):  # noqa: F811
    make_evaluate(uri, parent, candidate="random_forest")
    stops(uri, supabase, parent, "no finished run of random_forest")


def test_a_baseline_without_a_finished_run_stops(uri, parent, supabase):  # noqa: F811
    tune(uri, parent, "ridge", trials=2)
    make_evaluate(uri, parent)
    stops(uri, supabase, parent, "no finished run of baseline")


def test_check_gate_returns_the_candidate_and_the_cutoff(uri, ready):
    assert check_gate(ready, MlflowClient(uri), uri) == ("ridge", "2024-3")


# --- the rows ---


def load_all(flags=FLAGS, split=None):
    wide = wide_of(flags, test_target=7.0)
    return load_all_races(
        "run",
        client=object(),
        reader=lambda client, version: wide,
        splitter=lambda run_id, uri: split if split is not None else split_of(),
    )


def test_the_rows_are_the_valid_ones_of_the_train_and_the_test_races():
    rows = load_all()
    assert set(zip(rows.meta["year"], rows.meta["round"], strict=True)) == set(TRAIN) | set(TEST)
    assert sorted(rows.meta["driver"]) == ["HAM", "HAM", "HAM", "VER", "VER", "VER"]  # ALO is invalid
    assert len(rows.X) == len(rows.y) == len(rows.meta) == 6
    assert 7.0 in rows.y.tolist()  # the test target is training data now


def test_the_features_are_the_catalog_in_order():
    assert list(load_all().X.columns) == load_catalog().names


def test_a_race_outside_the_split_is_left_out():
    flags = FLAGS | {(2025, 1, "HAM"): True}
    assert (2025, 1) not in set(zip(load_all(flags).meta["year"], load_all(flags).meta["round"], strict=True))


def test_an_unknown_run_fails_before_supabase_is_touched():
    def splitter(run_id, uri):
        raise ValueError("unknown run")

    def reader(client, version):
        raise AssertionError("Supabase was touched")

    with pytest.raises(ValueError, match="unknown run"):
        load_all_races("nope", client=object(), reader=reader, splitter=splitter)


def test_no_valid_rows_raises():
    flags = {key: False for key in FLAGS}
    with pytest.raises(ValueError, match="no valid row"):
        load_all(flags)


def test_the_registered_model_name_is_the_one_of_training(uri, ready, supabase):  # noqa: F811
    run_refit(ready, supabase, uri, Loader())
    assert {v.name for v in mlflow_versions(uri)} == {REGISTERED_MODEL}
