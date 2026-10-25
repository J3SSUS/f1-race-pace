import optuna
import pytest

from common.config import Choice, Fixed, Range, load_config
from training.tuning import IMPUTATION, suggest

optuna.logging.set_verbosity(optuna.logging.WARNING)

SPECS = {
    "fixed": Fixed("l1"),
    "n": Range(2, 9),
    "depth_log": Range(1, 64, True),
    "alpha": Range(0.01, 100, True),
    "ratio": Range(0.1, 0.9),
    "kind": Choice(("a", "b", "c")),
}


def collect(n_trials: int = 30, seed: int = 0, specs=SPECS) -> tuple[list[dict], optuna.Study]:
    seen = []

    def objective(trial):
        seen.append(suggest(trial, specs, ["median", "mean"]))
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return seen, study


def test_fixed_value_makes_no_suggestion():
    seen, study = collect()
    assert all(p["fixed"] == "l1" for p in seen)
    assert "fixed" not in study.trials[0].params


def test_integer_range_gives_integers_and_float_range_gives_floats():
    seen, _ = collect()
    assert all(type(p["n"]) is int and type(p["depth_log"]) is int for p in seen)
    assert all(type(p["alpha"]) is float and type(p["ratio"]) is float for p in seen)


def test_every_value_is_inside_its_range():
    seen, _ = collect()
    for p in seen:
        assert 2 <= p["n"] <= 9
        assert 1 <= p["depth_log"] <= 64
        assert 0.01 <= p["alpha"] <= 100
        assert 0.1 <= p["ratio"] <= 0.9
        assert p["kind"] in {"a", "b", "c"}


def test_log_scale_is_honoured():
    _, study = collect()
    dists = study.trials[0].distributions
    assert dists["alpha"].log and dists["depth_log"].log
    assert not dists["ratio"].log and not dists["n"].log


def test_imputation_is_always_suggested():
    seen, study = collect()
    assert {p[IMPUTATION] for p in seen} <= {"median", "mean"}
    assert all(IMPUTATION in t.params for t in study.trials)
    seen_only_fixed, _ = collect(specs={"fixed": Fixed(1)})
    assert all(IMPUTATION in p for p in seen_only_fixed)


def test_same_seed_same_sequence():
    a, _ = collect(seed=3)
    b, _ = collect(seed=3)
    c, _ = collect(seed=4)
    assert a == b
    assert a != c


@pytest.mark.parametrize("model", list(load_config().enabled_models()))
def test_repo_config_spaces_are_valid(model):
    specs = load_config().enabled_models()[model]
    seen, _ = collect(n_trials=5, specs=specs)
    assert all(set(specs) | {IMPUTATION} == set(p) for p in seen)


# ---- tune_model ----------------------------------------------------------------------------------------------------

from datetime import UTC, datetime  # noqa: E402

import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from test_manifest import config, make_parts  # noqa: E402
from test_pipelines import table  # noqa: E402

import training.tuning as tuning  # noqa: E402
from training.data import TrainSet  # noqa: E402
from training.metrics import METRICS  # noqa: E402
from training.sampling import EXPERIMENT, register_run  # noqa: E402
from training.tuning import tune_model  # noqa: E402

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


def train_set(n_races: int = 12, per_race: int = 5) -> TrainSet:
    X, y = table(n_races * per_race)
    meta = pd.DataFrame(
        {
            "entry_id": range(len(X)),
            "year": 2024,
            "round": [i // per_race + 1 for i in range(len(X))],
            "driver": [f"D{i % per_race}" for i in range(len(X))],
        }
    )
    return TrainSet(X=X, y=y, meta=meta)


def cfg(trials: int = 3):
    base = load_config()
    return base.model_copy(update={"optuna": base.optuna.model_copy(update={"trials": trials})})


@pytest.fixture
def parent(uri, tmp_path) -> str:
    return register_run(make_parts(), config(), [], split_dir=tmp_path / "splits", tracking_uri=uri, now=NOW)


def snapshot(client: MlflowClient, run_id: str) -> dict:
    run = client.get_run(run_id)
    return {
        "status": run.info.status,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
        "datasets": [(i.dataset.name, i.dataset.digest) for i in (run.inputs.dataset_inputs or [])],
        "artifacts": sorted(a.path for a in client.list_artifacts(run_id)),
    }


def tune(uri, parent, model="ridge", trials=3, train=None):
    specs = load_config().enabled_models()[model]
    return tune_model(model, specs, train or train_set(), parent, cfg(trials), uri)


def trial_runs(client: MlflowClient, result):
    return client.search_runs(
        [client.get_run(result.run_id).info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{result.run_id}'"
    )


def test_model_run_is_named_tagged_and_linked_to_the_sampling_run(uri, parent):
    result = tune(uri, parent)
    client = MlflowClient(uri)
    run = client.get_run(result.run_id)
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "ridge"
    assert run.data.tags["step"] == "training"
    assert run.data.tags["algorithm"] == "ridge"
    assert run.data.tags["mlflow.parentRunId"] == parent
    assert client.get_experiment(run.info.experiment_id).name == EXPERIMENT


def test_algorithm_tag_uses_the_schema_name(uri, parent):
    result = tune(uri, parent, "random_forest", trials=2)
    assert MlflowClient(uri).get_run(result.run_id).data.tags["algorithm"] == "rf"


def test_one_trial_run_per_trial_with_tags_params_and_five_steps(uri, parent):
    result = tune(uri, parent, trials=3)
    client = MlflowClient(uri)
    runs = sorted(trial_runs(client, result), key=lambda r: int(r.data.tags["trial"]))
    assert [r.info.run_name for r in runs] == [f"ridge-trial-{i}" for i in range(3)]
    for r in runs:
        assert r.info.status == "FINISHED"
        assert r.data.tags["step"] == "tuning" and r.data.tags["algorithm"] == "ridge"
        assert r.data.tags["mlflow.parentRunId"] == result.run_id
        assert set(r.data.params) == {"alpha", "imputation"}
        assert not client.list_artifacts(r.info.run_id)
        for m in METRICS:
            assert f"cv_{m}_mean" in r.data.metrics and f"cv_{m}_std" in r.data.metrics
            history = client.get_metric_history(r.info.run_id, f"fold_{m}")
            assert sorted(h.step for h in history) == [1, 2, 3, 4, 5]


def test_model_run_params_metrics_and_artifacts(uri, parent, tmp_path):
    result = tune(uri, parent, trials=3)
    client = MlflowClient(uri)
    run = client.get_run(result.run_id)
    params = run.data.params
    assert params["n_trials"] == "3" and params["seed"] == "42" and params["n_folds"] == "5"
    assert params["imputation"] == result.best_params["imputation"]
    assert float(params["alpha"]) == pytest.approx(result.best_params["alpha"])
    for m in METRICS:
        assert run.data.metrics[f"cv_{m}_mean"] == pytest.approx(result.cv.mean[m])
        assert run.data.metrics[f"cv_{m}_std"] == pytest.approx(result.cv.std[m])
    assert run.data.metrics["best_trial"] == result.best_trial
    assert {a.path for a in client.list_artifacts(result.run_id)} >= {"oof_predictions.parquet", "trials.parquet"}
    # as in SPEC 07, `model/` is a logged model of the run, loaded through `runs:/<id>/model`
    assert mlflow.sklearn.load_model(f"runs:/{result.run_id}/model") is not None
    assert not client.search_logged_models([run.info.experiment_id], filter_string=f"source_run_id = '{parent}'")
    logged = pd.read_parquet(client.download_artifacts(result.run_id, "trials.parquet", str(tmp_path)))
    assert list(logged["number"]) == [0, 1, 2]
    assert {"run_id", "mae", "alpha", "imputation"} <= set(logged.columns)


def test_best_trial_has_the_lowest_cv_mae(uri, parent):
    result = tune(uri, parent, trials=4)
    assert result.best_trial == int(result.trials.loc[result.trials["mae"].idxmin(), "number"])
    assert result.cv.mean["mae"] == pytest.approx(result.trials["mae"].min())
    client = MlflowClient(uri)
    maes = {int(r.data.tags["trial"]): r.data.metrics["cv_mae_mean"] for r in trial_runs(client, result)}
    assert min(maes, key=maes.get) == result.best_trial


def test_refit_model_predicts_like_the_logged_one(uri, parent):
    train = train_set()
    result = tune(uri, parent, "random_forest", trials=2, train=train)
    logged = mlflow.sklearn.load_model(f"runs:/{result.run_id}/model")
    refit = tuning.build_pipeline(
        "random_forest",
        {k: v for k, v in result.best_params.items() if k != "imputation"},
        result.best_params["imputation"],
        42,
    ).fit(train.X, train.y)
    assert logged.predict(train.X) == pytest.approx(refit.predict(train.X))


def test_same_seed_repeats_the_trials_and_the_best_params(uri, parent):
    a = tune(uri, parent, trials=3)
    b = tune(uri, parent, trials=3)
    assert a.best_params == b.best_params
    assert a.best_trial == b.best_trial
    cols = [c for c in a.trials.columns if c != "run_id"]
    pd.testing.assert_frame_equal(a.trials[cols], b.trials[cols])


def test_failing_trial_fails_its_run_and_the_model_run_and_leaves_the_sampling_run(uri, parent, monkeypatch):
    client = MlflowClient(uri)
    before = snapshot(client, parent)
    real = tuning.build_pipeline
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(tuning, "build_pipeline", flaky)
    with pytest.raises(RuntimeError, match="boom"):
        tune(uri, parent, trials=3)
    experiment = client.get_run(parent).info.experiment_id
    runs = {r.info.run_name: r for r in client.search_runs([experiment]) if r.info.run_id != parent}
    assert runs["ridge"].info.status == "FAILED"
    assert runs["ridge-trial-1"].info.status == "FAILED"
    assert runs["ridge-trial-0"].info.status == "FINISHED"
    assert snapshot(client, parent) == before
