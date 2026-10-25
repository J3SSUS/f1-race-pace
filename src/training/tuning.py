"""Optuna tuning of the six models over the temporal CV of SPEC 07."""

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from mlflow import MlflowClient

from common.config import Choice, Fixed, ParamSpec, Range, TrainingConfig, load_config
from training.baseline import QUALIFYING
from training.cv import CvResult, cross_validate
from training.data import TrainSet, load_train
from training.pipelines import ALGORITHMS, build_pipeline
from training.runs import log_cv_metrics, log_train_params
from training.sampling import resolve_tracking_uri

IMPUTATION = "imputation"  # the one hyperparameter every model shares


def suggest(trial: optuna.Trial, specs: dict[str, ParamSpec], imputations: list[str]) -> dict[str, Any]:
    """The params of one trial: fixed values as they are, ranges and choices from Optuna, plus the imputation."""
    params: dict[str, Any] = {}
    for name, spec in specs.items():
        if isinstance(spec, Fixed):
            params[name] = spec.value
        elif isinstance(spec, Range):
            suggest_range = trial.suggest_int if spec.is_int else trial.suggest_float
            params[name] = suggest_range(name, spec.low, spec.high, log=spec.log)
        elif isinstance(spec, Choice):
            params[name] = trial.suggest_categorical(name, list(spec.options))
        else:
            raise TypeError(f"param '{name}': unexpected spec {spec!r}")
    params[IMPUTATION] = trial.suggest_categorical(IMPUTATION, list(imputations))
    return params


@dataclass(frozen=True)
class TuneResult:
    model: str  # config key
    run_id: str  # the model run
    best_trial: int
    best_params: dict[str, Any]  # includes "imputation"
    cv: CvResult  # cross-validation of the best trial
    trials: pd.DataFrame  # number, run_id, mae (cv mean), one column per param


def tune_model(
    model: str,
    specs: dict[str, ParamSpec],
    train: TrainSet,
    parent_run_id: str,
    config: TrainingConfig,
    tracking_uri: str | None = None,
) -> TuneResult:
    """One Optuna study for `model`: a run for the model under `parent_run_id` and a child run per trial.

    The objective is the mean MAE of `cross_validate`. The best trial is refit on all of train and logged in the model
    run with its OOF predictions and the trials table. If a trial raises, that trial and the model run end `FAILED`
    and the error propagates; the parent run is never changed.
    """
    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    experiment_id = MlflowClient(uri).get_run(parent_run_id).info.experiment_id
    algorithm = ALGORITHMS[model]
    n_folds, seed, n_trials = config.cv.folds, config.optuna.seed, config.optuna.trials
    tags = {"step": "training", "algorithm": algorithm}
    with mlflow.start_run(run_name=model, experiment_id=experiment_id, tags=tags, parent_run_id=parent_run_id) as run:
        rows: list[dict[str, Any]] = []
        results: dict[int, CvResult] = {}
        all_params: dict[int, dict[str, Any]] = {}

        def objective(trial: optuna.Trial) -> float:
            params = suggest(trial, specs, config.imputation)
            imputation = params[IMPUTATION]
            estimator_params = {k: v for k, v in params.items() if k != IMPUTATION}
            trial_tags = {"step": "tuning", "algorithm": algorithm, "trial": str(trial.number)}
            with mlflow.start_run(
                run_name=f"{model}-trial-{trial.number}", experiment_id=experiment_id, tags=trial_tags, nested=True
            ) as trial_run:
                pipeline = build_pipeline(model, estimator_params, imputation, seed)
                cv = cross_validate(pipeline, train, n_folds)
                mlflow.log_params(params)
                log_cv_metrics(cv)
            results[trial.number] = cv
            all_params[trial.number] = params
            rows.append({"number": trial.number, "run_id": trial_run.info.run_id, "mae": cv.mean["mae"], **params})
            return cv.mean["mae"]

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials)

        best = study.best_trial.number
        trials = pd.DataFrame(rows).sort_values("number", ignore_index=True)
        best_params = all_params[best]
        best_cv = results[best]
        fitted = build_pipeline(
            model, {k: v for k, v in best_params.items() if k != IMPUTATION}, best_params[IMPUTATION], seed
        ).fit(train.X, train.y)

        log_train_params(train, n_folds, {**best_params, "n_trials": n_trials, "seed": seed})
        log_cv_metrics(best_cv)
        mlflow.log_metric("best_trial", best)
        with tempfile.TemporaryDirectory() as folder:
            for name, frame in (("oof_predictions.parquet", best_cv.oof), ("trials.parquet", trials)):
                path = Path(folder) / name
                frame.to_parquet(path, index=False)
                mlflow.log_artifact(str(path))
        mlflow.sklearn.log_model(fitted, name="model", serialization_format="cloudpickle")
    return TuneResult(model, run.info.run_id, best, best_params, best_cv, trials)


def select_models(config: TrainingConfig, models: list[str] | None = None) -> dict[str, dict[str, ParamSpec]]:
    """The enabled models of the config in config order, or only the ones in `models`.

    A name that is not an enabled model raises `ValueError`.
    """
    enabled = config.enabled_models()
    if models is None:
        return enabled
    unknown = [m for m in models if m not in ALGORITHMS]
    if unknown:
        raise ValueError(f"unknown models {unknown}; allowed: {sorted(ALGORITHMS)}")
    disabled = [m for m in models if m not in enabled]
    if disabled:
        raise ValueError(f"models {disabled} are disabled in the config; enabled: {list(enabled)}")
    return {name: specs for name, specs in enabled.items() if name in models}


def tune_models(
    run_id: str,
    config: TrainingConfig | None = None,
    client=None,
    tracking_uri: str | None = None,
    loader: Callable = load_train,
    models: list[str] | None = None,
    on_model: Callable[[str, str], None] | None = None,
) -> dict[str, TuneResult]:
    """`tune_model` for the enabled models of the config, in config order, under the sampling run `run_id`.

    `models` keeps only the named ones; a name that is not an enabled model raises `ValueError` before anything is
    loaded. Loads the training set once. `on_model(name, model_run_id)` is called as each model run closes.
    """
    config = config or load_config()
    enabled = select_models(config, models)
    train = loader(run_id, client, tracking_uri=tracking_uri)
    results = {}
    for name, specs in enabled.items():
        results[name] = tune_model(name, specs, train, run_id, config, tracking_uri)
        if on_model:
            on_model(name, results[name].run_id)
    return results


def reference_mae(run_id: str, name: str = QUALIFYING, tracking_uri: str | None = None) -> float | None:
    """`cv_mae_mean` of the latest finished run `name` under the sampling run `run_id`, or None if there is none."""
    uri = resolve_tracking_uri(tracking_uri)
    client = MlflowClient(uri)
    runs = client.search_runs(
        [client.get_run(run_id).info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{run_id}' and attributes.run_name = '{name}' "
        "and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0].data.metrics["cv_mae_mean"] if runs else None
