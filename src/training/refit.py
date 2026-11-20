"""The `refit` step: train the candidate and the baseline again on every race up to the cutoff (SPEC 23).

It runs after `evaluate` in a retraining. The models were chosen on the train and the test was used once to report;
now the chosen recipe is trained again on **all** the races of the sampling run, train and test, so the newest races
enter the model. The pipelines are `sklearn.base.clone` of the ones already logged (same hyperparameters, no Optuna),
and each refit is a run `refit-<name>` and a version of `f1-race-pace` that nobody promotes: no `champion` alias, no
`champion_history`, and no `metrics` rows, because any number on rows the model trained on would mislead. Its
evaluation is the forward one, on races after the cutoff (SPEC 24).

The test is a gate. The step stops, and writes nothing, if the sampling run has no cutoff, if there is no finished
`evaluate` run, if that run has no `candidate`, or if the candidate's test MAE is not lower than the baseline's. The
test target is read here as training data, after `evaluate` used it once: that is the only exception to the rule that
`evaluation.py` is the only module that reads it.
"""

from collections.abc import Callable
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.base import clone

from common.supabase_io import get_client
from features.store import read_wide
from training.data import NOT_FEATURES, TARGET, TrainSet
from training.evaluation import EVALUATE
from training.registry import REFIT_STEP, Version, register_refit
from training.sampling import SAMPLING_TAG, read_split, resolve_tracking_uri
from training.selection import BASELINE
from training.step import finished_run_id

CANDIDATE_TAG = "candidate"  # tag of the `evaluate` run that names the chosen model
REFIT_OF_TAG = "refit_of"  # tag of a refit run: the run its pipeline was cloned from
NO_CUTOFF = "none"  # the value of the `cutoff` param of a sampling run made with no cutoff


class RefitError(ValueError):
    """The refit cannot run, and nothing was written."""


@dataclass(frozen=True)
class RefitResult:
    name: str  # "ridge", ..., or "baseline"
    version: Version
    rows: int  # valid rows the model was fitted on
    races: int  # distinct races among them
    reused: bool  # True if a finished refit run was already there and nothing was fitted


def load_all_races(
    run_id: str,
    client=None,
    *,
    version: str = "v1",
    tracking_uri: str | None = None,
    reader: Callable = read_wide,
    splitter: Callable = read_split,
) -> TrainSet:
    """Every valid row of every race of the sampling run `run_id`, train and test.

    Same shape and columns as `load_train`, on both sides of the split. The split is read first, so an unknown run
    stops before Supabase is touched. Invalid rows are dropped before the `TrainSet` is built.
    """
    split = splitter(run_id, tracking_uri)
    races = pd.MultiIndex.from_frame(split[["year", "round"]])
    wide = reader(client if client is not None else get_client(), version)

    keep = pd.MultiIndex.from_frame(wide[["year", "round"]]).isin(races) & wide["is_valid_row"].to_numpy()
    rows = wide[keep].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"The races of run '{run_id}' have no valid row in feature set {version}")

    features = [c for c in rows.columns if c not in NOT_FEATURES]
    return TrainSet(
        X=rows[features],
        y=rows[TARGET].astype("float64"),
        meta=rows[["entry_id", "year", "round", "driver"]],
    )


def check_gate(sampling_run_id: str, ml: MlflowClient, uri: str) -> tuple[str, str]:
    """`(candidate name, cutoff)` if the sampling run may be refit, or `RefitError` with the reason.

    The checks run in this order: the sampling run exists and has a cutoff; a finished `evaluate` run exists and
    names a `candidate`; the candidate's `test_mae` is lower than the `baseline_test_mae` of that run. The MAE are
    read from the `evaluate` run: the test target is not read to decide.
    """
    try:
        sampling = ml.get_run(sampling_run_id)
    except MlflowException as exc:
        raise RefitError(f"unknown sampling run {sampling_run_id}") from exc
    cutoff = sampling.data.params.get("cutoff", NO_CUTOFF)
    if cutoff == NO_CUTOFF:
        raise RefitError(
            f"the sampling run {sampling_run_id} has no cutoff: a refit is for a retraining, "
            "a first training is never refit"
        )
    evaluate_run_id = finished_run_id(sampling_run_id, EVALUATE, uri)
    if evaluate_run_id is None:
        raise RefitError(
            f"the sampling run {sampling_run_id} has no finished evaluate run: the test is used once to report "
            "and only after that becomes training data (run scripts/evaluate.py first)"
        )
    evaluate = ml.get_run(evaluate_run_id)
    candidate = evaluate.data.tags.get(CANDIDATE_TAG, "")
    if not candidate:
        raise RefitError(f"the evaluate run {evaluate_run_id} has no candidate: no model beat the others")
    metrics = evaluate.data.metrics
    missing = [m for m in ("test_mae", "baseline_test_mae") if m not in metrics]
    if missing:
        raise RefitError(f"the evaluate run {evaluate_run_id} has no {missing[0]} metric")
    if not metrics["test_mae"] < metrics["baseline_test_mae"]:
        raise RefitError(
            f"the gate: the test MAE of {candidate} ({metrics['test_mae']:.3f}) is not lower than the baseline's "
            f"({metrics['baseline_test_mae']:.3f}) on the test of {sampling_run_id}"
        )
    return candidate, cutoff


def _source_run(sampling_run_id: str, name: str, uri: str) -> str:
    run_id = finished_run_id(sampling_run_id, name, uri)
    if run_id is None:
        raise RefitError(f"no finished run of {name} for the sampling run {sampling_run_id}: run scripts/train.py")
    return run_id


def _finished_refit(ml: MlflowClient, experiment_id: str, sampling_run_id: str, source_run_id: str):
    """The latest finished refit run of `source_run_id` for the sampling run, or None."""
    runs = ml.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.step = '{REFIT_STEP}' and tags.{SAMPLING_TAG} = '{sampling_run_id}' "
            f"and tags.{REFIT_OF_TAG} = '{source_run_id}' and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def run_refit(
    sampling_run_id: str,
    client=None,
    tracking_uri: str | None = None,
    loader: Callable = load_all_races,
) -> list[RefitResult]:
    """Refit the candidate and then the baseline of `sampling_run_id` on every race of its split, and register them.

    Raises `RefitError` before anything is written if the gate stops it. Everything that can fail without writing
    (the gate, the source runs, loading both models and the rows, fitting) runs first. A finished refit run of the
    same sampling run and source is reused: nothing is fitted and only a missing registration is completed, so
    running it again is safe. Never promotes. Returns the candidate first, then the baseline.
    """
    uri = resolve_tracking_uri(tracking_uri)
    ml = MlflowClient(uri)
    candidate, cutoff = check_gate(sampling_run_id, ml, uri)
    names = [candidate, BASELINE]
    sources = {name: _source_run(sampling_run_id, name, uri) for name in names}
    experiment_id = ml.get_run(sampling_run_id).info.experiment_id
    client = client if client is not None else get_client()
    existing = {name: _finished_refit(ml, experiment_id, sampling_run_id, sources[name]) for name in names}

    train = None
    fitted = {}
    if any(run is None for run in existing.values()):
        mlflow.set_tracking_uri(uri)
        train = loader(sampling_run_id, client, tracking_uri=uri)
        for name in names:
            if existing[name] is None:
                recipe = clone(mlflow.sklearn.load_model(f"runs:/{sources[name]}/model"))
                fitted[name] = recipe.fit(train.X, train.y)

    results = []
    for name in names:
        source = ml.get_run(sources[name])
        if existing[name] is not None:
            run_id = existing[name].info.run_id
            rows = int(existing[name].data.params["refit_rows"])
            races = int(existing[name].data.params["refit_races"])
            reused = True
        else:
            rows = len(train.y)
            races = len(train.meta[["year", "round"]].drop_duplicates())
            tags = {
                "step": REFIT_STEP,
                "algorithm": source.data.tags.get("algorithm", ""),
                SAMPLING_TAG: sampling_run_id,
                REFIT_OF_TAG: sources[name],
            }
            with mlflow.start_run(run_name=f"{REFIT_STEP}-{name}", experiment_id=experiment_id, tags=tags) as run:
                mlflow.log_params(
                    {
                        "refit_rows": rows,
                        "refit_races": races,
                        "cutoff": cutoff,
                        "features": ",".join(train.X.columns),
                    }
                )
                mlflow.sklearn.log_model(fitted[name], name="model", serialization_format="cloudpickle")
            run_id = run.info.run_id
            reused = False
        results.append(RefitResult(name, register_refit(name, run_id, client, uri), rows, races, reused))
    return results
