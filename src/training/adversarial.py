"""Adversarial validation of the train/test split: can a classifier tell a train race from a test race?

It uses the Saturday features only, never the target or the identification columns, and the folds are grouped by
race so a race (one label) is never on both sides of a fold. An AUC near 0.5 means the two parts look alike. The
result is reported and changes nothing: the remedy for a real shift is a new split, not a model fitted to the test.
"""

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from common.config import TrainingConfig
from features.catalog import IDENTIFICATION, TARGET_COLUMNS
from training.sampling import resolve_tracking_uri

# The form features grow with the calendar (they need earlier races), so they separate train from test by design.
FORM_FEATURES = ("driver_pace_last3", "team_pace_last3", "team_pace_this_circuit_prev")
FORBIDDEN = {"entry_id", *IDENTIFICATION, *TARGET_COLUMNS}
CLASSIFIER = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15}


@dataclass(frozen=True)
class AucResult:
    mean: float
    std: float
    folds: list[float]  # AUC of each fold
    features: list[str]  # the columns the classifier received

    def to_dict(self) -> dict:
        return asdict(self)


def feature_variants(features: list[str]) -> dict[str, list[str]]:
    """The two runs: every feature, and every feature but the three form features."""
    return {"all": list(features), "no_form": [f for f in features if f not in FORM_FEATURES]}


def _folds(labels: np.ndarray, groups: np.ndarray, n_folds: int, seed: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    yield from splitter.split(np.zeros(len(labels)), labels, groups)


def _matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """The features as LightGBM reads them: numbers as float (a missing value is NaN), text as category."""
    matrix = frame[features].copy()
    for name in features:
        if matrix[name].dtype == object:
            matrix[name] = matrix[name].astype("category")
        else:
            matrix[name] = matrix[name].astype("float64")
    return matrix


def adversarial_auc(wide: pd.DataFrame, split: pd.DataFrame, features: list[str], config: TrainingConfig) -> AucResult:
    """AUC of a classifier that tells test races (label 1) from train races (label 0), over every entry of the races.

    `wide` is the output of `read_wide`, `split` the output of `read_split`. Only `features` reach the classifier.
    """
    forbidden = FORBIDDEN & set(features)
    if forbidden:
        raise ValueError(f"never use these columns to tell train from test: {sorted(forbidden)}")
    frame = wide.merge(split, on=["year", "round"], how="inner")
    if frame.empty:
        raise ValueError("no entry belongs to a race of the split")
    labels = (frame["split"] == "test").to_numpy().astype(int)
    groups = (frame["year"] * 100 + frame["round"]).to_numpy()
    matrix = _matrix(frame, features)
    seed = config.optuna.seed

    aucs = []
    for train_idx, valid_idx in _folds(labels, groups, config.cv.folds, seed):
        if len(set(labels[valid_idx])) < 2:
            raise ValueError("a validation fold holds one class only: too few races for this many folds")
        model = LGBMClassifier(**CLASSIFIER, random_state=seed, n_jobs=1, deterministic=True, verbose=-1)
        model.fit(matrix.iloc[train_idx], labels[train_idx])
        aucs.append(float(roc_auc_score(labels[valid_idx], model.predict_proba(matrix.iloc[valid_idx])[:, 1])))
    return AucResult(float(np.mean(aucs)), float(np.std(aucs)), aucs, list(features))


def adversarial_validation(
    wide: pd.DataFrame, split: pd.DataFrame, features: list[str], config: TrainingConfig
) -> dict[str, AucResult]:
    """The AUC with every feature (`all`) and without the form features (`no_form`)."""
    return {name: adversarial_auc(wide, split, cols, config) for name, cols in feature_variants(features).items()}


def log_to_run(
    run_id: str,
    results: dict[str, AucResult],
    config: TrainingConfig,
    races: int,
    entries: int,
    tracking_uri: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Write the two mean AUCs as metrics and `adversarial_validation.json` as an artifact of the sampling run.

    Nothing else in the run changes: no param, tag or status. Returns the JSON that was logged.
    """
    report = {
        "run_id": run_id,
        "generated_at": (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "races": races,
        "entries": entries,
        "folds": config.cv.folds,
        "seed": config.optuna.seed,
        "results": {name: result.to_dict() for name, result in results.items()},
    }
    client = MlflowClient(resolve_tracking_uri(tracking_uri))
    client.log_metric(run_id, "adversarial_auc_all", results["all"].mean)
    client.log_metric(run_id, "adversarial_auc_no_form", results["no_form"].mean)
    client.log_dict(run_id, report, "adversarial_validation.json")
    return report
