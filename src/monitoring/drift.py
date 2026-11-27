"""Drift report of the champion (SPEC 26): which races a report covers, the MAE change and the retraining rule.

This part is pure: it reads and writes nothing. A report covers the last `DRIFT_WINDOW` races after a start point, and
is due once that many races are waiting. The MAE change is the current MAE against the reference, and the rule holds if
it is at least `MAE_CHANGE_PCT` or `MIN_DRIFTED_FEATURES` features drifted. The thresholds are not training knobs, so
they are constants here and not in `config/training.json` (as in `training.promotion`).
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.catalog import Catalog

DRIFT_WINDOW = 5  # races per report, and races of the window
MAE_CHANGE_PCT = 15.0  # the rule holds if the MAE rose by at least this percentage
MIN_DRIFTED_FEATURES = 3  # or if at least this many features drifted
P_VALUE = 0.05  # a feature drifts if its p-value is below this
EXPERIMENT = "drift"  # MLflow experiment of the Evidently HTML reports
_TOLERANCE = 1e-9  # a change exactly on the 15% boundary must not fail on float rounding
# The test of each dtype of `config/features.yml`, forced (SPEC 26): Evidently's default would use distances without a
# p-value above 1,000 reference rows.
STATTESTS = {"float": "ks", "int": "ks", "flag": "chisquare", "category": "chisquare"}


@dataclass(frozen=True)
class Window:
    """The races waiting for a report: `pending` is how many, `races` the last `DRIFT_WINDOW` (oldest first)."""

    pending: int
    races: list[int]

    @property
    def due(self) -> bool:
        return self.pending >= DRIFT_WINDOW


def start_race(
    base_race_id: int | None, last_report_race_id: int | None, race_order: dict[int, tuple[int, int]]
) -> int | None:
    """The race after which the next report counts: the later (by year and round) of the champion's base race and its
    last report's `through_race_id`. `base_race_id` is `trained_through_race_id` for a champion that is not a refit, and
    the last race of the reference window for a refit one. None if both are None: every race counts."""
    given = [race for race in (base_race_id, last_report_race_id) if race is not None]
    for race in given:
        if race not in race_order:
            raise ValueError(f"race {race} is not in the calendar of races")
    return max(given, key=race_order.__getitem__) if given else None


def select_window(
    eligible: list[int] | set[int], race_order: dict[int, tuple[int, int]], after: int | None = None
) -> Window:
    """The races of `eligible` (valid ground truth and a champion prediction) that come after the race `after`, and
    the last `DRIFT_WINDOW` of them.

    Order is by `(year, round)`, never by race id (ids are not chronological). `after` is a race id, or None for "from
    the start". The window is empty while fewer than `DRIFT_WINDOW` races are waiting. A race that is not in
    `race_order` is an error: dropping it silently would move the window.
    """
    unknown = sorted(set(eligible) - set(race_order))
    if unknown:
        raise ValueError(f"races {unknown} are not in the calendar of races")
    if after is not None and after not in race_order:
        raise ValueError(f"race {after} is not in the calendar of races")
    floor = race_order[after] if after is not None else None
    waiting = sorted((r for r in set(eligible) if floor is None or race_order[r] > floor), key=race_order.__getitem__)
    return Window(pending=len(waiting), races=waiting[-DRIFT_WINDOW:] if len(waiting) >= DRIFT_WINDOW else [])


def mae_change_pct(current_mae: float, reference_mae: float) -> float:
    """`(current / reference - 1) * 100`: positive when the champion got worse than its reference."""
    if reference_mae <= 0:
        raise ValueError(f"the reference MAE is {reference_mae}: the change is undefined")
    return (current_mae / reference_mae - 1) * 100


def rule_met(change_pct: float, n_drifted_features: int) -> bool:
    """The retraining rule of 2026-10-15: MAE up `MAE_CHANGE_PCT`, or `MIN_DRIFTED_FEATURES` drifted features."""
    return change_pct >= MAE_CHANGE_PCT - _TOLERANCE or n_drifted_features >= MIN_DRIFTED_FEATURES


@dataclass(frozen=True)
class FeatureDrift:
    """The drift of one feature. `statistic` is always None (Evidently gives only the p-value with forced tests) and
    `p_value` is None when a side has no value for the feature: it is then not drifted."""

    feature: str
    feature_id: int
    stattest: str  # "ks" or "chisquare"
    statistic: float | None
    p_value: float | None
    is_drifted: bool


@dataclass(frozen=True)
class FeatureDriftResult:
    features: list[FeatureDrift]  # one per feature of the catalog, in catalog order
    html: str | None  # the Evidently report of the features it could test; None if none could be tested

    @property
    def n_drifted(self) -> int:
        return sum(f.is_drifted for f in self.features)


def _prepared(frame: pd.DataFrame, catalog: Catalog) -> pd.DataFrame:
    """The feature columns as Evidently wants them: float64 for numeric features, object (NaN for a missing value)
    for flags and categories. The nullable `Int64` of the catalog would carry `pd.NA`, which its tests do not take."""
    out = {}
    for feature in catalog.features:
        column = frame[feature.name]
        if STATTESTS[feature.dtype] == "ks":
            out[feature.name] = pd.to_numeric(column, errors="coerce").astype("float64")
        else:
            out[feature.name] = column.astype("object").where(column.notna(), np.nan)
    return pd.DataFrame(out)


def _evidently_p_values(
    reference: pd.DataFrame, current: pd.DataFrame, columns: dict[str, str]
) -> tuple[dict[str, float], str]:
    """The one place that calls Evidently: the p-value of each column in `columns` (name -> "ks" or "chisquare") and
    the HTML report. Imported here so the pure code and its tests do not pay for Evidently (or its warnings)."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    names = list(columns)
    definition = DataDefinition(
        numerical_columns=[c for c in names if columns[c] == "ks"],
        categorical_columns=[c for c in names if columns[c] == "chisquare"],
    )
    snapshot = Report([DataDriftPreset(columns=names, num_method="ks", cat_method="chisquare", threshold=P_VALUE)]).run(
        Dataset.from_pandas(current[names], data_definition=definition),
        Dataset.from_pandas(reference[names], data_definition=definition),
    )
    p_values = {}
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if config["type"].endswith("ValueDrift"):
            p_values[config["column"]] = float(metric["value"])
    missing = [c for c in names if c not in p_values]
    if missing:
        raise RuntimeError(f"Evidently returned no drift value for {missing}")
    return p_values, snapshot.get_html_str(as_iframe=False)


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, catalog: Catalog, feature_ids: dict[str, int]
) -> FeatureDriftResult:
    """The drift of every feature of the catalog between the reference rows and the current rows.

    Float and int features use KS, flags and categories chi-squared (`STATTESTS`); a feature drifts when its p-value is
    below `P_VALUE`. A feature with no value on either side is not tested (Evidently raises for the whole report): it
    is stored with no p-value and not drifted. `feature_ids` maps each feature name to its `features.feature_id`.
    """
    missing = [f.name for f in catalog.features if f.name not in feature_ids]
    if missing:
        raise ValueError(f"no feature_id for {missing}")
    for label, frame in (("reference", reference), ("current", current)):
        absent = [f.name for f in catalog.features if f.name not in frame.columns]
        if absent:
            raise ValueError(f"the {label} rows have no column for {absent}")

    reference, current = _prepared(reference, catalog), _prepared(current, catalog)
    testable = {
        f.name: STATTESTS[f.dtype]
        for f in catalog.features
        if reference[f.name].notna().any() and current[f.name].notna().any()
    }
    p_values, html = _evidently_p_values(reference, current, testable) if testable else ({}, None)

    features = []
    for f in catalog.features:
        p = p_values.get(f.name)
        p = p if p is not None and math.isfinite(p) else None
        features.append(
            FeatureDrift(
                feature=f.name,
                feature_id=feature_ids[f.name],
                stattest=STATTESTS[f.dtype],
                statistic=None,
                p_value=p,
                is_drifted=p is not None and p < P_VALUE,
            )
        )
    return FeatureDriftResult(features=features, html=html)
