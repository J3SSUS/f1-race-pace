"""Public functions: the Saturday features, and the full row of one race (Saturday features plus the Sunday target)."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from features.catalog import (
    IDENTIFICATION,
    TARGET_COLUMNS,
    Catalog,
    apply_dtypes,
    features_schema,
    load_catalog,
    row_schema,
)
from features.context import context_features
from features.entries import qualifying_entries
from features.form import form_features
from features.practice import fp2_features, sprint_features
from features.qualifying import qualifying_features
from features.race_info import RaceInfo
from features.target import race_target
from ingestion.session import RAW_DIR


@lru_cache
def _catalog() -> Catalog:
    return load_catalog()


def race_features(race: RaceInfo, history: pd.DataFrame, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """The Saturday features of every entry of a race: reads FP2, S (if any) and Q, and never the R session.

    `history` holds the valid rows of earlier races (see `form.HISTORY_COLUMNS`). The result has the identification
    columns and the features in the order of `config/features.yml`, and passes `features_schema`.
    """
    catalog = _catalog()
    entries = qualifying_entries(race, raw_dir)
    rows = pd.DataFrame(
        {
            "year": race.year,
            "round": race.round,
            "circuit": race.circuit,
            "driver": entries["driver"],
            "team": entries["team"],
        }
    )

    qualifying = qualifying_features(entries)
    for column in ("quali_delta_pct", "grid_position", "reached_q3", "teammate_quali_gap"):
        rows[column] = qualifying[column]
    form = form_features(race, entries, history)
    for column in ("driver_pace_last3", "team_pace_last3", "team_pace_this_circuit_prev"):
        rows[column] = form[column]
    for per_driver in (fp2_features(race, raw_dir), sprint_features(race, raw_dir)):
        for column in per_driver.columns:
            rows[column] = entries["driver_number"].map(per_driver[column])
    for name, value in context_features(race, raw_dir).items():
        rows[name] = value

    rows = apply_dtypes(rows[IDENTIFICATION + catalog.names], catalog)
    return features_schema(catalog).validate(rows)


def build_race(race: RaceInfo, history: pd.DataFrame, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """`race_features` joined with `race_target` on the driver: the full row, validated with `row_schema`."""
    catalog = _catalog()
    rows = race_features(race, history, raw_dir).merge(
        race_target(race, raw_dir), on="driver", how="left", validate="1:1"
    )
    return row_schema(catalog).validate(rows[IDENTIFICATION + TARGET_COLUMNS + catalog.names])
