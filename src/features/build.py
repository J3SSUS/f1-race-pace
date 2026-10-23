"""Public functions: the Saturday features, the Sunday target and the full row of one race."""

from pathlib import Path

import pandas as pd

from features.race_info import RaceInfo
from ingestion.session import RAW_DIR


def race_features(race: RaceInfo, history: pd.DataFrame, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Saturday only: reads FP2, S (if any) and Q, never the R session."""
    raise NotImplementedError


def build_race(race: RaceInfo, history: pd.DataFrame, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """`race_features` joined with `race_target` on the driver: the full row."""
    raise NotImplementedError
