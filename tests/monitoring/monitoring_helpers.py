"""The synthetic world of the tests of `monitoring`: Supabase tables, a fake MLflow and a wide matrix."""

import pandas as pd
from fake_supabase import FakeSupabase
from mlflow.exceptions import MlflowException

from features.catalog import apply_dtypes, load_catalog

CATALOG = load_catalog()
RACES = {  # (year, round) -> race_id, scrambled on purpose
    (2025, 22): 7,
    (2025, 23): 3,
    (2025, 24): 9,
    (2026, 1): 50,
    (2026, 2): 12,
    (2026, 3): 44,
    (2026, 4): 5,
    (2026, 5): 31,
    (2026, 6): 20,
    (2026, 7): 28,
    (2026, 8): 15,
}
DRIVERS = {"HAM": 1, "VER": 2, "ALO": 3}
SPLIT = pd.DataFrame([(2025, 22, "train"), (2025, 23, "train"), (2025, 24, "test")], columns=["year", "round", "split"])
CHAMPION_ID, SAMPLING = 4, "sampling-run"
PROMOTED_AT = "2026-04-15T12:00:00+00:00"


class FakeMlflow:
    """The `get_run(run_id).data.tags` of `MlflowClient`; an unknown run raises like MLflow does."""

    def __init__(self, runs: dict[str, dict]):
        self.runs = runs

    def get_run(self, run_id: str):
        if run_id not in self.runs:
            raise MlflowException(f"no run {run_id}")
        tags = self.runs[run_id]

        class Run:
            class data:  # noqa: N801
                pass

        Run.data.tags = tags
        return Run


def entry(race_id: int, driver: str) -> int:
    return race_id * 10 + DRIVERS[driver]


def wide() -> pd.DataFrame:
    """Shaped like `read_wide`: every race, three drivers, ALO invalid. A feature's value is its entry id, so a row can
    be told from another."""
    rows = []
    for (year, round_), race_id in RACES.items():
        for driver in DRIVERS:
            row = {
                "entry_id": entry(race_id, driver),
                "year": year,
                "round": round_,
                "circuit": "Monza",
                "driver": driver,
                "team": "Ferrari",
                "race_pace_delta_pct": 0.5,
                "clean_laps": 20,
                "is_valid_row": driver != "ALO",
            }
            for feature in CATALOG.features:
                row[feature.name] = feature.allowed[0] if feature.dtype == "category" else float(entry(race_id, driver))
            rows.append(row)
    return apply_dtypes(pd.DataFrame(rows), CATALOG).astype({"is_valid_row": "bool", "clean_laps": "int64"})


def tables(*, refit: bool = False, ingested: dict[int, str] | None = None, reports=(), test_mae=0.43) -> dict:
    """The Supabase tables of the world. `ingested` is `round of 2026 -> ingested_at` of its ground truth."""
    ingested = ingested or {}
    predictions, truth = [], []
    for (year, round_), race_id in RACES.items():
        if year != 2026:
            continue
        for driver in DRIVERS:
            predictions.append(
                {
                    "year": year,
                    "round": round_,
                    "driver": driver,
                    "model_version_id": CHAMPION_ID,
                    "abs_error": None if (round_, driver) == (1, "VER") else (0.1 if driver == "HAM" else 0.3) * round_,
                    "is_valid_row": driver != "ALO",
                }
            )
            truth.append(
                {"entry_id": entry(race_id, driver), "ingested_at": ingested.get(round_, "2026-01-01T00:00:00+00:00")}
            )
    metrics = [
        {"model_version_id": CHAMPION_ID, "split": "test", "metric_name": "mae", "value": test_mae},
        {"model_version_id": CHAMPION_ID, "split": "train", "metric_name": "mae", "value": 0.9},
        {"model_version_id": CHAMPION_ID, "split": "walk_forward", "metric_name": "mae", "value": 0.7},
    ]
    return {
        "races": [{"race_id": i, "year": y, "round": r} for (y, r), i in RACES.items()],
        "race_entries": [
            {"entry_id": entry(i, d), "race_id": i, "driver_id": n} for i in RACES.values() for d, n in DRIVERS.items()
        ],
        "ground_truth": truth,
        "prediction_results": predictions
        + [  # another version's row: not the champion's
            {"year": 2026, "round": 1, "driver": "HAM", "model_version_id": 99, "abs_error": 9.0, "is_valid_row": True},
        ],
        "champion_history": [
            {
                "champion_id": 1,
                "model_version_id": 1,
                "promoted_at": "2026-01-01T00:00:00+00:00",
                "demoted_at": PROMOTED_AT,
            },
            {"champion_id": 2, "model_version_id": CHAMPION_ID, "promoted_at": PROMOTED_AT, "demoted_at": None},
        ],
        "model_versions": [
            {
                "model_version_id": CHAMPION_ID,
                "mlflow_run_id": "run-champion",
                "feature_set_id": 1,
                "trained_through_race_id": RACES[(2026, 2)] if refit else RACES[(2025, 23)],
            }
        ],
        "feature_sets": [{"feature_set_id": 1, "version": CATALOG.version}],
        "metrics": metrics,
        "drift_reports": [dict(r) for r in reports],
    }


def mlflow_for(refit: bool = False) -> FakeMlflow:
    tags = {"sampling_run_id": SAMPLING} | ({"step": "refit"} if refit else {})
    return FakeMlflow({"run-champion": tags})


def splitter(run_id, uri):
    assert run_id == SAMPLING
    return SPLIT


def client(**kwargs) -> FakeSupabase:
    return FakeSupabase(tables(**kwargs))


# 2026 R1..R6 had their ground truth in before the promotion (2026-04-15), R7 and R8 after it.
INGESTED = {n: f"2026-03-{n:02d}T10:00:00+00:00" for n in range(1, 7)} | {
    7: "2026-04-20T10:00:00.123456+00:00",
    8: "2026-04-27T10:00:00+00:00",
}
