"""Run the `refit` step: train the candidate and the baseline again on every race up to the cutoff (SPEC 23).

For a retraining, after `scripts/evaluate.py`. The pipelines that `train.py` logged are cloned (same hyperparameters,
no Optuna) and fitted on all the valid rows of the sampling run, train and test, so the newest races enter the model.
Each refit is a run `refit-<name>` and a new version of `f1-race-pace` with a `model_versions` row whose
`trained_through_race_id` is the cutoff race; no `metrics` rows are written and nothing is promoted. Prints one line per
model. Safe to run again: a finished refit of the same sampling run is reused.
Exits with code 1 and a `STOPPED:` line, writing nothing, if it cannot: an unknown sampling run, a sampling run with no
cutoff (a first training is never refit), no finished `evaluate` run, no candidate, a candidate whose test MAE is not
lower than the baseline's, a missing run or a Supabase error.

    uv run python scripts/refit.py <sampling_run_id>
"""

import argparse
import logging
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlflow.exceptions import MlflowException  # noqa: E402

from features.store import StoreError  # noqa: E402
from training.refit import RefitResult, run_refit  # noqa: E402


def describe(result: RefitResult) -> str:
    year, round_ = result.trained_through
    reused = " (reused)" if result.reused else ""
    return (
        f"{result.name}: version {result.version.mlflow_version} (model_version_id {result.version.model_version_id}), "
        f"trained through {year} round {round_}, {result.rows} rows in {result.races} races{reused}"
    )


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable = run_refit,
    out: Callable[[str], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="the sampling run_id of the retraining that scripts/sample.py printed")
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)
    try:
        results = runner(args.run_id)
    except (ValueError, StoreError, RuntimeError, MlflowException) as exc:
        out(f"STOPPED: {exc}")
        return 1
    for result in results:
        out(describe(result))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())
