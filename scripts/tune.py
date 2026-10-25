"""Tune the enabled models of `config/training.json` with Optuna against a sampling run.

Reads the features from Supabase (`.env`) and the split from the MLflow run, runs one Optuna study per model over the
temporal CV inside train and registers each model as a child run of the sampling run, with one child run per trial.
Prints the run id of each model, then one line per model with its best cross-validated MAE (mean and standard
deviation over the folds) and the gain over the `baseline` run of the same sampling run, read from MLflow. Without a
`baseline` run it prints a note and no gain. Exits with code 1 and a `STOPPED:` line if it cannot (unknown run, missing
feature set, too few races, no valid row or an unknown model name).

    uv run python scripts/tune.py <run_id> [--models ridge,xgboost]
"""

import argparse
import logging
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.store import StoreError  # noqa: E402
from training.baseline import QUALIFYING  # noqa: E402
from training.tuning import TuneResult, reference_mae, tune_models  # noqa: E402


def describe(name: str, result: TuneResult, reference: float | None) -> str:
    line = f"{name}: mae {result.cv.mean['mae']:.3f} ± {result.cv.std['mae']:.3f} (trial {result.best_trial})"
    if reference is None:
        return line
    gain = reference - result.cv.mean["mae"]
    return f"{line} | gain over {QUALIFYING}: MAE {gain:+.3f} ({gain / reference:+.1%})"


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable = tune_models,
    baseline: Callable[[str], float | None] = reference_mae,
    out: Callable[[str], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="the run_id that scripts/sample.py printed")
    parser.add_argument("--models", help="comma-separated config keys to run (default: every enabled model)")
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    try:
        results = runner(args.run_id, models=models, on_model=lambda name, child: out(f"{name} run_id: {child}"))
        reference = baseline(args.run_id)
    except (ValueError, StoreError) as exc:
        out(f"STOPPED: {exc}")
        return 1
    if reference is None:
        out(f"note: no '{QUALIFYING}' run under {args.run_id}; run scripts/baseline.py to see the gain")
    for name, result in results.items():
        out(describe(name, result, reference))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())
