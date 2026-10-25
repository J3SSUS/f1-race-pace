"""The `training` step: tune the enabled models of `config/training.json` and register each as a model version.

Reads the features from Supabase (`.env`) and the split from the MLflow run of `scripts/sample.py`. Runs the qualifying
baseline and the floor if the sampling run has no `baseline` run yet, then one Optuna study per model, and registers the
baseline and each model as a version of `f1-race-pace` in MLflow and as a row of `model_versions` and `metrics` in
Supabase. Prints the run id and the version of each as it is registered, then one line per model with its
cross-validated MAE (mean and standard deviation over the folds) and the gain over the baseline. It never promotes: no
`champion` alias.
Exits with code 1 and a `STOPPED:` line if it cannot (unknown run, missing feature set, too few races, an unknown model
name or a Supabase error).

    uv run python scripts/train.py <run_id> [--models ridge,xgboost]
"""

import argparse
import logging
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlflow import MlflowClient  # noqa: E402

from features.store import StoreError  # noqa: E402
from training.baseline import QUALIFYING  # noqa: E402
from training.registry import Version  # noqa: E402
from training.sampling import resolve_tracking_uri  # noqa: E402
from training.step import run_training  # noqa: E402


def run_metrics(mlflow_run_id: str) -> dict[str, float]:
    """The metrics of a run in MLflow."""
    return MlflowClient(resolve_tracking_uri()).get_run(mlflow_run_id).data.metrics


def describe(name: str, metrics: dict[str, float], reference: float | None) -> str:
    mean = metrics["cv_mae_mean"]
    line = f"{name}: mae {mean:.3f} ± {metrics['cv_mae_std']:.3f}"
    if reference is None or name == QUALIFYING:
        return line
    gain = reference - mean
    return f"{line} | gain over {QUALIFYING}: MAE {gain:+.3f} ({gain / reference:+.1%})"


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable = run_training,
    metrics: Callable[[str], dict[str, float]] = run_metrics,
    out: Callable[[str], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="the run_id that scripts/sample.py printed")
    parser.add_argument("--models", help="comma-separated config keys to run (default: every enabled model)")
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None

    def registered(name: str, version: Version) -> None:
        out(f"{name} run_id: {version.mlflow_run_id} version: {version.mlflow_version}")

    try:
        versions = runner(args.run_id, models=models, on_version=registered)
        results = {name: metrics(version.mlflow_run_id) for name, version in versions.items()}
    except (ValueError, StoreError, RuntimeError) as exc:
        out(f"STOPPED: {exc}")
        return 1
    reference = results[QUALIFYING]["cv_mae_mean"] if QUALIFYING in results else None
    for name, values in results.items():
        out(describe(name, values, reference))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())
