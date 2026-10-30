"""Run the two references (the qualifying baseline and the median floor) against a sampling run.

Reads the features from Supabase (`.env`) and the split from the MLflow run, cross-validates both inside train and
registers each as a run tagged with the sampling run. Prints the run id of each, one line per candidate with the
mean and standard deviation over the folds of every metric, and the gain of the baseline over the floor. Exits with
code 1 and a `STOPPED:` line if it cannot (unknown run, missing feature set, too few races or no valid row).

    uv run python scripts/baseline.py <run_id>
"""

import argparse
import logging
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.store import StoreError  # noqa: E402
from training.baseline import FLOOR, QUALIFYING  # noqa: E402
from training.cv import CvResult  # noqa: E402
from training.metrics import METRICS  # noqa: E402
from training.runs import run_references  # noqa: E402


def describe(name: str, result: CvResult) -> str:
    cells = [f"{m} {result.mean[m]:.3f} ± {result.std[m]:.3f}" for m in METRICS]
    return f"{name}: " + " | ".join(cells)


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable = run_references,
    out: Callable[[str], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="the run_id that scripts/sample.py printed")
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)
    try:
        results = runner(args.run_id, on_run=lambda name, child: out(f"{name} run_id: {child}"))
    except (ValueError, StoreError) as exc:
        out(f"STOPPED: {exc}")
        return 1
    for name in (QUALIFYING, FLOOR):
        out(describe(name, results[name]))
    baseline, floor = results[QUALIFYING].mean["mae"], results[FLOOR].mean["mae"]
    out(f"gain of {QUALIFYING} over {FLOOR}: MAE {floor - baseline:+.3f} ({(floor - baseline) / floor:+.1%})")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())
