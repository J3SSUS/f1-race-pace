"""Promote a model version to `champion` by hand: the first champion of `f1-race-pace`.

The writes live in `src/training/promotion.py`. This script sets the `champion` alias on that version in MLflow, closes
the current row of `champion_history` (`demoted_at`) and inserts the new one with `promoted_by = manual` and the test
MAE improvement over the baseline of the same sampling run, read from `metrics`. It refuses a version with no
`split = 'test'` metrics, so a champion has always gone through `scripts/evaluate.py`. A non-positive improvement
prints a warning and still promotes: the decision is the user's. Promoting the version that is already champion
changes nothing. Exits with code 1 and a `STOPPED:` line if it cannot.

    uv run python scripts/promote.py <model_version_id>
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
from training.promotion import ALIAS, promote, version_test_mae  # noqa: E402, F401  (re-exported for the tests)


def main(
    argv: list[str] | None = None, *, promoter: Callable = promote, out: Callable[[str], None] | None = None
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_version_id", type=int, help="model_versions.model_version_id that evaluate.py printed")
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)
    try:
        result = promoter(args.model_version_id)
    except (ValueError, StoreError, RuntimeError, MlflowException) as exc:
        out(f"STOPPED: {exc}")
        return 1
    if not result["changed"]:
        out(f"model version {result['model_version_id']} is already the champion: nothing changed")
        return 0
    pct = result["improvement_pct"]
    out(
        f"promoted model version {result['model_version_id']} (MLflow version {result['mlflow_version']}) to champion: "
        f"test MAE {result['test_mae']:.3f} vs baseline {result['baseline_test_mae']:.3f} ({pct:+.1f}%)"
    )
    if not result["alias_set"]:
        out("the alias was already on this version: only champion_history was completed")
    if result["demoted"]:
        out(f"closed champion_history rows: {', '.join(str(i) for i in result['demoted'])}")
    if pct <= 0:
        out("WARNING: the improvement over the baseline is not positive; promoted anyway")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())
