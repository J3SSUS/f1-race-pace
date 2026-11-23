"""Promote a model version to `champion`: by hand, or automatically with the rule of `CLAUDE.md`.

The writes live in `src/training/promotion.py`. Every promotion sets the `champion` alias in MLflow, closes the current
row of `champion_history` (`demoted_at`), inserts the new one, marks the champion that leaves as the only
`is_shadow = true` and removes the `challenger` alias if it was on the promoted version.

By hand (`promoted_by = manual`): the version needs `split = 'test'` metrics, so a champion has always gone through
`scripts/evaluate.py`; the test MAE improvement over the baseline of the same sampling run is stored. A non-positive
improvement prints a warning and still promotes: the decision is the user's. Promoting the current champion changes
nothing.

    uv run python scripts/promote.py <model_version_id>

Automatically (`promoted_by = auto`): the version must hold the `challenger` alias. It is promoted only if, on the last
5 races both it and the champion predicted, its mean walk-forward MAE is at most 0.98 x the champion's and it is not
worse in at least 3 of the 5. "The rule is not met" is a normal result (exit code 0, with the reasons and the numbers).
`--dry-run` evaluates and prints and writes nothing.

    uv run python scripts/promote.py --auto <model_version_id> [--dry-run]

Exits with code 1 and a `STOPPED:` line if it cannot. After a promotion it reminds you to run `scripts/publish.py`.
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
from training.promotion import (  # noqa: E402, F401  (re-exported for the tests)
    ALIAS,
    auto_promote,
    promote,
    version_test_mae,
)

PUBLISH_REMINDER = "next: run `uv run python scripts/publish.py` to republish the page and the model card"


def describe_shadow(result: dict) -> list[str]:
    lines = []
    if result["shadow_set"] is not None:
        lines.append(f"model version {result['shadow_set']} is now the shadow (the champion that left)")
    if result["shadow_cleared"]:
        lines.append(f"shadow mark removed from: {', '.join(str(i) for i in result['shadow_cleared'])}")
    if result["challenger_alias_removed"]:
        lines.append("the `challenger` alias was removed from the promoted version")
    return lines


def describe_races(verdict: dict) -> list[str]:
    """The per-race numbers the rule looked at, oldest first, so a borderline result can be judged by hand."""
    lines = [f"races both predicted: {verdict['common_races']}"]
    for race_id, champion, challenger in verdict["per_race"]:
        mark = " (worse)" if challenger > champion else ""
        lines.append(f"  race {race_id}: champion {champion:.4f}, challenger {challenger:.4f}{mark}")
    if verdict["champion_mae"] is not None:
        lines.append(
            f"mean MAE over the last {len(verdict['window'])}: champion {verdict['champion_mae']:.4f}, "
            f"challenger {verdict['challenger_mae']:.4f} ({verdict['improvement_pct']:+.1f}%), "
            f"worse in {verdict['races_worse']} race(s)"
        )
    return lines


def run_manual(version_id: int, promoter: Callable, out: Callable[[str], None]) -> int:
    result = promoter(version_id)
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
    for line in describe_shadow(result):
        out(line)
    if pct <= 0:
        out("WARNING: the improvement over the baseline is not positive; promoted anyway")
    out(PUBLISH_REMINDER)
    return 0


def run_auto(version_id: int, dry_run: bool, auto_promoter: Callable, out: Callable[[str], None]) -> int:
    result = auto_promoter(version_id, dry_run=dry_run)
    out(f"challenger model version {version_id} against champion model version {result['champion_version_id']}")
    for line in describe_races(result):
        out(line)
    if result["promoted"]:
        out(
            f"promoted model version {version_id} to champion (automatic): forward improvement "
            f"{result['improvement_pct']:+.1f}% over the last {len(result['window'])} races"
        )
        for line in describe_shadow(result):
            out(line)
        out(PUBLISH_REMINDER)
    elif result["passes"]:
        out(f"dry run: the rule is met, model version {version_id} would be promoted; nothing was written")
    else:
        out(f"not promoted: {result['reason']}")
    return 0


def main(
    argv: list[str] | None = None,
    *,
    promoter: Callable = promote,
    auto_promoter: Callable = auto_promote,
    out: Callable[[str], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_version_id", type=int, help="model_versions.model_version_id that evaluate.py printed")
    parser.add_argument("--auto", action="store_true", help="promote only if the challenger passes the rule")
    parser.add_argument("--dry-run", action="store_true", help="with --auto: evaluate and print, write nothing")
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)
    if args.dry_run and not args.auto:
        out("STOPPED: --dry-run only goes with --auto")
        return 1
    try:
        if args.auto:
            return run_auto(args.model_version_id, args.dry_run, auto_promoter, out)
        return run_manual(args.model_version_id, promoter, out)
    except (ValueError, StoreError, RuntimeError, MlflowException) as exc:
        out(f"STOPPED: {exc}")
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())
