"""The `evaluate` step: pick the candidate from the cross-validated predictions and score it once on the test races.

Reads the out-of-fold predictions of the seven versions of `f1-race-pace` from MLflow (`.env`), runs the paired block
bootstrap by race and prints the table and the candidate. Without `--selection-only` it then scores the candidate and
the baseline once on the test races, writes 8 `metrics` rows (`split = 'test'`) to Supabase and logs one `evaluate` run
in MLflow, and prints the test result of the candidate against the baseline with its intervals. The test is touched
once: a second run for the same sampling run is refused. It never promotes: `scripts/promote.py` does that by hand.
`--selection-only` reads no test row and writes nothing, to review the choice before using the only evaluation.
Exits with code 1 and a `STOPPED:` line if it cannot (unknown run, a missing version, an evaluation already done or a
Supabase error).

    uv run python scripts/evaluate.py <sampling_run_id> [--selection-only]
"""

import argparse
import logging
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlflow.exceptions import MlflowException  # noqa: E402

from common.config import load_config  # noqa: E402
from features.store import StoreError  # noqa: E402
from training.evaluation import N_BOOT, TestResult, run_evaluation  # noqa: E402
from training.selection import BASELINE, Comparison, Selection, load_oof, select  # noqa: E402


def races_of(oof) -> int:
    """The number of races the bootstrap resamples: the distinct (year, round) of the OOF rows."""
    return len(oof[BASELINE][["year", "round"]].drop_duplicates())


def describe_selection(selection: Selection, n_rows: int, n_races: int, seed: int) -> list[str]:
    lines = [
        f"Selection on the out-of-fold predictions: {n_rows} rows in {n_races} races; "
        f"paired block bootstrap by race, {N_BOOT} resamples, seed {seed}",
        f"reference (lowest CV MAE of the six): {selection.reference}",
        f"{'model':<14}{'cv_mae':>8}{'diff':>9}{'ci_low':>9}{'ci_high':>9}{'width':>8}  not_worse",
    ]
    for c in selection.table:
        mark = "  <- candidate" if c.name == selection.candidate else ""
        lines.append(
            f"{c.name:<14}{c.cv_mae:>8.3f}{c.diff:>+9.3f}{c.ci_low:>+9.3f}{c.ci_high:>+9.3f}"
            f"{c.ci_high - c.ci_low:>8.3f}  {'yes' if c.not_worse else 'no'}{mark}"
        )
    lines.append(f"candidate (simplest not significantly worse): {selection.candidate}")
    lines.append(describe_baseline(selection.vs_baseline, selection.beats_baseline))
    return lines


def describe_baseline(vs: Comparison | None, beats: bool) -> str:
    if vs is None:
        return "no comparison with the baseline"
    interval = f"diff {vs.diff:+.3f} [{vs.ci_low:+.3f}, {vs.ci_high:+.3f}]"
    if beats:
        return f"{vs.name} beats the baseline in CV: {interval} (ci_high < 0)"
    return f"{vs.name} does NOT beat the baseline in CV: {interval}; promotion is not suggested"


def describe_test(result: TestResult) -> list[str]:
    lines = [f"{result.name}: {result.n_rows} rows in {result.n_races} test races"]
    for metric, value in result.metrics.items():
        low, high = result.intervals[metric]
        lines.append(f"  {metric:<18}{value:>8.3f}  [{low:.3f}, {high:.3f}]")
    q = result.error_quantiles
    lines.append(
        f"  bias {result.mean_error:+.3f} | |error| median {q['median']:.3f}, p90 {q['p90']:.3f}, "
        f"p95 {q['p95']:.3f}, max {q['max']:.3f}"
    )
    return lines


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable = run_evaluation,
    oof_loader: Callable = load_oof,
    out: Callable[[str], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="the sampling run_id that scripts/sample.py printed")
    parser.add_argument("--selection-only", action="store_true", help="select from the OOF; read no test row")
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)
    seed = load_config().optuna.seed

    try:
        if args.selection_only:
            oof = oof_loader(args.run_id)
            selection, result = select(oof, N_BOOT, seed), None
        else:
            result = runner(args.run_id)
            selection = result["selection"]
            oof = oof_loader(args.run_id)
    except (ValueError, StoreError, RuntimeError, MlflowException) as exc:
        out(f"STOPPED: {exc}")
        return 1

    for line in describe_selection(selection, len(oof[BASELINE]), races_of(oof), seed):
        out(line)
    if result is None:
        out("selection only: no test row was read and nothing was written")
        return 0
    out("Test result (touched once):")
    for name in (selection.candidate, BASELINE):
        for line in describe_test(result["results"][name]):
            out(line)
    candidate_id = result["versions"][selection.candidate]
    if selection.beats_baseline:
        out(f"next, by hand: uv run python scripts/promote.py {candidate_id}")
    else:
        out("the candidate does not beat the baseline in CV: promotion is not suggested")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())
