"""Make the drift reports of the champion that are due (SPEC 26): MAE change and feature drift against its reference.

A report covers the champion's last 5 races once 5 races are waiting after its last report (or after the end of its
training, if it has none). It compares the MAE of those races with the champion's reference MAE (its test MAE, or for a
refit champion its MAE just before its promotion) and each of the 15 features with the rows the champion trained on
(KS for numbers, chi-squared for flags and categories, drift at p < 0.05). It stores the Evidently HTML as an artifact
of a run in the MLflow experiment `drift`, one row in `drift_reports` and one per feature in `drift_feature_results`,
and says whether the retraining rule holds (MAE up 15% or 3+ drifted features); it never triggers a retraining.

  --year Y --round R   the check as of that race: the task of the `ground_truth` DAG
  --year Y             the backfill: every race of the year in order, one report each time 5 races pile up

Running it again for the same races writes nothing. Prints one line per report, or how many races are waiting. Exits
with code 0 in both cases, and with code 1 and a `STOPPED:` line if a report cannot be built (no champion, no test MAE,
a refit champion with fewer than 5 races before its promotion, Supabase or MLflow down).

    uv run python scripts/drift_report.py --year 2026
    uv run python scripts/drift_report.py --year 2026 --round 21 --dag-run-id manual__2026-11-26
"""

import argparse
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.supabase_io import get_client  # noqa: E402
from features.store import StoreError  # noqa: E402
from monitoring.drift import DRIFT_WINDOW  # noqa: E402
from monitoring.inputs import DriftError, Inputs, load_inputs  # noqa: E402
from monitoring.run import Made, RunResult, run_reports  # noqa: E402


def describe(made: Made, n_features: int) -> str:
    report, written = made.report, made.written
    state = "written" if written.created else "already stored"
    reference = "test MAE" if report.reference_source == "test" else "MAE before promotion"
    (first_year, first_round), (last_year, last_round) = made.first, made.last
    races = (
        f"{first_year} rounds {first_round}-{last_round}"
        if first_year == last_year
        else f"{first_year} round {first_round} to {last_year} round {last_round}"
    )
    return (
        f"report {races} (model version {report.model_version_id}): "
        f"MAE {report.current_mae:.4f} vs {report.reference_mae:.4f} ({reference}), {report.mae_change_pct:+.1f}%; "
        f"{report.n_drifted_features} of {n_features} features drifted; rule met: "
        f"{'yes' if report.rule_met else 'no'}; drift_report_id {written.drift_report_id}, "
        f"MLflow run {written.mlflow_run_id} ({state})"
    )


def main(
    argv: list[str] | None = None,
    *,
    client=None,
    tracking_uri: str | None = None,
    loader: Callable[..., Inputs] = load_inputs,
    runner: Callable[..., RunResult] = run_reports,
    out: Callable[[str], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True, help="season of the race, or of the backfill")
    parser.add_argument("--round", type=int, default=None, help="check as of this round; omit it for the whole year")
    parser.add_argument("--dag-run-id", default=None, help="Airflow run that made the report (left empty by hand)")
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)
    try:
        client = client or get_client()
        inputs = loader(client, tracking_uri)
        result = runner(
            client,
            inputs,
            year=args.year,
            round_=args.round,
            dag_run_id=args.dag_run_id,
            tracking_uri=tracking_uri,
        )
    except (DriftError, StoreError, RuntimeError, ValueError) as exc:
        out(f"STOPPED: {exc}")
        return 1
    for made in result.reports:
        out(describe(made, len(made.report.features)))
    if result.pending or not result.reports:
        out(f"{result.pending} of {DRIFT_WINDOW} races since the last report: nothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())
