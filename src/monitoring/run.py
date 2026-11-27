"""Make the drift reports that are due, for one race or for a whole season (SPEC 26).

A report is due when `DRIFT_WINDOW` races of the champion are waiting after its start point (`monitoring.drift`). For
one race (`round` given) the check is "as of that race": only the champion's scored races up to it count. For a season
(`round` None) every race of the year is a checkpoint, in order, so the backfill makes the reports the live DAG would
have made, and a report counts as the last one for the next checkpoint. Running it again finds the reports it stored
and writes nothing.
"""

from collections.abc import Callable
from dataclasses import dataclass

from features.catalog import load_catalog
from monitoring.drift import select_window, start_race
from monitoring.inputs import DriftError, Inputs, feature_ids
from monitoring.report import DriftReport, Written, build_report, write_report


@dataclass(frozen=True)
class Made:
    report: DriftReport
    written: Written
    first: tuple[int, int]  # (year, round) of the first and the last race of the window
    last: tuple[int, int]


@dataclass(frozen=True)
class RunResult:
    reports: list[Made]
    pending: int  # races waiting after the last checkpoint (0 right after a report)


def run_reports(
    client,
    inputs: Inputs,
    *,
    year: int,
    round_: int | None = None,
    dag_run_id: str | None = None,
    builder: Callable = build_report,
    writer: Callable = write_report,
    tracking_uri: str | None = None,
) -> RunResult:
    """Every report due as of `(year, round_)`, or as of each scored race of `year` when `round_` is None."""
    order = inputs.race_order
    scored = set(inputs.results["race_id"])
    if round_ is None:
        checkpoints = sorted((r for r in scored if order[r][0] == year), key=order.__getitem__)
        if not checkpoints:
            raise DriftError(
                f"the champion (model version {inputs.champion.model_version_id}) has no scored prediction in {year}"
            )
    else:
        match = [race for race, value in order.items() if value == (year, round_)]
        if not match:
            raise DriftError(f"race {year} round {round_} is not in races")
        checkpoints = match

    catalog, ids = load_catalog(), feature_ids(client)
    last_report = inputs.last_report_race_id
    made: list[Made] = []
    pending = 0
    for race in checkpoints:
        as_of = {r for r in scored if order[r] <= order[race]}
        after = start_race(inputs.reference.base_race_id, last_report, order)
        window = select_window(as_of, order, after)
        pending = window.pending
        if not window.due:
            continue
        report, html = builder(inputs, window.races, catalog, ids)
        written = writer(client, report, html, dag_run_id=dag_run_id, tracking_uri=tracking_uri)
        made.append(Made(report, written, order[window.races[0]], order[window.races[-1]]))
        last_report, pending = report.through_race_id, 0
    return RunResult(made, pending)
