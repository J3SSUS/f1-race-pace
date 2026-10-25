"""The `sampling` step: ingest the races up to the cutoff, label them train or test and register the run.

The split itself is defined in `training.split`. Here the sessions of every race are ingested and checked first, so
a race with a missing qualifying or race session stops the step before anything is split.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ingestion.reference import ReferenceDataError, ensure_race_entries
from ingestion.schedule import expected_sessions
from ingestion.session import RAW_DIR, ingest_session

# The sessions the features read, in the order they are listed in the race index. FP2 and S are read only when the
# weekend has them (a sprint weekend has no FP2); Q and R are always needed.
SESSION_ORDER = ("FP2", "Q", "S", "R")
OPTIONAL_SESSIONS = ("FP2", "S")


class SamplingError(Exception):
    """A Q or R session that failed or is invalid, or no races to split."""

    def __init__(self, message: str, problems: list[dict] | None = None):
        super().__init__(message)
        self.problems = problems or []


@dataclass
class IngestReport:
    sessions: dict[tuple[int, int], list[str]]  # sessions read per race, in `SESSION_ORDER`
    issues: list[dict] = field(default_factory=list)  # FP2 and S sessions that are failed or invalid


def _status(result) -> str | None:
    """`failed`, `invalid` or None when the session is usable."""
    if result.status == "failed":
        return "failed"
    return None if result.valid else "invalid"


def _problem_message(problems: list[dict]) -> str:
    lines = [f"{p['year']} round {p['round']} {p['session']}: {p['status']}" for p in problems]
    return "Q or R sessions that cannot be used, nothing was split:\n" + "\n".join(lines)


def ingest_races(
    races: list[tuple[int, int]],
    schedule: pd.DataFrame,
    *,
    client=None,
    raw_dir: Path = RAW_DIR,
    ingest: Callable = ingest_session,
    ensure: Callable = ensure_race_entries,
    log: Callable[[str], None] | None = None,
) -> IngestReport:
    """Ingest FP2, S (if the schedule has it), Q and R of every race and make sure its entries exist.

    A failed or invalid FP2 or S goes to `issues`. A failed R, a failed Q, an invalid R, or a Q that
    `ensure_race_entries` cannot use raises one `SamplingError` that lists every race affected. A Q that is invalid
    only because of a table the entries do not need is accepted (the Q rule of SPEC 02).
    """
    report = IngestReport(sessions={})
    problems: list[dict] = []
    for year, round in races:
        scheduled = set(expected_sessions(schedule, year, round))
        sessions = [s for s in SESSION_ORDER if s not in OPTIONAL_SESSIONS or s in scheduled]
        report.sessions[(year, round)] = sessions

        status = {s: _status(ingest(year, round, s, raw_dir=raw_dir)) for s in sessions}
        for session in OPTIONAL_SESSIONS:
            if status.get(session):
                report.issues.append({"year": year, "round": round, "session": session, "status": status[session]})
        if status["R"]:
            problems.append({"year": year, "round": round, "session": "R", "status": status["R"]})
        if status["Q"] == "failed":
            problems.append({"year": year, "round": round, "session": "Q", "status": "failed"})
        else:
            try:
                ensure(year, round, client=client, schedule=schedule, ingest=ingest, raw_dir=raw_dir)
            except ReferenceDataError:
                problems.append({"year": year, "round": round, "session": "Q", "status": "invalid"})
        if log:
            log(f"{year} round {round:>2}: {','.join(sessions)}")

    if problems:
        raise SamplingError(_problem_message(problems), problems)
    return report
