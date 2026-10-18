"""Coverage of the raw data: expected sessions of the held events against what `meta.json` says is on disk."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ingestion.schemas import TABLES
from ingestion.session import RAW_DIR, read_meta, session_dir

MIN_COVERAGE_PCT = 95.0  # phase 1 definition of done
REPORT_PATH = RAW_DIR.parent / "reports" / "coverage.csv"
COLUMNS = [
    "year",
    "round",
    "event_name",
    "session",
    "session_date_utc",
    "status",
    "rows_laps",
    "rows_session_result",
    "rows_weather",
    "errors",
]


def _row(event: pd.Series, raw_dir: Path) -> dict:
    path = session_dir(int(event["year"]), int(event["round"]), event["session"], raw_dir)
    meta = read_meta(path)
    row = {
        "year": int(event["year"]),
        "round": int(event["round"]),
        "event_name": event["event_name"],
        "session": event["session"],
        "session_date_utc": event["session_date_utc"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "missing",
        "rows_laps": None,
        "rows_session_result": None,
        "rows_weather": None,
        "errors": "",
    }
    if meta is None:
        return row
    absent = [name for name in TABLES if not (path / f"{name}.parquet").exists()]
    errors = list(meta["errors"]) + [f"{name}.parquet is missing" for name in absent]
    row.update(
        status="valid" if meta["valid"] and not absent else "invalid",
        rows_laps=meta["rows"].get("laps"),
        rows_session_result=meta["rows"].get("session_result"),
        rows_weather=meta["rows"].get("weather"),
        errors="; ".join(errors),
    )
    return row


def build_coverage(events: pd.DataFrame, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per expected session of the held events (`events` is the output of `held_events`)."""
    rows = [_row(event, raw_dir) for _, event in events.iterrows()]
    return pd.DataFrame(rows, columns=COLUMNS)


def write_report(coverage: pd.DataFrame, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(path, index=False)
    return path


@dataclass
class Summary:
    expected: int
    valid: int
    invalid: int
    missing: int
    percentage: float
    not_valid_by_race: list[tuple[int, int, str, list[tuple[str, str]]]]  # (year, round, event, [(session, status)])

    @property
    def meets_target(self) -> bool:
        return self.percentage >= MIN_COVERAGE_PCT


def summary(coverage: pd.DataFrame) -> Summary:
    counts = coverage["status"].value_counts()
    expected = len(coverage)
    valid = int(counts.get("valid", 0))
    bad = coverage[coverage["status"] != "valid"]
    by_race = [
        (int(year), int(rnd), event, list(zip(group["session"], group["status"], strict=True)))
        for (year, rnd, event), group in bad.groupby(["year", "round", "event_name"], sort=True)
    ]
    return Summary(
        expected=expected,
        valid=valid,
        invalid=int(counts.get("invalid", 0)),
        missing=int(counts.get("missing", 0)),
        percentage=round(100 * valid / expected, 1) if expected else 0.0,
        not_valid_by_race=by_race,
    )


def format_summary(result: Summary) -> str:
    lines = [
        f"Coverage: {result.percentage}% ({result.valid} valid of {result.expected} expected sessions; "
        f"{result.invalid} invalid, {result.missing} missing). Target: {MIN_COVERAGE_PCT}%."
    ]
    for year, rnd, event, sessions in result.not_valid_by_race:
        lines.append(f"  {year} round {rnd:02d} {event}: " + ", ".join(f"{s} ({status})" for s, status in sessions))
    return "\n".join(lines)
