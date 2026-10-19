import pandas as pd
from test_session import bad_loader, good_loader

from ingestion.coverage import MIN_COVERAGE_PCT, build_coverage, format_summary, summary, write_report
from ingestion.session import ingest_session


def events(*sessions, year=2024, rnd=8, name="Monaco Grand Prix"):
    return pd.DataFrame(
        {
            "year": year,
            "round": rnd,
            "event_name": name,
            "session": list(sessions),
            "session_date_utc": pd.Timestamp("2024-05-25T14:00:00Z"),
        }
    )


def ingest(tmp_path, session, loader, rnd=8, year=2024):
    return ingest_session(year, rnd, session, loader=loader, raw_dir=tmp_path, sleep=lambda s: None)


def test_valid_invalid_and_missing(tmp_path):
    ingest(tmp_path, "Q", good_loader())
    ingest(tmp_path, "R", bad_loader)
    cov = build_coverage(events("Q", "R", "FP2"), tmp_path)
    assert cov["session"].tolist() == ["Q", "R", "FP2"]
    assert cov["status"].tolist() == ["valid", "invalid", "missing"]
    assert cov.loc[0, ["rows_laps", "rows_session_result", "rows_weather"]].tolist() == [2, 2, 2]
    assert cov.loc[1, "errors"] != "" and cov.loc[2, "errors"] == ""
    assert cov.loc[2, ["rows_laps", "rows_weather"]].isna().all()
    assert cov.loc[0, "session_date_utc"] == "2024-05-25T14:00:00Z"


def test_a_valid_session_with_a_missing_file_is_invalid(tmp_path):
    res = ingest(tmp_path, "Q", good_loader())
    (res.path / "pit.parquet").unlink()
    cov = build_coverage(events("Q"), tmp_path)
    assert cov.loc[0, "status"] == "invalid" and "pit.parquet is missing" in cov.loc[0, "errors"]


def test_a_failed_session_is_missing(tmp_path):
    from test_session import failing_loader

    ingest(tmp_path, "Q", failing_loader)
    assert build_coverage(events("Q"), tmp_path)["status"].tolist() == ["missing"]


def test_summary_counts_and_lists_what_is_not_valid(tmp_path):
    ingest(tmp_path, "Q", good_loader())
    ingest(tmp_path, "R", bad_loader)
    result = summary(build_coverage(events("Q", "R", "FP2", "FP1"), tmp_path))
    assert (result.expected, result.valid, result.invalid, result.missing) == (4, 1, 1, 2)
    assert result.percentage == 25.0 and not result.meets_target
    assert result.not_valid_by_race == [
        (2024, 8, "Monaco Grand Prix", [("R", "invalid"), ("FP2", "missing"), ("FP1", "missing")])
    ]


def test_target_is_met_at_95_percent():
    cov = pd.DataFrame(
        {
            "year": 2024,
            "round": 1,
            "event_name": "x",
            "session": list("ABCDEFGHIJKLMNOPQRST"),
            "status": ["valid"] * 19 + ["missing"],
        }
    )
    result = summary(cov)
    assert result.percentage == MIN_COVERAGE_PCT and result.meets_target


def test_no_expected_sessions_never_meets_the_target():
    result = summary(build_coverage(events(), None))
    assert result.expected == 0 and result.percentage == 0.0 and not result.meets_target


def test_write_report_creates_the_folder(tmp_path):
    ingest(tmp_path, "Q", good_loader())
    cov = build_coverage(events("Q", "FP2"), tmp_path)
    path = write_report(cov, tmp_path / "reports" / "coverage.csv")
    back = pd.read_csv(path)
    assert back["status"].tolist() == ["valid", "missing"] and list(back.columns) == list(cov.columns)


def test_format_summary_names_the_races(tmp_path):
    ingest(tmp_path, "Q", good_loader())
    text = format_summary(summary(build_coverage(events("Q", "FP2"), tmp_path)))
    assert "50.0%" in text and "2024 round 08 Monaco Grand Prix: FP2 (missing)" in text
