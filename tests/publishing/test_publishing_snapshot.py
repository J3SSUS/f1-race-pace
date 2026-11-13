import copy
import json
from pathlib import Path

import pytest

from publishing.snapshot import build_snapshot, normalize

GOLDEN = Path(__file__).resolve().parents[1] / "site" / "golden"
GENERATED_AT = "2026-11-10T21:00:00Z"


@pytest.fixture
def raw():
    return json.loads((GOLDEN / "raw_rows.json").read_text())


@pytest.fixture
def expected():
    return json.loads((GOLDEN / "expected_snapshot.json").read_text())


def test_the_golden_rows_give_the_expected_snapshot(raw, expected):
    assert build_snapshot(raw, GENERATED_AT) == expected


def test_normalize_gives_the_five_parts_of_the_expected_file(raw, expected):
    parts = {key: expected[key] for key in ("champion", "baseline", "races", "metrics", "results")}
    assert normalize(raw) == parts


def test_the_input_rows_are_not_modified(raw):
    before = copy.deepcopy(raw)
    normalize(raw)
    assert raw == before


def test_a_newer_baseline_without_metrics_does_not_win(raw):
    # version 6 is the newest baseline but has no walk_forward metrics; versions 1 and 3 do, and 3 is the newest of them
    assert normalize(raw)["baseline"] == {"model_version_id": 3, "algorithm": "baseline"}


def test_only_the_champion_and_baseline_metrics_are_kept_in_order(raw):
    kept = normalize(raw)["metrics"]
    assert {row["model_version_id"] for row in kept} == {3, 4}
    assert [(r["model_version_id"], r["race_id"], r["metric_name"]) for r in kept] == [
        (4, None, "mae"),
        (4, None, "rmse"),
        (4, 89, "mae"),
        (3, None, "mae"),
        (3, None, "rmse"),
        (3, 89, "mae"),
    ]


def test_a_race_not_run_yet_keeps_its_nulls(raw):
    pending = [r for r in normalize(raw)["results"] if r["round"] == 18]
    assert len(pending) == 2
    assert all(r["actual_pace_delta_pct"] is None and r["is_valid_row"] is None for r in pending)


def test_no_current_champion_raises(raw):
    raw["championHistory"][1]["demoted_at"] = "2026-11-03T00:00:00+00:00"
    with pytest.raises(ValueError, match=r"champion_history: expected one current champion, found 0"):
        normalize(raw)


def test_two_current_champions_raise(raw):
    raw["championHistory"][0]["demoted_at"] = None
    with pytest.raises(ValueError, match=r"champion_history: expected one current champion, found 2"):
        normalize(raw)


def test_a_champion_missing_from_model_versions_raises(raw):
    raw["modelVersions"] = [v for v in raw["modelVersions"] if v["model_version_id"] != 4]
    with pytest.raises(ValueError, match=r"model_versions: no row for the champion \(version 4\)"):
        normalize(raw)


def test_no_baseline_with_metrics_raises(raw):
    raw["metrics"] = [m for m in raw["metrics"] if m["model_version_id"] not in (1, 3)]
    with pytest.raises(ValueError, match=r"model_versions: no baseline version has walk_forward metrics"):
        normalize(raw)


def test_empty_results_raise(raw):
    raw["results"] = []
    with pytest.raises(ValueError, match=r"prediction_results: no rows for the champion"):
        normalize(raw)


def test_errors_never_carry_row_content(raw):
    raw["results"] = []
    raw["races"][0]["event_name"] = "SECRET-EVENT"
    with pytest.raises(ValueError) as error:
        normalize(raw)
    assert "SECRET-EVENT" not in str(error.value)
