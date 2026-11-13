import copy
import json
from pathlib import Path

import pytest

from publishing.snapshot import SNAPSHOT_COLUMNS, build_snapshot, check_columns, normalize, write_snapshot

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


# --- columns and writing ---


def test_the_expected_snapshot_has_only_allowed_columns(expected):
    assert check_columns(expected) == []


def test_snapshot_columns_cover_the_five_parts():
    assert set(SNAPSHOT_COLUMNS) == {"champion", "baseline", "races", "metrics", "results"}


def test_an_extra_column_is_reported_by_part_and_column_once(expected):
    for row in expected["results"]:
        row["model_version_id"] = 4
    expected["champion"]["secret_note"] = "SECRET-VALUE"
    problems = check_columns(expected)
    assert problems == [
        "champion: unexpected column 'secret_note'",
        "results: unexpected column 'model_version_id'",
    ]
    assert "SECRET-VALUE" not in " ".join(problems)


def test_a_missing_column_is_reported_by_part_and_column(expected):
    del expected["metrics"][2]["value"]
    del expected["baseline"]["algorithm"]
    assert check_columns(expected) == ["baseline: missing column 'algorithm'", "metrics: missing column 'value'"]


def test_a_missing_or_unknown_top_level_key_is_reported(expected):
    del expected["generated_at"]
    expected["debug"] = {}
    assert check_columns(expected) == ["snapshot: unexpected key 'debug'", "snapshot: missing key 'generated_at'"]


def test_a_part_with_the_wrong_shape_is_reported(expected):
    expected["races"] = "not rows"
    assert check_columns(expected) == ["races: not a row or a list of rows"]


def test_write_snapshot_gives_compact_sorted_json_and_the_same_bytes(tmp_path, expected):
    target = tmp_path / "snapshot.json"
    write_snapshot(target, expected)
    first = target.read_bytes()
    text = first.decode("utf-8")
    assert text.startswith('{"baseline":{"algorithm":"baseline","model_version_id":3},"champion":')
    assert '": ' not in text and ", " not in text.split('"event_name"')[0]  # compact separators, before any text value
    assert json.loads(text) == expected
    write_snapshot(target, expected)
    assert target.read_bytes() == first
    assert [p.name for p in tmp_path.iterdir()] == ["snapshot.json"]  # no temporary file left behind


def test_non_ascii_text_is_written_as_is(tmp_path, expected):
    expected["races"][0]["event_name"] = "Gran Premio de São Paulo"
    target = tmp_path / "snapshot.json"
    write_snapshot(target, expected)
    assert "São Paulo" in target.read_text(encoding="utf-8")


def test_a_failed_write_leaves_the_previous_file_untouched(tmp_path, expected, monkeypatch):
    target = tmp_path / "snapshot.json"
    target.write_text("previous", encoding="utf-8")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("publishing.snapshot.os.replace", boom)
    with pytest.raises(OSError):
        write_snapshot(target, expected)
    assert target.read_text(encoding="utf-8") == "previous"
    assert [p.name for p in tmp_path.iterdir()] == ["snapshot.json"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {1, 2}])
def test_a_value_that_is_not_valid_json_writes_nothing(tmp_path, expected, bad):
    target = tmp_path / "snapshot.json"
    target.write_text("previous", encoding="utf-8")
    expected["metrics"][0]["value"] = bad
    with pytest.raises((ValueError, TypeError)):
        write_snapshot(target, expected)
    assert target.read_text(encoding="utf-8") == "previous"
    assert [p.name for p in tmp_path.iterdir()] == ["snapshot.json"]
