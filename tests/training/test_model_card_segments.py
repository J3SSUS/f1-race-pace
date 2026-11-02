import numpy as np
import pandas as pd
import pytest
from fake_supabase import FakeSupabase
from pytest import approx

from training.model_card import MIN_SEGMENT_ROWS, read_entry_segments, segment_errors


def oof_table(rows) -> pd.DataFrame:
    """`rows` are (entry_id, round, y_true, y_pred); every row is in the 2024 season."""
    frame = pd.DataFrame(rows, columns=["entry_id", "round", "y_true", "y_pred"])
    return frame.assign(year=2024, driver="XXX", fold=1)[
        ["entry_id", "year", "round", "driver", "fold", "y_true", "y_pred"]
    ]


# Six entries: team A has 1, 2, 3 and team B has 4, 5, 6; entries 1 and 4 are on a street circuit, the rest permanent.
SEGMENTS = pd.DataFrame(
    {
        "entry_id": [1, 2, 3, 4, 5, 6],
        "team": ["A", "A", "A", "B", "B", "B"],
        "circuit_type": ["street", "permanent", "permanent", "street", "permanent", "permanent"],
    }
)
CHAMPION = oof_table(
    [(1, 1, 1.0, 0.0), (2, 2, 0.0, 0.5), (3, 3, -1.0, -0.5), (4, 1, 2.0, 1.0), (5, 2, 0.0, 0.0), (6, 3, 1.0, 3.0)]
)
BASELINE = oof_table(
    [(1, 1, 1.0, 0.5), (2, 2, 0.0, 0.5), (3, 3, -1.0, 0.5), (4, 1, 2.0, 0.5), (5, 2, 0.0, 0.5), (6, 3, 1.0, 0.5)]
)


def by_key(errors):
    return {(e.segment_type, e.segment_value): e for e in errors}


def test_mae_bias_and_sizes_equal_the_values_computed_by_hand():
    found = by_key(segment_errors(CHAMPION, BASELINE, SEGMENTS))
    # team A: errors (pred - true) -1, +.5, +.5 ; team B: -1, 0, +2
    assert (found["team", "A"].mae, found["team", "A"].bias) == (approx(2 / 3), approx(0.0))
    assert (found["team", "B"].mae, found["team", "B"].bias) == (approx(1.0), approx(1 / 3))
    assert (found["circuit_type", "street"].mae, found["circuit_type", "street"].bias) == (approx(1.0), approx(-1.0))
    assert (found["circuit_type", "permanent"].mae, found["circuit_type", "permanent"].bias) == (
        approx(0.75),
        approx(0.75),
    )
    assert [found[k].n_rows for k in found] == [3, 3, 2, 4]
    assert [found[k].n_races for k in found] == [3, 3, 1, 2]


def test_the_baseline_mae_is_in_the_same_segments_on_the_same_rows():
    found = by_key(segment_errors(CHAMPION, BASELINE, SEGMENTS))
    assert found["team", "A"].baseline_mae == approx(2.5 / 3)
    assert found["team", "B"].baseline_mae == approx(2.5 / 3)
    assert found["circuit_type", "street"].baseline_mae == approx(1.0)
    assert found["circuit_type", "permanent"].baseline_mae == approx(0.75)


def test_the_order_is_teams_then_circuit_types_each_with_the_largest_mae_first():
    errors = segment_errors(CHAMPION, BASELINE, SEGMENTS)
    assert [(e.segment_type, e.segment_value) for e in errors] == [
        ("team", "B"),
        ("team", "A"),
        ("circuit_type", "street"),
        ("circuit_type", "permanent"),
    ]


def test_per_segment_type_the_rows_add_up_to_the_oof_rows():
    errors = segment_errors(CHAMPION, BASELINE, SEGMENTS)
    for kind in ("team", "circuit_type"):
        assert sum(e.n_rows for e in errors if e.segment_type == kind) == len(CHAMPION)


def test_the_row_order_of_the_baseline_does_not_matter():
    shuffled = BASELINE.sample(frac=1, random_state=3)
    assert segment_errors(CHAMPION, shuffled, SEGMENTS) == segment_errors(CHAMPION, BASELINE, SEGMENTS)


def test_a_segment_with_29_rows_is_few_rows_and_one_with_30_is_not():
    ids = np.arange(1, 60)  # 29 entries of team A and 30 of team B
    segments = pd.DataFrame({"entry_id": ids, "team": ["A"] * 29 + ["B"] * 30, "circuit_type": "permanent"})
    table = oof_table([(int(i), 1 + int(i) % 3, 0.0, 1.0) for i in ids])
    found = by_key(segment_errors(table, table, segments))
    assert MIN_SEGMENT_ROWS == 30
    assert (found["team", "A"].n_rows, found["team", "A"].few_rows) == (29, True)
    assert (found["team", "B"].n_rows, found["team", "B"].few_rows) == (30, False)


def test_an_entry_with_no_team_or_circuit_raises():
    no_team = SEGMENTS.assign(team=[None, "A", "A", "B", "B", "B"])
    with pytest.raises(ValueError, match="no team or circuit"):
        segment_errors(CHAMPION, BASELINE, no_team)
    no_circuit = SEGMENTS.assign(circuit_type=[np.nan, "permanent", "permanent", "street", "permanent", "permanent"])
    with pytest.raises(ValueError, match="no team or circuit"):
        segment_errors(CHAMPION, BASELINE, no_circuit)
    with pytest.raises(ValueError, match="no team or circuit"):
        segment_errors(CHAMPION, BASELINE, SEGMENTS[SEGMENTS["entry_id"] != 6])


def test_oof_that_cover_different_entries_raise():
    with pytest.raises(ValueError, match="do not cover the same entries"):
        segment_errors(CHAMPION, BASELINE[BASELINE["entry_id"] != 6], SEGMENTS)


def test_repeated_entries_raise():
    with pytest.raises(ValueError, match="repeated entry_id"):
        segment_errors(pd.concat([CHAMPION, CHAMPION.head(1)]), BASELINE, SEGMENTS)


def reference_tables() -> dict[str, list[dict]]:
    return {
        "teams": [{"team_id": 1, "name": "Alpha"}, {"team_id": 2, "name": "Beta"}],
        "circuits": [{"circuit_id": 10, "circuit_type": "street"}, {"circuit_id": 11, "circuit_type": "permanent"}],
        "races": [{"race_id": 100, "circuit_id": 10}, {"race_id": 101, "circuit_id": 11}],
        "race_entries": [
            {"entry_id": 1, "race_id": 100, "team_id": 1},
            {"entry_id": 2, "race_id": 100, "team_id": 2},
            {"entry_id": 3, "race_id": 101, "team_id": 1},
        ],
    }


def test_read_entry_segments_maps_each_entry_to_its_team_and_circuit_type():
    segments = read_entry_segments(FakeSupabase(reference_tables()))
    assert segments.to_dict("records") == [
        {"entry_id": 1, "team": "Alpha", "circuit_type": "street"},
        {"entry_id": 2, "team": "Beta", "circuit_type": "street"},
        {"entry_id": 3, "team": "Alpha", "circuit_type": "permanent"},
    ]


def test_only_the_four_reference_tables_are_read_and_nothing_is_written():
    tables = {**reference_tables(), "ground_truth": [{"entry_id": 1, "race_pace_delta_pct": 9.9}]}
    client = FakeSupabase(tables)
    read_entry_segments(client)
    assert {name for name, _, _ in client.reads} == {"race_entries", "teams", "races", "circuits"}
    assert client.writes == []


def test_an_entry_with_a_dangling_reference_keeps_a_missing_value_and_segment_errors_raises():
    tables = reference_tables()
    tables["race_entries"][0]["team_id"] = 99  # no such team
    segments = read_entry_segments(FakeSupabase(tables))
    assert segments.loc[segments["entry_id"] == 1, "team"].isna().all()
    oof = oof_table([(1, 1, 0.0, 1.0), (2, 1, 0.0, 1.0), (3, 2, 0.0, 1.0)])
    with pytest.raises(ValueError, match="no team or circuit"):
        segment_errors(oof, oof, segments)
