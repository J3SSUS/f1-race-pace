import pytest
from fake_supabase import FakeSupabase

from live.metrics import FRAME_COLUMNS, evaluation_frame


def seeded(*, extra_entries: int = 0) -> FakeSupabase:
    """2024 rounds 1 and 2 (race 11 and 12, drivers VER and NOR) and round 1 of 2025 (race 21), champion version 4.

    Entries: 1 and 2 of race 11, 3 and 4 of race 12, 5 of race 21. Every entry has a valid ground truth and a
    prediction, except what each test changes.
    """
    entries = [
        {"entry_id": 1, "race_id": 11, "driver_id": 1},
        {"entry_id": 2, "race_id": 11, "driver_id": 2},
        {"entry_id": 3, "race_id": 12, "driver_id": 1},
        {"entry_id": 4, "race_id": 12, "driver_id": 2},
        {"entry_id": 5, "race_id": 21, "driver_id": 1},
    ]
    entries += [{"entry_id": 100 + i, "race_id": 12, "driver_id": 1} for i in range(extra_entries)]
    return FakeSupabase(
        {
            "races": [
                {"race_id": 11, "year": 2024, "round": 1},
                {"race_id": 12, "year": 2024, "round": 2},
                {"race_id": 21, "year": 2025, "round": 1},
            ],
            "drivers": [{"driver_id": 1, "code": "VER"}, {"driver_id": 2, "code": "NOR"}],
            "race_entries": entries,
            "ground_truth": [
                {"entry_id": e["entry_id"], "actual_pace_delta_pct": 0.1 * e["entry_id"], "is_valid_row": True}
                for e in entries
            ],
            "predictions": [
                {
                    "entry_id": e["entry_id"],
                    "model_version_id": 4,
                    "predicted_pace_delta_pct": 0.2 * e["entry_id"],
                    "baseline_pace_delta_pct": 0.3 * e["entry_id"],
                }
                for e in entries
            ],
        }
    )


def test_one_row_per_prediction_of_the_year_with_its_columns():
    frame = evaluation_frame(seeded(), 2024)

    assert list(frame.columns) == FRAME_COLUMNS
    assert frame[["race_id", "driver"]].values.tolist() == [[11, "NOR"], [11, "VER"], [12, "NOR"], [12, "VER"]]
    row = frame[(frame["round"] == 2) & (frame["driver"] == "NOR")].iloc[0]
    assert (row["race_id"], row["year"], row["model_version_id"]) == (12, 2024, 4)
    assert row["predicted"] == pytest.approx(0.8)  # entry 4
    assert row["baseline"] == pytest.approx(1.2)
    assert row["actual"] == pytest.approx(0.4)
    assert list(frame["round"]) == [1, 1, 2, 2]


def test_rows_whose_ground_truth_is_not_valid_or_has_no_target_are_left_out():
    client = seeded()
    truth = {g["entry_id"]: g for g in client.tables["ground_truth"]}
    truth[1]["is_valid_row"] = False
    truth[2]["actual_pace_delta_pct"] = None
    client.tables["ground_truth"].remove(truth[3])  # no ground truth row at all

    frame = evaluation_frame(client, 2024)

    assert frame["actual"].tolist() == [pytest.approx(0.4)]


def test_rows_with_a_null_baseline_are_left_out():
    client = seeded()
    next(p for p in client.tables["predictions"] if p["entry_id"] == 2)["baseline_pace_delta_pct"] = None

    frame = evaluation_frame(client, 2024)

    assert len(frame) == 3
    assert frame[(frame["race_id"] == 11) & (frame["driver"] == "NOR")].empty


def test_another_year_is_ignored():
    frame = evaluation_frame(seeded(), 2025)

    assert frame["year"].tolist() == [2025] and frame["race_id"].tolist() == [21]
    assert evaluation_frame(seeded(), 2030).empty


def test_it_keeps_one_row_per_model_version_of_the_prediction():
    client = seeded()
    client.tables["predictions"].append(
        {"entry_id": 1, "model_version_id": 7, "predicted_pace_delta_pct": 9.0, "baseline_pace_delta_pct": 0.3}
    )

    frame = evaluation_frame(client, 2024)

    assert sorted(frame["model_version_id"].unique()) == [4, 7]
    assert len(frame[frame["model_version_id"] == 4]) == 4


def test_it_reads_past_the_first_page_of_each_table():
    client = seeded(extra_entries=1100)

    frame = evaluation_frame(client, 2024)

    assert len(frame) == 4 + 1100
    assert any(first >= 1000 for table, first, _ in client.reads if table == "predictions")


def test_nothing_is_written():
    client = seeded()
    evaluation_frame(client, 2024)
    assert client.writes == []


def test_with_nothing_to_score_the_frame_is_empty_with_its_columns():
    frame = evaluation_frame(FakeSupabase(), 2024)
    assert frame.empty and list(frame.columns) == FRAME_COLUMNS
