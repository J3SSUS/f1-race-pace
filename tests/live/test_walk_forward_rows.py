import pandas as pd
import pytest

from live.metrics import FRAME_COLUMNS, SPLIT, walk_forward_rows
from training.metrics import score

CHAMPION, BASELINE = 4, 3


def make_frame(races: dict[int, list[tuple[float, float, float]]]) -> pd.DataFrame:
    """`{round: [(actual, predicted, baseline), ...]}`: race_id is round + 10 and year 2024, all of version 4."""
    rows = [
        {
            "race_id": rnd + 10,
            "year": 2024,
            "round": rnd,
            "driver": f"D{i}",
            "model_version_id": CHAMPION,
            "predicted": predicted,
            "baseline": baseline,
            "actual": actual,
        }
        for rnd, entries in races.items()
        for i, (actual, predicted, baseline) in enumerate(entries)
    ]
    return pd.DataFrame(rows, columns=FRAME_COLUMNS)


def values(rows: list[dict], version: int, race_id: int | None) -> dict[str, float]:
    return {r["metric_name"]: r["value"] for r in rows if r["model_version_id"] == version and r["race_id"] == race_id}


# Round 1: actual 1, 2, 3 / champion 1, 2, 4 / baseline 3, 2, 1.  Round 2: actual 0, 2 / champion 1, 1 / baseline 0, 2.
TWO_RACES = {1: [(1, 1, 3), (2, 2, 2), (3, 4, 1)], 2: [(0, 1, 0), (2, 1, 2)]}


def test_each_model_gets_an_aggregate_and_one_set_per_race_with_the_four_names():
    rows = walk_forward_rows(make_frame(TWO_RACES), CHAMPION, BASELINE)

    assert len(rows) == 2 * 3 * 4  # two models, aggregate + two races, four metrics
    assert {r["split"] for r in rows} == {SPLIT} == {"walk_forward"}
    assert {r["segment_type"] for r in rows} == {"overall"} and {r["segment_value"] for r in rows} == {None}
    for version in (CHAMPION, BASELINE):
        for race_id in (None, 11, 12):
            mine = [r for r in rows if r["model_version_id"] == version and r["race_id"] == race_id]
            assert sorted(r["metric_name"] for r in mine) == ["mae", "pairwise_acc", "r2", "rmse"]


def test_values_derived_by_hand_and_the_aggregate_pools_the_rows():
    rows = walk_forward_rows(make_frame(TWO_RACES), CHAMPION, BASELINE)

    # champion: errors 0, 0, 1 in round 1 and 1, 1 in round 2
    assert values(rows, CHAMPION, 11)["mae"] == pytest.approx(1 / 3)
    assert values(rows, CHAMPION, 12)["mae"] == pytest.approx(1.0)
    assert values(rows, CHAMPION, None)["mae"] == pytest.approx(3 / 5)  # pooled, not (1/3 + 1) / 2
    assert values(rows, CHAMPION, None)["rmse"] == pytest.approx((3 / 5) ** 0.5)
    assert values(rows, CHAMPION, 11)["pairwise_acc"] == pytest.approx(1.0)  # every pair in order
    assert values(rows, CHAMPION, 12)["pairwise_acc"] == pytest.approx(0.5)  # the predictions tie
    assert values(rows, CHAMPION, None)["pairwise_acc"] == pytest.approx(0.75)
    # baseline: errors 2, 0, 2 in round 1 and 0, 0 in round 2
    assert values(rows, BASELINE, 11)["mae"] == pytest.approx(4 / 3)
    assert values(rows, BASELINE, 12)["mae"] == pytest.approx(0.0)
    assert values(rows, BASELINE, None)["mae"] == pytest.approx(4 / 5)
    assert values(rows, BASELINE, 11)["pairwise_acc"] == pytest.approx(0.0)  # reversed
    assert values(rows, BASELINE, None)["pairwise_acc"] == pytest.approx(0.5)


def test_the_baseline_rows_carry_the_baseline_version_and_the_champion_rows_its_own():
    rows = walk_forward_rows(make_frame(TWO_RACES), CHAMPION, BASELINE)

    assert {r["model_version_id"] for r in rows} == {CHAMPION, BASELINE}
    # the baseline column feeds the baseline version, the predicted column the champion
    assert values(rows, BASELINE, None)["mae"] != values(rows, CHAMPION, None)["mae"]


def test_a_race_with_one_row_or_equal_actuals_gets_no_set_but_counts_in_the_aggregate():
    races = TWO_RACES | {3: [(5, 5, 6)], 4: [(1, 2, 1), (1, 0, 1)]}  # one row; two rows with the same actual
    frame = make_frame(races)

    rows = walk_forward_rows(frame, CHAMPION, BASELINE)

    assert {r["race_id"] for r in rows} == {None, 11, 12}  # races 13 and 14 have no set of their own
    assert len(rows) == 24
    # champion errors: round 1 -> 1, round 2 -> 2, round 3 -> 0, round 4 -> 2, over 8 rows
    assert values(rows, CHAMPION, None)["mae"] == pytest.approx(5 / 8)
    assert values(rows, BASELINE, None)["mae"] == pytest.approx((4 + 0 + 1 + 0) / 8)


def test_the_values_equal_score_on_the_same_rows():
    frame = make_frame(TWO_RACES | {3: [(5, 5, 6)], 4: [(1, 2, 1), (1, 0, 1)]})

    rows = walk_forward_rows(frame, CHAMPION, BASELINE)

    for version, column in ((CHAMPION, "predicted"), (BASELINE, "baseline")):
        expected = score(frame["actual"], frame[column], frame)
        got = values(rows, version, None)
        assert got["mae"] == pytest.approx(expected["mae"])
        assert got["rmse"] == pytest.approx(expected["rmse"])
        assert got["r2"] == pytest.approx(expected["r2"])
        assert got["pairwise_acc"] == pytest.approx(expected["pairwise_accuracy"])
        assert "mean_error" not in {r["metric_name"] for r in rows}
        race = frame[frame["race_id"] == 11]
        expected = score(race["actual"], race[column], race)
        assert values(rows, version, 11)["rmse"] == pytest.approx(expected["rmse"])


def test_only_the_rows_of_the_champions_version_are_scored():
    frame = make_frame(TWO_RACES)
    other = frame.assign(model_version_id=9, predicted=99.0)

    rows = walk_forward_rows(pd.concat([frame, other], ignore_index=True), CHAMPION, BASELINE)

    assert values(rows, CHAMPION, None)["mae"] == pytest.approx(3 / 5)


def test_no_rows_for_the_champion_or_no_pair_of_different_actuals_raises():
    with pytest.raises(ValueError, match="no evaluated rows"):
        walk_forward_rows(make_frame({}), CHAMPION, BASELINE)
    with pytest.raises(ValueError, match="different actual"):
        walk_forward_rows(make_frame({1: [(1, 1, 1), (1, 2, 2)]}), CHAMPION, BASELINE)
