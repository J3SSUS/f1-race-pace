"""The promotion rule of SPEC 25 on synthetic per-race MAEs, with every verdict worked out by hand."""

import pytest

from training.promotion import PROMOTION_RATIO, PROMOTION_WINDOW, evaluate_rule

ORDER = {race: (2026, race) for race in range(1, 9)}  # race id == round, all in 2026


def verdict(champion: list[float], challenger: list[float], first_race: int = 1, order=ORDER) -> dict:
    """Both models on consecutive races starting at `first_race`."""
    return evaluate_rule(
        {first_race + i: v for i, v in enumerate(champion)},
        {first_race + i: v for i, v in enumerate(challenger)},
        order,
    )


def test_the_constants_are_the_rule_of_claude_md():
    assert PROMOTION_WINDOW == 5 and PROMOTION_RATIO == 0.98


def test_with_fewer_than_five_common_races_it_does_not_promote_and_says_how_many():
    out = verdict([1.0] * 4, [0.1] * 4)
    assert out["passes"] is False and out["common_races"] == 4 and out["window"] == []
    assert "only 4 common races" in out["reason"] and "1 more" in out["reason"]
    assert out["challenger_mae"] is None and out["improvement_pct"] is None


def test_a_mean_exactly_at_098_times_the_champions_promotes():
    out = verdict([1.0] * 5, [0.98] * 5)
    assert out["passes"] is True and out["reason"] is None
    assert out["improvement_pct"] == pytest.approx(2.0)


def test_a_mean_just_above_the_boundary_does_not_promote():
    out = verdict([1.0] * 5, [0.9801] * 5)
    assert out["passes"] is False and "mean MAE" in out["reason"] and "worse" not in out["reason"]


def test_a_good_mean_but_worse_in_three_races_does_not_promote():
    out = verdict([1.0] * 5, [1.1, 1.1, 1.1, 0.1, 0.1])  # mean 0.7, worse in 3
    assert out["passes"] is False and out["races_worse"] == 3
    assert "worse than the champion in 3 of 5" in out["reason"] and "mean MAE" not in out["reason"]
    assert out["challenger_mae"] == pytest.approx(0.7)


def test_worse_in_two_races_still_promotes():
    out = verdict([1.0] * 5, [1.1, 1.1, 0.1, 0.1, 0.1])  # mean 0.5, worse in 2
    assert out["passes"] is True and out["races_worse"] == 2


def test_a_tie_in_a_race_is_not_worse():
    out = verdict([1.0] * 5, [1.0, 1.0, 1.0, 0.1, 0.1])  # three ties: with ">=" it would be worse in 3
    assert out["passes"] is True and out["races_worse"] == 0


def test_both_reasons_are_reported_together():
    out = verdict([1.0] * 5, [1.5, 1.5, 1.5, 0.9, 0.9])  # mean 1.26 is above 0.98 and it is worse in 3 races
    assert out["passes"] is False
    assert "mean MAE" in out["reason"] and "worse than the champion in 3 of 5" in out["reason"]


def test_only_the_last_five_common_races_count():
    champion = [1.0] * 8
    challenger = [9.0, 9.0, 9.0, 0.5, 0.5, 0.5, 0.5, 0.5]  # the three old races are terrible, the last five good
    out = verdict(champion, challenger)
    assert out["passes"] is True and out["common_races"] == 8 and out["window"] == [4, 5, 6, 7, 8]
    assert out["challenger_mae"] == pytest.approx(0.5)


def test_only_races_both_predicted_count():
    champion = {race: 1.0 for race in range(1, 9)}
    challenger = {race: 0.5 for race in range(4, 9)}  # the challenger only has the races after its cutoff
    out = evaluate_rule(champion, challenger, ORDER)
    assert out["common_races"] == 5 and out["window"] == [4, 5, 6, 7, 8] and out["passes"] is True
    short = evaluate_rule(champion, {race: 0.5 for race in range(5, 9)}, ORDER)
    assert short["common_races"] == 4 and short["passes"] is False


def test_the_order_is_year_and_round_not_the_race_id():
    order = {10: (2026, 5), 9: (2026, 4), 8: (2026, 3), 7: (2026, 2), 6: (2026, 1), 5: (2025, 24), 4: (2025, 23)}
    champion = {race: 1.0 for race in order}
    challenger = {4: 9.0, 5: 9.0, 6: 0.5, 7: 0.5, 8: 0.5, 9: 0.5, 10: 0.5}  # the two 2025 races are the bad ones
    out = evaluate_rule(champion, challenger, order)
    assert out["window"] == [6, 7, 8, 9, 10] and out["passes"] is True


def test_a_race_the_races_table_does_not_know_is_ignored():
    out = evaluate_rule(
        {r: 1.0 for r in range(1, 7)}, {r: 0.5 for r in range(1, 7)}, {r: (2026, r) for r in range(1, 5)}
    )
    assert out["common_races"] == 4 and out["passes"] is False


def test_the_per_race_numbers_are_returned_oldest_first():
    out = verdict([1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 1.0, 1.5, 2.0, 2.5])
    assert out["per_race"] == [(1, 1.0, 0.5), (2, 2.0, 1.0), (3, 3.0, 1.5), (4, 4.0, 2.0), (5, 5.0, 2.5)]


def test_a_champion_with_a_zero_mean_does_not_divide_by_zero():
    out = verdict([0.0] * 5, [0.0] * 5)
    assert out["passes"] is True and out["improvement_pct"] == 0.0
