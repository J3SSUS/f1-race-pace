"""The window of a drift report, the MAE change and the rule of SPEC 26, with every value worked out by hand."""

import pytest

from monitoring.drift import (
    DRIFT_WINDOW,
    MAE_CHANGE_PCT,
    MIN_DRIFTED_FEATURES,
    P_VALUE,
    mae_change_pct,
    rule_met,
    select_window,
    start_race,
)

# race id == round for 2026 rounds 1..12, so "later" is easy to read.
ORDER = {race: (2026, race) for race in range(1, 13)}
# Ids that are NOT chronological (as in the real table): id 90 is round 1 of 2026, id 5 is round 2.
SCRAMBLED = {90: (2026, 1), 5: (2026, 2), 70: (2026, 3), 12: (2026, 4), 44: (2026, 5), 8: (2026, 6), 31: (2026, 7)}


def test_the_constants_are_the_decisions_of_spec_26():
    assert (DRIFT_WINDOW, MAE_CHANGE_PCT, MIN_DRIFTED_FEATURES, P_VALUE) == (5, 15.0, 3, 0.05)


def test_four_waiting_races_are_not_due_and_say_how_many():
    window = select_window([1, 2, 3, 4], ORDER)
    assert window.pending == 4 and window.races == [] and window.due is False


def test_exactly_five_waiting_races_are_due_and_all_of_them_are_the_window():
    window = select_window([5, 3, 1, 4, 2], ORDER)
    assert window.pending == 5 and window.races == [1, 2, 3, 4, 5] and window.due is True


def test_with_more_than_five_the_window_is_the_last_five_and_pending_counts_all():
    window = select_window(range(1, 9), ORDER)
    assert window.pending == 8 and window.races == [4, 5, 6, 7, 8]


def test_the_window_is_ordered_by_year_and_round_not_by_race_id():
    window = select_window(SCRAMBLED, SCRAMBLED)
    assert window.races == [70, 12, 44, 8, 31]  # rounds 3 to 7; the two lowest rounds (ids 90 and 5) drop out


def test_only_races_after_the_start_race_count():
    window = select_window(range(1, 13), ORDER, after=5)
    assert window.pending == 7 and window.races == [8, 9, 10, 11, 12]
    assert select_window(range(1, 11), ORDER, after=5).races == [6, 7, 8, 9, 10]


def test_the_start_race_itself_does_not_count_and_a_year_change_is_respected():
    order = {1: (2025, 24), 2: (2026, 1), 3: (2026, 2), 4: (2026, 3), 5: (2026, 4), 6: (2026, 5)}
    window = select_window(order, order, after=1)
    assert window.pending == 5 and window.races == [2, 3, 4, 5, 6]
    assert select_window(order, order, after=6).pending == 0


def test_after_by_order_not_by_id():
    # `after` is id 5 (round 2): races 70, 12, 44, 8, 31 (rounds 3 to 7) are after it; id 90 (round 1) is not.
    window = select_window(SCRAMBLED, SCRAMBLED, after=5)
    assert window.pending == 5 and window.races == [70, 12, 44, 8, 31]


def test_an_unknown_race_is_an_error_not_a_silent_drop():
    with pytest.raises(ValueError, match="not in the calendar"):
        select_window([1, 2, 99], ORDER)
    with pytest.raises(ValueError, match="not in the calendar"):
        select_window([1, 2], ORDER, after=99)


def test_start_race_is_the_later_of_the_base_and_the_last_report():
    assert start_race(3, 8, ORDER) == 8  # a report after the base: it counts from the report
    assert start_race(8, 3, ORDER) == 8  # a report before the base (the champion changed): from the base
    assert start_race(3, None, ORDER) == 3 and start_race(None, 8, ORDER) == 8
    assert start_race(None, None, ORDER) is None


def test_start_race_compares_by_year_and_round():
    assert start_race(90, 5, SCRAMBLED) == 5  # id 5 is round 2, later than id 90 (round 1) although smaller
    with pytest.raises(ValueError, match="not in the calendar"):
        start_race(99, None, ORDER)


def test_a_refit_champion_counts_after_its_reference_window():
    # Refit promoted after round 8 with a reference window of rounds 4 to 8; round 9 onwards is new.
    assert select_window(range(1, 13), ORDER, after=start_race(8, None, ORDER)).pending == 4


def test_mae_change_is_the_ratio_minus_one_in_percent():
    assert mae_change_pct(0.5, 0.4) == pytest.approx(25.0)
    assert mae_change_pct(0.4, 0.4) == 0.0
    assert mae_change_pct(0.3, 0.4) == pytest.approx(-25.0)


def test_a_zero_reference_has_no_change():
    with pytest.raises(ValueError, match="undefined"):
        mae_change_pct(0.5, 0.0)


@pytest.mark.parametrize(
    ("change", "drifted", "expected"),
    [
        (14.9, 2, False),  # both just below
        (15.0, 0, True),  # the MAE edge is inclusive
        (14.9, 3, True),  # the feature edge is inclusive
        (-30.0, 3, True),  # the MAE improved a lot but three features drifted
        (40.0, 0, True),  # no drifted feature but the MAE rose a lot
        (0.0, 0, False),
    ],
)
def test_the_rule_is_an_or_with_inclusive_edges(change, drifted, expected):
    assert rule_met(change, drifted) is expected


def test_a_change_of_exactly_15_percent_holds_despite_float_rounding():
    change = mae_change_pct(0.46, 0.4)  # 14.999999999999996 in floats
    assert change < 15.0
    assert rule_met(change, 0) is True
