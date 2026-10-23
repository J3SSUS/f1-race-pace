import numpy as np
import pandas as pd
import pytest

from features import RaceInfo
from features.form import HISTORY_COLUMNS, empty_history, form_features

NAN = np.nan
ENTRIES = pd.DataFrame({"driver": ["AAA", "BBB", "CCC", "DDD"], "team": ["Alpha", "Alpha", "Beta", "Gamma"]})


def history(*rows) -> pd.DataFrame:
    """Rows are (year, round, circuit, driver, team, target)."""
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def race(year, round, circuit="Circ5") -> RaceInfo:
    return RaceInfo(year=year, round=round, circuit=circuit, circuit_type="permanent")


def forms(race_info, hist, entries=ENTRIES) -> pd.DataFrame:
    return form_features(race_info, entries, hist).set_index("driver")


# 2024 rounds 1 to 4 and the same circuit in 2023. BBB has no valid row in round 4 (it is not in the history).
HISTORY = history(
    (2023, 5, "Circ5", "AAA", "Alpha", 1.0),
    (2023, 5, "Circ5", "BBB", "Alpha", 3.0),
    (2023, 5, "Circ5", "CCC", "Beta", 4.0),
    (2024, 1, "Circ1", "AAA", "Alpha", -1.0),
    (2024, 1, "Circ1", "BBB", "Alpha", 0.5),
    (2024, 1, "Circ1", "CCC", "Beta", 2.0),
    (2024, 2, "Circ2", "AAA", "Alpha", -2.0),
    (2024, 2, "Circ2", "BBB", "Alpha", 1.0),
    (2024, 2, "Circ2", "CCC", "Beta", 3.0),
    (2024, 3, "Circ3", "AAA", "Alpha", -3.0),
    (2024, 3, "Circ3", "BBB", "Alpha", 1.5),
    (2024, 3, "Circ3", "CCC", "Beta", 4.0),
    (2024, 4, "Circ4", "AAA", "Alpha", -4.0),
    (2024, 4, "Circ4", "CCC", "Beta", 5.0),
)


def test_form_features_match_the_hand_derived_values():
    result = forms(race(2024, 5), HISTORY)
    # driver: the last 3 rows of each driver
    assert result.loc["AAA", "driver_pace_last3"] == pytest.approx((-2 - 3 - 4) / 3, abs=1e-9)  # rounds 2 to 4
    assert result.loc["BBB", "driver_pace_last3"] == pytest.approx((0.5 + 1.0 + 1.5) / 3, abs=1e-9)  # rounds 1 to 3
    assert result.loc["CCC", "driver_pace_last3"] == pytest.approx((3 + 4 + 5) / 3, abs=1e-9)
    # team: the mean of the last 3 race means. Alpha: round 2 -0.5, round 3 -0.75, round 4 -4 (only AAA is valid)
    assert result.loc["AAA", "team_pace_last3"] == pytest.approx((-0.5 - 0.75 - 4.0) / 3, abs=1e-9)
    assert result.loc["BBB", "team_pace_last3"] == result.loc["AAA", "team_pace_last3"]
    assert result.loc["CCC", "team_pace_last3"] == pytest.approx(4.0, abs=1e-9)
    # circuit the year before (2023 round 5): Alpha (1 + 3) / 2, Beta 4; Gamma was not there
    assert result.loc["AAA", "team_pace_this_circuit_prev"] == pytest.approx(2.0, abs=1e-9)
    assert result.loc["CCC", "team_pace_this_circuit_prev"] == pytest.approx(4.0, abs=1e-9)
    assert np.isnan(result.loc["DDD", "team_pace_this_circuit_prev"])


def test_a_driver_and_team_without_history_get_nan():
    result = forms(race(2024, 5), HISTORY)
    assert result.loc["DDD", ["driver_pace_last3", "team_pace_last3"]].isna().all()


def test_the_first_race_has_no_form():
    result = forms(race(2023, 1), empty_history())
    assert result.isna().all().all()
    assert list(result.columns) == ["driver_pace_last3", "team_pace_last3", "team_pace_this_circuit_prev"]


def test_races_after_or_equal_to_this_one_are_ignored():
    later = history(
        (2024, 5, "Circ5", "AAA", "Alpha", 50.0),  # this very race
        (2024, 6, "Circ6", "AAA", "Alpha", 60.0),
        (2025, 1, "Circ1", "AAA", "Alpha", 70.0),
    )
    with_future = pd.concat([HISTORY, later], ignore_index=True)
    pd.testing.assert_frame_equal(forms(race(2024, 5), with_future), forms(race(2024, 5), HISTORY))


def test_the_order_of_the_history_rows_does_not_matter():
    shuffled = HISTORY.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(forms(race(2024, 5), shuffled), forms(race(2024, 5), HISTORY))


def test_rows_without_a_target_are_ignored():
    invalid = history((2024, 4, "Circ4", "BBB", "Alpha", NAN))
    with_invalid = pd.concat([HISTORY, invalid], ignore_index=True)
    pd.testing.assert_frame_equal(forms(race(2024, 5), with_invalid), forms(race(2024, 5), HISTORY))


def test_form_crosses_seasons_within_the_same_regulations():
    hist = history(
        (2023, 20, "C20", "AAA", "Alpha", 1.0),
        (2023, 21, "C21", "AAA", "Alpha", 2.0),
        (2023, 22, "C22", "AAA", "Alpha", 3.0),
    )
    result = forms(race(2024, 1), hist, ENTRIES.iloc[[0]])
    assert result.loc["AAA", "driver_pace_last3"] == pytest.approx(2.0, abs=1e-9)


def test_a_driver_who_changes_team_keeps_his_form_but_not_the_old_teams():
    hist = history(
        (2024, 1, "C1", "AAA", "Alpha", 1.0),
        (2024, 2, "C2", "AAA", "Alpha", 2.0),
        (2024, 3, "C3", "CCC", "Beta", 10.0),
    )
    entries = pd.DataFrame({"driver": ["AAA"], "team": ["Beta"]})
    result = forms(race(2024, 4), hist, entries)
    assert result.loc["AAA", "driver_pace_last3"] == pytest.approx(1.5, abs=1e-9)
    assert result.loc["AAA", "team_pace_last3"] == pytest.approx(10.0, abs=1e-9)  # Beta's only race


# 2025 history and the first two races of 2026
REGULATION = history(
    (2025, 22, "Circ5", "AAA", "Alpha", 5.0),
    (2025, 23, "Circ6", "AAA", "Alpha", 6.0),
    (2025, 24, "Circ7", "AAA", "Alpha", 7.0),
)


def test_the_first_2026_race_has_no_form_and_the_second_uses_only_the_first():
    first = forms(race(2026, 1), REGULATION, ENTRIES.iloc[[0]])
    assert first.isna().all().all()
    with_first = pd.concat([REGULATION, history((2026, 1, "Circ1", "AAA", "Alpha", -1.0))], ignore_index=True)
    second = forms(race(2026, 2), with_first, ENTRIES.iloc[[0]])
    assert second.loc["AAA", "driver_pace_last3"] == pytest.approx(-1.0, abs=1e-9)
    assert second.loc["AAA", "team_pace_last3"] == pytest.approx(-1.0, abs=1e-9)


def test_the_circuit_of_the_year_before_is_empty_for_every_2026_row():
    # Circ5 was raced in 2025 by Alpha, but 2026 has other regulations
    result = forms(race(2026, 9, "Circ5"), REGULATION, ENTRIES.iloc[[0]])
    assert np.isnan(result.loc["AAA", "team_pace_this_circuit_prev"])


def test_from_2027_the_circuit_of_2026_is_available():
    hist = history((2026, 5, "Circ5", "AAA", "Alpha", 2.5), (2026, 6, "Circ6", "AAA", "Alpha", 3.5))
    result = forms(race(2027, 5, "Circ5"), hist, ENTRIES.iloc[[0]])
    assert result.loc["AAA", "team_pace_this_circuit_prev"] == pytest.approx(2.5, abs=1e-9)
    assert result.loc["AAA", "driver_pace_last3"] == pytest.approx(3.0, abs=1e-9)
