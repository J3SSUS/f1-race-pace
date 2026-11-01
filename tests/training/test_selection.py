import numpy as np
import pandas as pd
import pytest

from training.selection import BASELINE, SIMPLICITY, select

N_RACES, DRIVERS = 8, 4
YEAR = np.repeat(2024, N_RACES * DRIVERS)
ROUND = np.repeat(np.arange(1, N_RACES + 1), DRIVERS)


def make_oof(errors: dict[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    """OOF frames with y_true = 0, so |y_pred| is the error of each row."""
    n = N_RACES * DRIVERS
    return {
        name: pd.DataFrame(
            {"entry_id": np.arange(n), "year": YEAR, "round": ROUND, "driver": 1, "fold": 1, "y_true": 0.0, "y_pred": e}
        )
        for name, e in errors.items()
    }


def flat(value: float) -> np.ndarray:
    return np.full(N_RACES * DRIVERS, value)


def all_equal(**overrides: np.ndarray) -> dict[str, np.ndarray]:
    errors = {name: flat(0.4) for name in (BASELINE, *SIMPLICITY)}
    errors[BASELINE] = flat(0.6)
    return errors | overrides


def test_reference_is_best_of_the_six_not_the_baseline():
    # the baseline is the best in the table, but it must not be the reference
    errors = all_equal(catboost=flat(0.3))
    errors[BASELINE] = flat(0.1)
    result = select(make_oof(errors))
    assert result.reference == "catboost"
    assert [row.name for row in result.table] == list(SIMPLICITY)


def test_a_tie_goes_to_the_simplest_model():
    result = select(make_oof(all_equal()))  # all six identical: the first in SIMPLICITY wins
    assert result.candidate == "ridge"
    assert all(row.not_worse for row in result.table)


def test_simpler_model_within_the_noise_beats_the_reference():
    # catboost is the best (0.399) but ridge, 0.001 worse on average, wins and loses by race: the interval of its
    # difference includes 0, so the simplest model is the candidate
    wiggle = np.repeat(np.tile([0.01, -0.01], N_RACES // 2), DRIVERS)
    errors = all_equal(ridge=flat(0.4) + wiggle, catboost=flat(0.399))
    result = select(make_oof(errors))
    by_name = {row.name: row for row in result.table}
    assert result.reference == "catboost"
    assert by_name["ridge"].diff == pytest.approx(0.001) and by_name["ridge"].ci_low < 0 < by_name["ridge"].ci_high
    assert result.candidate == "ridge"


def test_significantly_worse_model_is_never_the_candidate():
    # every model but catboost is worse in every race, so only the reference is not significantly worse
    errors = all_equal(ridge=flat(0.5), elasticnet=flat(0.5), catboost=flat(0.3))
    result = select(make_oof(errors))
    by_name = {row.name: row for row in result.table}
    assert result.reference == "catboost"
    assert not by_name["ridge"].not_worse and by_name["ridge"].ci_low > 0
    assert [row.not_worse for row in result.table] == [False] * 5 + [True]
    assert result.candidate == "catboost"


def test_beats_baseline_when_candidate_is_better_in_every_race():
    result = select(make_oof(all_equal()))
    assert result.beats_baseline
    assert result.vs_baseline.name == result.candidate
    assert result.vs_baseline.diff == pytest.approx(-0.2)
    assert result.vs_baseline.ci_high < 0


def test_does_not_beat_baseline_and_no_error():
    errors = all_equal()
    errors[BASELINE] = flat(0.4)  # same as every model
    result = select(make_oof(errors))
    assert not result.beats_baseline
    assert result.candidate == "ridge"  # stays as selected
    assert result.vs_baseline.diff == 0


def test_rows_may_come_in_another_order():
    errors = all_equal(catboost=flat(0.3))
    oof = make_oof(errors)
    oof["xgboost"] = oof["xgboost"].sample(frac=1, random_state=0)
    assert select(make_oof(errors)) == select(oof)


def test_different_entry_ids_raise_naming_the_model():
    oof = make_oof(all_equal())
    oof["lightgbm"] = oof["lightgbm"].assign(entry_id=lambda f: f["entry_id"] + 1)
    with pytest.raises(ValueError, match="lightgbm"):
        select(oof)


def test_missing_model_raises_naming_it():
    oof = make_oof(all_equal())
    del oof["xgboost"]
    with pytest.raises(ValueError, match="xgboost"):
        select(oof)


def test_repeated_entry_ids_raise():
    oof = make_oof(all_equal())
    oof["ridge"] = pd.concat([oof["ridge"], oof["ridge"].iloc[:1]])
    with pytest.raises(ValueError, match="ridge"):
        select(oof)


def test_same_seed_same_selection():
    rng = np.random.default_rng(3)
    errors = {name: np.abs(rng.normal(0.4, 0.1, N_RACES * DRIVERS)) for name in (BASELINE, *SIMPLICITY)}
    assert select(make_oof(errors)) == select(make_oof(errors))
