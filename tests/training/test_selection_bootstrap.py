import numpy as np
import pytest

from training.selection import paired_block_bootstrap

# 6 races of 3 drivers; the error of every row of race r is hand-made
RACES = np.repeat(np.arange(6), 3)


def test_identical_models_have_zero_difference_and_interval_around_zero():
    err = np.array([0.1, 0.5, 0.3] * 6)
    diff, low, high = paired_block_bootstrap(err, err.copy(), RACES)
    assert diff == 0.0
    assert low <= 0 <= high


def test_better_model_in_every_race_has_negative_interval():
    err_b = np.full(18, 0.5)
    err_a = err_b - np.repeat([0.10, 0.12, 0.15, 0.11, 0.13, 0.14], 3)  # 0.10 to 0.15 better in every race
    diff, low, high = paired_block_bootstrap(err_a, err_b, RACES)
    assert diff == pytest.approx(-0.125)
    assert high < 0
    assert low <= diff <= high


def test_worse_model_has_positive_difference():
    err_b = np.full(18, 0.2)
    diff, low, _ = paired_block_bootstrap(err_b + 0.1, err_b, RACES)
    assert diff == pytest.approx(0.1)
    assert low > 0


def test_difference_uses_absolute_errors():
    diff, _, _ = paired_block_bootstrap(np.full(18, -0.4), np.full(18, 0.1), RACES)
    assert diff == pytest.approx(0.3)


def test_same_seed_same_result_and_other_seed_differs():
    rng = np.random.default_rng(0)
    err_a, err_b = rng.random(18), rng.random(18)
    first = paired_block_bootstrap(err_a, err_b, RACES, seed=7)
    assert paired_block_bootstrap(err_a, err_b, RACES, seed=7) == first
    other = paired_block_bootstrap(err_a, err_b, RACES, seed=8)
    assert other[0] == first[0] and other[1:] != first[1:]


def test_rows_of_a_race_are_resampled_together():
    # Race 0 has a difference of +1 on all its rows, race 1 of -1: any resample is a mix of whole races, so the
    # bootstrap mean is always k/n_races*(+1) + (1-k/n_races)*(-1). With races of different sizes the weights
    # follow the rows of each race, not the races.
    races = np.array([0, 0, 0, 0, 1])
    err_b = np.zeros(5)
    err_a = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    diff, low, high = paired_block_bootstrap(err_a, err_b, races, n_boot=200)
    assert (diff, low, high) == (1.0, 1.0, 1.0)  # every resample has only rows with difference 1

    err_a = np.array([1.0, 1.0, 1.0, 1.0, -1.0])  # |err| is 1 everywhere: no difference to find
    assert paired_block_bootstrap(err_a, err_b, races)[0] == 1.0

    # races 0 (4 rows, +2) and 1 (1 row, -1): resample means can only be 2, -1 or (8-1)/5 = 1.4
    err_a = np.array([2.0, 2.0, 2.0, 2.0, 1.0])
    err_b = np.array([0.0, 0.0, 0.0, 0.0, 2.0])
    diff, low, high = paired_block_bootstrap(err_a, err_b, races, n_boot=500)
    assert diff == pytest.approx(1.4)
    assert -1.0 <= low <= high <= 2.0
    assert high == pytest.approx(2.0)


def test_different_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        paired_block_bootstrap(np.ones(18), np.ones(17), RACES)
    with pytest.raises(ValueError, match="same length"):
        paired_block_bootstrap(np.ones(18), np.ones(18), RACES[:-1])


def test_no_rows_raises():
    with pytest.raises(ValueError, match="at least one row"):
        paired_block_bootstrap([], [], [])


def test_races_must_be_one_dimensional():
    with pytest.raises(ValueError, match="one-dimensional"):
        paired_block_bootstrap(np.ones(4), np.ones(4), np.ones((2, 2)))
