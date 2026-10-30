import math

import pandas as pd
import pytest

from training.metrics import METRICS, score

# Race (2025, 1) has 3 drivers and race (2025, 2) has 2: five rows, every value derived by hand.
RACES = pd.DataFrame({"year": [2025] * 5, "round": [1, 1, 1, 2, 2]})
Y_TRUE = [1.0, 2.0, 3.0, 0.0, 1.0]


def test_hand_computed_values():
    # errors (pred - true) = 0, 1, -1, 0.5, -0.5
    pred = [1.0, 3.0, 2.0, 0.5, 0.5]
    result = score(Y_TRUE, pred, RACES)
    assert list(result) == list(METRICS)
    assert result["mae"] == pytest.approx(0.6)
    assert result["rmse"] == pytest.approx(math.sqrt(0.5))
    assert result["mean_error"] == pytest.approx(0.0)
    # SST around the mean 1.4 is 5.2 and SSE is 2.5
    assert result["r2"] == pytest.approx(1 - 2.5 / 5.2)
    # race 1: pairs (1,2) and (1,3) right, (2,3) reversed = 2/3; race 2: predictions tie = 0.5
    assert result["pairwise_accuracy"] == pytest.approx((2 / 3 + 0.5) / 2)


def test_perfect_predictor():
    result = score(Y_TRUE, Y_TRUE, RACES)
    assert result["mae"] == 0
    assert result["pairwise_accuracy"] == 1.0
    assert result["r2"] == 1.0


def test_reversed_order_scores_zero():
    pred = [-v for v in Y_TRUE]
    assert score(Y_TRUE, pred, RACES)["pairwise_accuracy"] == 0.0


def test_constant_prediction_is_exactly_chance():
    result = score(Y_TRUE, [0.7] * 5, RACES)
    assert result["pairwise_accuracy"] == 0.5
    assert result["r2"] <= 0


def test_mean_error_sign_is_pred_minus_true():
    assert score(Y_TRUE, [v + 2 for v in Y_TRUE], RACES)["mean_error"] == pytest.approx(2.0)


def test_pairs_with_the_same_actual_value_are_left_out():
    races = pd.DataFrame({"year": [2025] * 3, "round": [1] * 3})
    # actual 1, 1, 2: only the pairs against the third driver count, and both are ordered right
    assert score([1.0, 1.0, 2.0], [5.0, 0.0, 9.0], races)["pairwise_accuracy"] == 1.0


def test_a_race_with_one_driver_adds_nothing():
    races = pd.DataFrame({"year": [2025] * 3, "round": [1, 1, 2]})
    # race 2 has a single driver; race 1 is reversed, so the mean is 0 (not 0.5 from a phantom race)
    assert score([1.0, 2.0, 5.0], [2.0, 1.0, 5.0], races)["pairwise_accuracy"] == 0.0


def test_same_round_in_different_years_are_different_races():
    races = pd.DataFrame({"year": [2024, 2025], "round": [1, 1]})
    with pytest.raises(ValueError, match="at least one race"):
        score([1.0, 2.0], [1.0, 2.0], races)


def test_no_pairs_at_all_raises():
    races = pd.DataFrame({"year": [2025, 2025], "round": [1, 1]})
    with pytest.raises(ValueError, match="at least one race"):
        score([1.0, 1.0], [0.0, 1.0], races)
