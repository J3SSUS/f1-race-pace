import pandas as pd
import pytest
from fake_supabase import FakeSupabase

from features.store import StoreError
from training.registry import (
    METRIC_NAMES,
    feature_set_id_of,
    last_train_race_id,
    metric_rows,
    model_version_row,
)

CV = {"mae": 1.5, "rmse": 2.5, "r2": 0.1, "pairwise_accuracy": 0.6, "mean_error": -0.2}


def split_frame(rows):
    return pd.DataFrame(rows, columns=["year", "round", "split"])


def test_model_version_row_has_the_five_columns_and_is_not_shadow():
    row = model_version_row("ridge", "abc", 3, 7)
    assert row == {
        "mlflow_run_id": "abc",
        "algorithm": "ridge",
        "feature_set_id": 3,
        "trained_through_race_id": 7,
        "is_shadow": False,
    }


def test_metric_rows_are_four_train_overall_rows_with_schema_names():
    rows = metric_rows(9, CV)
    assert [r["metric_name"] for r in rows] == ["mae", "rmse", "r2", "pairwise_acc"]
    assert {r["split"] for r in rows} == {"train"}
    assert {r["segment_type"] for r in rows} == {"overall"}
    assert all(r["race_id"] is None and r["segment_value"] is None for r in rows)
    assert all(r["model_version_id"] == 9 for r in rows)
    assert {r["metric_name"]: r["value"] for r in rows} == {
        "mae": 1.5,
        "rmse": 2.5,
        "r2": 0.1,
        "pairwise_acc": 0.6,
    }


def test_mean_error_is_not_written():
    assert "mean_error" not in METRIC_NAMES
    assert "mean_error" not in {r["metric_name"] for r in metric_rows(1, CV)}


def test_metric_rows_name_the_missing_metric():
    with pytest.raises(ValueError, match="r2"):
        metric_rows(1, {"mae": 1.0, "rmse": 2.0, "pairwise_accuracy": 0.5})


def test_feature_set_id_is_the_one_of_the_version():
    client = FakeSupabase(
        {"feature_sets": [{"feature_set_id": 1, "version": "v0"}, {"feature_set_id": 2, "version": "v1"}]}
    )
    assert feature_set_id_of(client, "v1") == 2


def test_missing_feature_set_raises_with_a_clear_message():
    client = FakeSupabase({"feature_sets": [{"feature_set_id": 1, "version": "v0"}]})
    with pytest.raises(StoreError, match="feature set v1 does not exist"):
        feature_set_id_of(client, "v1")


def test_last_train_race_whatever_the_order_of_the_rows():
    client = FakeSupabase(
        {
            "races": [
                {"race_id": 30, "year": 2024, "round": 2},
                {"race_id": 10, "year": 2023, "round": 22},
                {"race_id": 20, "year": 2024, "round": 1},
                {"race_id": 40, "year": 2024, "round": 3},
            ]
        }
    )
    split = split_frame([(2024, 3, "test"), (2024, 2, "train"), (2023, 22, "train"), (2024, 1, "train")])
    assert last_train_race_id(client, split) == 30


def test_missing_race_raises_with_a_clear_message():
    client = FakeSupabase({"races": [{"race_id": 1, "year": 2023, "round": 1}]})
    with pytest.raises(StoreError, match="2024 round 5 is not in races"):
        last_train_race_id(client, split_frame([(2023, 1, "train"), (2024, 5, "train")]))


def test_split_without_train_races_raises():
    with pytest.raises(StoreError, match="no train race"):
        last_train_race_id(FakeSupabase({"races": []}), split_frame([(2024, 1, "test")]))
