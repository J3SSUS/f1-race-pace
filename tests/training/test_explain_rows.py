import pandas as pd
import pytest
from fake_supabase import FakeSupabase
from test_data import FLAGS, TEST, TRAIN, load, split_of, wide_of

from features.catalog import TARGET_COLUMNS, load_catalog
from features.store import StoreError
from training.evaluation import load_test
from training.explain import load_explain_rows, read_actuals

CATALOG = load_catalog()


def explain_rows(flags=FLAGS, split=None):
    wide = wide_of(flags, test_target=7.0)
    return load_explain_rows(
        "run",
        client=object(),
        reader=lambda client, version: wide,
        splitter=lambda run_id, uri: split if split is not None else split_of(),
    )


def test_valid_rows_of_train_and_test_come_back_labelled_and_without_the_target():
    X, meta = explain_rows(FLAGS | {(2024, 1, "ALO"): False})
    assert len(X) == len(meta) == 6  # the invalid ALO row is out
    assert X.index.equals(meta.index)
    assert list(meta.columns) == ["entry_id", "year", "round", "driver", "split"]
    labelled = dict(zip(zip(meta["year"], meta["round"], strict=True), meta["split"], strict=True))
    assert {key: labelled[key] for key in TRAIN + TEST} == {key: "train" for key in TRAIN} | {
        key: "test" for key in TEST
    }
    assert (meta["split"] == "test").sum() == 2
    assert "ALO" not in set(meta["driver"])


def test_no_returned_frame_has_a_target_column():
    X, meta = explain_rows()
    for frame in (X, meta):
        assert not set(TARGET_COLUMNS) & set(frame.columns)
    assert 7.0 not in X.to_numpy().ravel().tolist()  # the test target, set to 7.0 in the fake wide matrix


def test_features_are_the_catalog_in_order():
    X, _ = explain_rows()
    assert list(X.columns) == CATALOG.names


def test_it_is_the_rows_of_load_train_and_load_test_together():
    X, meta = explain_rows()
    train = load()
    wide = wide_of(FLAGS, test_target=7.0)
    test = load_test("run", client=object(), reader=lambda c, v: wide, splitter=lambda r, u: split_of())
    assert meta.loc[meta["split"] == "train", "entry_id"].tolist() == train.meta["entry_id"].tolist()
    assert meta.loc[meta["split"] == "test", "entry_id"].tolist() == test.meta["entry_id"].tolist()
    pd.testing.assert_frame_equal(X[meta["split"] == "train"].reset_index(drop=True), train.X)
    pd.testing.assert_frame_equal(X[meta["split"] == "test"].reset_index(drop=True), test.X)


def test_an_unknown_run_fails_before_supabase_is_touched():
    def splitter(run_id, uri):
        raise ValueError("unknown run")

    def reader(client, version):
        raise AssertionError("Supabase was touched")

    with pytest.raises(ValueError, match="unknown run"):
        load_explain_rows("nope", client=object(), reader=reader, splitter=splitter)


def test_no_valid_row_in_test_raises():
    flags = {key: valid for key, valid in FLAGS.items() if key[:2] not in TEST} | {(2024, 3, "HAM"): False}
    with pytest.raises(ValueError, match="test races.*no valid row"):
        explain_rows(flags)


def test_no_valid_row_in_train_raises():
    flags = {key: False if key[:2] in TRAIN else valid for key, valid in FLAGS.items()}
    with pytest.raises(ValueError, match="train races.*no valid row"):
        explain_rows(flags)


def test_a_race_outside_the_split_is_left_out():
    X, meta = explain_rows(FLAGS | {(2025, 1, "HAM"): True})
    assert 2025 not in set(meta["year"])


def ground_truth() -> FakeSupabase:
    return FakeSupabase(
        {
            "ground_truth": [
                {"entry_id": 1, "actual_pace_delta_pct": 0.5},
                {"entry_id": 2, "actual_pace_delta_pct": -0.25},
                {"entry_id": 3, "actual_pace_delta_pct": 1.5},
                {"entry_id": 4, "actual_pace_delta_pct": None},  # an invalid row: no actual value
            ]
        }
    )


def test_read_actuals_returns_only_the_entries_it_is_asked_for():
    client = ground_truth()
    assert read_actuals(client, [3, 1]) == {1: 0.5, 3: 1.5}
    assert client.reads == [("ground_truth", 0, 2)]  # Supabase sent two rows, not the table


def test_read_actuals_asks_the_server_for_those_entries_only():
    client = ground_truth()
    read_actuals(client, [2])
    assert client.reads == [("ground_truth", 0, 1)]


def test_a_repeated_entry_is_read_once():
    assert read_actuals(ground_truth(), [2, 2, 2]) == {2: -0.25}


def test_no_entries_reads_nothing():
    client = ground_truth()
    assert read_actuals(client, []) == {}
    assert client.reads == []


def test_an_entry_that_does_not_exist_raises():
    with pytest.raises(ValueError, match=r"\[99\]"):
        read_actuals(ground_truth(), [1, 99])


def test_an_entry_without_an_actual_value_raises():
    with pytest.raises(ValueError, match=r"\[4\]"):
        read_actuals(ground_truth(), [4])


def test_a_supabase_error_is_reported_by_type_only():
    class Broken:
        def table(self, name):
            raise ConnectionError("secret-service-key")

    with pytest.raises(StoreError, match="ConnectionError") as error:
        read_actuals(Broken(), [1])
    assert "secret-service-key" not in str(error.value)


def test_load_train_and_load_test_are_unchanged():
    train = load()
    assert not (train.y == 999.0).any()  # a test target never reaches load_train
    assert set(zip(train.meta["year"], train.meta["round"], strict=True)) == set(TRAIN)
