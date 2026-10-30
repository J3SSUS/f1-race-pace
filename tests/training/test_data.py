import numpy as np
import pandas as pd
import pytest

from features.catalog import apply_dtypes, load_catalog
from training.data import TrainSet, load_train

CATALOG = load_catalog()
TRAIN = [(2024, 1), (2024, 2)]
TEST = [(2024, 3)]


def split_of(train=TRAIN, test=TEST):
    rows = [(y, r, "train") for y, r in train] + [(y, r, "test") for y, r in test]
    return pd.DataFrame(rows, columns=["year", "round", "split"])


def wide_of(valid_flags: dict[tuple[int, int, str], bool], test_target: float = 999.0) -> pd.DataFrame:
    """One row per (year, round, driver) in the flags, shaped like `read_wide`. The target is 1, 2, 3...; test rows
    get `test_target`, so any leak of a test target is visible."""
    rows = []
    for i, ((year, round, driver), valid) in enumerate(valid_flags.items()):
        row = {
            "entry_id": i + 1,
            "year": year,
            "round": round,
            "circuit": "Monza",
            "driver": driver,
            "team": "Ferrari",
            "race_pace_delta_pct": float(i + 1) if valid else np.nan,
            "clean_laps": 20 if valid else 3,
            "is_valid_row": valid,
        }
        for feature in CATALOG.features:
            row[feature.name] = feature.allowed[0] if feature.dtype == "category" else float(i)
        if (year, round) in TEST:
            row["race_pace_delta_pct"] = test_target
        rows.append(row)
    return apply_dtypes(pd.DataFrame(rows), CATALOG).astype({"is_valid_row": "bool", "clean_laps": "int64"})


FLAGS = {
    (2024, 1, "HAM"): True,
    (2024, 1, "VER"): True,
    (2024, 1, "ALO"): False,
    (2024, 2, "HAM"): True,
    (2024, 2, "VER"): True,
    (2024, 3, "HAM"): True,
    (2024, 3, "VER"): True,
}


def load(flags=FLAGS, split=None) -> TrainSet:
    wide = wide_of(flags)
    return load_train(
        "run",
        client=object(),
        reader=lambda client, version: wide,
        splitter=lambda run_id, uri: split if split is not None else split_of(),
    )


def test_only_valid_rows_of_train_races_come_back():
    train = load()
    assert set(zip(train.meta["year"], train.meta["round"], strict=True)) == set(TRAIN)
    assert len(train.X) == len(train.y) == len(train.meta) == 4  # ALO is invalid and race 3 is test
    assert "ALO" not in set(train.meta["driver"])


def test_no_test_target_reaches_the_train_set():
    train = load()
    assert not (train.y == 999.0).any()
    assert train.y.tolist() == [1.0, 2.0, 4.0, 5.0]


def test_features_are_the_catalog_in_order_with_their_dtypes():
    train = load()
    assert list(train.X.columns) == CATALOG.names
    assert len(train.X.columns) == 15
    for feature in CATALOG.features:
        assert str(train.X[feature.name].dtype) == feature.pandas_dtype
    assert train.X["circuit_type"].dtype == object  # text: encoded inside each model


def test_y_has_no_nan_and_is_float64():
    train = load()
    assert train.y.dtype == "float64"
    assert not train.y.isna().any()


def test_meta_is_sorted_and_aligned_with_x_and_y():
    train = load()
    assert list(train.meta.columns) == ["entry_id", "year", "round", "driver"]
    assert train.X.index.equals(train.y.index) and train.X.index.equals(train.meta.index)
    keys = list(zip(train.meta["year"], train.meta["round"], train.meta["driver"], strict=True))
    assert keys == sorted(keys)
    assert train.meta["entry_id"].tolist() == [1, 2, 4, 5]  # entry 3 is the invalid ALO


def test_train_races_without_a_valid_row_raise():
    flags = {(2024, 1, "HAM"): False, (2024, 2, "HAM"): False, (2024, 3, "HAM"): True}
    with pytest.raises(ValueError, match="no valid row"):
        load(flags)


def test_unknown_run_id_raises_before_supabase_is_read(uri):
    def reader(client, version):
        raise AssertionError("Supabase must not be read for an unknown run")

    with pytest.raises(ValueError, match="No MLflow run 'nope'"):
        load_train("nope", client=object(), tracking_uri=uri, reader=reader)
