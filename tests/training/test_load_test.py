import pytest
from test_data import FLAGS, TEST, TRAIN, load, split_of, wide_of

from features.catalog import load_catalog
from training.evaluation import load_test

CATALOG = load_catalog()


def load_the_test(flags=FLAGS, split=None):
    wide = wide_of(flags, test_target=7.0)
    return load_test(
        "run",
        client=object(),
        reader=lambda client, version: wide,
        splitter=lambda run_id, uri: split if split is not None else split_of(),
    )


def test_only_valid_rows_of_test_races_come_back():
    flags = FLAGS | {(2024, 3, "ALO"): False}
    test = load_the_test(flags)
    assert set(zip(test.meta["year"], test.meta["round"], strict=True)) == set(TEST)
    assert sorted(test.meta["driver"]) == ["HAM", "VER"]  # ALO is invalid
    assert len(test.X) == len(test.y) == len(test.meta) == 2
    assert test.y.tolist() == [7.0, 7.0]


def test_no_train_row_comes_back():
    test = load_the_test()
    assert not set(zip(test.meta["year"], test.meta["round"], strict=True)) & set(TRAIN)


def test_features_are_the_catalog_in_order():
    test = load_the_test()
    assert list(test.X.columns) == CATALOG.names


def test_an_unknown_run_fails_before_supabase_is_touched():
    def splitter(run_id, uri):
        raise ValueError("unknown run")

    def reader(client, version):
        raise AssertionError("Supabase was touched")

    with pytest.raises(ValueError, match="unknown run"):
        load_test("nope", client=object(), reader=reader, splitter=splitter)


def test_no_valid_rows_raises():
    flags = {key: valid for key, valid in FLAGS.items() if key[:2] not in TEST} | {(2024, 3, "HAM"): False}
    with pytest.raises(ValueError, match="no valid row"):
        load_the_test(flags)


def test_no_test_races_in_the_split_raises():
    with pytest.raises(ValueError, match="no valid row"):
        load_the_test(split=split_of(test=[]))


def test_load_train_still_never_sees_a_test_row():
    train = load()
    assert set(zip(train.meta["year"], train.meta["round"], strict=True)) == set(TRAIN)
    assert not (train.y == 999.0).any()
