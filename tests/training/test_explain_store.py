import numpy as np
import pandas as pd
import pytest
from fake_supabase import FakeSupabase

from features.catalog import load_catalog
from training.explain import Champion, shap_rows, upsert_shap_values

FEATURES = [f.name for f in load_catalog().features]
VERSION = 7
N = 12


@pytest.fixture
def parts():
    rng = np.random.default_rng(1)
    meta = pd.DataFrame({"entry_id": np.arange(100, 100 + N), "split": ["train"] * (N - 2) + ["test"] * 2})
    values = pd.DataFrame(rng.normal(size=(N, 15)), columns=FEATURES)
    champion = Champion(VERSION, "1", "run", "sampling", "ridge", None)
    return champion, meta, values


@pytest.fixture
def client():
    return FakeSupabase({"features": [{"feature_id": i, "name": name} for i, name in enumerate(FEATURES, 1)]})


def test_there_is_one_row_per_valid_entry_and_feature_with_the_champions_version(parts, client):
    champion, meta, values = parts
    assert upsert_shap_values(client, champion, meta, values) == N * 15
    rows = client.tables["shap_values"]
    assert len(rows) == N * 15
    assert {r["model_version_id"] for r in rows} == {VERSION}
    assert {r["entry_id"] for r in rows} == set(meta["entry_id"])


def test_each_value_goes_to_its_entry_and_the_feature_id_of_its_name(parts, client):
    champion, meta, values = parts
    upsert_shap_values(client, champion, meta, values)
    ids = {r["name"]: r["feature_id"] for r in client.tables["features"]}
    stored = {(r["entry_id"], r["feature_id"]): r["shap_value"] for r in client.tables["shap_values"]}
    for i in (0, 5, N - 1):
        for name in ("quali_delta_pct", "circuit_type"):
            assert stored[(int(meta["entry_id"][i]), ids[name])] == values.loc[i, name]


def test_running_twice_leaves_the_same_rows(parts, client):
    champion, meta, values = parts
    upsert_shap_values(client, champion, meta, values)
    first = [dict(r) for r in client.tables["shap_values"]]
    upsert_shap_values(client, champion, meta, values)
    assert client.tables["shap_values"] == first


def test_a_feature_missing_from_the_features_table_raises_before_writing(parts):
    champion, meta, values = parts
    client = FakeSupabase({"features": [{"feature_id": i, "name": name} for i, name in enumerate(FEATURES[:-1], 1)]})
    with pytest.raises(ValueError, match=FEATURES[-1]):
        upsert_shap_values(client, champion, meta, values)
    assert client.count("shap_values") == 0
    assert client.writes == []


def test_a_value_that_is_not_finite_raises_before_writing(parts, client):
    champion, meta, values = parts
    values.iloc[3, 4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        upsert_shap_values(client, champion, meta, values)
    assert client.writes == []


def test_only_the_shap_values_table_is_written(parts, client):
    champion, meta, values = parts
    upsert_shap_values(client, champion, meta, values)
    assert {table for table, _ in client.writes} == {"shap_values"}
    assert set(client.tables) == {"features", "shap_values"}


def test_rows_follow_the_order_of_the_entries_and_then_of_the_features(parts):
    _, meta, values = parts
    rows = shap_rows(values, meta, {name: i + 1 for i, name in enumerate(FEATURES)}, VERSION)
    assert [r["entry_id"] for r in rows[:16]] == [100] * 15 + [101]
    assert [r["feature_id"] for r in rows[:15]] == list(range(1, 16))
