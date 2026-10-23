import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from features.catalog import (
    CATALOG_PATH,
    GROUPS,
    IDENTIFICATION,
    TARGET_COLUMNS,
    apply_dtypes,
    features_schema,
    load_catalog,
    row_schema,
)

# pandera raises `SchemaErrors` when it collects several failures (column order, dtypes)
REJECTS = (pa.errors.SchemaError, pa.errors.SchemaErrors)

NAMES = [
    "quali_delta_pct",
    "grid_position",
    "reached_q3",
    "fp_long_run_delta_pct",
    "fp_tyre_deg",
    "driver_pace_last3",
    "team_pace_last3",
    "teammate_quali_gap",
    "team_pace_this_circuit_prev",
    "circuit_type",
    "track_temp",
    "air_temp",
    "rain_flag",
    "sprint_pace_delta_pct",
    "is_sprint_weekend",
]


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def valid_row(**changes) -> dict:
    """One valid driver-race row; `changes` overrides columns."""
    row = {
        "year": 2024,
        "round": 1,
        "circuit": "Synthetic",
        "driver": "AAA",
        "team": "Alpha",
        "race_pace_delta_pct": -0.5,
        "clean_laps": 40,
        "is_valid_row": True,
        "quali_delta_pct": 0.4,
        "grid_position": 3,
        "reached_q3": 1,
        "fp_long_run_delta_pct": 0.1,
        "fp_tyre_deg": 0.05,
        "driver_pace_last3": -0.3,
        "team_pace_last3": -0.2,
        "teammate_quali_gap": 0.1,
        "team_pace_this_circuit_prev": -0.1,
        "circuit_type": "permanent",
        "track_temp": 35.0,
        "air_temp": 25.0,
        "rain_flag": 0,
        "sprint_pace_delta_pct": np.nan,
        "is_sprint_weekend": 0,
    }
    return row | changes


def frame(catalog, *rows) -> pd.DataFrame:
    df = pd.DataFrame(rows).astype({"year": "int64", "round": "int64", "clean_laps": "int64", "is_valid_row": bool})
    return apply_dtypes(df, catalog)


def test_the_catalog_lists_the_15_features_in_order(catalog):
    assert catalog.names == NAMES
    assert [f.position for f in catalog.features] == list(range(1, 16))
    assert len({f.name for f in catalog.features}) == 15


def test_groups_are_allowed_by_the_supabase_check(catalog):
    assert GROUPS == {"qualifying", "practice", "sprint", "form", "circuit", "weather", "context"}
    assert {f.group for f in catalog.features} <= GROUPS


def test_every_feature_is_known_on_saturday_and_exploration_columns_are_out(catalog):
    assert all(f.known_on_saturday for f in catalog.features)
    assert not {"humidity", "wind_speed"} & set(catalog.names)


def test_a_valid_row_passes_the_row_schema(catalog):
    row_schema(catalog).validate(frame(catalog, valid_row()))


def test_the_row_has_23_columns_in_order(catalog):
    columns = list(row_schema(catalog).columns)
    assert columns == IDENTIFICATION + TARGET_COLUMNS + NAMES
    assert len(columns) == 23


def test_the_features_schema_has_no_target_columns(catalog):
    columns = list(features_schema(catalog).columns)
    assert columns == IDENTIFICATION + NAMES
    df = frame(catalog, valid_row()).drop(columns=TARGET_COLUMNS)
    features_schema(catalog).validate(df)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("quali_delta_pct", 3.5),
        ("quali_delta_pct", -0.1),
        ("circuit_type", "oval"),
        ("reached_q3", 2),
        ("rain_flag", 3),
        ("is_sprint_weekend", -1),
        ("grid_position", 0),
        ("track_temp", 200.0),
    ],
)
def test_the_schema_rejects_values_out_of_the_limits(catalog, column, value):
    with pytest.raises(REJECTS):
        row_schema(catalog).validate(frame(catalog, valid_row(**{column: value})))


@pytest.mark.parametrize("column", ["reached_q3", "circuit_type", "is_sprint_weekend"])
def test_a_feature_that_is_not_nullable_rejects_a_missing_value(catalog, column):
    df = frame(catalog, valid_row()).astype({column: object})
    df[column] = None
    with pytest.raises(REJECTS):
        row_schema(catalog).validate(df)


def test_missing_values_are_allowed_where_the_catalog_says_so(catalog):
    row = valid_row(
        quali_delta_pct=np.nan,
        grid_position=pd.NA,
        fp_long_run_delta_pct=np.nan,
        team_pace_this_circuit_prev=np.nan,
        rain_flag=pd.NA,
        race_pace_delta_pct=np.nan,
        is_valid_row=False,
    )
    row_schema(catalog).validate(frame(catalog, row))


def test_a_repeated_key_is_rejected(catalog):
    with pytest.raises(REJECTS):
        row_schema(catalog).validate(frame(catalog, valid_row(), valid_row(team="Beta")))


def test_an_extra_or_missing_column_is_rejected(catalog):
    df = frame(catalog, valid_row())
    with pytest.raises(REJECTS):
        row_schema(catalog).validate(df.assign(humidity=50.0))
    with pytest.raises(REJECTS):
        row_schema(catalog).validate(df.drop(columns=["air_temp"]))


def test_the_columns_must_come_in_the_order_of_the_catalog(catalog):
    df = frame(catalog, valid_row())
    swapped = df[[*df.columns[:-2], df.columns[-1], df.columns[-2]]]
    with pytest.raises(REJECTS):
        row_schema(catalog).validate(swapped)


def test_the_target_is_empty_exactly_when_the_row_is_not_valid(catalog):
    with pytest.raises(REJECTS):
        row_schema(catalog).validate(frame(catalog, valid_row(race_pace_delta_pct=np.nan)))
    with pytest.raises(REJECTS):
        row_schema(catalog).validate(frame(catalog, valid_row(is_valid_row=False)))


def test_a_valid_row_needs_15_clean_laps(catalog):
    with pytest.raises(REJECTS):
        row_schema(catalog).validate(frame(catalog, valid_row(clean_laps=14)))
    row_schema(catalog).validate(frame(catalog, valid_row(clean_laps=15)))


def test_schema_hash_is_stable_and_changes_with_a_limit(catalog, tmp_path):
    text = CATALOG_PATH.read_text()
    same = tmp_path / "same.yml"
    same.write_text(text)
    assert load_catalog(same).schema_hash == catalog.schema_hash
    changed = tmp_path / "changed.yml"
    changed.write_text(text.replace("max: 3 # capped at 3%", "max: 4 # capped at 3%"))
    assert load_catalog(changed).schema_hash != catalog.schema_hash


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("position: 2\n", "position: 1\n", "positions"),
        ("group: qualifying", "group: telemetry", "group"),
        ("dtype: flag", "dtype: boolean", "dtype"),
        ("known_on_saturday: true", "known_on_saturday: yes please", "boolean"),
        ("name: grid_position", "name: quali_delta_pct", "repeated"),
    ],
)
def test_a_wrong_catalog_is_rejected(tmp_path, old, new, message):
    broken = tmp_path / "broken.yml"
    broken.write_text(CATALOG_PATH.read_text().replace(old, new, 1))
    with pytest.raises(ValueError, match=message):
        load_catalog(broken)
