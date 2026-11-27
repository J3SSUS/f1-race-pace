"""The feature drift of SPEC 26 on synthetic frames: which test each feature gets, what drifts and the edge cases.

No OpenF1 or Supabase data: the frames are built here from the real catalog (`config/features.yml`), so the 15
features and their dtypes are the real ones. The reference has more than 1,000 rows on purpose: it is where Evidently's
default would drop the p-value, and the forced tests must still give one.
"""

import numpy as np
import pandas as pd
import pytest

from features.catalog import load_catalog
from monitoring.drift import P_VALUE, STATTESTS, feature_drift

CATALOG = load_catalog()
IDS = {name: i for i, name in enumerate(CATALOG.names, start=1)}
FLAGS = ["reached_q3", "rain_flag", "is_sprint_weekend"]
CATEGORIES = ["circuit_type"]


def frame(rows: int, seed: int, **shift) -> pd.DataFrame:
    """One column per feature, drawn from the same distribution for every seed; `shift` moves a numeric column by that
    much (`track_temp=10`) or sets the probability of 1 of a flag (`rain_flag=0.9`) or the weights of a category."""
    rng = np.random.default_rng(seed)
    data = {}
    for feature in CATALOG.features:
        if feature.dtype == "float":
            data[feature.name] = rng.normal(shift.get(feature.name, 0.0), 1.0, rows)
        elif feature.dtype == "int":
            data[feature.name] = pd.array(rng.integers(1, 21, rows), dtype="Int64")
        elif feature.dtype == "flag":
            data[feature.name] = rng.binomial(1, shift.get(feature.name, 0.5), rows).astype("int64")
        else:
            data[feature.name] = rng.choice(["street", "high_speed", "mixed"], rows, p=shift.get(feature.name, None))
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return frame(1500, seed=1)


@pytest.fixture(scope="module")
def same(reference) -> object:
    return feature_drift(reference, frame(100, seed=2), CATALOG, IDS)


def by_name(result) -> dict:
    return {f.feature: f for f in result.features}


def test_the_catalog_has_the_four_categorical_features_the_spec_names():
    chi = sorted(f.name for f in CATALOG.features if STATTESTS[f.dtype] == "chisquare")
    assert chi == sorted(FLAGS + CATEGORIES)


def test_one_result_per_feature_in_catalog_order_with_its_id_and_test(same):
    assert [f.feature for f in same.features] == CATALOG.names
    assert [f.feature_id for f in same.features] == list(range(1, len(CATALOG.names) + 1))
    expected = {name: ("chisquare" if name in FLAGS + CATEGORIES else "ks") for name in CATALOG.names}
    assert {f.feature: f.stattest for f in same.features} == expected


def test_the_statistic_is_always_null_and_every_p_value_is_a_probability(same):
    assert all(f.statistic is None for f in same.features)
    assert all(f.p_value is not None and 0.0 <= f.p_value <= 1.0 for f in same.features)


def test_the_same_distribution_does_not_drift(same):
    assert same.n_drifted == 0
    assert all(f.p_value >= P_VALUE for f in same.features)


def test_a_shifted_numeric_feature_drifts_and_only_that_one(reference):
    result = feature_drift(reference, frame(100, seed=2, track_temp=3.0), CATALOG, IDS)
    drifted = [f.feature for f in result.features if f.is_drifted]
    assert drifted == ["track_temp"]
    assert by_name(result)["track_temp"].stattest == "ks" and by_name(result)["track_temp"].p_value < 1e-10
    assert result.n_drifted == 1


def test_a_shifted_flag_drifts_with_chi_squared(reference):
    result = feature_drift(reference, frame(100, seed=2, rain_flag=0.95), CATALOG, IDS)
    rain = by_name(result)["rain_flag"]
    assert rain.is_drifted and rain.stattest == "chisquare"


def test_a_shifted_category_drifts_with_chi_squared(reference):
    result = feature_drift(reference, frame(100, seed=2, circuit_type=[0.9, 0.05, 0.05]), CATALOG, IDS)
    circuit = by_name(result)["circuit_type"]
    assert circuit.is_drifted and circuit.stattest == "chisquare"


def test_a_feature_that_drifts_exactly_at_the_threshold_is_not_drifted(reference, monkeypatch):
    """`is_drifted` is `p < P_VALUE`: raising the threshold above a p-value turns the flag on, it is not Evidently's."""
    base = feature_drift(reference, frame(100, seed=2), CATALOG, IDS)
    smallest = min(f.p_value for f in base.features)
    monkeypatch.setattr("monitoring.drift.P_VALUE", smallest)  # p == threshold: not drifted
    assert feature_drift(reference, frame(100, seed=2), CATALOG, IDS).n_drifted == 0
    monkeypatch.setattr("monitoring.drift.P_VALUE", smallest * 1.01)  # p < threshold: drifted
    assert feature_drift(reference, frame(100, seed=2), CATALOG, IDS).n_drifted >= 1


def test_missing_values_are_handled_in_numeric_and_nullable_int_features(reference):
    current = frame(100, seed=2)
    current.loc[:9, "quali_delta_pct"] = np.nan
    current["grid_position"] = current["grid_position"].astype("Int64")
    current.loc[:4, "grid_position"] = pd.NA
    ref = reference.copy()
    ref.loc[:99, "fp_tyre_deg"] = np.nan
    result = feature_drift(ref, current, CATALOG, IDS)
    named = by_name(result)
    for name in ("quali_delta_pct", "grid_position", "fp_tyre_deg"):
        assert named[name].p_value is not None and 0.0 <= named[name].p_value <= 1.0


def test_a_feature_with_no_value_in_the_window_has_no_p_value_and_the_rest_are_still_tested(reference):
    current = frame(100, seed=2)
    current["fp_long_run_delta_pct"] = np.nan
    result = feature_drift(reference, current, CATALOG, IDS)
    empty = by_name(result)["fp_long_run_delta_pct"]
    assert empty.p_value is None and empty.is_drifted is False and empty.stattest == "ks"
    assert sum(f.p_value is not None for f in result.features) == len(CATALOG.names) - 1
    assert result.html is not None


def test_a_feature_with_no_value_in_the_reference_is_not_tested_either(reference):
    ref = reference.copy()
    ref["circuit_type"] = None
    result = feature_drift(ref, frame(100, seed=2), CATALOG, IDS)
    assert by_name(result)["circuit_type"].p_value is None and not by_name(result)["circuit_type"].is_drifted


def test_a_feature_constant_in_the_window_but_not_in_the_reference_is_a_drift(reference):
    current = frame(100, seed=2)
    current["rain_flag"] = 0
    result = feature_drift(reference, current, CATALOG, IDS)
    assert by_name(result)["rain_flag"].is_drifted


def test_with_no_testable_feature_there_is_no_html_and_nothing_drifts(reference):
    current = pd.DataFrame({name: np.nan for name in CATALOG.names}, index=range(100))
    result = feature_drift(reference, current, CATALOG, IDS)
    assert result.html is None and result.n_drifted == 0 and all(f.p_value is None for f in result.features)


def test_the_html_is_the_evidently_report(same):
    assert isinstance(same.html, str) and "<html" in same.html.lower() and "track_temp" in same.html


def test_a_missing_feature_id_or_column_is_an_error(reference):
    current = frame(100, seed=2)
    with pytest.raises(ValueError, match="no feature_id"):
        feature_drift(reference, current, CATALOG, {k: v for k, v in IDS.items() if k != "track_temp"})
    with pytest.raises(ValueError, match="current rows have no column"):
        feature_drift(reference, current.drop(columns=["track_temp"]), CATALOG, IDS)
    with pytest.raises(ValueError, match="reference rows have no column"):
        feature_drift(reference.drop(columns=["rain_flag"]), current, CATALOG, IDS)
