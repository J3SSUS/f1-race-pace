import numpy as np
import pandas as pd
import pytest

import training.adversarial as adversarial
from common.config import TrainingConfig, load_config
from features.catalog import load_catalog
from training.adversarial import FORM_FEATURES, _folds, adversarial_auc, adversarial_validation, feature_variants

NAMES = load_catalog().names
N_RACES, PER_RACE = 40, 8


def config(seed=42, folds=5) -> TrainingConfig:
    base = load_config().model_dump()
    base["optuna"]["seed"] = seed
    base["cv"]["folds"] = folds
    return TrainingConfig.model_validate(base)


def make_data(signal: str | None = None, seed=0):
    """40 races of 8 entries: the last 8 races are test. Every feature is noise, unless `signal` names one."""
    rng = np.random.default_rng(seed)
    rows = []
    for race in range(N_RACES):
        for d in range(PER_RACE):
            rows.append({"year": 2024 + race // 20, "round": race % 20 + 1, "driver": f"D{d}"})
    wide = pd.DataFrame(rows)
    wide["circuit"] = "C"
    wide["team"] = "T"
    n = len(wide)
    for name in NAMES:
        wide[name] = rng.normal(size=n)
    wide["circuit_type"] = rng.choice(["street", "permanent", "hybrid"], size=n)
    wide.loc[rng.random(n) < 0.2, "fp_tyre_deg"] = np.nan
    wide["race_pace_delta_pct"] = rng.normal(size=n)
    wide["clean_laps"] = 30
    wide["is_valid_row"] = True
    wide["entry_id"] = range(n)
    races = wide[["year", "round"]].drop_duplicates().reset_index(drop=True)
    races["split"] = ["train"] * (N_RACES - 8) + ["test"] * 8
    if signal:
        is_test = wide.merge(races, on=["year", "round"])["split"].eq("test").to_numpy()
        wide[signal] = is_test * 5.0 + rng.normal(size=n)
    return wide, races


def test_a_feature_that_separates_train_from_test_gives_an_auc_near_1():
    wide, split = make_data(signal="fp_tyre_deg")
    assert adversarial_auc(wide, split, NAMES, config()).mean > 0.95


def test_no_signal_gives_an_auc_near_one_half():
    wide, split = make_data()
    assert 0.35 < adversarial_auc(wide, split, NAMES, config()).mean < 0.65


def test_the_variant_without_form_drops_exactly_the_three_form_features():
    variants = feature_variants(NAMES)
    assert variants["all"] == NAMES and len(NAMES) == 15
    assert set(NAMES) - set(variants["no_form"]) == set(FORM_FEATURES)
    assert len(variants["no_form"]) == 12


def test_a_form_feature_that_grows_with_the_calendar_shows_only_in_the_all_variant():
    wide, split = make_data(signal="driver_pace_last3")
    result = adversarial_validation(wide, split, NAMES, config())
    assert result["all"].mean > 0.95
    assert 0.35 < result["no_form"].mean < 0.65


def test_the_classifier_receives_exactly_the_features(monkeypatch):
    seen = []

    class Spy(adversarial.LGBMClassifier):
        def fit(self, X, y, **kw):
            seen.append(list(X.columns))
            return super().fit(X, y, **kw)

    monkeypatch.setattr(adversarial, "LGBMClassifier", Spy)
    wide, split = make_data()
    result = adversarial_validation(wide, split, NAMES, config())
    assert {tuple(cols) for cols in seen} == {tuple(NAMES), tuple(feature_variants(NAMES)["no_form"])}
    assert result["all"].features == NAMES
    forbidden = {"race_pace_delta_pct", "clean_laps", "is_valid_row", "year", "round", "driver", "team", "circuit"}
    assert not forbidden & {c for cols in seen for c in cols}


@pytest.mark.parametrize("column", ["race_pace_delta_pct", "clean_laps", "is_valid_row", "year", "round", "driver"])
def test_a_forbidden_column_is_refused(column):
    wide, split = make_data()
    with pytest.raises(ValueError, match=column):
        adversarial_auc(wide, split, [*NAMES, column], config())


def test_no_race_is_in_both_sides_of_a_fold():
    wide, split = make_data()
    frame = wide.merge(split, on=["year", "round"])
    labels = (frame["split"] == "test").to_numpy().astype(int)
    groups = (frame["year"] * 100 + frame["round"]).to_numpy()
    folds = list(_folds(labels, groups, 5, 42))
    assert len(folds) == 5
    for train_idx, valid_idx in folds:
        assert not set(groups[train_idx]) & set(groups[valid_idx])
    assert sorted(i for _, valid in folds for i in valid) == list(range(len(frame)))


def test_the_same_seed_gives_the_same_auc_and_another_seed_changes_the_folds():
    wide, split = make_data(signal="fp_tyre_deg")
    first = adversarial_auc(wide, split, NAMES, config(seed=1))
    assert adversarial_auc(wide, split, NAMES, config(seed=1)) == first
    assert adversarial_auc(wide, split, NAMES, config(seed=2)).folds != first.folds


def test_missing_values_and_all_entries_are_used():
    wide, split = make_data()
    wide["is_valid_row"] = False  # validity is not used to pick rows
    assert wide["fp_tyre_deg"].isna().any()
    assert len(adversarial_auc(wide, split, NAMES, config()).folds) == 5


def test_too_few_test_races_for_the_folds_raises_a_clear_error():
    wide, split = make_data()
    split["split"] = ["train"] * (N_RACES - 1) + ["test"]
    with pytest.raises(ValueError, match="one class only"):
        adversarial_auc(wide, split, NAMES, config())
