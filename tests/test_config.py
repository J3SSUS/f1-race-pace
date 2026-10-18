import copy
import json

import pytest
from pydantic import ValidationError

from common.config import CONFIG_PATH, Choice, Fixed, Range, load_config, parse_param


@pytest.fixture
def raw():
    return json.loads(CONFIG_PATH.read_text())


def build(raw, tmp_path):
    p = tmp_path / "training.json"
    p.write_text(json.dumps(raw))
    return load_config(p)


def test_repo_config_loads():
    cfg = load_config()
    assert cfg.split.test_fraction == 0.2
    assert list(cfg.enabled_models()) == [
        "ridge",
        "elasticnet",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
    ]


def test_disabled_model_is_skipped(raw, tmp_path):
    raw["models"]["catboost"]["enabled"] = False
    assert "catboost" not in build(raw, tmp_path).enabled_models()


def test_no_enabled_model_is_rejected(raw, tmp_path):
    for m in raw["models"].values():
        m["enabled"] = False
    with pytest.raises(ValidationError, match="at least one model"):
        build(raw, tmp_path)


def test_unknown_model_is_rejected(raw, tmp_path):
    raw["models"]["svr"] = {"enabled": True, "params": {}}
    with pytest.raises(ValidationError, match="unknown models"):
        build(raw, tmp_path)


def test_unknown_param_is_rejected(raw, tmp_path):
    raw["models"]["ridge"]["params"]["alfa"] = [0.1, 1]
    with pytest.raises(ValidationError, match="unknown params"):
        build(raw, tmp_path)


def test_unknown_key_is_rejected(raw, tmp_path):
    raw["optuna"]["trial"] = 10
    with pytest.raises(ValidationError):
        build(raw, tmp_path)


@pytest.mark.parametrize(
    "path, value",
    [
        (("split", "test_fraction"), 0),
        (("split", "test_fraction"), 1),
        (("cv", "folds"), 1),
        (("optuna", "trials"), 0),
        (("imputation",), []),
        (("imputation",), ["median", "mode"]),
    ],
)
def test_out_of_range_values_are_rejected(raw, tmp_path, path, value):
    bad = copy.deepcopy(raw)
    target = bad
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        build(bad, tmp_path)


def test_years_must_be_in_order(raw, tmp_path):
    raw["data"]["first_year"], raw["data"]["last_year"] = 2025, 2022
    with pytest.raises(ValidationError, match="first_year"):
        build(raw, tmp_path)


def test_cutoff_is_parsed(raw, tmp_path):
    raw["data"]["cutoff"] = {"year": 2026, "round": 5}
    cfg = build(raw, tmp_path)
    assert (cfg.data.cutoff.year, cfg.data.cutoff.round) == (2026, 5)


def test_cutoff_round_must_be_positive(raw, tmp_path):
    raw["data"]["cutoff"] = {"year": 2026, "round": 0}
    with pytest.raises(ValidationError):
        build(raw, tmp_path)


def test_parse_param_kinds():
    assert parse_param("x", 0) == Fixed(0)
    assert parse_param("x", 0.5) == Fixed(0.5)
    assert parse_param("x", [1, 5]) == Range(1, 5)
    assert parse_param("x", [1, 5]).is_int
    assert not parse_param("x", [0.1, 5]).is_int
    assert parse_param("x", [0.01, 1, "log"]) == Range(0.01, 1, log=True)
    assert parse_param("x", ["a", "b"]) == Choice(("a", "b"))


@pytest.mark.parametrize(
    "value",
    [[5, 1], [1, 1], [0, 1, "log"], [1, 2, "linear"], ["a"], [1, "a"], [1, 2, 3, 4], {"low": 1}, None, []],
)
def test_parse_param_rejects_bad_values(value):
    with pytest.raises(ValueError):
        parse_param("x", value)
