import optuna
import pytest

from common.config import Choice, Fixed, Range, load_config
from training.tuning import IMPUTATION, suggest

optuna.logging.set_verbosity(optuna.logging.WARNING)

SPECS = {
    "fixed": Fixed("l1"),
    "n": Range(2, 9),
    "depth_log": Range(1, 64, True),
    "alpha": Range(0.01, 100, True),
    "ratio": Range(0.1, 0.9),
    "kind": Choice(("a", "b", "c")),
}


def collect(n_trials: int = 30, seed: int = 0, specs=SPECS) -> tuple[list[dict], optuna.Study]:
    seen = []

    def objective(trial):
        seen.append(suggest(trial, specs, ["median", "mean"]))
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return seen, study


def test_fixed_value_makes_no_suggestion():
    seen, study = collect()
    assert all(p["fixed"] == "l1" for p in seen)
    assert "fixed" not in study.trials[0].params


def test_integer_range_gives_integers_and_float_range_gives_floats():
    seen, _ = collect()
    assert all(type(p["n"]) is int and type(p["depth_log"]) is int for p in seen)
    assert all(type(p["alpha"]) is float and type(p["ratio"]) is float for p in seen)


def test_every_value_is_inside_its_range():
    seen, _ = collect()
    for p in seen:
        assert 2 <= p["n"] <= 9
        assert 1 <= p["depth_log"] <= 64
        assert 0.01 <= p["alpha"] <= 100
        assert 0.1 <= p["ratio"] <= 0.9
        assert p["kind"] in {"a", "b", "c"}


def test_log_scale_is_honoured():
    _, study = collect()
    dists = study.trials[0].distributions
    assert dists["alpha"].log and dists["depth_log"].log
    assert not dists["ratio"].log and not dists["n"].log


def test_imputation_is_always_suggested():
    seen, study = collect()
    assert {p[IMPUTATION] for p in seen} <= {"median", "mean"}
    assert all(IMPUTATION in t.params for t in study.trials)
    seen_only_fixed, _ = collect(specs={"fixed": Fixed(1)})
    assert all(IMPUTATION in p for p in seen_only_fixed)


def test_same_seed_same_sequence():
    a, _ = collect(seed=3)
    b, _ = collect(seed=3)
    c, _ = collect(seed=4)
    assert a == b
    assert a != c


@pytest.mark.parametrize("model", list(load_config().enabled_models()))
def test_repo_config_spaces_are_valid(model):
    specs = load_config().enabled_models()[model]
    seen, _ = collect(n_trials=5, specs=specs)
    assert all(set(specs) | {IMPUTATION} == set(p) for p in seen)
