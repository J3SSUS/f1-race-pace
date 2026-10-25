"""Optuna tuning of the six models over the temporal CV of SPEC 07."""

from typing import Any

import optuna

from common.config import Choice, Fixed, ParamSpec, Range

IMPUTATION = "imputation"  # the one hyperparameter every model shares


def suggest(trial: optuna.Trial, specs: dict[str, ParamSpec], imputations: list[str]) -> dict[str, Any]:
    """The params of one trial: fixed values as they are, ranges and choices from Optuna, plus the imputation."""
    params: dict[str, Any] = {}
    for name, spec in specs.items():
        if isinstance(spec, Fixed):
            params[name] = spec.value
        elif isinstance(spec, Range):
            suggest_range = trial.suggest_int if spec.is_int else trial.suggest_float
            params[name] = suggest_range(name, spec.low, spec.high, log=spec.log)
        elif isinstance(spec, Choice):
            params[name] = trial.suggest_categorical(name, list(spec.options))
        else:
            raise TypeError(f"param '{name}': unexpected spec {spec!r}")
    params[IMPUTATION] = trial.suggest_categorical(IMPUTATION, list(imputations))
    return params
