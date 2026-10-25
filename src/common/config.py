"""Loader and validation for config/training.json.

A `params` value is read as:
- a scalar: fixed value;
- `[low, high]`: Optuna range (integer if both are integers, float otherwise);
- `[low, high, "log"]`: log-scale range;
- a list of strings: categorical choice.

Unknown keys, models and params are rejected so a typo never passes silently.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "training.json"

# Params each model accepts. Extend a set here when a new param is needed.
ALLOWED_PARAMS: dict[str, set[str]] = {
    "ridge": {"alpha", "fit_intercept"},
    "elasticnet": {"alpha", "l1_ratio", "max_iter", "fit_intercept"},
    "random_forest": {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "criterion",
    },
    "xgboost": {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
        "objective",
    },
    "lightgbm": {
        "n_estimators",
        "num_leaves",
        "max_depth",
        "learning_rate",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "verbose",
        "objective",
    },
    "catboost": {"iterations", "depth", "learning_rate", "l2_leaf_reg", "subsample", "verbose", "loss_function"},
}
IMPUTATIONS = {"median", "mean"}


@dataclass(frozen=True)
class Fixed:
    value: Any


@dataclass(frozen=True)
class Range:
    low: float
    high: float
    log: bool = False

    @property
    def is_int(self) -> bool:
        return isinstance(self.low, int) and isinstance(self.high, int)


@dataclass(frozen=True)
class Choice:
    options: tuple[str, ...]


ParamSpec = Fixed | Range | Choice


def _is_number(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def parse_param(name: str, value: Any) -> ParamSpec:
    """Turn a raw `params` value into Fixed, Range or Choice."""
    if not isinstance(value, list):
        if isinstance(value, dict) or value is None:
            raise ValueError(f"param '{name}': expected a scalar or a list, got {value!r}")
        return Fixed(value)
    if value and all(isinstance(x, str) for x in value):
        if len(value) < 2:
            raise ValueError(f"param '{name}': a choice needs at least 2 options")
        return Choice(tuple(value))
    if len(value) in (2, 3) and all(_is_number(x) for x in value[:2]):
        low, high = value[:2]
        log = False
        if len(value) == 3:
            if value[2] != "log":
                raise ValueError(f"param '{name}': the third element must be \"log\", got {value[2]!r}")
            log = True
        if low >= high:
            raise ValueError(f"param '{name}': low must be below high, got [{low}, {high}]")
        if log and low <= 0:
            raise ValueError(f"param '{name}': a log range needs low > 0, got {low}")
        return Range(low, high, log)
    raise ValueError(
        f"param '{name}': cannot read {value!r} (use a scalar, [low, high], [low, high, \"log\"] or a list of strings)"
    )


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Cutoff(_Strict):
    year: int
    round: int = Field(ge=1)


class DataConfig(_Strict):
    first_year: int
    last_year: int
    # null: races through `last_year`. Set: races through that round (walk-forward); `last_year` is ignored.
    cutoff: Cutoff | None = None

    @model_validator(mode="after")
    def _years_in_order(self):
        if self.first_year > self.last_year:
            raise ValueError("first_year must not be after last_year")
        if self.cutoff is not None and self.cutoff.year < self.first_year:
            raise ValueError("cutoff.year must not be before first_year")
        return self


class SplitConfig(_Strict):
    test_fraction: float = Field(gt=0, lt=1)


class CvConfig(_Strict):
    folds: int = Field(ge=2)


class OptunaConfig(_Strict):
    trials: int = Field(ge=1)
    seed: int


class ModelConfig(_Strict):
    enabled: bool
    params: dict[str, Any] = {}

    def specs(self) -> dict[str, ParamSpec]:
        return {name: parse_param(name, value) for name, value in self.params.items()}


class TrainingConfig(_Strict):
    data: DataConfig
    split: SplitConfig
    cv: CvConfig
    optuna: OptunaConfig
    imputation: list[str]
    models: dict[str, ModelConfig]

    @field_validator("imputation")
    @classmethod
    def _known_imputation(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("imputation needs at least one option")
        unknown = set(v) - IMPUTATIONS
        if unknown:
            raise ValueError(f"unknown imputation {sorted(unknown)}; allowed: {sorted(IMPUTATIONS)}")
        return v

    @model_validator(mode="after")
    def _check_models(self):
        unknown = set(self.models) - set(ALLOWED_PARAMS)
        if unknown:
            raise ValueError(f"unknown models {sorted(unknown)}; allowed: {sorted(ALLOWED_PARAMS)}")
        for name, model in self.models.items():
            bad = set(model.params) - ALLOWED_PARAMS[name]
            if bad:
                raise ValueError(
                    f"model '{name}': unknown params {sorted(bad)}; allowed: {sorted(ALLOWED_PARAMS[name])}"
                )
            model.specs()  # validates every value
        if not any(m.enabled for m in self.models.values()):
            raise ValueError("at least one model must be enabled")
        return self

    def enabled_models(self) -> dict[str, dict[str, ParamSpec]]:
        """Enabled models with their parsed params, in config order."""
        return {name: m.specs() for name, m in self.models.items() if m.enabled}


def load_config(path: Path | str = CONFIG_PATH) -> TrainingConfig:
    with open(path) as f:
        return TrainingConfig.model_validate(json.load(f))
