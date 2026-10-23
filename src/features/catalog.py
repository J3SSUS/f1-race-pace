"""The feature catalog (`config/features.yml`) and the Pandera schemas generated from it."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import yaml

from features.constants import MIN_VALID_CLEAN_LAPS

CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "features.yml"
GROUPS = {"qualifying", "practice", "sprint", "form", "circuit", "weather", "context"}  # features.feature_group
DTYPES = {"float", "int", "flag", "category"}
IDENTIFICATION = ["year", "round", "circuit", "driver", "team"]
TARGET_COLUMNS = ["race_pace_delta_pct", "clean_laps", "is_valid_row"]
KEY = ["year", "round", "driver"]


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    dtype: str
    known_on_saturday: bool
    position: int
    nullable: bool
    min: float | None = None
    max: float | None = None
    allowed: tuple | None = None

    @property
    def pandas_dtype(self) -> str:
        if self.dtype == "float":
            return "float64"
        if self.dtype == "category":
            return "object"
        return "Int64" if self.nullable else "int64"


@dataclass(frozen=True)
class Catalog:
    version: str
    features: tuple[Feature, ...]  # ordered by position
    schema_hash: str  # SHA-256 of the text of the file

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]


def load_catalog(path: Path = CATALOG_PATH) -> Catalog:
    """Read and check the catalog; a typo, a repeated name or a gap in the positions is rejected."""
    text = path.read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or not isinstance(data.get("features"), list) or not data.get("version"):
        raise ValueError(f"{path.name}: expected a `version` and a list of `features`")
    features = []
    for raw in data["features"]:
        unknown = set(raw) - {
            "name",
            "group",
            "dtype",
            "known_on_saturday",
            "position",
            "nullable",
            "min",
            "max",
            "allowed",
        }
        if unknown:
            raise ValueError(f"{path.name}: unknown keys {sorted(unknown)} in {raw.get('name')}")
        allowed = tuple(raw["allowed"]) if raw.get("allowed") is not None else None
        features.append(Feature(**{**raw, "allowed": allowed}))
    features.sort(key=lambda f: f.position)

    names = [f.name for f in features]
    if len(set(names)) != len(names):
        raise ValueError(f"{path.name}: repeated feature names")
    if [f.position for f in features] != list(range(1, len(features) + 1)):
        raise ValueError(f"{path.name}: positions must be 1 to {len(features)} with no gaps or repeats")
    for f in features:
        if f.group not in GROUPS:
            raise ValueError(f"{path.name}: {f.name} has the group {f.group!r}, not one of {sorted(GROUPS)}")
        if f.dtype not in DTYPES:
            raise ValueError(f"{path.name}: {f.name} has the dtype {f.dtype!r}, not one of {sorted(DTYPES)}")
        if not isinstance(f.known_on_saturday, bool) or not isinstance(f.nullable, bool):
            raise ValueError(f"{path.name}: {f.name} needs boolean `known_on_saturday` and `nullable`")
    return Catalog(str(data["version"]), tuple(features), hashlib.sha256(text.encode()).hexdigest())


def apply_dtypes(frame: pd.DataFrame, catalog: Catalog) -> pd.DataFrame:
    """The frame with every feature column cast to the dtype of the catalog."""
    return frame.astype({f.name: f.pandas_dtype for f in catalog.features if f.name in frame.columns})


def _column(feature: Feature) -> pa.Column:
    checks = []
    if feature.allowed is not None:
        checks.append(pa.Check.isin(list(feature.allowed)))
    if feature.min is not None or feature.max is not None:
        checks.append(pa.Check.in_range(feature.min, feature.max, include_min=True, include_max=True))
    dtype = str if feature.dtype == "category" else feature.pandas_dtype
    return pa.Column(dtype, checks=checks, nullable=feature.nullable, required=True)


def _identification() -> dict[str, pa.Column]:
    return {
        "year": pa.Column("int64"),
        "round": pa.Column("int64"),
        "circuit": pa.Column(str),
        "driver": pa.Column(str),
        "team": pa.Column(str),
    }


def _target() -> dict[str, pa.Column]:
    return {
        "race_pace_delta_pct": pa.Column("float64", nullable=True),
        "clean_laps": pa.Column("int64", pa.Check.ge(0)),
        "is_valid_row": pa.Column(bool),
    }


_TARGET_CHECKS = [
    pa.Check(
        lambda df: df["is_valid_row"] == df["race_pace_delta_pct"].notna(),
        error="race_pace_delta_pct must be empty exactly when the row is not valid",
    ),
    pa.Check(
        lambda df: ~df["is_valid_row"] | (df["clean_laps"] >= MIN_VALID_CLEAN_LAPS),
        error=f"a valid row needs at least {MIN_VALID_CLEAN_LAPS} clean laps",
    ),
]


def features_schema(catalog: Catalog) -> pa.DataFrameSchema:
    """Output of `race_features`: the identification columns and the features, in the order of the catalog."""
    columns = _identification() | {f.name: _column(f) for f in catalog.features}
    return pa.DataFrameSchema(columns, strict=True, ordered=True, unique=KEY)


def row_schema(catalog: Catalog) -> pa.DataFrameSchema:
    """Output of `build_race`: identification, target and features, with the rules between target and validity."""
    columns = _identification() | _target() | {f.name: _column(f) for f in catalog.features}
    return pa.DataFrameSchema(columns, checks=_TARGET_CHECKS, strict=True, ordered=True, unique=KEY)
