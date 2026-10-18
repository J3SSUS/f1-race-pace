"""Pandera schemas for the raw tables: only the columns later phases read (presence and dtype)."""

import pandas as pd
import pandera.pandas as pa

_NOT_EMPTY = pa.Check(lambda df: len(df) >= 1, error="table has no rows")


def _col(dtype, nullable: bool = False) -> pa.Column:
    return pa.Column(dtype, nullable=nullable, required=True)


LAPS_SCHEMA = pa.DataFrameSchema(
    {
        "Driver": _col(str),
        "DriverNumber": _col(str),
        "Team": _col(str),
        "LapNumber": _col(float),
        "LapTime": _col("timedelta64[ns]", nullable=True),
        "Stint": _col(float, nullable=True),
        "Compound": _col(str, nullable=True),
        "PitInTime": _col("timedelta64[ns]", nullable=True),
        "PitOutTime": _col("timedelta64[ns]", nullable=True),
        "TrackStatus": _col(str, nullable=True),
        "IsAccurate": _col(bool),
    },
    checks=_NOT_EMPTY,
    strict=False,
)

RESULTS_SCHEMA = pa.DataFrameSchema(
    {
        "DriverNumber": _col(str),
        "Abbreviation": _col(str),
        "FullName": _col(str),
        "TeamName": _col(str),
        "Position": _col(float, nullable=True),
        "Q1": _col("timedelta64[ns]", nullable=True),
        "Q2": _col("timedelta64[ns]", nullable=True),
        "Q3": _col("timedelta64[ns]", nullable=True),
    },
    checks=_NOT_EMPTY,
    strict=False,
)

WEATHER_SCHEMA = pa.DataFrameSchema(
    {
        "Time": _col("timedelta64[ns]"),
        "AirTemp": _col(float),
        "TrackTemp": _col(float),
        "Rainfall": _col(bool),
    },
    strict=False,
)


def _errors(name: str, schema: pa.DataFrameSchema, df: pd.DataFrame) -> list[str]:
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        return [
            f"{name}.{row.column}: {row.check} (failure case: {row.failure_case})"
            if row.column is not None
            else f"{name}: {row.check} (failure case: {row.failure_case})"
            for row in exc.failure_cases.itertuples()
        ]
    return []


def validate_tables(laps: pd.DataFrame, results: pd.DataFrame, weather: pd.DataFrame) -> list[str]:
    """Error messages of the three tables; empty when all pass."""
    return (
        _errors("laps", LAPS_SCHEMA, laps)
        + _errors("results", RESULTS_SCHEMA, results)
        + _errors("weather", WEATHER_SCHEMA, weather)
    )
