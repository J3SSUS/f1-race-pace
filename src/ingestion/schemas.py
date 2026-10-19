"""Pandera schemas for the raw OpenF1 tables: only the columns later phases read (presence and dtype).

The loader normalises types before validation (see `openf1_loader.py`): timestamps are UTC, integers that
can be null are pandas `Int64`, and a table with no rows keeps its declared columns.
"""

import pandas as pd
import pandera.pandas as pa

TS = "datetime64[ns, UTC]"

_NOT_EMPTY = pa.Check(lambda df: len(df) >= 1, error="table has no rows")


def _col(dtype, nullable: bool = False) -> pa.Column:
    return pa.Column(dtype, nullable=nullable, required=True)


def _schema(columns: dict, not_empty: bool = False) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(columns, checks=_NOT_EMPTY if not_empty else None, strict=False)


SCHEMAS = {
    "laps": _schema(
        {
            "driver_number": _col("int64"),
            "lap_number": _col("int64"),
            "lap_duration": _col("float64", nullable=True),
            "is_pit_out_lap": _col(bool),
            "date_start": _col(TS, nullable=True),
            "duration_sector_1": _col("float64", nullable=True),
            "duration_sector_2": _col("float64", nullable=True),
            "duration_sector_3": _col("float64", nullable=True),
        },
        not_empty=True,
    ),
    "pit": _schema(
        {
            "driver_number": _col("int64"),
            "lap_number": _col("int64"),
            "date": _col(TS),
            "pit_duration": _col("float64", nullable=True),
        }
    ),
    "stints": _schema(
        {
            "driver_number": _col("int64"),
            "stint_number": _col("int64"),
            "lap_start": _col("Int64", nullable=True),
            "lap_end": _col("Int64", nullable=True),
            "compound": _col(str, nullable=True),
            "tyre_age_at_start": _col("Int64", nullable=True),
        }
    ),
    "race_control": _schema(
        {
            "date": _col(TS),
            "lap_number": _col("Int64", nullable=True),
            "category": _col(str),
            "flag": _col(str, nullable=True),
            "scope": _col(str, nullable=True),
            "message": _col(str),
        }
    ),
    "weather": _schema(
        {
            "date": _col(TS),
            "air_temperature": _col("float64"),
            "track_temperature": _col("float64"),
            "rainfall": _col("int64"),
        }
    ),
    "session_result": _schema(
        {
            "driver_number": _col("int64"),
            "position": _col("Int64", nullable=True),
            "dnf": _col(bool),
            "dns": _col(bool),
            "dsq": _col(bool),
        },
        not_empty=True,
    ),
    "drivers": _schema(
        {
            "driver_number": _col("int64"),
            "name_acronym": _col(str),
            "team_name": _col(str),
            "full_name": _col(str),
        },
        not_empty=True,
    ),
}

TABLES = tuple(SCHEMAS)


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


def validate_tables(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Error messages of the seven tables; empty when all pass."""
    errors = []
    for name, schema in SCHEMAS.items():
        if name not in tables:
            errors.append(f"{name}: table is missing")
        else:
            errors += _errors(name, schema, tables[name])
    return errors
