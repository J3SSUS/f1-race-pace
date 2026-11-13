"""Build the `site/snapshot.json` the page falls back to when Supabase does not answer (format of SPEC 16).

The rules here (who is the champion, who is the baseline, which metrics are kept) are the same as `findChampion`,
`findBaseline` and `normalizeLive` in `docs/site/site.js`. One golden fixture in `tests/site/golden/` is checked by
pytest and by Node, so the two implementations cannot drift apart.

Input rows are a dictionary with the keys of the JavaScript: `championHistory`, `modelVersions`, `races`, `metrics`
and `results`. Every `ValueError` names the rule that failed and never carries a row's content.
"""

import json
import os
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
PAGE_SIZE = 1000  # PostgREST returns at most 1,000 rows per request

SNAPSHOT_COLUMNS = {  # the only keys a row of each part may have; anything else stops publish
    "champion": {"model_version_id", "algorithm", "promoted_at"},
    "baseline": {"model_version_id", "algorithm"},
    "races": {"race_id", "year", "round", "event_name", "race_date"},
    "metrics": {"model_version_id", "race_id", "metric_name", "value"},
    "results": {
        "year",
        "round",
        "driver",
        "team",
        "predicted_pace_delta_pct",
        "actual_pace_delta_pct",
        "is_valid_row",
        "is_late",
    },
}
TOP_LEVEL_KEYS = {"schema_version", "generated_at", *SNAPSHOT_COLUMNS}


def find_champion(champion_history: list[dict], model_versions: list[dict]) -> dict:
    """The champion is the `champion_history` row with no `demoted_at`; its algorithm comes from `model_versions`."""
    current = [row for row in champion_history if row.get("demoted_at") is None]
    if len(current) != 1:
        raise ValueError(f"champion_history: expected one current champion, found {len(current)}")
    version_id = current[0]["model_version_id"]
    version = next((row for row in model_versions if row["model_version_id"] == version_id), None)
    if version is None:
        raise ValueError(f"model_versions: no row for the champion (version {version_id})")
    return {"model_version_id": version_id, "algorithm": version["algorithm"], "promoted_at": current[0]["promoted_at"]}


def find_baseline(model_versions: list[dict], metrics: list[dict]) -> dict:
    """The newest `baseline` version that has walk-forward metrics (as `find_baseline` does in SPEC 14)."""
    scored = {row["model_version_id"] for row in metrics}
    candidates = [r for r in model_versions if r["algorithm"] == "baseline" and r["model_version_id"] in scored]
    if not candidates:
        raise ValueError("model_versions: no baseline version has walk_forward metrics")
    newest = max(candidates, key=lambda row: row["model_version_id"])
    return {"model_version_id": newest["model_version_id"], "algorithm": "baseline"}


def normalize(rows: dict) -> dict:
    """The five parts of the snapshot: the champion, the baseline, the races, only their metrics and the results."""
    champion = find_champion(rows["championHistory"], rows["modelVersions"])
    baseline = find_baseline(rows["modelVersions"], rows["metrics"])
    scored = {champion["model_version_id"], baseline["model_version_id"]}
    results = [dict(row) for row in rows["results"]]
    if not results:
        raise ValueError("prediction_results: no rows for the champion")
    return {
        "champion": champion,
        "baseline": baseline,
        "races": [dict(row) for row in rows["races"]],
        "metrics": [dict(row) for row in rows["metrics"] if row["model_version_id"] in scored],
        "results": results,
    }


def build_snapshot(rows: dict, generated_at: str) -> dict:
    """`schema_version` 1, the time of the run and the five parts."""
    return {"schema_version": SCHEMA_VERSION, "generated_at": generated_at, **normalize(rows)}


def check_columns(snapshot: dict) -> list[str]:
    """Problems found in the keys of a snapshot, one line per part and column; empty when it has only allowed columns.

    A line names the part and the column, never a value: `results: unexpected column 'x'`, `metrics: missing column
    'value'`. A part of rows is checked row by row, but each problem is reported once.
    """
    problems = [f"snapshot: unexpected key '{key}'" for key in sorted(snapshot.keys() - TOP_LEVEL_KEYS)]
    problems += [f"snapshot: missing key '{key}'" for key in sorted(TOP_LEVEL_KEYS - snapshot.keys())]
    for part, allowed in SNAPSHOT_COLUMNS.items():
        if part not in snapshot:
            continue
        value = snapshot[part]
        rows = [value] if isinstance(value, dict) else value
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            problems.append(f"{part}: not a row or a list of rows")
            continue
        seen = set().union(*(row.keys() for row in rows)) if rows else set()
        extra = seen - allowed
        missing = {column for column in allowed if any(column not in row for row in rows)}
        problems += [f"{part}: unexpected column '{column}'" for column in sorted(extra)]
        problems += [f"{part}: missing column '{column}'" for column in sorted(missing)]
    return problems


def write_snapshot(path: Path, snapshot: dict) -> None:
    """Write compact JSON with sorted keys through a temporary file, so a failure never leaves half a file.

    The same snapshot always gives the same bytes. `NaN` and `Infinity` are refused (they are not valid JSON and the
    page could not read them). On any failure the temporary file is removed and the previous file stays untouched.
    """
    text = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path = Path(path)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False)
    try:
        with handle:
            handle.write(text)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


class SupabaseReadError(RuntimeError):
    """A read failed. The message holds the table and the type of the error, never the error's own text, which could
    carry a URL or a key."""


def read_table(
    client, table: str, columns: str, eq: dict | None = None, null: str | None = None, order=()
) -> list[dict]:
    """Every row of `table`, `PAGE_SIZE` at a time, until a short page comes back.

    `eq` filters by equality, `null` keeps the rows where that column is NULL and `order` lists the sort columns.
    """
    rows: list[dict] = []
    while True:
        query = client.table(table).select(columns)
        for column, value in (eq or {}).items():
            query = query.eq(column, value)
        if null:
            query = query.is_(null, "null")
        for column in order:
            query = query.order(column)
        try:
            page = query.range(len(rows), len(rows) + PAGE_SIZE - 1).execute().data
        except Exception as error:
            raise SupabaseReadError(f"{table}: {type(error).__name__}") from None
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            return rows


def read_rows(client) -> dict:
    """The five public reads of the page, as the dictionary `normalize` takes (the keys of the JavaScript).

    The champion is looked up first because `prediction_results` is read only for its `model_version_id`; with no
    champion, two of them or a champion missing from `model_versions` this raises the `ValueError` of
    `find_champion` before the big read.
    """
    champion_history = read_table(
        client, "champion_history", "model_version_id,promoted_at,demoted_at", null="demoted_at"
    )
    model_versions = read_table(client, "model_versions", "model_version_id,algorithm,is_shadow")
    champion = find_champion(champion_history, model_versions)
    return {
        "championHistory": champion_history,
        "modelVersions": model_versions,
        "races": read_table(client, "races", "race_id,year,round,event_name,race_date", order=("year", "round")),
        "metrics": read_table(
            client,
            "metrics",
            "model_version_id,race_id,metric_name,value",
            eq={"split": "walk_forward", "segment_type": "overall"},
        ),
        "results": read_table(
            client,
            "prediction_results",
            "year,round,driver,team,predicted_pace_delta_pct,actual_pace_delta_pct,is_valid_row,is_late",
            eq={"model_version_id": champion["model_version_id"]},
            order=("year", "round", "driver"),
        ),
    }
