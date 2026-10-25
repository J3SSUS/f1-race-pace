"""The feature store in Supabase: load the features and the actual target of every race, and read them back.

Every write is an upsert, so a rerun never duplicates. Errors name the table and the key columns only, never the
service key. This is the only module of `features` that talks to Supabase.
"""

from dataclasses import dataclass

from common.supabase_io import upsert_rows
from features.catalog import Catalog
from features.race_info import RaceInfo

PAGE_SIZE = 1000  # PostgREST returns at most this many rows per request
LOAD_REFERENCE = "run scripts/load_reference.py first"


class StoreError(Exception):
    """A missing race, circuit or entry, a feature set whose hash changed, or a race that failed."""


@dataclass(frozen=True)
class LoadResult:
    run_id: str  # uuid4 hex of this run
    races: int  # races built and written
    entries: int  # race_entries covered
    feature_values: int  # rows written to feature_values
    ground_truth: int  # rows written to ground_truth
    valid_rows: int  # ground_truth rows with is_valid_row = true


def read_table(client, table: str, columns: str = "*") -> list[dict]:
    """Every row of `table`, read in pages of 1,000 until a short page, in the order the server returns them."""
    rows: list[dict] = []
    while True:
        try:
            page = client.table(table).select(columns).range(len(rows), len(rows) + PAGE_SIZE - 1).execute().data
        except Exception as exc:
            raise StoreError(f"read from {table} failed: {type(exc).__name__}") from None
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            return rows


@dataclass(frozen=True)
class FeatureSet:
    feature_set_id: int
    feature_ids: dict[str, int]  # feature name -> features.feature_id


def write_feature_set(client, catalog: Catalog, version: str = "v1") -> FeatureSet:
    """Write the 15 `features`, the feature set `version` and its `feature_set_items`; rerunning changes nothing.

    Two things stop it before writing anything: a feature set that already exists with another `schema_hash`
    (it is never overwritten; ask for a new version), and a feature that already exists with another group, dtype
    or `known_on_saturday` (`features.name` is unique and older sets share that row, so it needs a new name).
    """
    existing = next((s for s in read_table(client, "feature_sets") if s["version"] == version), None)
    if existing is not None and existing["schema_hash"] != catalog.schema_hash:
        raise StoreError(
            f"feature set {version} already exists with another schema_hash: "
            f"changing a feature needs a new version in config/features.yml"
        )

    rows = [
        {"name": f.name, "feature_group": f.group, "dtype": f.dtype, "known_on_saturday": f.known_on_saturday}
        for f in catalog.features
    ]
    stored = {row["name"]: row for row in read_table(client, "features")}
    for row in rows:
        old = stored.get(row["name"])
        changed = [c for c in ("feature_group", "dtype", "known_on_saturday") if old and old[c] != row[c]]
        if changed:
            raise StoreError(
                f"feature {row['name']} already exists with another {', '.join(changed)}: older feature sets share "
                f"that row, so a feature defined differently needs a new name"
            )

    features = upsert_rows(client, "features", rows, "name")
    feature_ids = {row["name"]: row["feature_id"] for row in features}
    (feature_set,) = upsert_rows(
        client, "feature_sets", [{"version": version, "schema_hash": catalog.schema_hash}], "version"
    )
    upsert_rows(
        client,
        "feature_set_items",
        [
            {"feature_set_id": feature_set["feature_set_id"], "feature_id": feature_ids[f.name], "position": f.position}
            for f in catalog.features
        ],
        "feature_set_id,feature_id",
    )
    return FeatureSet(feature_set["feature_set_id"], feature_ids)


@dataclass(frozen=True)
class RaceRef:
    info: RaceInfo
    entry_ids: dict[str, int]  # driver code -> race_entries.entry_id


def load_reference_index(client) -> dict[tuple[int, int], RaceRef]:
    """Every race in Supabase as `(year, round)` -> its `RaceInfo` and the `entry_id` of each driver code.

    A race without entries, a race whose circuit is missing or an entry whose driver is missing raises
    `StoreError` before anything is written; the remedy is `scripts/load_reference.py`.
    """
    circuits = {c["circuit_id"]: c for c in read_table(client, "circuits", "circuit_id,name,circuit_type")}
    codes = {d["driver_id"]: d["code"] for d in read_table(client, "drivers", "driver_id,code")}
    entries: dict[int, dict[str, int]] = {}
    for entry in read_table(client, "race_entries", "entry_id,race_id,driver_id"):
        if entry["driver_id"] not in codes:
            raise StoreError(f"entry {entry['entry_id']} has a driver that is not in drivers: {LOAD_REFERENCE}")
        entries.setdefault(entry["race_id"], {})[codes[entry["driver_id"]]] = entry["entry_id"]

    races = sorted(read_table(client, "races", "race_id,year,round,circuit_id"), key=lambda r: (r["year"], r["round"]))
    index = {}
    for race in races:
        label = f"{race['year']} round {race['round']}"
        circuit = circuits.get(race["circuit_id"])
        if circuit is None:
            raise StoreError(f"race {label} has a circuit that is not in circuits: {LOAD_REFERENCE}")
        if race["race_id"] not in entries:
            raise StoreError(f"race {label} has no race_entries: {LOAD_REFERENCE}")
        info = RaceInfo(race["year"], race["round"], circuit["name"], circuit["circuit_type"])
        index[(race["year"], race["round"])] = RaceRef(info, entries[race["race_id"]])
    return index
