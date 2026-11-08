"""Ground truth of one race: what actually happened, computed from the race session and stored in `ground_truth`.

It makes no prediction. The prediction script uses data up to Saturday and never writes `ground_truth`; neither
touches the other's table.
"""

from features.race_info import RaceInfo
from features.store import LOAD_REFERENCE, RaceRef, read_table


class GroundTruthError(Exception):
    """A reason to stop, worded for the user. Never carries a response body or a key."""


def read_race_ref(client, year: int, round: int) -> RaceRef:
    """The `RaceInfo` and the `entry_id` of each driver code of this one race.

    Unlike `load_reference_index`, a race that is not loaded yet (no entries) elsewhere does not matter: only the
    requested race is checked. A missing race, circuit or entries, or an entry whose driver is missing, raises
    `GroundTruthError` naming the race; the remedy is `scripts/load_reference.py`. Nothing is written.
    """
    label = f"{year} round {round}"
    races = read_table(client, "races", "race_id,year,round,circuit_id")
    race = next((r for r in races if r["year"] == year and r["round"] == round), None)
    if race is None:
        raise GroundTruthError(f"race {label} is not in races: {LOAD_REFERENCE}")

    circuits = read_table(client, "circuits", "circuit_id,name,circuit_type")
    circuit = next((c for c in circuits if c["circuit_id"] == race["circuit_id"]), None)
    if circuit is None:
        raise GroundTruthError(f"race {label} has a circuit that is not in circuits: {LOAD_REFERENCE}")

    entries = [
        e for e in read_table(client, "race_entries", "entry_id,race_id,driver_id") if e["race_id"] == race["race_id"]
    ]
    if not entries:
        raise GroundTruthError(f"race {label} has no race_entries: {LOAD_REFERENCE}")

    codes = {d["driver_id"]: d["code"] for d in read_table(client, "drivers", "driver_id,code")}
    entry_ids: dict[str, int] = {}
    for entry in entries:
        if entry["driver_id"] not in codes:
            raise GroundTruthError(
                f"race {label}: entry {entry['entry_id']} has a driver that is not in drivers: {LOAD_REFERENCE}"
            )
        entry_ids[codes[entry["driver_id"]]] = entry["entry_id"]
    return RaceRef(RaceInfo(year, round, circuit["name"], circuit["circuit_type"]), entry_ids)
