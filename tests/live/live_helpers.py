"""The synthetic race of `tests/features` as the live scripts see it: its reference tables and its raw sessions."""

import json
from pathlib import Path

from synthetic_race import DRIVERS, write_race

ENTRY = {d.code: 100 + d.number for d in DRIVERS}  # driver code -> entry_id of race 2024 round 1


def reference() -> dict[str, list[dict]]:
    """Supabase tables with 2024 round 1 (the synthetic race): 8 drivers, 8 entries."""
    return {
        "circuits": [{"circuit_id": 1, "name": "Synthetic", "circuit_type": "permanent"}],
        "drivers": [{"driver_id": d.number, "code": d.code} for d in DRIVERS],
        "races": [{"race_id": 10, "year": 2024, "round": 1, "circuit_id": 1}],
        "race_entries": [{"entry_id": ENTRY[d.code], "race_id": 10, "driver_id": d.number} for d in DRIVERS],
    }


def write_valid_race(raw_dir: Path) -> None:
    """The Q and R sessions of 2024 round 1 on disk, marked valid as `ingest_session` would leave them."""
    write_race(raw_dir, 2024, 1, "clean")
    for session in ("Q", "R"):
        meta_path = raw_dir / "2024" / "01" / session / "meta.json"
        meta_path.write_text(json.dumps(json.loads(meta_path.read_text()) | {"valid": True, "errors": []}))
