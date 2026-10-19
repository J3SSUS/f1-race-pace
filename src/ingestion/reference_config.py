"""Loader and validation for the hand-checked reference files in config/ (circuits.yml, teams.yml).

`circuit_type` must be one of the three values the schema accepts; a draft entry still marked `review`
is rejected, so nothing loads until a person has decided. A duplicate `circuit_key` or `name` is rejected
too, and a missing key fails with the field name. `teams.yml` maps OpenF1 team names to one name per
lineage; a name that is not listed is stored as it comes.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
CIRCUITS_PATH = CONFIG_DIR / "circuits.yml"
TEAMS_PATH = CONFIG_DIR / "teams.yml"

CircuitType = Literal["street", "permanent", "hybrid"]


class Circuit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    circuit_key: int
    name: str
    country: str
    circuit_type: CircuitType

    @field_validator("circuit_type", mode="before")
    @classmethod
    def _decided(cls, value):
        if value == "review":
            raise ValueError("still marked 'review': decide street, permanent or hybrid")
        return value


def load_circuits(path: Path = CIRCUITS_PATH) -> dict[int, Circuit]:
    """Circuits of the file by OpenF1 `circuit_key`."""
    entries = yaml.safe_load(path.read_text()) or []
    if not isinstance(entries, list):
        raise ValueError(f"{path.name}: expected a list of circuits")
    circuits: dict[int, Circuit] = {}
    names: set[str] = set()
    for entry in entries:
        circuit = Circuit.model_validate(entry)
        if circuit.circuit_key in circuits:
            raise ValueError(f"{path.name}: duplicate circuit_key {circuit.circuit_key}")
        if circuit.name in names:
            raise ValueError(f"{path.name}: duplicate circuit name {circuit.name!r}")
        circuits[circuit.circuit_key] = circuit
        names.add(circuit.name)
    return circuits


def load_team_aliases(path: Path = TEAMS_PATH) -> dict[str, str]:
    """OpenF1 `team_name` -> stored team name. An alias target that is itself an alias is rejected."""
    data = yaml.safe_load(path.read_text()) or {}
    aliases = data.get("aliases") if isinstance(data, dict) else None
    if not isinstance(aliases, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in aliases.items()):
        raise ValueError(f"{path.name}: expected an `aliases` map of team names")
    chained = sorted(v for v in aliases.values() if v in aliases)
    if chained:
        raise ValueError(f"{path.name}: alias target is also an alias: {chained}")
    return aliases


def canonical_team(name: str, aliases: dict[str, str] | None = None) -> str:
    """The stored name of a team: its alias if it has one, otherwise the name as it comes."""
    aliases = load_team_aliases() if aliases is None else aliases
    return aliases.get(name, name)
