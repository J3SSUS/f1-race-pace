"""Loader and validation for the hand-checked reference files in config/ (circuits.yml).

`circuit_type` must be one of the three values the schema accepts; a draft entry still marked `review`
is rejected, so nothing loads until a person has decided. A duplicate `circuit_key` or `name` is rejected
too, and a missing key fails with the field name.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
CIRCUITS_PATH = CONFIG_DIR / "circuits.yml"

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
