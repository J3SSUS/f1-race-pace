"""What the raw Parquets do not say about a race; the caller reads it from `races` and `circuits`."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RaceInfo:
    year: int
    round: int
    circuit: str  # circuits.name, from config/circuits.yml
    circuit_type: str  # street | permanent | hybrid
