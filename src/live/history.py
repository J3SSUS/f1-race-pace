"""What the prediction script reads from Supabase about the races before the one it predicts.

Both functions only read. The history is the same table the training load fed to `form_features` (valid rows of
earlier races with their target), so a race predicted live gets the form features training would have given it.
"""

import pandas as pd

from features.form import HISTORY_COLUMNS, empty_history
from features.store import read_table


def _earlier_races(client, year: int, round: int) -> list[dict]:
    """The `races` rows before `(year, round)`, in order."""
    races = read_table(client, "races", "race_id,year,round,circuit_id")
    earlier = [r for r in races if (r["year"], r["round"]) < (year, round)]
    return sorted(earlier, key=lambda r: (r["year"], r["round"]))


def missing_ground_truth(client, year: int, round: int) -> list[tuple[int, int]]:
    """`(year, round)` of the earlier races, in order, with an entry that has no `ground_truth` row.

    A race with no entries has nothing to fill and is not listed. The same race and later races are ignored.
    """
    earlier = {r["race_id"]: (r["year"], r["round"]) for r in _earlier_races(client, year, round)}
    with_truth = {g["entry_id"] for g in read_table(client, "ground_truth", "entry_id")}
    missing = {
        earlier[e["race_id"]]
        for e in read_table(client, "race_entries", "entry_id,race_id")
        if e["race_id"] in earlier and e["entry_id"] not in with_truth
    }
    return sorted(missing)


def read_history(client, year: int, round: int) -> pd.DataFrame:
    """`HISTORY_COLUMNS` of the valid `ground_truth` rows (with a target) of the races before `(year, round)`.

    `circuit` is `circuits.name`, `team` is `teams.name` (the team of the entry) and `driver` is `drivers.code`.
    Sorted by `(year, round, driver)`; with no earlier rows it is the empty history of the first race.
    """
    races = pd.DataFrame(_earlier_races(client, year, round), columns=["race_id", "year", "round", "circuit_id"])
    truth = pd.DataFrame(
        read_table(client, "ground_truth", "entry_id,actual_pace_delta_pct,is_valid_row"),
        columns=["entry_id", "actual_pace_delta_pct", "is_valid_row"],
    )
    truth = truth[truth["is_valid_row"].fillna(False).astype(bool) & truth["actual_pace_delta_pct"].notna()]
    if races.empty or truth.empty:
        return empty_history()

    entries = pd.DataFrame(read_table(client, "race_entries", "entry_id,race_id,driver_id,team_id"))
    circuits = pd.DataFrame(read_table(client, "circuits", "circuit_id,name")).rename(columns={"name": "circuit"})
    drivers = pd.DataFrame(read_table(client, "drivers", "driver_id,code")).rename(columns={"code": "driver"})
    teams = pd.DataFrame(read_table(client, "teams", "team_id,name")).rename(columns={"name": "team"})
    frame = (
        truth.merge(entries, on="entry_id")
        .merge(races, on="race_id")
        .merge(circuits, on="circuit_id")
        .merge(drivers, on="driver_id")
        .merge(teams, on="team_id")
        .rename(columns={"actual_pace_delta_pct": "race_pace_delta_pct"})
    )
    frame = frame[HISTORY_COLUMNS].astype({"year": "int64", "round": "int64", "race_pace_delta_pct": "float64"})
    return frame.sort_values(["year", "round", "driver"], ignore_index=True)
