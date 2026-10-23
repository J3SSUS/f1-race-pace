"""Recent-form features: they look only at races before the one being built, and never across a regulation change."""

import pandas as pd

from features.constants import FORM_RACES, REGULATION_CHANGE_YEAR
from features.race_info import RaceInfo

HISTORY_COLUMNS = ["year", "round", "circuit", "driver", "team", "race_pace_delta_pct"]


def empty_history() -> pd.DataFrame:
    """The history of the first race: no earlier rows."""
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def _new_regulations(year):
    """True from the regulation change on; works on a year or on a column of years."""
    return year >= REGULATION_CHANGE_YEAR


def form_features(race: RaceInfo, entries: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """`driver_pace_last3`, `team_pace_last3` and `team_pace_this_circuit_prev` of each entry (`driver` and `team`).

    `history` has `HISTORY_COLUMNS`. Only its rows with a target, from races before this one and in the same
    regulation era, are read; order does not matter.

    - `driver_pace_last3`: mean target of the driver's last `FORM_RACES` rows.
    - `team_pace_last3`: mean of the team's last `FORM_RACES` race means (each over the team's rows of that race).
    - `team_pace_this_circuit_prev`: mean target of the team at this circuit the year before.

    Each is NaN without earlier values.
    """
    prior = history.dropna(subset=["race_pace_delta_pct"])
    earlier = (prior["year"] < race.year) | ((prior["year"] == race.year) & (prior["round"] < race.round))
    prior = prior[earlier & (_new_regulations(prior["year"]) == _new_regulations(race.year))].sort_values(
        ["year", "round"], kind="stable"
    )

    def last_mean(values: pd.Series) -> float:
        return float(values.tail(FORM_RACES).mean())

    driver_form = prior.groupby("driver")["race_pace_delta_pct"].agg(last_mean)
    team_race_means = prior.groupby(["team", "year", "round"])["race_pace_delta_pct"].mean()
    team_form = team_race_means.groupby("team").agg(last_mean)
    last_year_here = prior[(prior["circuit"] == race.circuit) & (prior["year"] == race.year - 1)]
    circuit_prev = last_year_here.groupby("team")["race_pace_delta_pct"].mean()

    return pd.DataFrame(
        {
            "driver": entries["driver"],
            "driver_pace_last3": entries["driver"].map(driver_form).astype(float),
            "team_pace_last3": entries["team"].map(team_form).astype(float),
            "team_pace_this_circuit_prev": entries["team"].map(circuit_prev).astype(float),
        }
    )
