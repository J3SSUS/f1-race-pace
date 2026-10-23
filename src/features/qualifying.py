"""Qualifying features: all of them come from `session_result`, so a Q session without laps still gives them."""

import numpy as np
import pandas as pd

from features.constants import QUALI_CAP_PCT


def _reached_q3(durations) -> int:
    times = np.asarray(durations, dtype=float)
    return int(len(times) >= 3 and not np.isnan(times[2]))


def qualifying_features(entries: pd.DataFrame) -> pd.DataFrame:
    """The qualifying features of the entries from `entries.qualifying_entries`.

    `quali_delta_pct` is the best time against the pole, in %, capped at `QUALI_CAP_PCT`. `teammate_quali_gap` is the
    driver's best time against the teammate's, in %, computed from the capped times (each time is limited to the
    pole plus the cap, so the teammate who drove normally does not inherit an extreme); NaN unless the team has
    exactly two entries, both with a time. `grid_position` is the Q position (null if the source has none) and
    `reached_q3` is 1 if the driver has a Q3 time.
    """
    best = entries["best_quali_time"]
    pole = best.min()
    capped = best.clip(upper=pole * (1 + QUALI_CAP_PCT / 100))
    teammate = capped.groupby(entries["team"]).transform(
        lambda times: times.to_numpy()[::-1] if len(times) == 2 else np.nan
    )
    return pd.DataFrame(
        {
            "driver_number": entries["driver_number"],
            "driver": entries["driver"],
            "quali_delta_pct": ((best / pole - 1) * 100).clip(upper=QUALI_CAP_PCT),
            "grid_position": entries["position"].astype("Int64"),
            "reached_q3": entries["duration"].map(_reached_q3),
            "teammate_quali_gap": (capped / teammate - 1) * 100,
        }
    )
