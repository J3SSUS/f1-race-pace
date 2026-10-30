"""The metrics of the training, all on the original scale of the target and over valid rows.

`score` is the only place they are computed. MAE is the main one; the others say whether a model is also unbiased,
explains variance and orders the drivers of a race.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

METRICS = ("mae", "rmse", "r2", "mean_error", "pairwise_accuracy")


def pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray, races: pd.DataFrame) -> float:
    """Mean over races of the share of driver pairs ordered as in reality. Pairs with the same actual value are
    left out; a pair counts 1 if the predicted order matches, 0 if it is reversed and 0.5 if the predictions tie.
    A race without pairs adds nothing; if no race has a pair it raises.
    """
    per_race = []
    keys = pd.DataFrame({"year": races["year"].to_numpy(), "round": races["round"].to_numpy()})
    for rows in keys.groupby(["year", "round"]).indices.values():
        t, p = y_true[rows], y_pred[rows]
        upper = np.triu(np.ones((len(t), len(t)), dtype=bool), k=1)
        true_sign = np.sign(t[:, None] - t[None, :])[upper]
        pred_sign = np.sign(p[:, None] - p[None, :])[upper]
        pairs = true_sign != 0
        if pairs.any():
            per_race.append(np.mean((1 + true_sign[pairs] * pred_sign[pairs]) / 2))
    if not per_race:
        raise ValueError("pairwise accuracy needs at least one race with two drivers of different actual value")
    return float(np.mean(per_race))


def score(y_true, y_pred, races: pd.DataFrame) -> dict[str, float]:
    """The `METRICS` of a set of rows. `races` has the `year` and `round` of each row, in the same order.
    `mean_error` is `mean(y_pred - y_true)`: positive leans toward "slower than the field". `r2` takes the mean of
    the evaluated `y_true` as its reference.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_error": float(np.mean(y_pred - y_true)),
        "pairwise_accuracy": pairwise_accuracy(y_true, y_pred, races),
    }
