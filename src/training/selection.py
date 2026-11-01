"""Model selection on the out-of-fold errors of train.

The test never enters here: the candidate is chosen from cross-validated predictions only, so the test can still
be touched once, to report.
"""

import numpy as np


def paired_block_bootstrap(err_a, err_b, races, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Mean difference of absolute errors `mean(|err_a|) - mean(|err_b|)` and its 2.5 and 97.5 percentiles.

    The blocks are whole races: the drivers of a race share conditions (rain, safety car, circuit), so their errors
    are dependent and a race is resampled with all its rows. Both models share the same rows, so they are
    resampled together (paired). `races` holds one label per row. A positive difference means `a` is worse.
    """
    err_a = np.abs(np.asarray(err_a, dtype="float64"))
    err_b = np.abs(np.asarray(err_b, dtype="float64"))
    races = np.asarray(races)
    if err_a.ndim != 1 or err_b.ndim != 1 or races.ndim != 1:
        raise ValueError("err_a, err_b and races must be one-dimensional")
    if not len(err_a) == len(err_b) == len(races):
        raise ValueError(
            f"err_a, err_b and races must have the same length, got {len(err_a)}, {len(err_b)} and {len(races)}"
        )
    if len(err_a) == 0:
        raise ValueError("the bootstrap needs at least one row")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")

    diff = err_a - err_b
    _, block = np.unique(races, return_inverse=True)
    n_races = int(block.max()) + 1
    diff_sum = np.bincount(block, weights=diff, minlength=n_races)
    rows = np.bincount(block, minlength=n_races)

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_races, size=(n_boot, n_races))  # races drawn with replacement
    boot = diff_sum[picks].sum(axis=1) / rows[picks].sum(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)
