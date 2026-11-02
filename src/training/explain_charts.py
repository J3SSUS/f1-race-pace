"""The charts of the explanation, one function per chart, written as reproducible `.svg` files.

Each function takes what it draws, the model version (named in the title) and the file to write, and returns that
path. The SHAP charts (summary, bar, waterfall) are drawn by `shap.plots` in its own format; the others are built
with `Figure` by hand. Running one twice gives the same bytes: `svg.hashsalt` is fixed so the element ids do not
change, text stays text (`svg.fonttype = none`) and the date is left out of the metadata. In the hand-made charts
blue and red mean sign (negative = faster than the field, positive = slower).
"""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from training.explain import Case

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"  # magnitude; negative contributions (faster than the field)
RED = "#e34948"  # positive contributions (slower than the field)
ORANGE = "#eb6834"  # the fitted line
RAMP = LinearSegmentedColormap.from_list("blue", ["#86b6ef", "#104281"])  # sequential: low -> high feature value
RC = {
    "svg.hashsalt": "f1-race-pace",
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
}
METADATA = {"Creator": "f1-race-pace explain", "Date": None}  # no date: a rerun does not change the file


def _version(model_version_id: int) -> str:
    return f"model version {model_version_id}"


def _draw(title: str, size: tuple[float, float], subtitle: str | None = None):
    """A figure and one axes on the surface, with the title; the caller draws and then calls `_save`."""
    fig = Figure(figsize=size, layout="constrained", facecolor=SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    if subtitle:
        ax.set_title(subtitle, loc="left", fontsize=8.5, color=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig: Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata=METADATA, facecolor=SURFACE)
    return path


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def _horizontal_bars(ax, labels: list[str], values: list[float], colors: list[str], base=0.0):
    """Bars from `base` (a number, or one start per bar) with the first label on top and the signed value at the tip."""
    positions = np.arange(len(labels))
    ax.barh(positions, values, left=base, height=0.6, color=colors)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)
    starts = np.broadcast_to(np.asarray(base, dtype="float64"), len(values))
    edges = np.concatenate([starts, starts + np.asarray(values)])
    pad = 0.12 * float(edges.max() - edges.min())  # room for the value at each tip (margins() is blocked by bar edges)
    ax.set_xlim(float(edges.min()) - pad, float(edges.max()) + pad)
    for position, start, value in zip(positions, starts, values, strict=True):
        ax.annotate(
            _signed(value),
            (start + value, position),
            xytext=(4 if value >= 0 else -4, 0),
            textcoords="offset points",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=8,
            color=INK_2,
        )


def coefficients_chart(coefficients: dict[str, float] | None, model_version_id: int, path: Path) -> Path | None:
    """The coefficients of a linear champion by preprocessed column, largest in size first. Nothing for a tree model."""
    if coefficients is None:
        return None
    with matplotlib.rc_context(RC):
        ordered = sorted(coefficients.items(), key=lambda item: -abs(item[1]))
        fig, ax = _draw(
            f"Coefficients of the champion - {_version(model_version_id)}",
            (8, 0.4 * len(ordered) + 1.6),
            "negative = faster than the field, positive = slower",
        )
        _horizontal_bars(
            ax,
            [name for name, _ in ordered],
            [value for _, value in ordered],
            [BLUE if value < 0 else RED for _, value in ordered],
        )
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_xlabel("coefficient (numeric features standardized; circuit_type columns are 0/1)")
        return _save(fig, path)


def permutation_chart(importance: pd.DataFrame, model_version_id: int, path: Path) -> Path:
    """The rise of the MAE when each feature is permuted (mean, with one standard deviation), largest first."""
    with matplotlib.rc_context(RC):
        ordered = importance.sort_values("mae_increase_mean", ascending=False, kind="stable")
        fig, ax = _draw(
            f"Permutation importance - {_version(model_version_id)}",
            (8, 0.4 * len(ordered) + 1.6),
            "valid rows of train, refit champion; bars are the mean and the line one standard deviation",
        )
        positions = np.arange(len(ordered))
        ax.barh(positions, ordered["mae_increase_mean"], height=0.6, color=BLUE)
        ax.errorbar(
            ordered["mae_increase_mean"], positions, xerr=ordered["mae_increase_std"], fmt="none", ecolor=INK_2, lw=1
        )
        ax.set_yticks(positions, list(ordered["feature"]))
        ax.invert_yaxis()
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", length=0)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_xlabel("increase of the MAE when the feature is permuted (race_pace_delta_pct points)")
        return _save(fig, path)


def _explanation(shap_values: pd.DataFrame, X: pd.DataFrame | None, base_value: float):
    """A `shap.Explanation` with numbers only: a text column (`circuit_type`) has no color scale (NaN, gray)."""
    import shap

    data = None if X is None else X[shap_values.columns].apply(pd.to_numeric, errors="coerce").to_numpy("float64")
    return shap.Explanation(
        values=shap_values.to_numpy("float64"),
        base_values=np.full(len(shap_values), base_value, dtype="float64"),
        data=data,
        feature_names=list(shap_values.columns),
    )


def _shap_figure(draw, title: str, subtitle: str, path: Path, size: tuple[float, float]) -> Path:
    """Run a `shap.plots` call on a fresh pyplot figure, add the title and write the reproducible SVG."""
    import matplotlib.pyplot as plt

    with matplotlib.rc_context({**RC, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE}):
        fig = plt.figure(figsize=size)
        state = np.random.get_state()
        np.random.seed(0)  # the beeswarm jitter comes from the global generator: same file on every run
        try:
            draw()
            fig = plt.gcf()
            fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
            fig.text(0.01, 0.945, subtitle, ha="left", fontsize=8, color=INK_2)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            return _save(fig, path)
        finally:
            np.random.set_state(state)
            plt.close("all")


def shap_summary_chart(shap_values: pd.DataFrame, X: pd.DataFrame, model_version_id: int, path: Path) -> Path:
    """The `shap` beeswarm: one row per feature ordered by mean |SHAP|, one dot per driver-race, colored by value."""
    import shap

    explanation = _explanation(shap_values, X, 0.0)
    return _shap_figure(
        lambda: shap.plots.beeswarm(explanation, max_display=len(shap_values.columns), show=False, color_bar=True),
        f"SHAP summary - {_version(model_version_id)}",
        "each dot is a driver in a race (train and test); right = pushes the prediction toward slower",
        path,
        (8.5, 0.45 * len(shap_values.columns) + 1.8),
    )


def shap_bar_chart(shap_values: pd.DataFrame, model_version_id: int, path: Path) -> Path:
    """The `shap` bar: mean |SHAP| of each feature."""
    import shap

    explanation = _explanation(shap_values, None, 0.0)
    return _shap_figure(
        lambda: shap.plots.bar(explanation, max_display=len(shap_values.columns), show=False),
        f"Mean |SHAP| - {_version(model_version_id)}",
        "average absolute contribution of each feature (race_pace_delta_pct points)",
        path,
        (8.5, 0.4 * len(shap_values.columns) + 1.8),
    )


def waterfall_chart(case: "Case", model_version_id: int, path: Path) -> Path:
    """The `shap` waterfall of one case: from the base value to the prediction, one bar per feature."""
    import shap

    names = list(case.contributions)
    explanation = shap.Explanation(
        values=np.array([case.contributions[name] for name in names], dtype="float64"),
        base_values=case.base_value,
        feature_names=names,
    )
    who = f"{case.label.replace('_', ' ')}: {case.driver}, {case.year} round {case.round}"
    return _shap_figure(
        lambda: shap.plots.waterfall(explanation, max_display=len(names), show=False),
        f"{who} - {_version(model_version_id)}",
        f"base value {case.base_value:+.3f}, prediction {case.prediction:+.3f}, actual {case.actual:+.3f}",
        path,
        (8.5, 0.4 * len(names) + 1.8),
    )


def predicted_vs_actual_chart(oof: pd.DataFrame, calibration: dict[str, float], model_version_id: int, path: Path):
    """Out-of-fold predicted against actual, with the line `y = x` and the Mincer-Zarnowitz fit."""
    with matplotlib.rc_context(RC):
        fig, ax = _draw(
            f"Predicted vs actual - {_version(model_version_id)}",
            (6.5, 6.5),
            f"{calibration['n_rows']} out-of-fold rows in {calibration['n_races']} races",
        )
        low = float(min(oof["y_pred"].min(), oof["y_true"].min()))
        high = float(max(oof["y_pred"].max(), oof["y_true"].max()))
        pad = 0.05 * (high - low)
        grid = np.array([low - pad, high + pad])
        ax.scatter(oof["y_pred"], oof["y_true"], s=14, color=BLUE, alpha=0.45, linewidths=0, label="out-of-fold rows")
        ax.plot(grid, grid, color=MUTED, linewidth=1, label="perfect calibration (y = x)")
        ax.plot(
            grid,
            calibration["intercept"] + calibration["slope"] * grid,
            color=ORANGE,
            linewidth=2,
            label=f"Mincer-Zarnowitz fit: slope {calibration['slope']:.2f}, intercept {calibration['intercept']:+.2f}",
        )
        ax.set_xlim(grid)
        ax.set_ylim(grid)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("predicted race_pace_delta_pct")
        ax.set_ylabel("actual race_pace_delta_pct")
        ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_2)
        return _save(fig, path)


def residuals_chart(oof: pd.DataFrame, model_version_id: int, path: Path) -> Path:
    """Out-of-fold residuals (prediction - actual, as everywhere else) against the prediction."""
    with matplotlib.rc_context(RC):
        fig, ax = _draw(
            f"Residuals vs predicted - {_version(model_version_id)}",
            (8, 5),
            "out-of-fold rows; above zero the model was slower than the actual, below zero faster",
        )
        ax.scatter(oof["y_pred"], oof["y_pred"] - oof["y_true"], s=14, color=BLUE, alpha=0.45, linewidths=0)
        ax.axhline(0, color=INK_2, linewidth=1)
        ax.set_xlabel("predicted race_pace_delta_pct")
        ax.set_ylabel("prediction - actual")
        return _save(fig, path)
