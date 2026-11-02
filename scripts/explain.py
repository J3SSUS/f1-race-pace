"""The `explain` step: explain the current champion of `f1-race-pace` and publish the result.

Reads the `champion` alias, computes SHAP by feature, permutation importance, the Mincer-Zarnowitz calibration line
and two local cases, upserts the SHAP values into Supabase (`shap_values`), writes the charts and `explain.json` to
`site/charts/` and logs an `explain` run in MLflow. It can run any number of times and never promotes. Exits with code 1
and a `STOPPED:` line if it cannot (no `champion` alias, an algorithm without an explainer, a missing version or a
Supabase error); nothing is written to `site/charts/` when Supabase fails.

    uv run python scripts/explain.py
"""

import argparse
import logging
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlflow.exceptions import MlflowException  # noqa: E402

from features.store import StoreError  # noqa: E402
from training.explain import CHARTS_DIR, Explanation, explainer_kind, run_explain  # noqa: E402

TOP = 5  # the most important features that are printed


def describe(result: dict, charts_dir: Path) -> list[str]:
    explanation: Explanation = result["explanation"]
    champion = result["champion"]
    lines = [
        f"champion: version {champion.model_version_id} ({champion.algorithm}), "
        f"explained with the {explainer_kind(champion.algorithm)} explainer"
    ]
    lines.append(f"SHAP rows upserted: {result['shap_rows']} (base value {explanation.base_value:+.3f})")
    lines.append(f"top {TOP} features by permutation importance (rise of the MAE when permuted):")
    for row in explanation.importance.head(TOP).itertuples():
        lines.append(f"  {row.feature:<28}{row.mae_increase_mean:>+8.3f} ± {row.mae_increase_std:.3f}")
    c = explanation.calibration
    lines.append(
        f"calibration (Mincer-Zarnowitz): actual = {c['intercept']:+.3f} + {c['slope']:.3f} × predicted "
        f"({int(c['n_rows'])} out-of-fold rows, {int(c['n_races'])} races)"
    )
    for case in explanation.cases:
        lines.append(
            f"case {case.label}: {case.driver}, {case.year} round {case.round}: "
            f"prediction {case.prediction:+.3f}, actual {case.actual:+.3f}, error {case.error:+.3f}"
        )
    lines.append(f"MLflow run: {result['run_id']}")
    lines.append(f"files written to {charts_dir}:")
    lines.extend(f"  {path.name}" for path in result["files"])
    return lines


def main(
    argv: list[str] | None = None, *, runner: Callable = run_explain, out: Callable[[str], None] | None = None
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--charts-dir", type=Path, default=ROOT / CHARTS_DIR, help="where the charts are written")
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)

    try:
        result = runner(charts_dir=args.charts_dir)
    except (ValueError, StoreError, RuntimeError, MlflowException) as exc:
        out(f"STOPPED: {exc}")
        return 1
    for line in describe(result, args.charts_dir):
        out(line)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())
