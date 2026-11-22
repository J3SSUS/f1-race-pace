"""Recompute the `walk_forward` metrics of a season from what Supabase holds (SPEC 24).

Scores every model version that has predictions in the year, and the baseline of each, on its own rows (valid ground
truth with a target), and upserts the result into `metrics` with `split = 'walk_forward'`. It reads `predictions`,
`ground_truth` and `model_versions`, and MLflow to find the baseline of each version (a refit gets the baseline of its
refit); it writes only `metrics`, so running it again duplicates nothing. It is the task `metrics` of the DAG
`ground_truth`, and what `scripts/backtest.py` runs at its end. Prints one line per model and the number of rows
written. Exits with code 1 and a `STOPPED:` line, writing nothing, if no race of the year has a prediction and a valid
ground truth, a version has no `model_versions` row or no baseline of its kind, two versions share a baseline, or
Supabase or MLflow fail.

    uv run python scripts/walk_forward_metrics.py --year 2026
"""

import argparse
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.supabase_io import get_client  # noqa: E402
from features.store import StoreError  # noqa: E402
from live.metrics import MetricsError, MetricsResult, ModelMetrics, walk_forward_metrics  # noqa: E402


def describe(model: ModelMetrics) -> str:
    counts = f"{model.races} races, {model.rows} rows, MAE {model.scores['mae']:.3f}"
    if model.baseline_of is None:
        return f"model version {model.model_version_id} ({model.algorithm}): {counts}"
    return f"baseline version {model.model_version_id} (of model version {model.baseline_of}): {counts}"


def main(
    argv: list[str] | None = None,
    *,
    client=None,
    tracking_uri: str | None = None,
    runner: Callable[..., MetricsResult] = walk_forward_metrics,
    out: Callable[[str], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True, help="season whose metrics are recomputed")
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)
    try:
        client = client or get_client()
        result = runner(client, year=args.year, tracking_uri=tracking_uri)
    except (MetricsError, StoreError, RuntimeError) as exc:
        out(f"STOPPED: {exc}")
        return 1
    for model in result.models:
        out(describe(model))
    out(f"metrics rows written: {result.rows_written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
