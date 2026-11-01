"""Promote a model version to `champion` by hand: the first champion of `f1-race-pace`.

Sets the `champion` alias on that version in MLflow, closes the current row of `champion_history` (`demoted_at`) and
inserts the new one with `promoted_by = manual` and the test MAE improvement over the baseline of the same sampling
run, read from `metrics`. It refuses a version with no `split = 'test'` metrics, so a champion has always gone through
`scripts/evaluate.py`. A non-positive improvement prints a warning and still promotes: the decision is the user's.
Promoting the version that is already champion changes nothing. Exits with code 1 and a `STOPPED:` line if it cannot.

    uv run python scripts/promote.py <model_version_id>
"""

import argparse
import logging
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlflow import MlflowClient  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402

from common.supabase_io import get_client, insert_rows, update_rows  # noqa: E402
from features.store import StoreError, read_table  # noqa: E402
from training.registry import REGISTERED_MODEL  # noqa: E402
from training.sampling import SAMPLING_TAG, resolve_tracking_uri  # noqa: E402

ALIAS = "champion"


def version_test_mae(metrics: list[dict], model_version_id: int) -> float | None:
    """The `split = 'test'` MAE of a version in the `metrics` rows, or None."""
    values = [
        r["value"]
        for r in metrics
        if r["model_version_id"] == model_version_id and r["split"] == "test" and r["metric_name"] == "mae"
    ]
    return values[0] if values else None


def promote(model_version_id: int, client=None, tracking_uri: str | None = None) -> dict:
    """Make `model_version_id` the champion: MLflow alias first, then `champion_history`.

    Everything that can be wrong is checked before anything is written: an unknown version, a version that is not in
    MLflow, a version with no test metrics, or a sampling run without exactly one baseline with test metrics. The
    alias is set only if it is not already on the version, and the history is closed and inserted only if the version
    is not already the current row, so a call after a Supabase failure completes the row without a second alias
    change. Returns what it did: `changed`, `alias_set`, `demoted` (the closed `champion_id`s), the test MAEs and
    `improvement_pct = (baseline - champion) / baseline * 100`.
    """
    client = client if client is not None else get_client()
    ml = MlflowClient(resolve_tracking_uri(tracking_uri))

    versions = {
        r["model_version_id"]: r for r in read_table(client, "model_versions", "model_version_id,mlflow_run_id")
    }
    if model_version_id not in versions:
        raise ValueError(f"model version {model_version_id} does not exist in model_versions")
    metrics = read_table(client, "metrics", "model_version_id,split,metric_name,value")
    champion_mae = version_test_mae(metrics, model_version_id)
    if champion_mae is None:
        raise ValueError(f"model version {model_version_id} has no test metrics: run scripts/evaluate.py first")

    by_run = {v.tags.get("mlflow_run_id"): v for v in ml.search_model_versions(f"name = '{REGISTERED_MODEL}'")}
    mlflow_version = by_run.get(versions[model_version_id]["mlflow_run_id"])
    if mlflow_version is None:
        raise ValueError(f"model version {model_version_id} is not a version of {REGISTERED_MODEL} in MLflow")
    sampling_run_id = mlflow_version.tags.get(SAMPLING_TAG, "")
    baselines = [
        row["model_version_id"]
        for row in versions.values()
        if (v := by_run.get(row["mlflow_run_id"])) is not None
        and v.tags.get(SAMPLING_TAG) == sampling_run_id
        and v.tags.get("algorithm") == "baseline"
        and version_test_mae(metrics, row["model_version_id"]) is not None
    ]
    if len(baselines) != 1:
        raise ValueError(
            f"the sampling run {sampling_run_id} has {len(baselines)} baseline versions with test metrics, expected 1"
        )
    baseline_mae = version_test_mae(metrics, baselines[0])
    if baseline_mae == 0:
        raise ValueError("the baseline test MAE is 0: the improvement is undefined")
    improvement = (baseline_mae - champion_mae) / baseline_mae * 100

    try:
        on_alias = str(ml.get_model_version_by_alias(REGISTERED_MODEL, ALIAS).version)
    except MlflowException:
        on_alias = None
    history = read_table(client, "champion_history", "champion_id,model_version_id,demoted_at")
    current = [r for r in history if r["demoted_at"] is None]
    result = {
        "model_version_id": model_version_id,
        "mlflow_version": str(mlflow_version.version),
        "test_mae": champion_mae,
        "baseline_test_mae": baseline_mae,
        "improvement_pct": improvement,
        "changed": True,
        "alias_set": False,
        "demoted": [],
    }
    if on_alias == result["mlflow_version"] and any(r["model_version_id"] == model_version_id for r in current):
        return result | {"changed": False}

    if on_alias != result["mlflow_version"]:
        ml.set_registered_model_alias(REGISTERED_MODEL, ALIAS, result["mlflow_version"])
        result["alias_set"] = True
    has_row = any(r["model_version_id"] == model_version_id for r in current)
    for row in current:  # the unique index allows one current row: close the others before inserting
        if row["model_version_id"] != model_version_id:
            now = datetime.now(UTC).isoformat()
            update_rows(client, "champion_history", {"demoted_at": now}, {"champion_id": row["champion_id"]})
            result["demoted"].append(row["champion_id"])
    if not has_row:
        insert_rows(
            client,
            "champion_history",
            [{"model_version_id": model_version_id, "promoted_by": "manual", "test_mae_improvement_pct": improvement}],
        )
    return result


def main(
    argv: list[str] | None = None, *, promoter: Callable = promote, out: Callable[[str], None] | None = None
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_version_id", type=int, help="model_versions.model_version_id that evaluate.py printed")
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)
    try:
        result = promoter(args.model_version_id)
    except (ValueError, StoreError, RuntimeError, MlflowException) as exc:
        out(f"STOPPED: {exc}")
        return 1
    if not result["changed"]:
        out(f"model version {result['model_version_id']} is already the champion: nothing changed")
        return 0
    pct = result["improvement_pct"]
    out(
        f"promoted model version {result['model_version_id']} (MLflow version {result['mlflow_version']}) to champion: "
        f"test MAE {result['test_mae']:.3f} vs baseline {result['baseline_test_mae']:.3f} ({pct:+.1f}%)"
    )
    if not result["alias_set"]:
        out("the alias was already on this version: only champion_history was completed")
    if result["demoted"]:
        out(f"closed champion_history rows: {', '.join(str(i) for i in result['demoted'])}")
    if pct <= 0:
        out("WARNING: the improvement over the baseline is not positive; promoted anyway")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())
