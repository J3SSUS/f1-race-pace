"""Promote a model version to `champion`: the writes shared by the manual and the automatic promotion.

Sets the `champion` alias on that version in MLflow, closes the current row of `champion_history` (`demoted_at`) and
inserts the new one with `promoted_by` (`manual` by default) and the test MAE improvement over the baseline of the same
sampling run, read from `metrics`. It refuses a version with no `split = 'test'` metrics, so a champion has always
gone through `scripts/evaluate.py`. Promoting the version that is already champion changes nothing.
"""

from datetime import UTC, datetime

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from common.supabase_io import get_client, insert_rows, update_rows
from features.store import read_table
from training.registry import REGISTERED_MODEL
from training.sampling import SAMPLING_TAG, resolve_tracking_uri

ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"  # set by the refit (SPEC 24); a promoted version must not keep it
PROMOTED_BY = ("manual", "auto")  # `champion_history.promoted_by`


def version_test_mae(metrics: list[dict], model_version_id: int) -> float | None:
    """The `split = 'test'` MAE of a version in the `metrics` rows, or None."""
    values = [
        r["value"]
        for r in metrics
        if r["model_version_id"] == model_version_id and r["split"] == "test" and r["metric_name"] == "mae"
    ]
    return values[0] if values else None


def outgoing_champion(history: list[dict], model_version_id: int, mine: dict | None) -> int | None:
    """The champion that `model_version_id` replaces: the latest other row of `champion_history`, or None.

    `mine` is the current row of this version, if it already has one (a call after a failure): then only the rows
    before it count. A version that was champion earlier and comes back replaces the one that held the alias last.
    """
    before = [
        r
        for r in history
        if r["model_version_id"] != model_version_id and (mine is None or r["champion_id"] < mine["champion_id"])
    ]
    return max(before, key=lambda r: r["champion_id"])["model_version_id"] if before else None


def promote(model_version_id: int, promoted_by: str = "manual", client=None, tracking_uri: str | None = None) -> dict:
    """Make `model_version_id` the champion: MLflow alias first, then `champion_history`.

    Everything that can be wrong is checked before anything is written: an unknown version, a version that is not in
    MLflow, a version with no test metrics, or a sampling run without exactly one baseline with test metrics. The
    alias is set only if it is not already on the version, and the history is closed and inserted only if the version
    is not already the current row, so a call after a Supabase failure completes the row without a second alias
    change. The writes go in this order: the `champion` alias, the `challenger` alias (removed if it is on this
    version), `champion_history`, then the shadow mark: the champion that leaves gets `is_shadow = true` and every
    other version `false`. The outgoing champion is read from `champion_history` (the row before the one of this
    version), not from what this call closed, so a call after a failure completes the mark. The first champion has no
    outgoing one and marks nothing. Returns what it did: `changed` (anything was written), `alias_set`, `demoted` (the
    closed `champion_id`s), `shadow_set` (the version that is the shadow after the call, None for the first
    champion), `shadow_cleared` (versions whose mark this call set back to false), `challenger_alias_removed`, the
    test MAEs and `improvement_pct = (baseline - champion) / baseline * 100`.
    """
    if promoted_by not in PROMOTED_BY:
        raise ValueError(f"promoted_by must be one of {', '.join(PROMOTED_BY)}, not {promoted_by!r}")
    client = client if client is not None else get_client()
    ml = MlflowClient(resolve_tracking_uri(tracking_uri))

    versions = {
        r["model_version_id"]: r
        for r in read_table(client, "model_versions", "model_version_id,mlflow_run_id,is_shadow")
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
    try:
        on_challenger = str(ml.get_model_version_by_alias(REGISTERED_MODEL, CHALLENGER_ALIAS).version)
    except MlflowException:
        on_challenger = None
    history = read_table(client, "champion_history", "champion_id,model_version_id,demoted_at")
    current = [r for r in history if r["demoted_at"] is None]
    mine = next((r for r in current if r["model_version_id"] == model_version_id), None)
    outgoing = outgoing_champion(history, model_version_id, mine)
    marked = [i for i, row in versions.items() if row.get("is_shadow")]
    to_clear = [i for i in marked if i != outgoing]
    mark_missing = outgoing is not None and outgoing not in marked
    result = {
        "model_version_id": model_version_id,
        "mlflow_version": str(mlflow_version.version),
        "test_mae": champion_mae,
        "baseline_test_mae": baseline_mae,
        "improvement_pct": improvement,
        "changed": True,
        "alias_set": False,
        "demoted": [],
        "shadow_set": outgoing,
        "shadow_cleared": [],
        "challenger_alias_removed": False,
    }
    remove_challenger = on_challenger == result["mlflow_version"]
    only_mine = all(r["model_version_id"] == model_version_id for r in current)
    if on_alias == result["mlflow_version"] and mine is not None and only_mine:
        if not (remove_challenger or to_clear or mark_missing):
            return result | {"changed": False}

    if on_alias != result["mlflow_version"]:
        ml.set_registered_model_alias(REGISTERED_MODEL, ALIAS, result["mlflow_version"])
        result["alias_set"] = True
    if remove_challenger:
        ml.delete_registered_model_alias(REGISTERED_MODEL, CHALLENGER_ALIAS)
        result["challenger_alias_removed"] = True
    for row in current:  # the unique index allows one current row: close the others before inserting
        if row["model_version_id"] != model_version_id:
            now = datetime.now(UTC).isoformat()
            update_rows(client, "champion_history", {"demoted_at": now}, {"champion_id": row["champion_id"]})
            result["demoted"].append(row["champion_id"])
    if mine is None:
        insert_rows(
            client,
            "champion_history",
            [
                {
                    "model_version_id": model_version_id,
                    "promoted_by": promoted_by,
                    "test_mae_improvement_pct": improvement,
                }
            ],
        )
    for version_id in to_clear:
        update_rows(client, "model_versions", {"is_shadow": False}, {"model_version_id": version_id})
        result["shadow_cleared"].append(version_id)
    if mark_missing:
        update_rows(client, "model_versions", {"is_shadow": True}, {"model_version_id": outgoing})
    return result
