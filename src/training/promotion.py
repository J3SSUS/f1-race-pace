"""Promote a model version to `champion`: the writes shared by the manual and the automatic promotion.

Sets the `champion` alias on that version in MLflow, closes the current row of `champion_history` (`demoted_at`) and
inserts the new one with `promoted_by` (`manual` by default) and the test MAE improvement over the baseline of the same
sampling run, read from `metrics`. It refuses a version with no `split = 'test'` metrics, so a champion has always
gone through `scripts/evaluate.py`. Promoting the version that is already champion changes nothing.
"""

import math
from datetime import UTC, datetime

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from common.supabase_io import get_client, insert_rows, update_rows
from features.store import read_table
from training.registry import REGISTERED_MODEL
from training.sampling import SAMPLING_TAG, resolve_tracking_uri

ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"  # set by the refit (SPEC 24); a promoted version must not keep it

# The promotion rule of `CLAUDE.md`: not a training knob, so it is not in `config/training.json`.
PROMOTION_WINDOW = 5  # last common races compared
PROMOTION_RATIO = 0.98  # the challenger's mean MAE must be at most 0.98 * the champion's
PROMOTION_MIN_WINS = 3  # and it must not be worse than the champion in at least 3 of the 5 races
_TOLERANCE = 1e-12  # a mean exactly on the 0.98 boundary must not fail on float rounding
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


def promote(
    model_version_id: int,
    promoted_by: str = "manual",
    client=None,
    tracking_uri: str | None = None,
    improvement_pct: float | None = None,
) -> dict:
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

    `improvement_pct` is given only by an automatic promotion (`auto_promote`): a challenger is a refit and has no
    test metrics, so the number that justified the promotion is the forward one, over the races of the rule. Then the
    test metrics are not read or required, `test_mae` and `baseline_test_mae` are None, and the value goes to
    `champion_history.test_mae_improvement_pct`. A manual promotion always needs test metrics.
    """
    if promoted_by not in PROMOTED_BY:
        raise ValueError(f"promoted_by must be one of {', '.join(PROMOTED_BY)}, not {promoted_by!r}")
    if improvement_pct is not None and promoted_by != "auto":
        raise ValueError("improvement_pct is only given by an automatic promotion")
    client = client if client is not None else get_client()
    ml = MlflowClient(resolve_tracking_uri(tracking_uri))

    versions = {
        r["model_version_id"]: r
        for r in read_table(client, "model_versions", "model_version_id,mlflow_run_id,is_shadow")
    }
    if model_version_id not in versions:
        raise ValueError(f"model version {model_version_id} does not exist in model_versions")
    champion_mae = baseline_mae = None
    if improvement_pct is None:
        metrics = read_table(client, "metrics", "model_version_id,split,metric_name,value")
        champion_mae = version_test_mae(metrics, model_version_id)
        if champion_mae is None:
            raise ValueError(f"model version {model_version_id} has no test metrics: run scripts/evaluate.py first")

    by_run = {v.tags.get("mlflow_run_id"): v for v in ml.search_model_versions(f"name = '{REGISTERED_MODEL}'")}
    mlflow_version = by_run.get(versions[model_version_id]["mlflow_run_id"])
    if mlflow_version is None:
        raise ValueError(f"model version {model_version_id} is not a version of {REGISTERED_MODEL} in MLflow")
    if improvement_pct is None:
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
                f"the sampling run {sampling_run_id} has {len(baselines)} baseline versions with test metrics, "
                "expected 1"
            )
        baseline_mae = version_test_mae(metrics, baselines[0])
        if baseline_mae == 0:
            raise ValueError("the baseline test MAE is 0: the improvement is undefined")
        improvement_pct = (baseline_mae - champion_mae) / baseline_mae * 100

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
        "improvement_pct": improvement_pct,
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
                    "test_mae_improvement_pct": improvement_pct,
                }
            ],
        )
    for version_id in to_clear:
        update_rows(client, "model_versions", {"is_shadow": False}, {"model_version_id": version_id})
        result["shadow_cleared"].append(version_id)
    if mark_missing:
        update_rows(client, "model_versions", {"is_shadow": True}, {"model_version_id": outgoing})
    return result


def evaluate_rule(
    champion_by_race: dict[int, float],
    challenger_by_race: dict[int, float],
    race_order: dict[int, tuple[int, int]],
) -> dict:
    """The promotion rule on the per-race MAE of the champion and the challenger. Pure: it reads and writes nothing.

    Only the races both have (and `race_order` knows, as `(year, round)`) count, and only the last `PROMOTION_WINDOW`
    of them. It passes if there are that many, the challenger's mean MAE over them is at most `PROMOTION_RATIO` times
    the champion's, and the challenger is not worse (a tie is not worse) in at least `PROMOTION_MIN_WINS` of them.
    Returns `passes`, `reason` (None if it passes, else why not, with the numbers), `common_races`, `window` (race ids,
    oldest first), `per_race` (`(race_id, champion_mae, challenger_mae)`), the two means, `races_worse` and
    `improvement_pct = (champion_mean - challenger_mean) / champion_mean * 100` (None below the window).
    """
    common = sorted(set(champion_by_race) & set(challenger_by_race) & set(race_order), key=race_order.__getitem__)
    out = {
        "passes": False,
        "reason": None,
        "common_races": len(common),
        "window": [],
        "per_race": [],
        "challenger_mae": None,
        "champion_mae": None,
        "races_worse": None,
        "improvement_pct": None,
    }
    if len(common) < PROMOTION_WINDOW:
        missing = PROMOTION_WINDOW - len(common)
        return out | {
            "reason": f"only {len(common)} common races, {PROMOTION_WINDOW} needed ({missing} more): not promoted yet"
        }

    window = common[-PROMOTION_WINDOW:]
    per_race = [(race, champion_by_race[race], challenger_by_race[race]) for race in window]
    champion_mean = math.fsum(row[1] for row in per_race) / len(per_race)
    challenger_mean = math.fsum(row[2] for row in per_race) / len(per_race)
    worse = sum(1 for _, champion, challenger in per_race if challenger > champion)
    improvement = (champion_mean - challenger_mean) / champion_mean * 100 if champion_mean else 0.0
    out |= {
        "window": window,
        "per_race": per_race,
        "challenger_mae": challenger_mean,
        "champion_mae": champion_mean,
        "races_worse": worse,
        "improvement_pct": improvement,
    }
    reasons = []
    if challenger_mean > PROMOTION_RATIO * champion_mean + _TOLERANCE:
        reasons.append(
            f"mean MAE {challenger_mean:.4f} is above {PROMOTION_RATIO} x the champion's {champion_mean:.4f} "
            f"({PROMOTION_RATIO * champion_mean:.4f})"
        )
    if PROMOTION_WINDOW - worse < PROMOTION_MIN_WINS:
        reasons.append(
            f"worse than the champion in {worse} of {PROMOTION_WINDOW} races "
            f"(not worse is needed in at least {PROMOTION_MIN_WINS})"
        )
    if reasons:
        return out | {"reason": "; ".join(reasons)}
    return out | {"passes": True}


def per_race_mae(metrics: list[dict], model_version_id: int) -> dict[int, float]:
    """`race_id -> MAE` of a version from its per-race `walk_forward` rows (`segment_type = 'overall'`)."""
    return {
        r["race_id"]: r["value"]
        for r in metrics
        if r["model_version_id"] == model_version_id
        and r["split"] == "walk_forward"
        and r["metric_name"] == "mae"
        and r["segment_type"] == "overall"
        and r["race_id"] is not None
    }


def auto_promote(
    challenger_version_id: int, client=None, tracking_uri: str | None = None, dry_run: bool = False
) -> dict:
    """Promote the challenger if the rule of `evaluate_rule` passes against the current champion.

    It stops with an error, before anything is written, for a version that does not exist, one that does not hold the
    `challenger` alias (SPEC 24), a missing or double current champion, or the challenger being the champion. "The
    rule is not met" is not an error: nothing is written and the result says why. With `dry_run` it evaluates and
    writes nothing even when the rule passes. When it promotes, the result carries the keys of `promote` too; the
    forward improvement goes to `champion_history` and no test metrics are needed (a refit has none).
    """
    client = client if client is not None else get_client()
    ml = MlflowClient(resolve_tracking_uri(tracking_uri))

    versions = {
        r["model_version_id"]: r for r in read_table(client, "model_versions", "model_version_id,mlflow_run_id")
    }
    if challenger_version_id not in versions:
        raise ValueError(f"model version {challenger_version_id} does not exist in model_versions")
    by_run = {v.tags.get("mlflow_run_id"): v for v in ml.search_model_versions(f"name = '{REGISTERED_MODEL}'")}
    mlflow_version = by_run.get(versions[challenger_version_id]["mlflow_run_id"])
    if mlflow_version is None:
        raise ValueError(f"model version {challenger_version_id} is not a version of {REGISTERED_MODEL} in MLflow")
    try:
        on_challenger = str(ml.get_model_version_by_alias(REGISTERED_MODEL, CHALLENGER_ALIAS).version)
    except MlflowException:
        on_challenger = None
    if on_challenger != str(mlflow_version.version):
        raise ValueError(
            f"model version {challenger_version_id} does not hold the `{CHALLENGER_ALIAS}` alias: "
            "only the challenger can be promoted automatically"
        )

    history = read_table(client, "champion_history", "champion_id,model_version_id,demoted_at")
    current = [r["model_version_id"] for r in history if r["demoted_at"] is None]
    if len(current) != 1:
        raise ValueError(f"champion_history: expected one current champion, found {len(current)}")
    champion_version_id = current[0]
    if champion_version_id == challenger_version_id:
        raise ValueError(f"model version {challenger_version_id} is already the champion")

    metrics = read_table(client, "metrics", "model_version_id,race_id,split,segment_type,metric_name,value")
    race_order = {r["race_id"]: (r["year"], r["round"]) for r in read_table(client, "races", "race_id,year,round")}
    verdict = evaluate_rule(
        per_race_mae(metrics, champion_version_id), per_race_mae(metrics, challenger_version_id), race_order
    )
    result = verdict | {
        "challenger_version_id": challenger_version_id,
        "champion_version_id": champion_version_id,
        "dry_run": dry_run,
        "promoted": False,
    }
    if not verdict["passes"]:
        return result
    if dry_run:
        return result | {"reason": "dry run: the rule is met, nothing was written"}
    promotion = promote(
        challenger_version_id,
        "auto",
        client=client,
        tracking_uri=tracking_uri,
        improvement_pct=verdict["improvement_pct"],
    )
    return result | promotion | {"promoted": True}
