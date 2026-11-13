"""Build the `site/snapshot.json` the page falls back to when Supabase does not answer (format of SPEC 16).

The rules here (who is the champion, who is the baseline, which metrics are kept) are the same as `findChampion`,
`findBaseline` and `normalizeLive` in `docs/site/site.js`. One golden fixture in `tests/site/golden/` is checked by
pytest and by Node, so the two implementations cannot drift apart.

Input rows are a dictionary with the keys of the JavaScript: `championHistory`, `modelVersions`, `races`, `metrics`
and `results`. Every `ValueError` names the rule that failed and never carries a row's content.
"""

SCHEMA_VERSION = 1


def find_champion(champion_history: list[dict], model_versions: list[dict]) -> dict:
    """The champion is the `champion_history` row with no `demoted_at`; its algorithm comes from `model_versions`."""
    current = [row for row in champion_history if row.get("demoted_at") is None]
    if len(current) != 1:
        raise ValueError(f"champion_history: expected one current champion, found {len(current)}")
    version_id = current[0]["model_version_id"]
    version = next((row for row in model_versions if row["model_version_id"] == version_id), None)
    if version is None:
        raise ValueError(f"model_versions: no row for the champion (version {version_id})")
    return {"model_version_id": version_id, "algorithm": version["algorithm"], "promoted_at": current[0]["promoted_at"]}


def find_baseline(model_versions: list[dict], metrics: list[dict]) -> dict:
    """The newest `baseline` version that has walk-forward metrics (as `find_baseline` does in SPEC 14)."""
    scored = {row["model_version_id"] for row in metrics}
    candidates = [r for r in model_versions if r["algorithm"] == "baseline" and r["model_version_id"] in scored]
    if not candidates:
        raise ValueError("model_versions: no baseline version has walk_forward metrics")
    newest = max(candidates, key=lambda row: row["model_version_id"])
    return {"model_version_id": newest["model_version_id"], "algorithm": "baseline"}


def normalize(rows: dict) -> dict:
    """The five parts of the snapshot: the champion, the baseline, the races, only their metrics and the results."""
    champion = find_champion(rows["championHistory"], rows["modelVersions"])
    baseline = find_baseline(rows["modelVersions"], rows["metrics"])
    scored = {champion["model_version_id"], baseline["model_version_id"]}
    results = [dict(row) for row in rows["results"]]
    if not results:
        raise ValueError("prediction_results: no rows for the champion")
    return {
        "champion": champion,
        "baseline": baseline,
        "races": [dict(row) for row in rows["races"]],
        "metrics": [dict(row) for row in rows["metrics"] if row["model_version_id"] in scored],
        "results": results,
    }


def build_snapshot(rows: dict, generated_at: str) -> dict:
    """`schema_version` 1, the time of the run and the five parts."""
    return {"schema_version": SCHEMA_VERSION, "generated_at": generated_at, **normalize(rows)}
