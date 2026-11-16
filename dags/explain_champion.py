"""Explain the current champion, on demand: `explain` (SHAP, charts) and then `model_card`, both on the `ml` image.

No schedule and no params: the DAG explains whatever the `champion` alias points to, so trigger it after
`scripts/promote.py`. The scripts keep their own code and write to the volume `pipeline-data` through their existing
options: the charts to `outputs/site/charts` and the card to `outputs/MODEL_CARD.md`. `k8s/fetch-outputs.sh` copies them
to the repository, where the user reviews and commits. `explain` also upserts `shap_values` in Supabase. No automatic
retries: a `STOPPED:` is something to read and fix.
"""

from __future__ import annotations

from datetime import datetime

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sdk import DAG
from f1_pods import pod_kwargs

CHARTS = "/app/data/outputs/site/charts"
CARD = "/app/data/outputs/MODEL_CARD.md"

with DAG(
    dag_id="explain_champion",
    schedule=None,
    start_date=datetime(2026, 11, 1),
    catchup=False,
    default_args={"retries": 0},
    tags=["explain", "cluster", "kind"],
) as dag:
    explain = KubernetesPodOperator(
        task_id="explain",
        name="explain",
        arguments=["scripts/explain.py", "--charts-dir", CHARTS],
        **pod_kwargs("ml"),
    )
    model_card = KubernetesPodOperator(
        task_id="model_card",
        name="model-card",
        arguments=["scripts/model_card.py", "--charts-dir", CHARTS, "--output", CARD],
        **pod_kwargs("ml"),
    )
    explain >> model_card
