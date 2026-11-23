"""Store what actually happened in one race, on demand: `resolve_race` picks the race, `ground_truth` writes it.

Params `year` and `round` are both optional; with neither, the race is the last one held. The first task prints the race
as JSON to the pod's XCom file and the second takes `year` and `round` from it. The third, `metrics` (SPEC 24),
recomputes the `walk_forward` metrics of that year for every model version with predictions, so the champion and the
challenger are scored after each race; it runs in the `ml` image because it finds the baselines through MLflow, which
the `data` image of the ground truth does not have. A failure of `metrics` leaves the ground truth stored. No automatic
retries: a `STOPPED:` shows as a failed task, and the user triggers the DAG again (30 minutes after a session ends, if
it was live).
"""

from __future__ import annotations

from datetime import datetime

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sdk import DAG, Param
from f1_pods import pod_kwargs

# A null param is rendered as an empty string, which `next_race.py` reads as "not given".
YEAR = "{{ params.year if params.year is not none else '' }}"
ROUND = "{{ params.round if params.round is not none else '' }}"

with DAG(
    dag_id="ground_truth",
    schedule=None,
    start_date=datetime(2026, 11, 1),
    catchup=False,
    params={
        "year": Param(None, type=["null", "integer"], description="Season; empty for the last race held"),
        "round": Param(None, type=["null", "integer"], description="Round; empty for the last race held"),
    },
    default_args={"retries": 0},
    tags=["ground-truth", "cluster", "kind"],
) as dag:
    resolve_race = KubernetesPodOperator(
        task_id="resolve_race",
        name="resolve-race",
        arguments=[
            "scripts/next_race.py",
            "--which",
            "last",
            "--year",
            YEAR,
            "--round",
            ROUND,
            "--output",
            "/airflow/xcom/return.json",
        ],
        do_xcom_push=True,
        **pod_kwargs("data"),
    )
    ground_truth = KubernetesPodOperator(
        task_id="ground_truth",
        name="ground-truth",
        arguments=[
            "scripts/ground_truth.py",
            "--year",
            "{{ ti.xcom_pull(task_ids='resolve_race')['year'] }}",
            "--round",
            "{{ ti.xcom_pull(task_ids='resolve_race')['round'] }}",
        ],
        **pod_kwargs("data"),
    )
    metrics = KubernetesPodOperator(
        task_id="metrics",
        name="metrics",
        arguments=[
            "scripts/walk_forward_metrics.py",
            "--year",
            "{{ ti.xcom_pull(task_ids='resolve_race')['year'] }}",
        ],
        **pod_kwargs("ml"),
    )
    resolve_race >> ground_truth >> metrics
