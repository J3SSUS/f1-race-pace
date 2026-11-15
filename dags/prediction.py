"""Predict one race with the champion, on demand: `resolve_race` picks the race, `predict` stores the prediction.

Params `year` and `round` are both optional; with neither, the race is the next one (the first whose race session has
not started, so from the qualifying Saturday until the lights go out). To predict a race that already started, give its
`year` and `round`: the prediction is stored as late. The first task prints the race as JSON to the pod's XCom file and
the second takes `year` and `round` from it. No automatic retries: a `STOPPED:` shows as a failed task (a live session:
trigger the DAG again 30 minutes after it ends; already predicted or no champion: it will not change by itself).
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
    dag_id="prediction",
    schedule=None,
    start_date=datetime(2026, 11, 1),
    catchup=False,
    params={
        "year": Param(None, type=["null", "integer"], description="Season; empty for the next race"),
        "round": Param(None, type=["null", "integer"], description="Round; empty for the next race"),
    },
    default_args={"retries": 0},
    tags=["prediction", "cluster", "kind"],
) as dag:
    resolve_race = KubernetesPodOperator(
        task_id="resolve_race",
        name="resolve-race",
        arguments=[
            "scripts/next_race.py",
            "--which",
            "next",
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
    predict = KubernetesPodOperator(
        task_id="predict",
        name="predict",
        arguments=[
            "scripts/predict.py",
            "--year",
            "{{ ti.xcom_pull(task_ids='resolve_race')['year'] }}",
            "--round",
            "{{ ti.xcom_pull(task_ids='resolve_race')['round'] }}",
        ],
        **pod_kwargs("ml"),
    )
    resolve_race >> predict
