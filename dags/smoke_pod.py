"""Smoke test of the cluster: one pod of each pipeline image, launched by Airflow.

`data_pod` lists /app/data and checks that the Supabase variables exist (never printing them);
`ml_pod` loads the champion from MLflow. Both only read. Run it on demand from the UI.
"""

from __future__ import annotations

from datetime import datetime

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sdk import DAG
from f1_pods import pod_kwargs

DATA_CHECK = """
import os
print("/app/data:", sorted(os.listdir("/app/data")))
for name in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY"):
    print(name, "is set:", bool(os.environ.get(name)))
"""

ML_CHECK = (
    "import sys; sys.path.insert(0, 'src'); "
    "from training.explain import load_champion; "
    "c = load_champion(); "
    "print(c.model_version_id, c.algorithm, type(c.model).__name__)"
)


with DAG(
    dag_id="smoke_pod",
    schedule=None,
    start_date=datetime(2026, 11, 1),
    catchup=False,
    tags=["smoke", "cluster", "kind"],
) as dag:
    data_pod = KubernetesPodOperator(
        task_id="data_pod",
        name="smoke-data",
        arguments=["-c", DATA_CHECK],
        **pod_kwargs("data"),
    )
    ml_pod = KubernetesPodOperator(
        task_id="ml_pod",
        name="smoke-ml",
        arguments=["-c", ML_CHECK],
        **pod_kwargs("ml"),
    )
    data_pod >> ml_pod
