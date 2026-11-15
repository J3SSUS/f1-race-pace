"""Smoke test of the cluster: one pod of each pipeline image, launched by Airflow.

`data_pod` lists /app/data and checks that the Supabase variables exist (never printing them);
`ml_pod` loads the champion from MLflow. Both only read. Run it on demand from the UI.
"""

from __future__ import annotations

from datetime import datetime

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sdk import DAG
from kubernetes.client import models as k8s

NAMESPACE = "f1"
TAG = "0.1.0"

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


def secret_env(name: str, secret: str) -> k8s.V1EnvVar:
    return k8s.V1EnvVar(
        name=name,
        value_from=k8s.V1EnvVarSource(secret_key_ref=k8s.V1SecretKeySelector(name=secret, key=name)),
    )


# The pod contract of SPEC 18/19: Parquets at /app/data, uid 1000, secrets and MLflow URL in the env.
POD = dict(
    namespace=NAMESPACE,
    in_cluster=True,
    image_pull_policy="IfNotPresent",
    env_vars=[
        secret_env("SUPABASE_URL", "supabase"),
        secret_env("SUPABASE_SERVICE_KEY", "supabase"),
        k8s.V1EnvVar(name="MLFLOW_TRACKING_URI", value="http://mlflow:5000"),
    ],
    volumes=[
        k8s.V1Volume(
            name="data",
            persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name="pipeline-data"),
        )
    ],
    volume_mounts=[k8s.V1VolumeMount(name="data", mount_path="/app/data")],
    security_context=k8s.V1PodSecurityContext(run_as_user=1000, run_as_group=1000, run_as_non_root=True, fs_group=1000),
    get_logs=True,
    on_finish_action="delete_pod",
    startup_timeout_seconds=300,
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
        image=f"f1-pipeline-data:{TAG}",
        arguments=["-c", DATA_CHECK],
        container_resources=k8s.V1ResourceRequirements(
            requests={"memory": "256Mi", "cpu": "100m"}, limits={"memory": "1Gi"}
        ),
        **POD,
    )
    ml_pod = KubernetesPodOperator(
        task_id="ml_pod",
        name="smoke-ml",
        image=f"f1-pipeline-ml:{TAG}",
        arguments=["-c", ML_CHECK],
        container_resources=k8s.V1ResourceRequirements(
            requests={"memory": "1Gi", "cpu": "500m"}, limits={"memory": "3Gi"}
        ),
        **POD,
    )
    data_pod >> ml_pod
