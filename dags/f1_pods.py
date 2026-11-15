"""The pod contract of SPEC 18/19, written once: what every `KubernetesPodOperator` of the project shares.

`pod_kwargs("data")` or `pod_kwargs("ml")` returns the keyword arguments for the operator: the image, the Parquets
at /app/data from the PVC `pipeline-data`, the Supabase Secret and the MLflow URL in the environment, uid 1000, the
memory of each image and the logs. A change to the contract is one edit here. Imported by the pipeline files next to it.
"""

from __future__ import annotations

from kubernetes.client import models as k8s

NAMESPACE = "f1"
TAG = "0.1.0"

# Memory of each image: (request, limit), and the CPU request.
RESOURCES = {
    "data": {"requests": {"memory": "256Mi", "cpu": "100m"}, "limits": {"memory": "1Gi"}},
    "ml": {"requests": {"memory": "1Gi", "cpu": "500m"}, "limits": {"memory": "3Gi"}},
}


def secret_env(name: str, secret: str) -> k8s.V1EnvVar:
    return k8s.V1EnvVar(
        name=name,
        value_from=k8s.V1EnvVarSource(secret_key_ref=k8s.V1SecretKeySelector(name=secret, key=name)),
    )


def pod_kwargs(kind: str, memory: dict | None = None) -> dict:
    """Keyword arguments of a `KubernetesPodOperator` for the `data` or `ml` image.

    `memory` replaces the default requests and limits of the image when a step needs other values.
    """
    if kind not in RESOURCES:
        raise ValueError(f"kind must be one of {sorted(RESOURCES)}, not {kind!r}")
    resources = memory or RESOURCES[kind]
    return dict(
        namespace=NAMESPACE,
        in_cluster=True,
        image=f"f1-pipeline-{kind}:{TAG}",
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
        security_context=k8s.V1PodSecurityContext(
            run_as_user=1000, run_as_group=1000, run_as_non_root=True, fs_group=1000
        ),
        container_resources=k8s.V1ResourceRequirements(**resources),
        get_logs=True,
        on_finish_action="delete_pod",
        startup_timeout_seconds=300,
    )
