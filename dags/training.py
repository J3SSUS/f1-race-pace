"""Train and evaluate a candidate, on demand: `sample`, one `train` pod per model in series, and `evaluate`.

Param `models` is a comma-separated list of config keys. Empty trains the six models and runs `evaluate`; filled trains
only those, in config order, and skips `evaluate` because the selection needs the baseline and the six models. The
first training pod also runs the baseline and the floor (`train.py` does that when the sampling run has none). Every
DAG run starts with its own sampling run, so every run is one "iteration" of the test. Nothing here promotes: the
champion changes only with `scripts/promote.py`, by hand. `choose_models` runs before `sample`, so a wrong name fails
in seconds and not after the OpenF1 download. `sample` returns its `run_id` through the pod's XCom file. No automatic
retries: a `STOPPED:` is something to read and fix.
"""

from __future__ import annotations

from datetime import datetime

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sdk import DAG, Param, task
from f1_pods import pod_kwargs

# The six models of `config/training.json`, in config order. Fixed here: the Airflow image has no project code.
MODELS = ["ridge", "elasticnet", "random_forest", "xgboost", "lightgbm", "catboost"]


def select_models(text: str | None) -> list[str]:
    """The models to train for the param `models`: all of them if it is empty, otherwise the names given.

    A wrong name raises, which fails the task; `retries=0` makes it a single attempt.
    """
    names = [name.strip() for name in (text or "").split(",") if name.strip()]
    if not names:
        return list(MODELS)
    unknown = [name for name in dict.fromkeys(names) if name not in MODELS]
    if unknown:
        raise ValueError(f"unknown model(s): {', '.join(unknown)}. Valid keys: {', '.join(MODELS)}")
    return [name for name in MODELS if name in names]


with DAG(
    dag_id="training",
    schedule=None,
    start_date=datetime(2026, 11, 1),
    catchup=False,
    params={
        # `null` is allowed so the form can be sent empty (a plain string param is required there).
        "models": Param(
            None,
            type=["null", "string"],
            description="Comma-separated config keys (e.g. ridge,elasticnet); empty trains all six and evaluates",
        ),
    },
    default_args={"retries": 0},
    tags=["training", "cluster", "kind"],
) as dag:

    @task
    def choose_models(params=None) -> list[str]:
        return select_models((params or {}).get("models"))

    @task
    def train_arguments(models: list[str], sampled: dict) -> list[list[str]]:
        return [["scripts/train.py", sampled["run_id"], "--models", name] for name in models]

    @task.short_circuit
    def all_models(params=None) -> bool:
        return not ((params or {}).get("models") or "").strip()

    models = choose_models()
    sample = KubernetesPodOperator(
        task_id="sample",
        name="sample",
        arguments=["scripts/sample.py", "--output", "/airflow/xcom/return.json"],
        do_xcom_push=True,
        **pod_kwargs("ml"),
    )
    train = KubernetesPodOperator.partial(
        task_id="train",
        name="train",
        max_active_tis_per_dagrun=1,
        **pod_kwargs("ml"),
    ).expand(arguments=train_arguments(models, sample.output))
    evaluate = KubernetesPodOperator(
        task_id="evaluate",
        name="evaluate",
        arguments=["scripts/evaluate.py", "{{ ti.xcom_pull(task_ids='sample')['run_id'] }}"],
        **pod_kwargs("ml"),
    )
    models >> sample
    train >> all_models() >> evaluate
