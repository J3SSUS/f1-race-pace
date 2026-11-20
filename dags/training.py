"""Train and evaluate a candidate, on demand: `sample`, one `train` pod per model in series, and `evaluate`.

Param `models` is a comma-separated list of config keys. Empty trains the six models and runs `evaluate`; filled trains
only those, in config order, and skips `evaluate` because the selection needs the baseline and the six models. The
first training pod also runs the baseline and the floor (`train.py` does that when the sampling run has none). Every
DAG run starts with its own sampling run, so every run is one "iteration" of the test. Nothing here promotes: the
champion changes only with `scripts/promote.py`, by hand. `choose_models` and `check_cutoff` run before `sample`, so
a wrong name or a badly written cutoff fails in seconds and not after the OpenF1 download. `sample` returns its
`run_id` through the pod's XCom file. No automatic retries: a `STOPPED:` is something to read and fix.

Param `cutoff` (`YYYY-R`, for example `2026-10`) makes the run a walk-forward retraining: the sampling uses the races
through that round (SPEC 22). Empty uses `data` of `config/training.json`. Here only the format is checked, because
the Airflow image has no project code; `sample.py` stops if the cutoff is not a race already held.
"""

from __future__ import annotations

import re
from datetime import datetime

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sdk import DAG, Param, task
from f1_pods import pod_kwargs

# The six models of `config/training.json`, in config order. Fixed here: the Airflow image has no project code.
MODELS = ["ridge", "elasticnet", "random_forest", "xgboost", "lightgbm", "catboost"]


CUTOFF_FORMAT = re.compile(r"(\d{4})-(\d{1,2})")


def parse_cutoff_param(text: str | None) -> str | None:
    """The cutoff of the param `cutoff` as `YYYY-R`, or None if it is empty (the config decides).

    Only the format is checked: four digits, a dash and a round of at least 1. A wrong value raises, which fails the
    task; `retries=0` makes it a single attempt. `sample.py` is the authority and also checks that the race was held.
    """
    value = (text or "").strip()
    if not value:
        return None
    match = CUTOFF_FORMAT.fullmatch(value)
    if match is None or int(match.group(2)) < 1:
        raise ValueError(f"cutoff {value!r} is not YYYY-R with a round of at least 1 (for example 2026-10)")
    return value


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
        "cutoff": Param(
            None,
            type=["null", "string"],
            description="Races through this round, YYYY-R (e.g. 2026-10); empty uses the data range of the config",
        ),
    },
    default_args={"retries": 0},
    tags=["training", "cluster", "kind"],
) as dag:

    @task
    def choose_models(params=None) -> list[str]:
        return select_models((params or {}).get("models"))

    @task
    def check_cutoff(params=None) -> str:
        # "" and not None for an empty cutoff: a task that returns None may leave no XCom for the next one to read.
        return parse_cutoff_param((params or {}).get("cutoff")) or ""

    @task
    def sample_arguments(cutoff: str) -> list[str]:
        arguments = ["scripts/sample.py", "--output", "/airflow/xcom/return.json"]
        return arguments + ["--cutoff", cutoff] if cutoff else arguments

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
        arguments=sample_arguments(check_cutoff()),
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
