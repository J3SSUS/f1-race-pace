"""The `training` step: the baseline and the tuned models of a sampling run, each registered as a version.

Training never promotes: no `champion` alias, no `champion_history`, no prediction and no test race.
"""

from collections.abc import Callable

from mlflow import MlflowClient

from common.config import TrainingConfig, load_config
from common.supabase_io import get_client
from features.catalog import load_catalog
from training.baseline import QUALIFYING
from training.data import load_train
from training.registry import Version, feature_set_id_of, last_train_race_id, register_version
from training.runs import run_references
from training.sampling import SAMPLING_TAG, read_split, resolve_tracking_uri
from training.tuning import select_models, tune_models


def finished_run_id(run_id: str, name: str, tracking_uri: str | None = None) -> str | None:
    """The latest finished run `name` tagged with the sampling run `run_id`, or None if there is none."""
    client = MlflowClient(resolve_tracking_uri(tracking_uri))
    runs = client.search_runs(
        [client.get_run(run_id).info.experiment_id],
        filter_string=f"tags.{SAMPLING_TAG} = '{run_id}' and attributes.run_name = '{name}' "
        "and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def run_training(
    run_id: str,
    config: TrainingConfig | None = None,
    client=None,
    tracking_uri: str | None = None,
    loader: Callable = load_train,
    models: list[str] | None = None,
    on_version: Callable[[str, Version], None] | None = None,
) -> dict[str, Version]:
    """Register the baseline and then tune and register each enabled model (or only the ones in `models`).

    Before anything runs it checks the model names, the split of `run_id` and the feature set and races in
    Supabase. The references (baseline and floor) run only if the sampling run has no finished `baseline` run.
    `on_version(name, version)` is called as each version is registered. A model that raises leaves the versions
    already registered in place and the error propagates. Returns the versions by name: the baseline first, then
    the models in config order.
    """
    config = config or load_config()
    select_models(config, models)
    client = client if client is not None else get_client()
    feature_set_id_of(client, load_catalog().version)
    last_train_race_id(client, read_split(run_id, tracking_uri))

    versions: dict[str, Version] = {}

    def register(name: str, mlflow_run_id: str) -> None:
        versions[name] = register_version(name, mlflow_run_id, client, tracking_uri)
        if on_version:
            on_version(name, versions[name])

    baseline = finished_run_id(run_id, QUALIFYING, tracking_uri)
    if baseline is None:
        run_references(run_id, config, client, tracking_uri, loader)
        baseline = finished_run_id(run_id, QUALIFYING, tracking_uri)
    register(QUALIFYING, baseline)
    tune_models(run_id, config, client, tracking_uri, loader, models, on_model=register)
    return versions
