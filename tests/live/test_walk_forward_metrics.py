from datetime import UTC, datetime

import pytest
from fake_supabase import FakeSupabase
from mlflow import MlflowClient
from model_helpers import log_version, promote
from test_evaluation_frame import seeded
from test_manifest import config, make_parts

from features.catalog import load_catalog
from live.metrics import MetricsError, MetricsResult, walk_forward_metrics
from training.registry import REGISTERED_MODEL
from training.sampling import register_run

pytestmark = pytest.mark.slow  # every test logs real models in a local MLflow store: needs --runslow

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
SECRET = "sb_secret_hunter2"


@pytest.fixture
def parent(uri, tmp_path) -> str:
    return register_run(make_parts(), config(), [], split_dir=tmp_path / "splits", tracking_uri=uri, now=NOW)


@pytest.fixture
def other_parent(uri, tmp_path) -> str:
    return register_run(make_parts(), config(), [], split_dir=tmp_path / "other", tracking_uri=uri, now=NOW)


@pytest.fixture
def supabase(uri, parent) -> FakeSupabase:
    """The synthetic 2024 and 2025 predictions plus a champion (ridge) and a baseline of one sampling run."""
    client = seeded()
    client.tables["feature_sets"] = [{"feature_set_id": 5, "version": load_catalog().version}]
    client.tables["races"] += [{"race_id": 100 + i, "year": 2023, "round": i + 1} for i in range(3)]
    champion = log_version(uri, parent, client, "ridge")
    baseline = log_version(uri, parent, client, "baseline")
    for p in client.tables["predictions"]:
        p["model_version_id"] = champion.model_version_id
    client.champion, client.baseline = champion, baseline
    client.writes.clear()
    return client


def walk_forward(client) -> list[dict]:
    return [m for m in client.tables["metrics"] if m["split"] == "walk_forward"]


def test_it_scores_the_champion_and_its_baseline_and_upserts_the_rows(uri, supabase):
    result = walk_forward_metrics(supabase, year=2024, tracking_uri=uri)

    assert isinstance(result, MetricsResult) and result.year == 2024
    assert result.champion.model_version_id == supabase.champion.model_version_id
    assert result.champion.algorithm == "ridge"
    assert result.baseline.model_version_id == supabase.baseline.model_version_id
    assert result.baseline.algorithm == "baseline"
    assert (result.champion.races, result.champion.rows) == (2, 4)
    assert (result.baseline.races, result.baseline.rows) == (2, 4)
    # 2024: entries 1-4, actual 0.1 * id, champion 0.2 * id, baseline 0.3 * id
    assert result.champion.scores["mae"] == pytest.approx(0.1 * (1 + 2 + 3 + 4) / 4)
    assert result.baseline.scores["mae"] == pytest.approx(0.2 * (1 + 2 + 3 + 4) / 4)
    assert set(result.champion.scores) == {"mae", "rmse", "r2", "pairwise_accuracy"}
    assert result.rows_written == len(walk_forward(supabase)) == 2 * 3 * 4
    assert {m["split"] for m in walk_forward(supabase)} == {"walk_forward"}
    assert {t for t, _ in supabase.writes} == {"metrics"}


def test_running_it_again_duplicates_nothing(uri, supabase):
    walk_forward_metrics(supabase, year=2024, tracking_uri=uri)
    before = [dict(m) for m in walk_forward(supabase)]

    walk_forward_metrics(supabase, year=2024, tracking_uri=uri)

    assert walk_forward(supabase) == before


def test_a_new_race_changes_the_aggregate_and_adds_only_its_own_rows(uri, supabase):
    walk_forward_metrics(supabase, year=2024, tracking_uri=uri)
    before = {(m["model_version_id"], m["race_id"], m["metric_name"]): m["value"] for m in walk_forward(supabase)}
    supabase.tables["races"].append({"race_id": 13, "year": 2024, "round": 3})
    for entry_id, driver_id in ((6, 1), (7, 2)):
        supabase.tables["race_entries"].append({"entry_id": entry_id, "race_id": 13, "driver_id": driver_id})
        supabase.tables["ground_truth"].append(
            {"entry_id": entry_id, "actual_pace_delta_pct": 2.0 - entry_id, "is_valid_row": True}
        )
        supabase.tables["predictions"].append(
            {
                "entry_id": entry_id,
                "model_version_id": supabase.champion.model_version_id,
                "predicted_pace_delta_pct": 0.0,
                "baseline_pace_delta_pct": 5.0,
            }
        )

    walk_forward_metrics(supabase, year=2024, tracking_uri=uri)

    after = {(m["model_version_id"], m["race_id"], m["metric_name"]): m["value"] for m in walk_forward(supabase)}
    assert len(after) == len(before) + 2 * 4  # the rows of race 13 for both models
    for key, value in before.items():
        if key[1] is not None:
            assert after[key] == value  # the earlier races are untouched
    assert (
        after[(supabase.champion.model_version_id, None, "mae")]
        != before[(supabase.champion.model_version_id, None, "mae")]
    )


def test_more_than_one_champion_version_stops_it_and_writes_nothing(uri, parent, supabase):
    other = log_version(uri, parent, supabase, "elasticnet")
    supabase.tables["predictions"][0]["model_version_id"] = other.model_version_id
    supabase.writes.clear()

    with pytest.raises(MetricsError, match="several model versions"):
        walk_forward_metrics(supabase, year=2024, tracking_uri=uri)
    assert supabase.writes == [] and walk_forward(supabase) == []


@pytest.mark.parametrize("year", [2030, 2024])
def test_no_race_with_a_prediction_and_a_valid_ground_truth_stops_it_and_writes_nothing(uri, supabase, year):
    if year == 2024:
        for g in supabase.tables["ground_truth"]:
            g["is_valid_row"] = False

    with pytest.raises(MetricsError, match="both a prediction and a valid ground truth"):
        walk_forward_metrics(supabase, year=year, tracking_uri=uri)
    assert supabase.writes == [] and walk_forward(supabase) == []


def test_a_champion_without_a_baseline_of_its_sampling_run_stops_it(uri, other_parent, supabase):
    supabase.tables["model_versions"] = [r for r in supabase.tables["model_versions"] if r["algorithm"] != "baseline"]
    log_version(uri, other_parent, supabase, "baseline")  # of another sampling run
    supabase.writes.clear()

    with pytest.raises(MetricsError, match="no baseline"):
        walk_forward_metrics(supabase, year=2024, tracking_uri=uri)
    assert supabase.writes == []


def test_a_supabase_error_is_reported_by_type_without_its_text(uri, supabase):
    class Failing(FakeSupabase):
        def table(self, name):
            if name == "ground_truth":
                raise ConnectionError(f"boom {SECRET}")
            return super().table(name)

    failing = Failing(supabase.tables)

    with pytest.raises(MetricsError) as error:
        walk_forward_metrics(failing, year=2024, tracking_uri=uri)
    assert "ground_truth" in str(error.value) and "ConnectionError" in str(error.value)
    assert SECRET not in str(error.value)


def test_no_mlflow_run_is_written_and_the_alias_is_untouched(uri, supabase):
    promote(uri, supabase.champion)
    ml = MlflowClient(uri)
    runs = len(ml.search_runs(experiment_ids=[e.experiment_id for e in ml.search_experiments()]))
    alias = ml.get_model_version_by_alias(REGISTERED_MODEL, "champion").version
    history = list(supabase.tables.get("champion_history", []))

    walk_forward_metrics(supabase, year=2024, tracking_uri=uri)

    assert len(ml.search_runs(experiment_ids=[e.experiment_id for e in ml.search_experiments()])) == runs
    assert ml.get_model_version_by_alias(REGISTERED_MODEL, "champion").version == alias
    assert supabase.tables.get("champion_history", []) == history
