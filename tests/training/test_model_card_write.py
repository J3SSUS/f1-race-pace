import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fake_supabase import FakeSupabase
from mlflow import MlflowClient
from test_manifest import config
from test_model_card_inputs import PARAMS, REFERENCE, TRAINED, experiment_of, log_run, parent, register  # noqa: F401
from test_model_card_render import METRICS as RENDER_METRICS
from test_model_card_render import NOTES, TEMPLATE, make_inputs, result
from test_model_card_segments import BASELINE as BASELINE_OOF
from test_model_card_segments import CHAMPION as CHAMPION_OOF

from features.store import StoreError
from training.model_card import NOTE_SECTIONS, TEMPLATE_FILE, write_model_card
from training.registry import REGISTERED_MODEL

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "model_card.py"
KEY = "service-key-that-must-never-be-printed"
NOW = datetime(2026, 11, 2, 9, 30, tzinfo=UTC)
CHARTS = [
    "permutation_importance",
    "shap_bar",
    "shap_summary",
    "coefficients",
    "case_fastest",
    "case_slowest",
    "predicted_vs_actual",
    "residuals",
]


def log_oof(uri, run_id, frame, tmp_path):
    folder = tmp_path / f"oof-{run_id}"
    folder.mkdir()
    frame.to_parquet(folder / "oof_predictions.parquet", index=False)
    MlflowClient(uri).log_artifact(run_id, str(folder / "oof_predictions.parquet"))


def reference_tables(supabase_extra) -> dict:
    tables = {name: [dict(r) for r in rows] for name, rows in REFERENCE.items()}
    tables["races"] = [{**r, "circuit_id": 11} for r in tables["races"]] + [
        {"race_id": 100, "year": 2024, "round": 8, "circuit_id": 10},
        {"race_id": 101, "year": 2024, "round": 9, "circuit_id": 11},
    ]
    tables["teams"] = [{"team_id": 1, "name": "A"}, {"team_id": 2, "name": "B"}]
    tables["circuits"] = [{"circuit_id": 10, "circuit_type": "street"}, {"circuit_id": 11, "circuit_type": "permanent"}]
    street, team = {1, 4}, {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
    tables["race_entries"] = [
        {"entry_id": i, "race_id": 100 if i in street else 101, "team_id": team[i]} for i in range(1, 7)
    ]
    return {**tables, **supabase_extra}


@pytest.fixture
def world(uri, parent, tmp_path):  # noqa: F811
    """A champion (ridge, model version 7) with every stored result the card reads, notes and charts on disk."""
    inputs = make_inputs()
    base_tags = {"step": "training", "sampling_run_id": parent}
    training = log_run(uri, parent, "ridge", {**base_tags, "algorithm": "ridge"}, PARAMS)
    log_oof(uri, training, CHAMPION_OOF, tmp_path)
    baseline = log_run(uri, parent, "baseline", {**base_tags, "algorithm": "baseline"})
    log_oof(uri, baseline, BASELINE_OOF, tmp_path)
    register(uri, parent, training)

    explain_tags = {"step": "explain", "sampling_run_id": parent, "model_version_id": "7"}
    explain = log_run(uri, parent, "explain", explain_tags, artifacts={"explain.json": inputs.explain})
    evaluate_tags = {"step": "evaluate", "sampling_run_id": parent}
    test_metrics = {"ridge": result(RENDER_METRICS), "baseline": result(RENDER_METRICS)}
    evaluate_artifacts = {"selection.json": inputs.selection, "test_metrics.json": test_metrics}
    evaluate = log_run(uri, parent, "evaluate", evaluate_tags, artifacts=evaluate_artifacts)

    supabase = FakeSupabase(
        reference_tables(
            {
                "model_versions": [{"model_version_id": 7, "mlflow_run_id": training, "algorithm": "ridge", **TRAINED}],
                "champion_history": [
                    {"model_version_id": 7, "promoted_at": "2026-10-10T00:00:00+00:00", "demoted_at": None}
                ],
                "ground_truth": [{"entry_id": 1, "race_pace_delta_pct": 9.9}],  # never to be read
            }
        )
    )
    charts = tmp_path / "site" / "charts"
    charts.mkdir(parents=True)
    for name in CHARTS:
        (charts / f"{name}.svg").write_text("<svg/>")
    (charts / "explain.json").write_text(json.dumps({"run_id": explain}))
    notes = tmp_path / "notes.md"
    notes.write_text("\n\n".join(f"## {name}\n\n{text}" for name, text in NOTES.items()) + "\n")
    return {
        "uri": uri,
        "supabase": supabase,
        "charts": charts,
        "notes": notes,
        "card": tmp_path / "MODEL_CARD.md",
        "explain": explain,
        "evaluate": evaluate,
        "training": training,
        "tmp": tmp_path,
    }


def write(world, **kwargs):
    kwargs.setdefault("now", NOW)
    return write_model_card(
        config(),
        kwargs.pop("client", world["supabase"]),
        world["uri"],
        world["card"],
        template=TEMPLATE,
        notes_path=world["notes"],
        charts_dir=world["charts"],
        **kwargs,
    )


assert TEMPLATE.name == Path(TEMPLATE_FILE).name


def test_the_card_is_written_with_the_five_parts_and_the_segments_from_the_oof(world):
    out = write(world)
    text = world["card"].read_text()
    assert out["path"] == world["card"] and out["text"] == text
    for title in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5."):
        assert title in text
    assert "# Model card: f1-race-pace, version 7 (ridge)" in text
    assert world["explain"] in text and world["evaluate"] in text
    assert "| B | 3 | 3 | 1.000 | 0.833 | +0.333 | few rows |" in text  # hand values of the segment tests
    assert "| street | 2 | 1 | 1.000 | 1.000 | -1.000 | few rows |" in text
    assert "Feature set: v1. The model was trained through 2024 round 2." in text
    assert "promoted 2026-10-10" in text


def test_it_writes_only_the_card_and_changes_nothing_else(world):
    before_files = sorted(p for p in world["tmp"].rglob("*") if p.is_file() and "mlflow" not in p.name)
    before_tables = {name: [dict(r) for r in rows] for name, rows in world["supabase"].tables.items()}
    ml = MlflowClient(world["uri"])
    alias = ml.get_model_version_by_alias(REGISTERED_MODEL, "champion").version
    runs = len(ml.search_runs([e.experiment_id for e in ml.search_experiments()]))
    write(world)
    after_files = sorted(p for p in world["tmp"].rglob("*") if p.is_file() and "mlflow" not in p.name)
    assert [p for p in after_files if p not in before_files] == [world["card"]]
    assert world["supabase"].tables == before_tables and world["supabase"].writes == []
    assert len(ml.search_runs([e.experiment_id for e in ml.search_experiments()])) == runs
    assert ml.get_model_version_by_alias(REGISTERED_MODEL, "champion").version == alias


def test_only_stored_results_and_reference_tables_are_read_and_no_model_is_loaded(world, monkeypatch):
    import mlflow.sklearn

    def refuse(*args, **kwargs):
        raise AssertionError("the champion must not be loaded")

    monkeypatch.setattr(mlflow.sklearn, "load_model", refuse)
    write(world)
    read = {table for table, _, _ in world["supabase"].reads}
    assert read == {
        "model_versions",
        "champion_history",
        "feature_sets",
        "races",
        "race_entries",
        "teams",
        "circuits",
    }


def test_running_twice_gives_the_same_file_except_for_the_date(world):
    write(world, now=NOW)
    first = world["card"].read_text()
    write(world, now=datetime(2027, 3, 1, tzinfo=UTC))
    second = world["card"].read_text()
    assert first != second
    assert first == second.replace("2027-03-01", "2026-11-02")


def test_the_card_never_contains_the_service_key(world, monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", KEY)
    write(world)
    assert KEY not in world["card"].read_text()


def break_alias(world):
    MlflowClient(world["uri"]).delete_registered_model_alias(REGISTERED_MODEL, "champion")


def break_explain(world):
    MlflowClient(world["uri"]).delete_run(world["explain"])


def break_evaluate(world):
    MlflowClient(world["uri"]).delete_run(world["evaluate"])


def break_notes(world):
    world["notes"].write_text(world["notes"].read_text().replace(f"## {NOTE_SECTIONS[2]}", "## Other"))


def break_oof(world):
    ml = MlflowClient(world["uri"])
    ml.delete_run(
        next(r.info.run_id for r in ml.search_runs([experiment_of_world(world)]) if r.info.run_name == "baseline")
    )


def experiment_of_world(world):
    ml = MlflowClient(world["uri"])
    return ml.get_run(world["training"]).info.experiment_id


def break_charts_run(world):
    (world["charts"] / "explain.json").write_text(json.dumps({"run_id": "another-explain-run"}))


def break_charts_missing_file(world):
    (world["charts"] / "residuals.svg").unlink()


def break_charts_no_report(world):
    (world["charts"] / "explain.json").unlink()


@pytest.mark.parametrize(
    ("breaker", "message"),
    [
        (break_alias, "no 'champion' alias"),
        (break_explain, "no finished explain run"),
        (break_evaluate, "no finished evaluate run"),
        (break_notes, NOTE_SECTIONS[2]),
        (break_oof, "baseline"),
        (break_charts_run, "another-explain-run"),
        (break_charts_missing_file, "residuals.svg"),
        (break_charts_no_report, "explain.json"),
    ],
)
def test_each_failure_raises_and_leaves_no_file(world, breaker, message):
    breaker(world)
    with pytest.raises(ValueError, match=message):
        write(world)
    assert not world["card"].exists()


def test_a_failure_leaves_an_existing_card_untouched(world):
    world["card"].write_text("the old card")
    break_charts_run(world)
    with pytest.raises(ValueError):
        write(world)
    assert world["card"].read_text() == "the old card"


class FailingSupabase(FakeSupabase):
    def __init__(self, tables, fail_on):
        super().__init__(tables)
        self.fail_on = fail_on

    def table(self, name):
        if name == self.fail_on:
            raise RuntimeError(f"boom with {KEY}")
        return super().table(name)


@pytest.mark.parametrize("table", ["model_versions", "feature_sets", "teams"])
def test_a_supabase_error_raises_a_store_error_without_the_response_and_leaves_no_file(world, table):
    broken = FailingSupabase(world["supabase"].tables, table)
    with pytest.raises(StoreError, match=table) as raised:
        write(world, client=broken)
    assert KEY not in str(raised.value)
    assert not world["card"].exists()


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("model_card_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(cli, world, **overrides):
    lines: list[str] = []

    def writer(**kwargs):
        return write_model_card(config(), overrides.get("client", world["supabase"]), world["uri"], now=NOW, **kwargs)

    args = ["--output", str(world["card"]), "--notes", str(world["notes"]), "--charts-dir", str(world["charts"])]
    return cli.main(args, writer=writer, out=lines.append), lines


def test_the_cli_prints_what_it_used_and_writes_the_card(cli, world):
    code, lines = run_cli(cli, world)
    text = "\n".join(lines)
    assert code == 0 and world["card"].exists()
    assert "model version 7 (ridge)" in text and "MLflow version" in text
    assert world["explain"] in text and world["evaluate"] in text
    assert "test MAE 0.412 vs baseline 0.412 (777 rows, 13 races)" in text
    assert "feature set v1, trained through 2024 round 2" in text
    assert "segments: 4 (teams and circuit types), 4 with few rows" in text
    assert str(world["card"]) in text


@pytest.mark.parametrize(
    "breaker", [break_alias, break_explain, break_evaluate, break_notes, break_oof, break_charts_run]
)
def test_the_cli_stops_with_one_line_exit_1_and_no_file(cli, world, breaker):
    breaker(world)
    code, lines = run_cli(cli, world)
    assert code == 1
    assert len(lines) == 1 and lines[0].startswith("STOPPED: ")
    assert not world["card"].exists()


def test_the_cli_reports_a_supabase_error_without_the_key(cli, world, monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", KEY)
    code, lines = run_cli(cli, world, client=FailingSupabase(world["supabase"].tables, "races"))
    assert code == 1 and len(lines) == 1 and lines[0].startswith("STOPPED: ")
    assert KEY not in lines[0]
    assert not world["card"].exists()


def test_the_cli_defaults_point_at_the_repository(cli, tmp_path):
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        raise ValueError("stop here")

    cli.main([], writer=spy, out=lambda line: None)
    root = SCRIPT.parents[1]
    assert seen["path"] == root / "MODEL_CARD.md"
    assert seen["notes_path"] == root / "docs" / "model_card_notes.md"
    assert seen["charts_dir"] == root / "site" / "charts"
    assert seen["template"] == root / "docs" / "model_card.md.j2"
