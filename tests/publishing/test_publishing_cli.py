import importlib.util
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pytest
from publishing_helpers import golden_client

from publishing.snapshot import build_snapshot
from sitegen.inputs import read_inputs

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "publish.py"
GOLDEN = Path(__file__).resolve().parents[1] / "site" / "golden"
URL = "https://projectref.supabase.co"
ANON = "anon-key-value"
SERVICE = "service-key-must-never-be-looked-up"
ENV = {"SUPABASE_URL": URL, "SUPABASE_ANON_KEY": ANON, "SUPABASE_SERVICE_KEY": SERVICE}
NOW = datetime(2026, 11, 12, 15, 30, 5, tzinfo=UTC)
GENERATED_AT = "2026-11-12T15:30:05Z"


class Spy(dict):
    """An environment that records every name looked up."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.looked_up = []

    def get(self, key, default=None):
        self.looked_up.append(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.looked_up.append(key)
        return super().__getitem__(key)


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("publish_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def raw():
    return json.loads((GOLDEN / "raw_rows.json").read_text())


def run(cli, site_root, out_dir, raw, *, environ=ENV, client=None, **kw):
    lines = []
    kw.setdefault("client_factory", lambda url, key: client or golden_client(raw))
    kw.setdefault("reader", partial(read_inputs, site_root, {"SUPABASE_URL": URL, "SUPABASE_ANON_KEY": ANON}))
    code = cli.main(["--output-dir", str(out_dir)], environ=environ, clock=lambda: NOW, out=lines.append, **kw)
    return code, lines


def test_it_writes_the_snapshot_and_the_four_pages(cli, site_root, tmp_path, raw):
    out_dir = tmp_path / "out"
    code, lines = run(cli, site_root, out_dir, raw)
    assert code == 0, lines
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "fonts",
        "index.html",
        "model.html",
        "site.css",
        "site.js",
        "snapshot.json",
    ]
    expected = json.loads((GOLDEN / "expected_snapshot.json").read_text())
    expected["generated_at"] = GENERATED_AT
    assert json.loads((out_dir / "snapshot.json").read_text()) == expected
    assert json.loads((out_dir / "snapshot.json").read_text()) == build_snapshot(raw, GENERATED_AT)


def test_it_prints_the_champion_the_baseline_the_counts_and_the_date(cli, site_root, tmp_path, raw):
    code, lines = run(cli, site_root, tmp_path / "out", raw)
    assert code == 0
    assert lines[:4] == [
        "champion: version 4 (ridge)",
        "baseline: version 3",
        "races: 4, metrics: 6, results: 5",
        f"generated_at: {GENERATED_AT}",
    ]
    assert lines[4].startswith("wrote ") and "snapshot.json" in lines[4]
    assert any(line.endswith("model.html") for line in lines)


def test_the_page_built_by_publish_has_no_run_id(cli, site_root, tmp_path, raw):
    out_dir = tmp_path / "out"
    assert run(cli, site_root, out_dir, raw)[0] == 0
    assert "d5bb2da8b777423eb46c949c88b44739" not in (out_dir / "model.html").read_text()


def test_it_never_looks_up_the_service_key_and_never_prints_a_key_or_the_url(cli, site_root, tmp_path, raw):
    spy = Spy(ENV)
    code, lines = run(cli, site_root, tmp_path / "out", raw, environ=spy)
    assert code == 0
    assert set(spy.looked_up) == {"SUPABASE_URL", "SUPABASE_ANON_KEY"}
    printed = "\n".join(lines)
    assert SERVICE not in printed and ANON not in printed and URL not in printed and "projectref" not in printed


def test_the_client_is_created_with_the_anon_key(cli, site_root, tmp_path, raw):
    seen = []

    def factory(url, key):
        seen.append((url, key))
        return golden_client(raw)

    assert run(cli, site_root, tmp_path / "out", raw, client_factory=factory)[0] == 0
    assert seen == [(URL, ANON)]


@pytest.mark.parametrize(
    "mutate, text",
    [
        (
            lambda raw: raw["championHistory"].__setitem__(1, {**raw["championHistory"][1], "demoted_at": "x"}),
            "found 0",
        ),
        (lambda raw: raw["modelVersions"].__delitem__(3), "no row for the champion"),
        (lambda raw: raw["metrics"].clear(), "no baseline version has walk_forward metrics"),
        (lambda raw: raw["results"].clear(), "no rows for the champion"),
    ],
)
def test_a_rule_that_does_not_hold_stops_it_and_writes_nothing(cli, site_root, tmp_path, raw, mutate, text):
    out_dir = tmp_path / "out"
    mutate(raw)
    code, lines = run(cli, site_root, out_dir, raw)
    assert code == 1
    assert len(lines) == 1 and lines[0].startswith("STOPPED: ") and text in lines[0]
    assert not out_dir.exists()


def test_a_column_that_is_not_allowed_stops_it_and_keeps_the_previous_snapshot(cli, site_root, tmp_path, raw):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "snapshot.json").write_text("previous")
    raw["results"][0]["internal_note"] = "SECRET-VALUE"
    code, lines = run(cli, site_root, out_dir, raw)
    assert code == 1
    assert lines == [
        "STOPPED: the snapshot has columns the page does not read: results: unexpected column 'internal_note'"
    ]
    assert (out_dir / "snapshot.json").read_text() == "previous"
    assert [p.name for p in out_dir.iterdir()] == ["snapshot.json"]


def test_a_supabase_error_stops_it_by_type_and_writes_nothing(cli, site_root, tmp_path, raw):
    out_dir = tmp_path / "out"
    client = golden_client(raw, error=ConnectionError(f"{URL} {ANON}"))
    code, lines = run(cli, site_root, out_dir, raw, client=client)
    assert code == 1
    assert lines == ["STOPPED: champion_history: ConnectionError"]
    assert not out_dir.exists()


def test_a_client_that_cannot_be_created_stops_it_without_its_text(cli, site_root, tmp_path, raw):
    def factory(url, key):
        raise RuntimeError(f"bad url {url} key {key}")

    code, lines = run(cli, site_root, tmp_path / "out", raw, client_factory=factory)
    assert code == 1
    assert lines == ["STOPPED: could not create the Supabase client (RuntimeError)"]


@pytest.mark.parametrize("missing", ["SUPABASE_URL", "SUPABASE_ANON_KEY"])
def test_a_missing_variable_stops_it_and_makes_no_network_call(cli, site_root, tmp_path, raw, missing):
    calls = []
    env = {k: v for k, v in ENV.items() if k != missing}
    code, lines = run(cli, site_root, tmp_path / "out", raw, environ=env, client_factory=lambda *a: calls.append(a))
    assert code == 1 and lines[0].startswith(f"STOPPED: Missing {missing}")
    assert calls == []


def test_a_missing_input_of_the_build_leaves_no_new_snapshot(cli, site_root, tmp_path, raw):
    out_dir = tmp_path / "out"
    (site_root / "MODEL_CARD.md").unlink()
    code, lines = run(cli, site_root, out_dir, raw)
    assert code == 1 and lines[0].startswith("STOPPED: ") and "MODEL_CARD.md" in lines[0]
    assert not out_dir.exists()


def test_the_same_data_and_time_give_the_same_bytes(cli, site_root, tmp_path, raw):
    first, second = tmp_path / "a", tmp_path / "b"
    assert run(cli, site_root, first, raw)[0] == 0
    assert run(cli, site_root, second, raw)[0] == 0
    assert (first / "snapshot.json").read_bytes() == (second / "snapshot.json").read_bytes()
    assert (first / "model.html").read_bytes() == (second / "model.html").read_bytes()
