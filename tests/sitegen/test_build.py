import json
import re

import pytest
from jinja2 import UndefinedError

from sitegen.build import SOURCE_DIR, build_site
from sitegen.inputs import SiteConfig, SiteInputs

SECTIONS = ["hero", "results", "race", "scatter", "features", "explanation"]


def make_inputs(**overrides):
    values = {
        "team_colors": {"Ferrari": "#E8002D"},
        "explain": {"run_id": "secret-run-id", "model_version_id": 4, "importance": [{"feature": "grid_position"}]},
        "card_html": "<h1>Card</h1><p>Text</p>",
        "config": SiteConfig("https://abc.supabase.co", "anon-key"),
    }
    return SiteInputs(**{**values, **overrides})


def site_config(js):
    """The object literal of `const SITE_CONFIG = {...};` at the top of the built site.js."""
    head = js.split("\n\n", 1)[0]
    assert head.startswith("const SITE_CONFIG = ") and head.endswith(";")
    return json.loads(head.removeprefix("const SITE_CONFIG = ").removesuffix(";"))


def test_it_writes_the_three_files(tmp_path):
    written = build_site(make_inputs(), tmp_path)
    assert [p.name for p in written] == ["index.html", "site.css", "site.js"]
    assert all(p.is_file() for p in written)


def test_the_page_has_the_six_sections_in_order(tmp_path):
    build_site(make_inputs(), tmp_path)
    html = (tmp_path / "index.html").read_text()
    positions = [html.index(f'<section id="{name}"') for name in SECTIONS]
    assert positions == sorted(positions)


def test_the_card_is_inserted_unescaped(tmp_path):
    build_site(make_inputs(), tmp_path)
    assert "<h1>Card</h1><p>Text</p>" in (tmp_path / "index.html").read_text()


def test_the_template_holds_no_figure():
    template = (SOURCE_DIR / "index.html.j2").read_text()
    text = re.sub(r"<[^>]*>|\{\{.*?\}\}", "", template.split("<body>", 1)[1])  # the visible text, no tags or variables
    assert not re.search(r"(?<![A-Za-z])\d", text)  # no figure; the '1' of "F1" is part of a word


def test_site_js_starts_with_the_config_and_keeps_the_source(tmp_path):
    build_site(make_inputs(), tmp_path)
    js = (tmp_path / "site.js").read_text()
    assert js.endswith((SOURCE_DIR / "site.js").read_text())
    config = site_config(js)
    assert config["supabaseUrl"] == "https://abc.supabase.co"
    assert config["anonKey"] == "anon-key"
    assert config["timeoutMs"] == 8000
    assert config["snapshotUrl"] == "snapshot.json"
    assert config["teamColors"] == {"Ferrari": "#E8002D"}


def test_only_the_model_version_and_importance_of_explain_reach_the_page(tmp_path):
    build_site(make_inputs(), tmp_path)
    js = (tmp_path / "site.js").read_text()
    assert site_config(js)["championExplain"] == {"model_version_id": 4, "importance": [{"feature": "grid_position"}]}
    assert "secret-run-id" not in js


def test_an_explain_file_without_an_importance_list_raises(tmp_path):
    inputs = make_inputs(explain={"model_version_id": 4})
    with pytest.raises(ValueError, match="importance"):
        build_site(inputs, tmp_path)
    assert not list(tmp_path.iterdir())


def test_a_variable_missing_from_the_template_context_raises(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "index.html.j2").write_text("<p>{{ card_html }} {{ not_defined }}</p>")
    (source / "site.css").write_text("")
    (source / "site.js").write_text("")
    out = tmp_path / "out"
    with pytest.raises(UndefinedError, match="not_defined"):
        build_site(make_inputs(), out, source)
    assert not out.exists()


def test_the_same_inputs_give_identical_bytes(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    build_site(make_inputs(), first)
    build_site(make_inputs(), second)
    for name in ("index.html", "site.css", "site.js"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
