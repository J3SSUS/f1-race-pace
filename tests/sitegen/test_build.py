import json
import re

import pytest
from jinja2 import UndefinedError

from sitegen.build import SOURCE_DIR, build_site
from sitegen.inputs import SiteConfig, SiteInputs

SECTIONS = ["hero", "results", "race", "scatter", "features"]


def make_inputs(**overrides):
    values = {
        "team_colors": {"Ferrari": "#E8002D"},
        "explain": {"run_id": "secret-run-id", "model_version_id": 4, "importance": [{"feature": "grid_position"}]},
        "card_summary_html": "<h1>Card</h1><p>Summary</p>",
        "card_details_html": "<h2>2. Results</h2><p>Details</p>",
        "config": SiteConfig("https://abc.supabase.co", "anon-key"),
    }
    return SiteInputs(**{**values, **overrides})


def site_config(js):
    """The object literal of `const SITE_CONFIG = {...};` at the top of the built site.js."""
    head = js.split("\n\n", 1)[0]
    assert head.startswith("const SITE_CONFIG = ") and head.endswith(";")
    return json.loads(head.removeprefix("const SITE_CONFIG = ").removesuffix(";"))


def test_it_writes_the_four_files(tmp_path):
    written = build_site(make_inputs(), tmp_path)
    assert [p.name for p in written] == ["index.html", "model.html", "site.css", "site.js"]
    assert all(p.is_file() for p in written)


def test_the_landing_has_its_five_sections_in_order(tmp_path):
    build_site(make_inputs(), tmp_path)
    html = (tmp_path / "index.html").read_text()
    positions = [html.index(f'<section id="{name}"') for name in SECTIONS]
    assert positions == sorted(positions)


def test_the_landing_links_to_the_model_page_and_holds_no_card_text(tmp_path):
    build_site(make_inputs(), tmp_path)
    html = (tmp_path / "index.html").read_text()
    assert 'href="model.html"' in html
    assert "Summary" not in html and "Details" not in html


def test_the_model_page_shows_the_summary_and_folds_the_rest(tmp_path):
    build_site(make_inputs(), tmp_path)
    html = (tmp_path / "model.html").read_text()
    assert "<h1>Card</h1><p>Summary</p>" in html  # inserted unescaped
    details = html[html.index('<details id="model-card-full">') :]
    assert "<summary>Read the full model card</summary>" in details
    assert "<h2>2. Results</h2><p>Details</p>" in details
    assert "Summary</p>" not in details
    assert "open" not in html[html.index("<details") : html.index(">", html.index("<details"))]
    assert 'href="index.html"' in html


@pytest.mark.parametrize("name", ["index.html.j2", "model.html.j2"])
def test_the_templates_hold_no_figure(name):
    template = (SOURCE_DIR / name).read_text()
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
    (source / "index.html.j2").write_text("<p>{{ not_defined }}</p>")
    (source / "model.html.j2").write_text("<p>{{ card_summary_html }} {{ card_details_html }}</p>")
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
    for name in ("index.html", "model.html", "site.css", "site.js"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
