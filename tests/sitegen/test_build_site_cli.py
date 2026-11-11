import importlib.util
import re
from functools import partial
from pathlib import Path

import pytest
from sitegen_helpers import ENV

from sitegen.inputs import read_inputs

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_site.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("build_site_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(cli, *args, **kw):
    lines = []
    code = cli.main(list(args), out=lines.append, **kw)
    return code, lines


def test_it_writes_the_page_and_prints_how_to_serve_it(cli, root, tmp_path):
    out_dir = tmp_path / "out"
    code, lines = run(cli, "--output-dir", str(out_dir), reader=partial(read_inputs, root, ENV))
    assert code == 0
    assert sorted(p.name for p in out_dir.iterdir()) == ["fonts", "index.html", "model.html", "site.css", "site.js"]
    assert [line for line in lines if line.startswith("wrote ") and "fonts" not in line] == [
        f"wrote {out_dir / name}" for name in ("index.html", "model.html", "site.css", "site.js")
    ]
    assert re.search(r"wrote .*fonts \(\d+ files\)", lines[-2])
    assert "python -m http.server -d" in lines[-1]


def test_a_missing_variable_stops_it_and_writes_nothing(cli, root, tmp_path):
    out_dir = tmp_path / "out"
    env = {k: v for k, v in ENV.items() if k != "SUPABASE_ANON_KEY"}
    code, lines = run(cli, "--output-dir", str(out_dir), reader=partial(read_inputs, root, env))
    assert code == 1
    assert len(lines) == 1 and lines[0].startswith("STOPPED:") and "SUPABASE_ANON_KEY" in lines[0]
    assert not out_dir.exists()


def test_a_missing_file_stops_it_and_writes_nothing(cli, root, tmp_path):
    (root / "MODEL_CARD.md").unlink()
    out_dir = tmp_path / "out"
    code, lines = run(cli, "--output-dir", str(out_dir), reader=partial(read_inputs, root, ENV))
    assert code == 1
    assert lines[0].startswith("STOPPED:") and "MODEL_CARD.md" in lines[0]
    assert not out_dir.exists()


def test_a_template_error_is_a_stopped_line_too(cli, root, tmp_path):
    (tmp_path / "source").mkdir()
    (tmp_path / "source" / "index.html.j2").write_text("{{ not_defined }}")
    (tmp_path / "source" / "model.html.j2").write_text("")
    from sitegen.build import build_site

    builder = partial(build_site, source_dir=tmp_path / "source")
    code, lines = run(
        cli, "--output-dir", str(tmp_path / "out"), reader=partial(read_inputs, root, ENV), builder=builder
    )
    assert code == 1
    assert lines[0].startswith("STOPPED:") and "not_defined" in lines[0]
