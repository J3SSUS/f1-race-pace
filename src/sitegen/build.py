"""Write the static page to `site/`: render `docs/site/index.html.j2` and copy `site.css` and `site.js`.

`site.js` is the source file with `const SITE_CONFIG = {...};` put in front of it, so the source stays plain JavaScript
that Node can test. Nothing here queries Supabase or reads the service key; it only writes what `SiteInputs` holds.
"""

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from sitegen.inputs import ROOT, SiteInputs

SOURCE_DIR = ROOT / "docs" / "site"
OUT_DIR = ROOT / "site"
TIMEOUT_MS = 8000
SNAPSHOT_URL = "snapshot.json"


def _site_config_js(inputs: SiteInputs) -> str:
    """`const SITE_CONFIG = {...};` with only what the page needs from `explain.json`: no run ids."""
    importance = inputs.explain.get("importance")
    if not isinstance(importance, list):
        raise ValueError("explain.json: expected an importance list")
    config = {
        "supabaseUrl": inputs.config.supabase_url,
        "anonKey": inputs.config.anon_key,
        "timeoutMs": TIMEOUT_MS,
        "snapshotUrl": SNAPSHOT_URL,
        "teamColors": inputs.team_colors,
        "championExplain": {"model_version_id": inputs.explain["model_version_id"], "importance": importance},
    }
    return f"const SITE_CONFIG = {json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False)};\n\n"


def build_site(inputs: SiteInputs, out_dir: Path = OUT_DIR, source_dir: Path = SOURCE_DIR) -> list[Path]:
    """Write `index.html`, `site.css` and `site.js` to `out_dir`; the same inputs give the same bytes."""
    env = Environment(
        loader=FileSystemLoader(source_dir),
        autoescape=select_autoescape(["html", "j2"]),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    html = env.get_template("index.html.j2").render(card_html=inputs.card_html)
    css = (source_dir / "site.css").read_text()
    js = _site_config_js(inputs) + (source_dir / "site.js").read_text()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in (("index.html", html), ("site.css", css), ("site.js", js)):
        path = out_dir / name
        path.write_bytes(text.encode("utf-8"))
        written.append(path)
    return written
