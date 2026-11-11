"""Write the static page to `site/` (`index.html`, `site.css` and `site.js`).

Reads `config/team_colors.json`, `site/charts/explain.json`, `MODEL_CARD.md` and `SUPABASE_URL` and
`SUPABASE_ANON_KEY` (environment or `.env`), and nothing else: it never queries Supabase and never reads the service
key. Exits with code 1 and a `STOPPED:` line, writing no file, when an input is missing or malformed.

    uv run python scripts/build_site.py
    python -m http.server -d site     # then open http://localhost:8000 (file:// cannot load snapshot.json)
"""

import argparse
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jinja2 import TemplateError  # noqa: E402

from sitegen.build import OUT_DIR, build_site  # noqa: E402
from sitegen.inputs import read_inputs  # noqa: E402


def shown(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def main(
    argv: list[str] | None = None,
    *,
    reader: Callable = read_inputs,
    builder: Callable = build_site,
    out: Callable[[str], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR, help="where the page is written (default: site/)")
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)

    try:
        written = builder(reader(), args.output_dir)
    except (ValueError, TemplateError, OSError) as exc:
        out(f"STOPPED: {exc}")
        return 1
    pages = [path for path in written if path.parent == args.output_dir]
    for path in pages:
        out(f"wrote {shown(path)}")
    if len(pages) < len(written):
        out(f"wrote {shown(args.output_dir / 'fonts')} ({len(written) - len(pages)} files)")
    out(f"serve it with: python -m http.server -d {shown(args.output_dir)}  (then open http://localhost:8000)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
