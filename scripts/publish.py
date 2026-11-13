"""Write `site/snapshot.json` from Supabase and rebuild `site/`.

Reads `SUPABASE_URL` and `SUPABASE_ANON_KEY` (environment or `.env`) and never looks up the service key. Reads the
same five public objects the page reads, builds the snapshot of SPEC 16 (`schema_version` 1), checks that it has only
public columns, writes it through a temporary file and rebuilds the pages with the same `build_site` as
`scripts/build_site.py`. Exits with code 1 and a `STOPPED:` line, writing no file, when it cannot. It never prints a
key or the URL of the project.

    uv run python scripts/publish.py
"""

import argparse
import os
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import dotenv_values  # noqa: E402
from jinja2 import TemplateError  # noqa: E402
from supabase import create_client  # noqa: E402

from publishing.snapshot import (  # noqa: E402
    SupabaseReadError,
    build_snapshot,
    check_columns,
    read_rows,
    write_snapshot,
)
from sitegen.build import OUT_DIR, build_site  # noqa: E402
from sitegen.inputs import read_inputs  # noqa: E402

ENV_NAMES = ("SUPABASE_URL", "SUPABASE_ANON_KEY")  # the only names looked up


def shown(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read_env(environ: Mapping[str, str] | None) -> dict[str, str]:
    """URL and anon key from the process environment, then `.env`. Only those two names are looked up."""
    if environ is None:
        environ = {**dotenv_values(ROOT / ".env"), **os.environ}
    values = {name: (environ.get(name) or "") for name in ENV_NAMES}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"Missing {', '.join(missing)}: set it in .env or the environment (see .env.example)")
    return values


def utc_now() -> datetime:
    return datetime.now(UTC)


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    client_factory: Callable = create_client,
    reader: Callable = read_inputs,
    builder: Callable = build_site,
    clock: Callable[[], datetime] = utc_now,
    out: Callable[[str], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR, help="where the site is written (default: site/)")
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)

    try:
        env = read_env(environ)
        try:
            client = client_factory(env["SUPABASE_URL"], env["SUPABASE_ANON_KEY"])
        except Exception as exc:  # the text of the error could carry the URL or the key
            raise ValueError(f"could not create the Supabase client ({type(exc).__name__})") from None
        rows = read_rows(client)
        snapshot = build_snapshot(rows, clock().strftime("%Y-%m-%dT%H:%M:%SZ"))
        problems = check_columns(snapshot)
        if problems:
            raise ValueError("the snapshot has columns the page does not read: " + "; ".join(problems))
        inputs = reader()  # before writing anything: a missing input must not leave a new snapshot behind
        target = args.output_dir / "snapshot.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_snapshot(target, snapshot)
        written = builder(inputs, args.output_dir)
    except (ValueError, SupabaseReadError, TemplateError) as exc:
        out(f"STOPPED: {exc}")
        return 1
    except OSError as exc:
        out(f"STOPPED: {type(exc).__name__}: {exc.strerror or 'file error'}")
        return 1

    champion, baseline = snapshot["champion"], snapshot["baseline"]
    out(f"champion: version {champion['model_version_id']} ({champion['algorithm']})")
    out(f"baseline: version {baseline['model_version_id']}")
    out(f"races: {len(snapshot['races'])}, metrics: {len(snapshot['metrics'])}, results: {len(snapshot['results'])}")
    out(f"generated_at: {snapshot['generated_at']}")
    out(f"wrote {shown(target)} ({target.stat().st_size} bytes)")
    for path in (p for p in written if p.parent == args.output_dir):
        out(f"wrote {shown(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
