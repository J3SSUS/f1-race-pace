"""Write `site/snapshot.json` from Supabase and rebuild `site/`.

Reads `SUPABASE_URL` and `SUPABASE_ANON_KEY` (environment or `.env`) and never looks up the service key. Checks that
the anon key is denied on the six private tables, reads the same five public objects the page reads, builds the
snapshot of SPEC 16 (`schema_version` 1), checks that it has only public columns, writes it through a temporary file,
rebuilds the pages with the same `build_site` as `scripts/build_site.py` and runs the control of exposure over the
files git would publish under `site/`. Exits with code 1 and a `STOPPED:` line when it cannot or when the control
finds something. It never prints a key, the URL of the project or the text of a finding. Commit and push stay manual.

    uv run python scripts/publish.py
    uv run python scripts/publish.py --check-only    # only the control, no network and no file written
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

from common.supabase_tables import PRIVATE_TABLES, can_select  # noqa: E402
from publishing.exposure import PublishedFilesError, check_files, published_files  # noqa: E402
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


def check_private_tables_denied(client) -> None:
    """The anon key must be denied on every private table; a table it can read stops `publish` before any write.

    A failure that is not a permission error (the network, say) is not read as "denied": it stops as well.
    """
    for table in PRIVATE_TABLES:
        try:
            readable = can_select(client, table)
        except Exception as exc:
            raise SupabaseReadError(f"{table}: {type(exc).__name__}") from None
        if readable:
            raise ValueError(f"the anon key can read the private table {table}; fix its RLS before publishing")


def run_control(root: Path, out: Callable[[str], None]) -> int:
    """The control of exposure over what git would publish under `site/`; 0 when it finds nothing.

    A finding is printed as the file and the rule, never the text that matched."""
    files = published_files(root)
    findings = check_files(root, files)
    for finding in findings:
        out(f"found: {finding.path} ({finding.rule})")
    if findings:
        out(f"STOPPED: the control of exposure found {len(findings)} problem(s) in the files git would publish")
        return 1
    out(f"control of exposure: nothing found in the {len(files)} files git would publish under site/")
    return 0


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    client_factory: Callable = create_client,
    reader: Callable = read_inputs,
    builder: Callable = build_site,
    root: Path = ROOT,
    clock: Callable[[], datetime] = utc_now,
    out: Callable[[str], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR, help="where the site is written (default: site/)")
    parser.add_argument("--check-only", action="store_true", help="only run the control: no network, no file written")
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)

    if args.check_only:
        try:
            return run_control(root, out)
        except (PublishedFilesError, OSError) as exc:
            out(f"STOPPED: {exc if isinstance(exc, PublishedFilesError) else type(exc).__name__}")
            return 1

    try:
        env = read_env(environ)
        try:
            client = client_factory(env["SUPABASE_URL"], env["SUPABASE_ANON_KEY"])
        except Exception as exc:  # the text of the error could carry the URL or the key
            raise ValueError(f"could not create the Supabase client ({type(exc).__name__})") from None
        check_private_tables_denied(client)  # before anything is written
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
    except (ValueError, SupabaseReadError, PublishedFilesError, TemplateError) as exc:
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
    try:
        if run_control(root, out):
            return 1
    except (PublishedFilesError, OSError) as exc:
        out(f"STOPPED: {exc if isinstance(exc, PublishedFilesError) else type(exc).__name__}")
        return 1
    out("commit and push are manual: git add site && git commit && git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
