"""Check that the Supabase project matches supabase/migrations: tables exist and permissions hold.

Reads SUPABASE_URL, SUPABASE_ANON_KEY and SUPABASE_SERVICE_KEY from `.env` and never prints them.
Read-only except for one insert attempt as `anon`, which must be rejected.

    uv run python scripts/check_supabase.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from postgrest.exceptions import APIError  # noqa: E402
from supabase import create_client  # noqa: E402

from common.supabase_tables import PRIVATE_TABLES, PUBLIC_TABLES, can_select  # noqa: E402

failures: list[str] = []


def report(ok: bool, label: str) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        failures.append(label)


def main() -> int:
    load_dotenv(ROOT / ".env")
    values = {k: os.environ.get(k, "") for k in ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_KEY")}
    missing = [k for k, v in values.items() if not v]
    if missing:
        print(f"Missing in .env: {', '.join(missing)}")
        return 1

    service = create_client(values["SUPABASE_URL"], values["SUPABASE_SERVICE_KEY"])
    anon = create_client(values["SUPABASE_URL"], values["SUPABASE_ANON_KEY"])

    print("service key: reads every table and the view")
    for table in PUBLIC_TABLES + PRIVATE_TABLES:
        report(can_select(service, table), f"service can select {table}")

    print("\nanon key: reads only the public tables and the view")
    for table in PUBLIC_TABLES:
        report(can_select(anon, table), f"anon can select {table}")
    for table in PRIVATE_TABLES:
        report(not can_select(anon, table), f"anon is denied {table}")

    print("\nanon key: cannot write or call the RLS helper")
    try:
        anon.table("drivers").insert({"code": "ZZZ"}).execute()
        report(False, "anon insert into drivers was rejected")
        service.table("drivers").delete().eq("code", "ZZZ").execute()  # undo the stray row
    except APIError:
        report(True, "anon insert into drivers was rejected")
    try:
        anon.rpc("rls_auto_enable").execute()
        report(False, "anon cannot call rls_auto_enable()")
    except APIError:
        report(True, "anon cannot call rls_auto_enable()")

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
