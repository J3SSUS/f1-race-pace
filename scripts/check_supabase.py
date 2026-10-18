"""Check that the Supabase project matches supabase/migrations: tables exist and permissions hold.

Reads SUPABASE_URL, SUPABASE_ANON_KEY and SUPABASE_SERVICE_KEY from `.env` and never prints them.
Read-only except for one insert attempt as `anon`, which must be rejected.

    uv run python scripts/check_supabase.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from postgrest.exceptions import APIError
from supabase import create_client

PUBLIC_TABLES = [
    "circuits",
    "drivers",
    "teams",
    "races",
    "race_entries",
    "features",
    "model_versions",
    "champion_history",
    "predictions",
    "ground_truth",
    "metrics",
    "shap_values",
    "prediction_results",  # view
]
PRIVATE_TABLES = [
    "feature_sets",
    "feature_set_items",
    "feature_values",
    "prediction_runs",
    "drift_reports",
    "drift_feature_results",
]

failures: list[str] = []


def report(ok: bool, label: str) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        failures.append(label)


def can_select(client, table: str) -> bool:
    try:
        client.table(table).select("*").limit(1).execute()
        return True
    except APIError:
        return False


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
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
