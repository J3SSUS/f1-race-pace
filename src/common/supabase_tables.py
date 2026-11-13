"""Which Supabase tables are public and which are private, and a read probe shared by the scripts.

`scripts/check_supabase.py` and `scripts/publish.py` both need the same two lists: keeping them in one place stops
"what is private" from drifting apart.
"""

from postgrest.exceptions import APIError

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


def can_select(client, table: str) -> bool:
    """True when the client can read one row of `table`; False on a permission error (`APIError`)."""
    try:
        client.table(table).select("*").limit(1).execute()
        return True
    except APIError:
        return False
