"""Load the features and the actual target of every race into Supabase and print one line per race.

Rebuilds every race in `races` in order (the form features need the earlier ones) and upserts `features`,
`feature_sets`, `feature_set_items`, `feature_values` and `ground_truth`. Stops at the first race that fails, with
exit code 1; rows already written are valid and a rerun continues without duplicates. Run
`scripts/load_reference.py` first.

    uv run python scripts/load_features.py                          # every race in Supabase
    uv run python scripts/load_features.py --from 2024 --to 2025    # write only these seasons
"""

import argparse
import sys
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.supabase_io import get_client  # noqa: E402
from features.store import StoreError, load_features  # noqa: E402
from ingestion.session import RAW_DIR  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--from", dest="first_year", type=int, default=None, help="first season to write (default: all)"
    )
    parser.add_argument("--to", dest="last_year", type=int, default=None, help="last season to write (default: all)")
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    client=None,
    raw_dir: Path = RAW_DIR,
    out: Callable[[str], None] | None = None,
) -> int:
    out = out or partial(print, flush=True)
    args = parse_args(argv)
    run_id = uuid.uuid4().hex
    out(f"run {run_id}")
    try:
        result = load_features(
            client or get_client(),
            raw_dir=raw_dir,
            first_year=args.first_year,
            last_year=args.last_year,
            run_id=run_id,
            log=out,
        )
    except (StoreError, RuntimeError) as exc:
        out(f"STOPPED: {exc}")
        return 1
    out(
        f"{result.races} races loaded: {result.entries} entries, {result.feature_values} feature values, "
        f"{result.ground_truth} ground truth rows ({result.valid_rows} valid)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
