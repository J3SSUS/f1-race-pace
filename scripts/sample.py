"""Run the `sampling` step: ingest the races up to the cutoff, split them 80/20 by race and register the MLflow run.

Reads `data`, `split` and the cutoff from `config/training.json` and the MLflow server from `MLFLOW_TRACKING_URI`
(`.env`). Prints one line per race, the train and test counts and the `run_id`, and exits with code 1 if the step
stops (a Q or R session that failed or is invalid, or no races to split). Safe to run again: ingestion skips the
sessions already on disk and every run is a new MLflow run. With `--output PATH` the run is also written there as
JSON, `{"run_id": "..."}` (the DAGs pass `/airflow/xcom/return.json`); what is printed does not change.

With `--cutoff YEAR-ROUND` (for example `2026-10`) the run uses the races through that round and not the `data` range
of the config, which is not edited: a walk-forward retraining. A cutoff that is not a held race stops the step.

    uv run python scripts/sample.py
    uv run python scripts/sample.py --output /airflow/xcom/return.json
    uv run python scripts/sample.py --cutoff 2026-10
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.sampling import SamplingError, run_sampling  # noqa: E402


def main(
    argv: list[str] | None = None,
    *,
    sampler: Callable = run_sampling,
    out: Callable[[str], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help='also write {"run_id": ...} as JSON to this file')
    parser.add_argument(
        "--cutoff",
        metavar="YEAR-ROUND",
        help="use the races through this round (for example 2026-10) instead of the range of the config",
    )
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)
    try:
        result = sampler(log=out, cutoff=args.cutoff)
    except SamplingError as exc:
        out(f"STOPPED: {exc}")
        return 1
    out(f"train: {result.train_races} races, digest {result.train_digest}")
    out(f"test: {result.test_races} races, digest {result.test_digest}")
    out(f"run_id: {result.run_id}")
    if args.output:
        args.output.write_text(json.dumps({"run_id": result.run_id}))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())
