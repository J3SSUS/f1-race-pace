"""The `model_card` step: write `MODEL_CARD.md` for the current champion of `f1-race-pace`.

Reads only stored results: the `champion` alias, the `explain` and `evaluate` runs, the params and out-of-fold
predictions of the champion's training run, `model_versions`, `feature_sets`, `races` and the team and circuit of every
entry, plus the hand-written `docs/model_card_notes.md`. It never loads the champion and writes nothing to Supabase or
MLflow; the only file it writes is `MODEL_CARD.md`. Exits with code 1 and a `STOPPED:` line, leaving no file, if it
cannot (no `champion` alias, no `explain` or `evaluate` run for that version, a missing artifact or notes section,
`site/charts/` from another `explain` run, or a Supabase error). After a promotion the order is `explain` and then this.

    uv run python scripts/model_card.py
"""

import argparse
import logging
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlflow.exceptions import MlflowException  # noqa: E402

from features.store import StoreError  # noqa: E402
from training.model_card import (  # noqa: E402
    CARD_FILE,
    CHARTS_DIR,
    CONFIG_NAMES,
    NOTES_FILE,
    TEMPLATE_FILE,
    write_model_card,
)


def describe(result: dict) -> list[str]:
    inputs, errors = result["inputs"], result["errors"]
    champion = inputs.test_metrics[CONFIG_NAMES.get(inputs.algorithm, inputs.algorithm)]
    baseline = inputs.test_metrics["baseline"]
    few = sum(e.few_rows for e in errors)
    return [
        f"champion: model version {inputs.model_version_id} ({inputs.algorithm}), "
        f"MLflow version {inputs.mlflow_version}",
        f"explain run {inputs.explain_run_id}, evaluate run {inputs.evaluate_run_id}, "
        f"sampling run {inputs.sampling_run_id}",
        f"test MAE {champion['metrics']['mae']:.3f} vs baseline {baseline['metrics']['mae']:.3f} "
        f"({champion['n_rows']} rows, {champion['n_races']} races)",
        f"segments: {len(errors)} (teams and circuit types), {few} with few rows",
        f"feature set {inputs.feature_set_version}, trained through "
        f"{inputs.trained_through['year']} round {inputs.trained_through['round']}",
        f"wrote {result['path']}",
    ]


def main(
    argv: list[str] | None = None, *, writer: Callable = write_model_card, out: Callable[[str], None] | None = None
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / CARD_FILE, help="where MODEL_CARD.md is written")
    parser.add_argument("--notes", type=Path, default=ROOT / NOTES_FILE, help="the hand-written notes file")
    parser.add_argument(
        "--charts-dir", type=Path, default=ROOT / CHARTS_DIR, help="where the explain step wrote charts"
    )
    args = parser.parse_args(argv)
    out = out or partial(print, flush=True)

    try:
        result = writer(
            path=args.output, template=ROOT / TEMPLATE_FILE, notes_path=args.notes, charts_dir=args.charts_dir
        )
    except (ValueError, StoreError, RuntimeError, MlflowException) as exc:
        out(f"STOPPED: {exc}")
        return 1
    for line in describe(result):
        out(line)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())
