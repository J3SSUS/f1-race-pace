"""One-time adversarial validation of the split of a sampling run: can a classifier tell train from test?

Reads the labels of the run with `read_split` and the features of every race with `read_wide`, never the target.
Prints the AUC with the 15 features and without the three form features, and logs both as metrics of the sampling
run, with `adversarial_validation.json` as an artifact. It changes nothing else: the result is read by a person and
goes to DECISIONS.md. Run it once, after `scripts/sample.py` and `scripts/load_features.py`.

    uv run python scripts/adversarial_validation.py <run_id>
"""

import argparse
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.config import TrainingConfig, load_config  # noqa: E402
from common.supabase_io import get_client  # noqa: E402
from features.catalog import load_catalog  # noqa: E402
from features.store import StoreError, read_wide  # noqa: E402
from training.adversarial import AucResult, adversarial_validation, log_to_run  # noqa: E402
from training.sampling import read_split  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="the run_id printed by scripts/sample.py")
    parser.add_argument("--version", default="v1", help="feature set version to read (default v1)")
    return parser.parse_args(argv)


def _line(label: str, result: AucResult) -> str:
    folds = ", ".join(f"{auc:.3f}" for auc in result.folds)
    return f"{label}: AUC {result.mean:.3f} ± {result.std:.3f}  (folds {folds})"


def main(
    argv: list[str] | None = None,
    *,
    split_reader: Callable = read_split,
    wide_reader: Callable = read_wide,
    client=None,
    config: TrainingConfig | None = None,
    tracking_uri: str | None = None,
    out: Callable[[str], None] | None = None,
) -> int:
    out = out or partial(print, flush=True)
    args = parse_args(argv)
    config = config or load_config()
    features = load_catalog().names
    try:
        split = split_reader(args.run_id, tracking_uri)
        wide = wide_reader(client or get_client(), args.version)
        results = adversarial_validation(wide, split, features, config)
        races, entries = len(split), len(wide.merge(split, on=["year", "round"]))
        log_to_run(args.run_id, results, config, races, entries, tracking_uri)
    except (StoreError, RuntimeError, ValueError) as exc:
        out(f"STOPPED: {exc}")
        return 1
    counts = split["split"].value_counts()
    out(f"run {args.run_id}: {entries} entries, {races} races ({counts['train']} train, {counts['test']} test)")
    out(_line(f"with the {len(results['all'].features)} features", results["all"]))
    out(_line(f"without form ({len(results['no_form'].features)})", results["no_form"]))
    out("logged adversarial_auc_all, adversarial_auc_no_form and adversarial_validation.json in the run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
