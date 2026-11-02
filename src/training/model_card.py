"""The model card of the champion of `f1-race-pace`: `MODEL_CARD.md` from stored results only.

Nothing is refitted and the champion model is never loaded: the card reads the `champion` alias and its tags, the
`explain` and `evaluate` runs, the params of the champion's training run and the config. It only reads: no MLflow run
and no Supabase write.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from common.config import TrainingConfig, load_config
from common.supabase_io import get_client
from features.store import read_table
from training.evaluation import EVALUATE
from training.explain import ALIAS
from training.pipelines import ALGORITHMS
from training.registry import REGISTERED_MODEL
from training.sampling import SAMPLING_TAG, read_split, resolve_tracking_uri
from training.step import finished_run_id

CARD_FILE = Path("MODEL_CARD.md")  # repository root
TEMPLATE_FILE = Path("docs/model_card.md.j2")
NOTES_FILE = Path("docs/model_card_notes.md")
MIN_SEGMENT_ROWS = 30  # fewer rows than this: the segment is marked "few rows"
NOTE_SECTIONS = ("Intended use", "Known limitations", "2026 regulations")  # the H2 headings of the notes file

EXPLAIN = "explain"  # run name of the explain step (the name `explain.py` gives its root run)
TUNING_KEYS = ("n_trials", "seed", "n_folds")  # params of the training run that are the tuning budget
BOOKKEEPING_KEYS = (*TUNING_KEYS, "train_races", "train_rows", "features", "metric_main")  # not hyperparameters
CONFIG_NAMES = {algorithm: name for name, algorithm in ALGORITHMS.items()}  # model_versions.algorithm -> config key


@dataclass(frozen=True)
class CardInputs:
    model_version_id: int
    mlflow_version: str  # version of f1-race-pace holding the `champion` alias
    algorithm: str
    sampling_run_id: str
    explain_run_id: str  # latest FINISHED `explain` run of this model version
    evaluate_run_id: str  # the `evaluate` run of the sampling run
    promoted_at: str | None  # champion_history.promoted_at of the current champion
    explain: dict  # explain.json
    selection: dict  # selection.json of the evaluate run
    test_metrics: dict  # test_metrics.json: champion and baseline; mae, rmse, r2, pairwise_accuracy with intervals
    hyperparameters: dict[str, str]  # params of the champion's training run: the best Optuna trial, incl. `imputation`
    tuning: dict[str, str]  # n_trials, seed, n_folds of that run
    data_range: dict  # first_year, last_year, cutoff (config/training.json)
    split: dict  # test_fraction, n_train_races, n_test_races, first and last race of each
    cv_folds: int


def _artifact_json(ml: MlflowClient, run_id: str, name: str, what: str) -> dict:
    """The JSON artifact `name` of a run, or a `ValueError` that names the run and the file."""
    try:
        path = ml.download_artifacts(run_id, name)
    except MlflowException as exc:
        raise ValueError(f"the {what} run {run_id} has no {name}") from exc
    return json.loads(Path(path).read_text())


def _latest_explain_run(ml: MlflowClient, sampling_run_id: str, model_version_id: int) -> str:
    """The latest FINISHED `explain` run of this model version; a FAILED one and other versions are ignored."""
    experiment_id = ml.get_run(sampling_run_id).info.experiment_id
    runs = ml.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.step = '{EXPLAIN}' and tags.model_version_id = '{model_version_id}' "
            f"and tags.{SAMPLING_TAG} = '{sampling_run_id}' and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"no finished explain run for model version {model_version_id}: run scripts/explain.py first")
    return runs[0].info.run_id


def _champion_test_metrics(test_metrics: dict, algorithm: str, evaluate_run_id: str) -> None:
    """Raise if the champion is not in `test_metrics.json` (never fall back to the candidate's numbers)."""
    name = CONFIG_NAMES.get(algorithm, algorithm)
    if name not in test_metrics:
        raise ValueError(
            f"the champion '{name}' is not in test_metrics.json of the evaluate run {evaluate_run_id} "
            f"(it holds {sorted(test_metrics)}): evaluate the champion before writing the card"
        )


def _hyperparameters(params: dict[str, str], run_id: str) -> tuple[dict[str, str], dict[str, str]]:
    """`(hyperparameters, tuning)` of a training run: the bookkeeping params leave the first; three go to the second."""
    hyperparameters = {k: v for k, v in params.items() if k not in BOOKKEEPING_KEYS}
    if not hyperparameters:
        raise ValueError(f"the training run {run_id} of the champion has no hyperparameters")
    missing = [k for k in TUNING_KEYS if k not in params]
    if missing:
        raise ValueError(f"the training run {run_id} of the champion has no {', '.join(missing)} param")
    return hyperparameters, {k: params[k] for k in TUNING_KEYS}


def _race(split, index: int) -> dict[str, int]:
    row = split.iloc[index]
    return {"year": int(row["year"]), "round": int(row["round"])}


def _split_summary(sampling_run_id: str, config: TrainingConfig, tracking_uri: str) -> dict:
    """Race counts and first and last race of train and test, from the manifest of the sampling run."""
    races = read_split(sampling_run_id, tracking_uri)
    summary: dict = {"test_fraction": config.split.test_fraction}
    for part in ("train", "test"):
        rows = races[races["split"] == part].reset_index(drop=True)
        summary[f"n_{part}_races"] = len(rows)
        summary[f"first_{part}_race"] = _race(rows, 0)
        summary[f"last_{part}_race"] = _race(rows, len(rows) - 1)
    return summary


def load_card_inputs(config: TrainingConfig | None = None, client=None, tracking_uri: str | None = None) -> CardInputs:
    """Every stored result the card needs, for the current champion. Nothing is refitted, loaded or written.

    Raises `ValueError` naming what is missing: no `champion` alias, a version without its tags or its
    `model_versions` row, no FINISHED `explain` run for the champion's version, no `evaluate` run of its sampling
    run, a champion that is not in `test_metrics.json`, a missing artifact or a training run without hyperparameters.
    """
    config = config or load_config()
    uri = resolve_tracking_uri(tracking_uri)
    ml = MlflowClient(uri)
    try:
        version = ml.get_model_version_by_alias(REGISTERED_MODEL, ALIAS)
    except MlflowException as exc:
        raise ValueError(f"{REGISTERED_MODEL} has no '{ALIAS}' alias: run scripts/promote.py first") from exc
    run_id = version.tags.get("mlflow_run_id", "")
    sampling_run_id = version.tags.get(SAMPLING_TAG, "")
    if not run_id or not sampling_run_id:
        raise ValueError(f"version {version.version} of {REGISTERED_MODEL} has no mlflow_run_id or sampling_run_id tag")

    client = client if client is not None else get_client()
    row = next(
        (
            r
            for r in read_table(client, "model_versions", "model_version_id,mlflow_run_id,algorithm")
            if r["mlflow_run_id"] == run_id
        ),
        None,
    )
    if row is None:
        raise ValueError(f"the champion (run {run_id}) has no model_versions row: run scripts/train.py")
    model_version_id, algorithm = row["model_version_id"], row["algorithm"]

    history = read_table(client, "champion_history", "model_version_id,promoted_at,demoted_at")
    current = [r for r in history if r["model_version_id"] == model_version_id and r["demoted_at"] is None]
    promoted_at = current[0]["promoted_at"] if current else None

    explain_run_id = _latest_explain_run(ml, sampling_run_id, model_version_id)
    explain = _artifact_json(ml, explain_run_id, "explain.json", EXPLAIN)

    evaluate_run_id = finished_run_id(sampling_run_id, EVALUATE, uri)
    if evaluate_run_id is None:
        raise ValueError(f"no finished evaluate run for the sampling run {sampling_run_id}: run scripts/evaluate.py")
    selection = _artifact_json(ml, evaluate_run_id, "selection.json", EVALUATE)
    test_metrics = _artifact_json(ml, evaluate_run_id, "test_metrics.json", EVALUATE)
    _champion_test_metrics(test_metrics, algorithm, evaluate_run_id)

    hyperparameters, tuning = _hyperparameters(dict(ml.get_run(run_id).data.params), run_id)
    cutoff = config.data.cutoff
    data_range = {
        "first_year": config.data.first_year,
        "last_year": config.data.last_year,
        "cutoff": None if cutoff is None else {"year": cutoff.year, "round": cutoff.round},
    }
    return CardInputs(
        model_version_id=model_version_id,
        mlflow_version=str(version.version),
        algorithm=algorithm,
        sampling_run_id=sampling_run_id,
        explain_run_id=explain_run_id,
        evaluate_run_id=evaluate_run_id,
        promoted_at=promoted_at,
        explain=explain,
        selection=selection,
        test_metrics=test_metrics,
        hyperparameters=hyperparameters,
        tuning=tuning,
        data_range=data_range,
        split=_split_summary(sampling_run_id, config, uri),
        cv_folds=config.cv.folds,
    )


@dataclass(frozen=True)
class SegmentError:
    segment_type: str  # "team" | "circuit_type"
    segment_value: str
    n_rows: int
    n_races: int
    mae: float  # |y_true - y_pred| mean of the champion, out-of-fold rows
    baseline_mae: float  # the same for the baseline, on the same rows
    bias: float  # mean(y_pred - y_true) of the champion: positive = predicted slower than it was
    few_rows: bool  # n_rows < MIN_SEGMENT_ROWS


SEGMENT_TYPES = ("team", "circuit_type")  # in the order the card shows them


def read_entry_segments(client) -> pd.DataFrame:
    """`entry_id`, `team` and `circuit_type` of every race entry. The only Supabase read of the card.

    Reads four public reference tables (`race_entries`, `teams`, `races`, `circuits`), read-only. An entry whose team
    or circuit cannot be found keeps a missing value; `segment_errors` raises on it.
    """
    entries = pd.DataFrame(read_table(client, "race_entries", "entry_id,race_id,team_id"))
    teams = pd.DataFrame(read_table(client, "teams", "team_id,name")).rename(columns={"name": "team"})
    races = pd.DataFrame(read_table(client, "races", "race_id,circuit_id"))
    circuits = pd.DataFrame(read_table(client, "circuits", "circuit_id,circuit_type"))
    merged = (
        entries.merge(teams, on="team_id", how="left")
        .merge(races, on="race_id", how="left")
        .merge(circuits, on="circuit_id", how="left")
    )
    return merged[["entry_id", "team", "circuit_type"]].reset_index(drop=True)


def _unique_entries(frame: pd.DataFrame, what: str) -> pd.Index:
    ids = pd.Index(frame["entry_id"])
    if ids.has_duplicates:
        raise ValueError(f"the {what} out-of-fold predictions have repeated entry_id values")
    return ids


def segment_errors(oof: pd.DataFrame, baseline_oof: pd.DataFrame, segments: pd.DataFrame) -> list[SegmentError]:
    """MAE, bias and size of the champion by team and by `circuit_type`, with the baseline's MAE on the same rows.

    Both OOF tables must cover the same entries and every entry must have a team and a circuit type, or it raises.
    The result lists the teams first and then the circuit types, each with the largest champion MAE first (ties by
    name). Nothing here reads a test row.
    """
    champion_ids = _unique_entries(oof, "champion's")
    baseline_ids = _unique_entries(baseline_oof, "baseline's")
    if set(champion_ids) != set(baseline_ids):
        only = sorted(set(champion_ids) ^ set(baseline_ids))
        raise ValueError(
            f"the champion's and the baseline's out-of-fold predictions do not cover the same entries "
            f"({len(only)} differ, e.g. entry_id {only[0]})"
        )
    segments = segments.drop_duplicates("entry_id").set_index("entry_id")
    unknown = sorted(set(champion_ids) - set(segments.index))
    if unknown:
        raise ValueError(f"{len(unknown)} entries have no team or circuit type (e.g. entry_id {unknown[0]})")
    mapped = segments.loc[champion_ids, list(SEGMENT_TYPES)]
    empty = mapped[mapped.isna().any(axis=1)]
    if not empty.empty:
        raise ValueError(f"{len(empty)} entries have no team or circuit type (e.g. entry_id {empty.index[0]})")

    frame = oof[["entry_id", "year", "round", "y_true", "y_pred"]].copy()
    frame["error"] = frame["y_pred"] - frame["y_true"]
    frame["abs_error"] = frame["error"].abs()
    baseline = baseline_oof.set_index("entry_id")
    frame["baseline_abs_error"] = (
        (baseline.loc[frame["entry_id"], "y_pred"] - baseline.loc[frame["entry_id"], "y_true"]).abs().to_numpy()
    )
    for kind in SEGMENT_TYPES:
        frame[kind] = mapped[kind].to_numpy()

    result = []
    for kind in SEGMENT_TYPES:
        rows = []
        for value, part in frame.groupby(kind, sort=True):
            rows.append(
                SegmentError(
                    segment_type=kind,
                    segment_value=str(value),
                    n_rows=len(part),
                    n_races=len(part[["year", "round"]].drop_duplicates()),
                    mae=float(part["abs_error"].mean()),
                    baseline_mae=float(part["baseline_abs_error"].mean()),
                    bias=float(part["error"].mean()),
                    few_rows=len(part) < MIN_SEGMENT_ROWS,
                )
            )
        result.extend(sorted(rows, key=lambda r: (-r.mae, r.segment_value)))
    return result


def read_notes(path: Path = NOTES_FILE) -> dict[str, str]:
    """The three hand-written sections of the notes file, by H2 heading, as written (only the blank lines around trim).

    Text before the first H2 (the header comment) and sections other than `NOTE_SECTIONS` are ignored. A missing file,
    a missing or repeated section or an empty one raises a `ValueError` that names it, so the card never goes out with
    a hole.
    """
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        raise ValueError(f"the notes file {path} does not exist") from None
    sections: dict[str, list[str]] = {}
    current = None
    for line in lines:
        if line.startswith("## "):
            current = line[3:].strip()
            if current in NOTE_SECTIONS and current in sections:
                raise ValueError(f"the notes file {path} has the section '{current}' twice")
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line)
    notes = {}
    for name in NOTE_SECTIONS:
        if name not in sections:
            raise ValueError(f"the notes file {path} has no '## {name}' section")
        text = "\n".join(sections[name]).strip("\n")
        if not text.strip():
            raise ValueError(f"the section '{name}' of the notes file {path} is empty")
        notes[name] = text
    return notes
