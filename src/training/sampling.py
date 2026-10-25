"""The `sampling` step: ingest the races up to the cutoff, label them train or test and register the run.

The split itself is defined in `training.split`. Here the sessions of every race are ingested and checked first, so
a race with a missing qualifying or race session stops the step before anything is split.
"""

import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.data
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from common.config import TrainingConfig, load_config
from common.supabase_io import ENV_PATH
from ingestion.reference import ReferenceDataError, ensure_race_entries
from ingestion.schedule import expected_sessions, load_schedules
from ingestion.session import RAW_DIR, ingest_session, session_dir
from training.split import select_races, split_races

# The sessions the features read, in the order they are listed in the race index. FP2 and S are read only when the
# weekend has them (a sprint weekend has no FP2); Q and R are always needed.
EXPERIMENT = "f1-race-pace"
RUN_NAME = "sampling"
PROJECT_DIR = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_DIR / "data" / "splits"
SPLIT_RULE = "chronological by race; the first floor((1 - test_fraction) * n) races are train"
SESSION_ORDER = ("FP2", "Q", "S", "R")
OPTIONAL_SESSIONS = ("FP2", "S")


class SamplingError(Exception):
    """A Q or R session that failed or is invalid, or no races to split."""

    def __init__(self, message: str, problems: list[dict] | None = None):
        super().__init__(message)
        self.problems = problems or []


@dataclass
class IngestReport:
    sessions: dict[tuple[int, int], list[str]]  # sessions read per race, in `SESSION_ORDER`
    issues: list[dict] = field(default_factory=list)  # FP2 and S sessions that are failed or invalid


def _status(result) -> str | None:
    """`failed`, `invalid` or None when the session is usable."""
    if result.status == "failed":
        return "failed"
    return None if result.valid else "invalid"


def _problem_message(problems: list[dict]) -> str:
    lines = [f"{p['year']} round {p['round']} {p['session']}: {p['status']}" for p in problems]
    return "Q or R sessions that cannot be used, nothing was split:\n" + "\n".join(lines)


def ingest_races(
    races: list[tuple[int, int]],
    schedule: pd.DataFrame,
    *,
    client=None,
    raw_dir: Path = RAW_DIR,
    ingest: Callable = ingest_session,
    ensure: Callable = ensure_race_entries,
    log: Callable[[str], None] | None = None,
) -> IngestReport:
    """Ingest FP2, S (if the schedule has it), Q and R of every race and make sure its entries exist.

    A failed or invalid FP2 or S goes to `issues`. A failed R, a failed Q, an invalid R, or a Q that
    `ensure_race_entries` cannot use raises one `SamplingError` that lists every race affected. A Q that is invalid
    only because of a table the entries do not need is accepted (the Q rule of SPEC 02).
    """
    report = IngestReport(sessions={})
    problems: list[dict] = []
    for year, round in races:
        scheduled = set(expected_sessions(schedule, year, round))
        sessions = [s for s in SESSION_ORDER if s not in OPTIONAL_SESSIONS or s in scheduled]
        report.sessions[(year, round)] = sessions

        status = {s: _status(ingest(year, round, s, raw_dir=raw_dir)) for s in sessions}
        for session in OPTIONAL_SESSIONS:
            if status.get(session):
                report.issues.append({"year": year, "round": round, "session": session, "status": status[session]})
        if status["R"]:
            problems.append({"year": year, "round": round, "session": "R", "status": status["R"]})
        if status["Q"] == "failed":
            problems.append({"year": year, "round": round, "session": "Q", "status": "failed"})
        else:
            try:
                ensure(year, round, client=client, schedule=schedule, ingest=ingest, raw_dir=raw_dir)
            except ReferenceDataError:
                problems.append({"year": year, "round": round, "session": "Q", "status": "invalid"})
        if log:
            log(f"{year} round {round:>2}: {','.join(sessions)}")

    if problems:
        raise SamplingError(_problem_message(problems), problems)
    return report


PART_COLUMNS = ["year", "round", "split", "sessions", "digest"]
PART_DIGEST_LENGTH = 16


@dataclass(frozen=True)
class Parts:
    train: pd.DataFrame  # one row per train race, `PART_COLUMNS`
    test: pd.DataFrame
    train_digest: str
    test_digest: str


def race_digest(year: int, round: int, sessions: list[str], raw_dir: Path = RAW_DIR) -> str:
    """SHA-256 of the Parquet tables of the sessions of a race, in sorted path order.

    The path inside the race (`Q/laps.parquet`) goes in with the bytes. `meta.json` is left out: it carries the run
    id and the download date, so a forced download of identical data would change the digest. A session with no
    Parquets on disk (a failed FP2 or S) adds nothing.
    """
    digest = hashlib.sha256()
    race_dir = session_dir(year, round, "Q", raw_dir).parent
    files = sorted(path for session in sessions for path in (race_dir / session).glob("*.parquet"))
    for path in files:
        digest.update(f"{path.relative_to(race_dir).as_posix()}\n".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def part_digest(frame: pd.DataFrame) -> str:
    """First 16 hex characters of the SHA-256 of the rows (`year`, `round`, `split`, race digest), in order."""
    digest = hashlib.sha256()
    for year, round, split, race in zip(frame["year"], frame["round"], frame["split"], frame["digest"], strict=True):
        digest.update(f"{year},{round},{split},{race}\n".encode())
    return digest.hexdigest()[:PART_DIGEST_LENGTH]


def build_parts(labels: pd.DataFrame, sessions: dict[tuple[int, int], list[str]], raw_dir: Path = RAW_DIR) -> Parts:
    """The train and test race indexes from the labels of `split_races` and the sessions read per race."""
    races = labels.copy()
    keys = list(zip(races["year"], races["round"], strict=True))
    races["sessions"] = [",".join(sessions[(int(y), int(r))]) for y, r in keys]
    races["digest"] = [race_digest(int(y), int(r), sessions[(int(y), int(r))], raw_dir) for y, r in keys]
    races = races[PART_COLUMNS]
    train = races[races["split"] == "train"].reset_index(drop=True)
    test = races[races["split"] == "test"].reset_index(drop=True)
    return Parts(train, test, part_digest(train), part_digest(test))


def _shown(path: Path) -> str:
    """The path as the manifest shows it: relative to the project when it is inside it."""
    return path.relative_to(PROJECT_DIR).as_posix() if path.is_relative_to(PROJECT_DIR) else str(path)


def _part_entry(frame: pd.DataFrame, digest: str, path: Path) -> dict:
    races = [[int(y), int(r)] for y, r in zip(frame["year"], frame["round"], strict=True)]
    return {"path": _shown(path), "rows": len(frame), "digest": digest, "races": races}


def build_manifest(
    run_id: str,
    parts: Parts,
    config: TrainingConfig,
    issues: list[dict],
    split_dir: Path = SPLIT_DIR,
    now: datetime | None = None,
) -> dict:
    """The manifest of a sampling run, as in the data model of SPEC 06."""
    cutoff = config.data.cutoff
    folder = split_dir / run_id
    return {
        "run_id": run_id,
        "generated_at": (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split_rule": SPLIT_RULE,
        "config": {
            "first_year": config.data.first_year,
            "last_year": config.data.last_year,
            "cutoff": None if cutoff is None else {"year": cutoff.year, "round": cutoff.round},
            "test_fraction": config.split.test_fraction,
        },
        "versions": {
            "python": platform.python_version(),
            "mlflow": version("mlflow"),
            "pandas": version("pandas"),
        },
        "train": _part_entry(parts.train, parts.train_digest, folder / "train.parquet"),
        "test": _part_entry(parts.test, parts.test_digest, folder / "test.parquet"),
        "session_issues": issues,
    }


def write_manifest(manifest: dict, parts: Parts, split_dir: Path = SPLIT_DIR) -> Path:
    """Write `train.parquet`, `test.parquet` and `manifest.json` in `split_dir/<run_id>/`; returns the manifest."""
    folder = split_dir / manifest["run_id"]
    folder.mkdir(parents=True, exist_ok=True)
    parts.train.to_parquet(folder / "train.parquet", index=False)
    parts.test.to_parquet(folder / "test.parquet", index=False)
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    """The argument, or `MLFLOW_TRACKING_URI` from the environment or `.env`. Never falls back to a local folder."""
    if tracking_uri:
        return tracking_uri
    load_dotenv(ENV_PATH)
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("Missing MLFLOW_TRACKING_URI: set it in .env or the environment (see .env.example)")
    return uri


def _cutoff_param(config: TrainingConfig) -> str:
    cutoff = config.data.cutoff
    return "none" if cutoff is None else f"{cutoff.year}-{cutoff.round}"


def register_run(
    parts: Parts,
    config: TrainingConfig,
    issues: list[dict],
    *,
    split_dir: Path = SPLIT_DIR,
    tracking_uri: str | None = None,
    now: datetime | None = None,
) -> str:
    """Open the parent run, write the manifest and the two Parquets, log params, datasets and manifest, and close it.

    The run ends `FINISHED`, or `FAILED` if anything inside raises (the exception goes on). Returns the run id.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=RUN_NAME, tags={"step": "sampling"}) as run:
        run_id = run.info.run_id
        manifest = build_manifest(run_id, parts, config, issues, split_dir, now)
        manifest_path = write_manifest(manifest, parts, split_dir)
        mlflow.log_params(
            {
                "first_year": config.data.first_year,
                "last_year": config.data.last_year,
                "cutoff": _cutoff_param(config),
                "test_fraction": config.split.test_fraction,
                "races": len(parts.train) + len(parts.test),
                "train_races": len(parts.train),
                "test_races": len(parts.test),
            }
        )
        for name, frame, digest in (
            ("train", parts.train, parts.train_digest),
            ("test", parts.test, parts.test_digest),
        ):
            source = str(manifest_path.parent / f"{name}.parquet")
            dataset = mlflow.data.from_pandas(frame, source=source, name=name, digest=digest)
            mlflow.log_input(dataset, context=name)
        mlflow.log_artifact(str(manifest_path))
    return run_id


def read_split(run_id: str, tracking_uri: str | None = None) -> pd.DataFrame:
    """`year`, `round` and `split` of every race of a sampling run, sorted by `(year, round)`.

    Read from the `manifest.json` artifact of the run, so the split travels with it and nobody recomputes it.
    """
    client = MlflowClient(resolve_tracking_uri(tracking_uri))
    try:
        client.get_run(run_id)
    except MlflowException as exc:
        raise ValueError(f"No MLflow run '{run_id}': run scripts/sample.py and use the run_id it prints") from exc
    with tempfile.TemporaryDirectory() as folder:
        try:
            path = client.download_artifacts(run_id, "manifest.json", folder)
        except MlflowException as exc:
            raise ValueError(f"MLflow run '{run_id}' has no manifest.json: it is not a sampling run") from exc
        manifest = json.loads(Path(path).read_text())
    rows = [(y, r, name) for name in ("train", "test") for y, r in manifest[name]["races"]]
    frame = pd.DataFrame(rows, columns=["year", "round", "split"]).astype({"year": "int64", "round": "int64"})
    return frame.sort_values(["year", "round"]).reset_index(drop=True)


@dataclass(frozen=True)
class SamplingResult:
    run_id: str
    train_races: int
    test_races: int
    train_digest: str
    test_digest: str
    manifest_path: Path


def run_sampling(
    config: TrainingConfig | None = None,
    *,
    client=None,
    raw_dir: Path = RAW_DIR,
    split_dir: Path = SPLIT_DIR,
    tracking_uri: str | None = None,
    schedule: pd.DataFrame | None = None,
    ingest: Callable = ingest_session,
    ensure: Callable = ensure_race_entries,
    today: datetime | None = None,
    log: Callable[[str], None] | None = None,
) -> SamplingResult:
    """The `sampling` step: select the races, split them, ingest and check their sessions, register the run.

    Raises `SamplingError` before any MLflow run is opened if there are no races, if the split leaves the train or
    the test part empty, or if a Q or R session cannot be used. `config=None` loads `config/training.json`.
    """
    config = config or load_config()
    if schedule is None:
        cutoff = config.data.cutoff
        last_year = config.data.last_year if cutoff is None else cutoff.year
        schedule = load_schedules(config.data.first_year, last_year)
    races = select_races(schedule, config, today)
    labels = split_races(races, config.split.test_fraction)
    counts = labels["split"].value_counts()
    if counts.get("train", 0) == 0 or counts.get("test", 0) == 0:
        raise SamplingError(
            f"no races to split: {len(races)} race(s) held, test_fraction {config.split.test_fraction} leaves "
            f"{counts.get('train', 0)} for train and {counts.get('test', 0)} for test"
        )

    report = ingest_races(races, schedule, client=client, raw_dir=raw_dir, ingest=ingest, ensure=ensure, log=log)
    parts = build_parts(labels, report.sessions, raw_dir)
    run_id = register_run(parts, config, report.issues, split_dir=split_dir, tracking_uri=tracking_uri)
    return SamplingResult(
        run_id=run_id,
        train_races=len(parts.train),
        test_races=len(parts.test),
        train_digest=parts.train_digest,
        test_digest=parts.test_digest,
        manifest_path=split_dir / run_id / "manifest.json",
    )
