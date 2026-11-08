import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
sys.path.append(str(TESTS / "features"))  # the in-memory Supabase fake and the synthetic races
sys.path.append(str(TESTS / "training"))  # the model helpers and the synthetic feature frames
sys.path.append(str(TESTS / "ingestion"))  # the synthetic OpenF1 tables and the loaders of test_session

import pytest  # noqa: E402
from live_helpers import write_valid_race  # noqa: E402


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    """A raw folder with the synthetic race 2024 round 1 (Q and R, valid)."""
    write_valid_race(tmp_path)
    return tmp_path


@pytest.fixture
def uri(tmp_path: Path) -> str:
    """A SQLite MLflow store in the temporary folder, with the experiment's artifacts there too."""
    from mlflow import MlflowClient

    from training.sampling import EXPERIMENT

    store = f"sqlite:///{tmp_path / 'mlflow.db'}"
    MlflowClient(store).create_experiment(EXPERIMENT, artifact_location=(tmp_path / "artifacts").as_uri())
    return store


@pytest.fixture
def season_raw(tmp_path: Path) -> Path:
    from season_helpers import write_season

    write_season(tmp_path)
    return tmp_path


@pytest.fixture
def season_schedule_frame():
    from season_helpers import season_schedule

    return season_schedule()


@pytest.fixture
def season_db(season_raw, season_schedule_frame):
    from season_helpers import season_client

    return season_client(season_raw, season_schedule_frame)


@pytest.fixture
def season_config(monkeypatch):
    """The reference configuration of the synthetic season instead of `config/circuits.yml`."""
    from season_helpers import CIRCUITS

    monkeypatch.setattr("ingestion.reference.load_circuits", lambda: CIRCUITS)
    monkeypatch.setattr("ingestion.reference.load_team_aliases", lambda: {})
