import sys
from pathlib import Path

import pytest
from mlflow import MlflowClient

from training.sampling import EXPERIMENT

sys.path.append(str(Path(__file__).resolve().parents[1] / "features"))  # the in-memory Supabase fake


@pytest.fixture
def uri(tmp_path: Path) -> str:
    """A SQLite store in the temporary folder, with the experiment's artifacts there too (not in the cwd)."""
    store = f"sqlite:///{tmp_path / 'mlflow.db'}"
    MlflowClient(store).create_experiment(EXPERIMENT, artifact_location=(tmp_path / "artifacts").as_uri())
    return store
