import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "features"))  # the in-memory Supabase fake

import pytest  # noqa: E402


@pytest.fixture
def uri(tmp_path: Path) -> str:
    """A SQLite MLflow store in the temporary folder with the `drift` experiment, its artifacts there too."""
    from mlflow import MlflowClient

    from monitoring.drift import EXPERIMENT

    store = f"sqlite:///{tmp_path / 'mlflow.db'}"
    MlflowClient(store).create_experiment(EXPERIMENT, artifact_location=(tmp_path / "artifacts").as_uri())
    return store
