"""Test options shared by every test folder.

Tests marked `slow` take seconds (they log real models in a local MLflow store, no network). They are skipped by
default so `uv run pytest tests` stays fast; `uv run pytest tests --runslow` runs everything, and is the run that
closes a spec.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="also run the tests marked slow")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: takes seconds (real models in a local MLflow store); needs --runslow")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow: run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
