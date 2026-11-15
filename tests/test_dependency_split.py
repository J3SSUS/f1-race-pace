"""The `data` image has no ML libraries, so the data scripts must never reach one.

`f1-pipeline-data` installs only `[project].dependencies`; the group `ml` (scikit-learn, mlflow, ...) is left out. The
five steps that image runs (`ingest`, `load_reference`, `load_features`, `ground_truth`, `next_race`) would break at
run time in the cluster, not here, if a module they reach imported one of those packages. This test reads the code (it
imports nothing) and follows every import of those scripts through `src/`, including the `__init__.py` Python runs on
the way and imports inside functions, and fails naming the chain that reaches an `ml` package.
"""

import ast
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA_SCRIPTS = ("ingest", "load_reference", "load_features", "ground_truth", "next_race")

# The import name when it differs from the distribution name in pyproject.toml.
IMPORT_NAMES = {"scikit-learn": "sklearn"}


def ml_packages(pyproject: Path = ROOT / "pyproject.toml") -> set[str]:
    """Top-level import names of the `ml` dependency group."""
    group = tomllib.loads(pyproject.read_text())["dependency-groups"]["ml"]
    names = {requirement.split("==")[0].strip().lower() for requirement in group}
    return {IMPORT_NAMES.get(name, name.replace("-", "_")) for name in names}


def module_file(module: str, src: Path) -> Path | None:
    """The file of a module under `src`, or None when it is not ours (standard library or third party)."""
    base = src.joinpath(*module.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def imported_modules(path: Path, src: Path) -> set[str]:
    """Every module a file imports, anywhere in it (also inside functions), as an absolute dotted name."""
    found: set[str] = set()
    package = path.relative_to(src).parent.parts if path.is_relative_to(src) else ()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(package[: len(package) - node.level + 1])
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            if module:
                found.add(module)
                # `from pkg import name` may name a submodule of ours.
                found.update(
                    f"{module}.{alias.name}" for alias in node.names if module_file(f"{module}.{alias.name}", src)
                )
    return found


def reached_third_party(entries: list[Path], src: Path = SRC) -> dict[str, list[str]]:
    """Every non-local top-level package the entry files reach, each with the chain of files that imports it."""
    third_party: dict[str, list[str]] = {}
    seen: set[Path] = set()
    queue: list[tuple[Path, list[str]]] = [(entry, [entry.name]) for entry in entries]
    while queue:
        path, chain = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        for module in sorted(imported_modules(path, src)):
            parts = module.split(".")
            local = [f for i in range(1, len(parts) + 1) if (f := module_file(".".join(parts[:i]), src))]
            if not local:
                top = parts[0]
                if top not in sys.stdlib_module_names:
                    third_party.setdefault(top, chain + [module])
                continue
            # Importing `a.b.c` runs a/__init__.py, a/b/__init__.py and c itself.
            queue.extend((file, chain + [file.relative_to(src).as_posix()]) for file in local if file not in seen)
    return third_party


def test_data_scripts_reach_no_ml_package():
    entries = [ROOT / "scripts" / f"{name}.py" for name in DATA_SCRIPTS]
    reached = reached_third_party(entries)
    leaks = {top: chain for top, chain in reached.items() if top in ml_packages()}
    assert not leaks, "the `data` image has no ML libraries, but these are reached: " + "; ".join(
        f"{top} via {' -> '.join(chain)}" for top, chain in sorted(leaks.items())
    )


def test_the_walk_really_follows_the_code():
    """Guards the guard: if the walk found nothing, the test above would pass for the wrong reason."""
    entries = [ROOT / "scripts" / f"{name}.py" for name in DATA_SCRIPTS]
    reached = reached_third_party(entries)
    assert {"pandas", "supabase"} <= set(reached)
    assert ml_packages() == {
        "sklearn",
        "xgboost",
        "lightgbm",
        "catboost",
        "optuna",
        "mlflow",
        "shap",
        "matplotlib",
        "jinja2",
        "markdown",
    }


@pytest.fixture
def fake_src(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "common").mkdir(parents=True)
    (src / "common" / "__init__.py").write_text("")
    (src / "common" / "util.py").write_text("import pandas\n")
    return src


@pytest.mark.parametrize(
    "body",
    [
        "import sklearn\n",
        "from mlflow import log_metric\n",
        "import matplotlib.pyplot as plt\n",
        "def late():\n    import catboost\n",
    ],
)
def test_an_ml_import_is_detected_through_the_chain(fake_src: Path, body: str):
    """A planted import is found at any depth: in the script's module, in a function and via an `__init__`."""
    (fake_src / "common" / "util.py").write_text(body)
    script = fake_src.parent / "step.py"
    script.write_text("from common.util import something\n")
    reached = reached_third_party([script], src=fake_src)
    assert set(reached) & ml_packages()


def test_an_ml_import_in_a_package_init_is_detected(fake_src: Path):
    (fake_src / "common" / "__init__.py").write_text("import xgboost\n")
    script = fake_src.parent / "step.py"
    script.write_text("from common.util import something\n")
    assert "xgboost" in reached_third_party([script], src=fake_src)


def test_a_clean_chain_passes(fake_src: Path):
    script = fake_src.parent / "step.py"
    script.write_text("from common.util import something\n")
    reached = reached_third_party([script], src=fake_src)
    assert reached == {"pandas": ["step.py", "common/util.py", "pandas"]}
    assert not set(reached) & ml_packages()
