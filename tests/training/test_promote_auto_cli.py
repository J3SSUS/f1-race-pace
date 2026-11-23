"""The CLI of `scripts/promote.py` with injected fakes: what it prints, the exit codes and what it never calls."""

import importlib.util
from pathlib import Path

import pytest

from features.store import StoreError

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "promote.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("promote_cli_auto", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verdict(**over) -> dict:
    base = {
        "passes": True,
        "reason": None,
        "common_races": 5,
        "window": [1, 2, 3, 4, 5],
        "per_race": [(i, 1.0, 0.5 if i != 5 else 1.1) for i in range(1, 6)],
        "champion_mae": 1.0,
        "challenger_mae": 0.62,
        "races_worse": 1,
        "improvement_pct": 38.0,
        "challenger_version_id": 7,
        "champion_version_id": 4,
        "dry_run": False,
        "promoted": True,
        "shadow_set": 4,
        "shadow_cleared": [],
        "challenger_alias_removed": True,
    }
    return base | over


class Recorder:
    """A stand-in for `promote` or `auto_promote` that records its calls and returns or raises what it is given."""

    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.result


def run(cli, argv, *, promoter=None, auto_promoter=None):
    lines: list[str] = []
    code = cli.main(
        argv,
        promoter=promoter or Recorder(error=AssertionError("the manual promoter must not run")),
        auto_promoter=auto_promoter or Recorder(error=AssertionError("the automatic promoter must not run")),
        out=lines.append,
    )
    return code, lines


def test_auto_promotes_prints_the_races_the_shadow_and_the_publish_reminder(cli):
    auto = Recorder(verdict())
    code, lines = run(cli, ["--auto", "7"], auto_promoter=auto)
    assert code == 0 and auto.calls == [((7,), {"dry_run": False})]
    text = "\n".join(lines)
    assert (
        "race 5: champion 1.0000, challenger 1.1000 (worse)" in text
        and "race 1: champion 1.0000, challenger 0.5000" in text
    )
    assert "promoted model version 7 to champion (automatic)" in text and "+38.0%" in text
    assert "model version 4 is now the shadow" in text and "`challenger` alias was removed" in text
    assert lines[-1] == cli.PUBLISH_REMINDER and "scripts/publish.py" in lines[-1]


def test_a_rule_that_is_not_met_exits_0_with_the_reasons_and_no_reminder(cli):
    not_met = verdict(passes=False, promoted=False, reason="worse than the champion in 3 of 5 races (...)")
    code, lines = run(cli, ["--auto", "7"], auto_promoter=Recorder(not_met))
    assert code == 0 and lines[-1].startswith("not promoted: worse than the champion in 3 of 5")
    assert cli.PUBLISH_REMINDER not in lines


def test_too_few_races_exits_0_and_says_how_many(cli):
    few = verdict(
        passes=False,
        promoted=False,
        common_races=3,
        window=[],
        per_race=[],
        champion_mae=None,
        challenger_mae=None,
        races_worse=None,
        improvement_pct=None,
        reason="only 3 common races, 5 needed (2 more): not promoted yet",
    )
    code, lines = run(cli, ["--auto", "7"], auto_promoter=Recorder(few))
    assert (
        code == 0 and "races both predicted: 3" in lines and lines[-1].startswith("not promoted: only 3 common races")
    )


def test_a_dry_run_passes_the_flag_and_says_nothing_was_written(cli):
    auto = Recorder(verdict(promoted=False, dry_run=True, reason="dry run: the rule is met, nothing was written"))
    code, lines = run(cli, ["--auto", "7", "--dry-run"], auto_promoter=auto)
    assert code == 0 and auto.calls == [((7,), {"dry_run": True})]
    assert lines[-1] == "dry run: the rule is met, model version 7 would be promoted; nothing was written"
    assert cli.PUBLISH_REMINDER not in lines  # nothing was promoted


def test_dry_run_without_auto_stops_before_calling_anything(cli):
    code, lines = run(cli, ["4", "--dry-run"])
    assert code == 1 and lines == ["STOPPED: --dry-run only goes with --auto"]


@pytest.mark.parametrize(
    "error", [ValueError("it is not the challenger"), StoreError("read failed"), RuntimeError("x")]
)
def test_an_error_prints_stopped_and_exits_1(cli, error):
    code, lines = run(cli, ["--auto", "7"], auto_promoter=Recorder(error=error))
    assert code == 1 and lines == [f"STOPPED: {error}"]


def manual_result(**over) -> dict:
    base = {
        "model_version_id": 7,
        "mlflow_version": "7",
        "test_mae": 0.8,
        "baseline_test_mae": 1.0,
        "improvement_pct": 20.0,
        "changed": True,
        "alias_set": True,
        "demoted": [1],
        "shadow_set": 4,
        "shadow_cleared": [],
        "challenger_alias_removed": False,
    }
    return base | over


def test_a_manual_promotion_prints_the_shadow_and_the_publish_reminder(cli):
    manual = Recorder(manual_result())
    code, lines = run(cli, ["7"], promoter=manual)
    assert code == 0 and manual.calls == [((7,), {})]
    assert any("model version 4 is now the shadow" in line for line in lines)
    assert lines[-1] == cli.PUBLISH_REMINDER


def test_the_first_manual_champion_prints_no_shadow_line(cli):
    code, lines = run(cli, ["4"], promoter=Recorder(manual_result(shadow_set=None, demoted=[])))
    assert code == 0 and not any("shadow" in line for line in lines) and lines[-1] == cli.PUBLISH_REMINDER


def test_a_manual_call_that_changes_nothing_has_no_reminder(cli):
    code, lines = run(cli, ["4"], promoter=Recorder(manual_result(changed=False)))
    assert code == 0 and len(lines) == 1 and "already the champion" in lines[0]
