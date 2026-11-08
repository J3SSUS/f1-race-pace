import json

import pytest
from test_session import bad_loader, failing_loader, good_loader, loaded, not_published_loader

from ingestion.openf1_client import SessionNotAvailable
from live.ground_truth import GroundTruthError, ingest_race_sessions


def run(tmp_path, loader, run_id="r1"):
    return ingest_race_sessions(2024, 8, run_id=run_id, raw_dir=tmp_path, loader=loader, sleep=lambda s: None)


def recording(loader):
    calls = []

    def wrapper(year, rnd, session):
        calls.append(session)
        return loader(year, rnd, session)

    return wrapper, calls


def live_loader(year, rnd, session):
    raise SessionNotAvailable("sessions: a live F1 session is in progress")


def test_q_and_r_are_ingested_with_the_given_run_id(tmp_path):
    calls = []
    assert run(tmp_path, good_loader(calls), run_id="abc") == {"Q": "downloaded", "R": "downloaded"}
    assert calls == [(2024, 8, "Q"), (2024, 8, "R")]
    for session in ("Q", "R"):
        meta = json.loads((tmp_path / "2024" / "08" / session / "meta.json").read_text())
        assert meta["run_id"] == "abc" and meta["valid"] is True


def test_a_session_already_on_disk_is_skipped(tmp_path):
    run(tmp_path, good_loader())
    calls = []
    assert run(tmp_path, good_loader(calls)) == {"Q": "skipped", "R": "skipped"}
    assert calls == []


def test_only_the_missing_session_is_downloaded(tmp_path):
    run(tmp_path, good_loader())
    (tmp_path / "2024" / "08" / "R" / "meta.json").unlink()
    assert run(tmp_path, good_loader()) == {"Q": "skipped", "R": "downloaded"}


def test_a_session_that_is_not_published_names_the_session_and_stops(tmp_path):
    loader, calls = recording(not_published_loader)
    with pytest.raises(GroundTruthError, match=r"session Q of 2024 round 8 is not available.*retry later"):
        run(tmp_path, loader)
    assert calls == ["Q"]  # R is not tried
    assert not (tmp_path / "2024").exists()


def test_a_download_that_fails_after_its_retries_names_the_session(tmp_path):
    with pytest.raises(GroundTruthError, match=r"session Q of 2024 round 8 is not available"):
        run(tmp_path, failing_loader)
    assert not (tmp_path / "2024").exists()


def test_a_session_blocked_by_a_live_session_says_when_to_retry(tmp_path):
    with pytest.raises(GroundTruthError, match=r"session Q of 2024 round 8 is blocked.*live.*retry 30 minutes after"):
        run(tmp_path, live_loader)
    assert not (tmp_path / "2024").exists()


def test_the_race_blocked_after_a_good_qualifying_names_r(tmp_path):
    def loader(year, rnd, session):
        return good_loader()(year, rnd, session) if session == "Q" else live_loader(year, rnd, session)

    with pytest.raises(GroundTruthError, match=r"session R of 2024 round 8 is blocked"):
        run(tmp_path, loader)


def test_an_invalid_session_lists_its_errors_and_the_way_to_download_it_again(tmp_path):
    with pytest.raises(
        GroundTruthError, match=r"session Q of 2024 round 8 is not valid: .*laps.*scripts/ingest\.py.*--force"
    ):
        run(tmp_path, bad_loader)


def test_an_invalid_session_already_on_disk_is_still_refused(tmp_path):
    with pytest.raises(GroundTruthError):
        run(tmp_path, bad_loader)
    with pytest.raises(GroundTruthError, match=r"not valid"):
        run(tmp_path, good_loader())  # skipped: its meta.json says valid = false


def test_an_invalid_race_session_names_r(tmp_path):
    def loader(year, rnd, session):
        return loaded() if session == "Q" else bad_loader(year, rnd, session)

    with pytest.raises(GroundTruthError, match=r"session R of 2024 round 8 is not valid"):
        run(tmp_path, loader)
