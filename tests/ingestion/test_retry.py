import pytest

from ingestion.retry import with_backoff


def flaky(failures: int):
    calls = []

    def fn():
        calls.append(1)
        if len(calls) <= failures:
            raise ConnectionError(f"fail {len(calls)}")
        return "ok"

    return fn, calls


def test_succeeds_on_the_third_try():
    fn, calls = flaky(2)
    waits = []
    assert with_backoff(fn, sleep=waits.append) == "ok"
    assert len(calls) == 3
    assert waits == [2, 4]


def test_fails_after_three_attempts_with_the_last_error():
    fn, calls = flaky(10)
    waits = []
    with pytest.raises(ConnectionError, match="fail 3"):
        with_backoff(fn, sleep=waits.append)
    assert len(calls) == 3
    assert waits == [2, 4]


def test_no_wait_when_first_try_works():
    fn, _ = flaky(0)
    waits = []
    assert with_backoff(fn, sleep=waits.append) == "ok"
    assert waits == []
