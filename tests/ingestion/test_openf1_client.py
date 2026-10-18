import pytest

from ingestion.openf1_client import MAX_RATE_LIMIT_RETRIES, OpenF1Client, OpenF1Error, SessionNotAvailable


class Response:
    def __init__(self, status=200, body=None, headers=None, text=""):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, text

    def json(self):
        return self._body


class Fake:
    """Queued responses, a fake clock that only moves when the client sleeps, and a log of both."""

    def __init__(self, *responses):
        self.responses, self.calls, self.sleeps, self.now = list(responses), [], [], 0.0

    def http_get(self, url, params, timeout):
        self.calls.append((url, params, self.now))
        return self.responses.pop(0)

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def client(self):
        return OpenF1Client(self.http_get, self.sleep, lambda: self.now)


def test_returns_json_and_passes_params():
    fake = Fake(Response(body=[{"a": 1}]))
    assert fake.client().get("laps", session_key=9519) == [{"a": 1}]
    assert fake.calls[0][:2] == ("https://api.openf1.org/v1/laps", {"session_key": 9519})


def test_requests_are_spaced():
    fake = Fake(Response(body=[]), Response(body=[]), Response(body=[]))
    client = fake.client()
    for _ in range(3):
        client.get("laps")
    times = [t for _, _, t in fake.calls]
    assert times[1] - times[0] >= 2.2 and times[2] - times[1] >= 2.2


def test_no_wait_when_enough_time_has_passed():
    fake = Fake(Response(body=[]), Response(body=[]))
    client = fake.client()
    client.get("laps")
    fake.now += 10
    client.get("laps")
    assert fake.sleeps == []


def test_429_waits_retry_after_then_succeeds():
    fake = Fake(Response(429, headers={"Retry-After": "3"}), Response(body=[{"ok": True}]))
    assert fake.client().get("laps") == [{"ok": True}]
    assert 4.0 in fake.sleeps and len(fake.calls) == 2


def test_429_forever_raises():
    fake = Fake(*[Response(429) for _ in range(MAX_RATE_LIMIT_RETRIES + 1)])
    with pytest.raises(OpenF1Error, match="rate limited"):
        fake.client().get("laps")
    assert len(fake.calls) == MAX_RATE_LIMIT_RETRIES + 1


def test_404_is_an_empty_result():
    assert Fake(Response(404)).client().get("pit", session_key=1) == []


def test_401_is_session_not_available():
    fake = Fake(Response(401, text='{"detail":"Session in progress, access is restricted"}'))
    with pytest.raises(SessionNotAvailable, match="Session in progress"):
        fake.client().get("laps")


def test_other_errors_raise():
    with pytest.raises(OpenF1Error, match="HTTP 500"):
        Fake(Response(500)).client().get("laps")
