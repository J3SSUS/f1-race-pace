"""The only module that calls OpenF1: spaced requests, 429 waits, 404 as empty, 401 as "not available yet"."""

import time
from collections.abc import Callable

import requests

BASE_URL = "https://api.openf1.org/v1"
MIN_GAP_SECONDS = 2.2  # free tier: 30 requests per minute
MAX_RATE_LIMIT_RETRIES = 5
TIMEOUT_SECONDS = 60


class OpenF1Error(Exception):
    pass


class SessionNotAvailable(OpenF1Error):
    """HTTP 401: OpenF1 restricts access during a session and for about 30 minutes after it."""


class OpenF1Client:
    def __init__(
        self,
        http_get: Callable = requests.get,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        min_gap: float = MIN_GAP_SECONDS,
    ):
        self._http_get, self._sleep, self._clock, self._min_gap = http_get, sleep, clock, min_gap
        self._last_request: float | None = None

    def _space_requests(self) -> None:
        if self._last_request is not None:
            wait = self._min_gap - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def get(self, endpoint: str, **params) -> list[dict]:
        url = f"{BASE_URL}/{endpoint}"
        for _ in range(MAX_RATE_LIMIT_RETRIES + 1):
            self._space_requests()
            response = self._http_get(url, params=params, timeout=TIMEOUT_SECONDS)
            status = response.status_code
            if status == 429:
                self._sleep(float(response.headers.get("Retry-After", 2)) + 1)
                continue
            if status == 404:  # OpenF1 answers 404 when a query has no rows
                return []
            if status == 401:
                raise SessionNotAvailable(f"{endpoint} {params}: {response.text[:200]}")
            if status >= 400:
                raise OpenF1Error(f"{endpoint} {params}: HTTP {status}")
            return response.json()
        raise OpenF1Error(f"{endpoint} {params}: still rate limited after {MAX_RATE_LIMIT_RETRIES} retries")


_client = OpenF1Client()  # shared, so the spacing holds across every caller in the process


def get(endpoint: str, **params) -> list[dict]:
    return _client.get(endpoint, **params)
