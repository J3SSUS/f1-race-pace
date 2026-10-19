"""Hand-written retry with backoff (no tenacity)."""

import time
from collections.abc import Callable


def with_backoff[T](
    fn: Callable[[], T],
    attempts: int = 3,
    delays: tuple[float, ...] = (2, 4, 8),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn` up to `attempts` times, sleeping `delays[i]` after failed attempt i; re-raise the last error."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt == attempts - 1:
                raise
            sleep(delays[min(attempt, len(delays) - 1)])
    raise ValueError("attempts must be at least 1")
