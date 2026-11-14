"""Injected clock and the single timestamp format used for storage and the API.

The format is fixed width with microseconds and a ``Z`` suffix, so string
order equals time order.
"""

from collections.abc import Callable
from datetime import UTC, datetime

Clock = Callable[[], datetime]
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def utc_now() -> datetime:
    return datetime.now(UTC)


def format_ts(moment: datetime) -> str:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(UTC).strftime(TIMESTAMP_FORMAT)


def parse_ts(text: str) -> datetime:
    """Parse a timestamp produced by ``format_ts``; any other spelling is rejected."""
    moment = datetime.strptime(text, TIMESTAMP_FORMAT).replace(tzinfo=UTC)
    if format_ts(moment) != text:
        raise ValueError(f"timestamp {text!r} is not in the canonical format {TIMESTAMP_FORMAT}")
    return moment
