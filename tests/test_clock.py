from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.clock import TIMESTAMP_FORMAT, format_ts, parse_ts, utc_now


def test_format_uses_microseconds_and_z_suffix():
    moment = datetime(2026, 11, 14, 12, 0, 1, 200000, tzinfo=UTC)
    assert format_ts(moment) == "2026-11-14T12:00:01.200000Z"


def test_round_trip_preserves_microseconds():
    text = "2026-11-14T12:00:00.000001Z"
    moment = parse_ts(text)
    assert moment.microsecond == 1
    assert moment.tzinfo is UTC
    assert format_ts(moment) == text


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        format_ts(datetime(2026, 11, 14, 12, 0, 0))


def test_non_utc_datetime_is_converted():
    plus_two = timezone(timedelta(hours=2))
    assert format_ts(datetime(2026, 11, 14, 14, 0, 0, tzinfo=plus_two)) == "2026-11-14T12:00:00.000000Z"


@pytest.mark.parametrize(
    "text",
    [
        "2026-11-14T12:00:00Z",  # no fraction
        "2026-11-14T12:00:00.1Z",  # short fraction
        "2026-11-14T12:00:00.000000",  # no Z
        "2026-11-14T12:00:00.000000+00:00",
        "2026-11-14 12:00:00.000000Z",
        "2026-1-14T12:00:00.000000Z",  # unpadded month
    ],
)
def test_non_canonical_text_is_rejected(text):
    with pytest.raises(ValueError):
        parse_ts(text)


def test_string_order_equals_time_order():
    base = datetime(2026, 11, 14, 23, 59, 59, 999998, tzinfo=UTC)
    moments = [base + timedelta(microseconds=n) for n in (0, 1, 2, 3)] + [
        base - timedelta(days=40),
        base + timedelta(days=400),
        base + timedelta(seconds=0.5),
    ]
    texts = [format_ts(m) for m in moments]
    assert sorted(texts) == [format_ts(m) for m in sorted(moments)]
    assert all(len(t) == len(texts[0]) for t in texts)


def test_utc_now_is_aware_utc():
    now = utc_now()
    assert now.utcoffset() == timedelta(0)
    assert parse_ts(format_ts(now)) == now


def test_timestamp_format_is_the_documented_one():
    assert TIMESTAMP_FORMAT == "%Y-%m-%dT%H:%M:%S.%fZ"


def test_fake_clock_advances_without_sleeping(clock):
    start = clock()
    clock.advance(90.5)
    assert clock() - start == timedelta(seconds=90.5)
    assert clock() is clock()
