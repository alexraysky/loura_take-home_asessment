import dataclasses
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


class FakeClock:
    """Clock that only moves when told to."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


# Every Settings field is read from the upper-cased variable of the same name.
CONFIG_VARS = tuple(f.name.upper() for f in dataclasses.fields(Settings))


@pytest.fixture(autouse=True)
def clean_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's shell configuration out of the tests."""
    for name in CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(datetime(2026, 11, 14, 12, 0, 0, tzinfo=UTC))


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(worker_count=0, db_path=str(tmp_path / "tickets.db"))


@pytest.fixture
def app(settings: Settings, clock: FakeClock) -> FastAPI:
    return create_app(settings, clock=clock)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
