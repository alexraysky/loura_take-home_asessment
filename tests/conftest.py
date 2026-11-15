import asyncio
import dataclasses
import time
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import Database
from app.main import create_app
from app.repository import Repository


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


@pytest.fixture
def db(settings: Settings) -> Database:
    """The database at settings.db_path with the schema in place."""
    database = Database(settings.db_path)
    database.create_schema()
    return database


@pytest.fixture
def repository(db: Database, clock: FakeClock) -> Repository:
    return Repository(db, clock)


# Shared by the lifespan tests that run real workers (test_app.py, test_reclassify.py).

DEADLINE_S = 5.0


class GateLLM:
    """Parks every call forever; counts the parked calls."""

    def __init__(self) -> None:
        self.parked = 0

    async def complete(self, system: str, user: str) -> str:
        self.parked += 1
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.fixture
def worker_settings(settings: Settings) -> Settings:
    # backoff 0: a retry is claimable at once, so the frozen FakeClock never needs to move.
    return replace(
        settings, worker_count=2, poll_interval_s=0.001, backoff_base_s=0.0, llm_timeout_s=1.0, lease_timeout_s=2.0
    )


def wait_until(predicate, what: str):
    deadline = time.monotonic() + DEADLINE_S
    while time.monotonic() < deadline:
        if result := predicate():
            return result
        time.sleep(0.005)
    raise AssertionError(f"timed out waiting for {what}")


def wait_for_status(client: TestClient, id: str, status: str) -> dict:
    def check():
        body = client.get(f"/tickets/{id}").json()
        return body if body["status"] == status else None

    return wait_until(check, f"{id} to be {status}")
