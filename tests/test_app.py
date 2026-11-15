import logging
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.api import ApiError
from app.clock import format_ts, utc_now
from app.config import Settings
from app.db import Database
from app.llm import FakeLLM, Mode
from app.main import create_app
from app.repository import Repository
from app.worker import WorkerPool
from conftest import GateLLM, wait_for_status, wait_until
from tests.sample_data import TICKET_PAYLOADS


class Item(BaseModel):
    name: str = Field(min_length=1)
    count: int


@pytest.fixture
def probe_app(app: FastAPI) -> FastAPI:
    """The app with test-only routes that exercise each error handler."""

    @app.get("/_conflict")
    async def conflict():
        raise ApiError(409, "ticket_conflict", "Ticket t-1001 already exists with different content.")

    @app.get("/_detailed")
    async def detailed():
        raise ApiError(400, "invalid_cursor", "Cursor is malformed.", {"cursor": "abc"})

    @app.post("/_items")
    async def create_item(item: Item):
        return item

    @app.get("/_not_modified")
    async def not_modified():
        raise HTTPException(304, headers={"etag": '"v1"'})

    @app.get("/_structured")
    async def structured():
        raise HTTPException(400, detail={"field": "x"})

    @app.get("/_odd_status")
    async def odd_status():
        raise HTTPException(499)

    @app.get("/_api_no_body")
    async def api_no_body():
        raise ApiError(304, "not_modified", "unused")

    @app.get("/_boom")
    async def boom():
        raise RuntimeError("secret internals")

    return app


@pytest.fixture
def probe(probe_app: FastAPI):
    with TestClient(probe_app, raise_server_exceptions=False) as c:
        yield c


def test_health_is_ok_with_workers_disabled(client, settings):
    assert settings.worker_count == 0
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_path_is_not_found_envelope(client):
    response = client.get("/missing")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found", "message": "Not Found", "details": None}}


def test_wrong_method_is_method_not_allowed_envelope(client):
    response = client.post("/health")
    assert response.status_code == 405
    assert response.json() == {
        "error": {"code": "method_not_allowed", "message": "Method Not Allowed", "details": None}
    }
    assert {m.strip() for m in response.headers["allow"].split(",")} == {"GET", "HEAD"}


def test_health_answers_head(client):
    response = client.head("/health")
    assert response.status_code == 200
    assert response.content == b""


def test_openapi_schema_builds_without_warnings(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/health" not in response.json()["paths"]


def test_unknown_status_code_still_has_a_message(probe):
    response = probe.get("/_odd_status")
    assert response.status_code == 499
    assert response.json() == {"error": {"code": "http_error", "message": "HTTP error", "details": None}}


def test_api_error_with_bodyless_status_has_no_body(probe):
    response = probe.get("/_api_no_body")
    assert response.status_code == 304
    assert response.content == b""


def test_bodyless_status_has_no_body(probe):
    response = probe.get("/_not_modified")
    assert response.status_code == 304
    assert response.content == b""
    assert response.headers["etag"] == '"v1"'


def test_structured_http_detail_goes_to_details(probe):
    response = probe.get("/_structured")
    assert response.status_code == 400
    assert response.json() == {"error": {"code": "http_error", "message": "Bad Request", "details": {"field": "x"}}}


def test_api_error_produces_exact_envelope(probe):
    response = probe.get("/_conflict")
    assert response.status_code == 409
    assert response.json() == {
        "error": {
            "code": "ticket_conflict",
            "message": "Ticket t-1001 already exists with different content.",
            "details": None,
        }
    }


def test_api_error_details_are_passed_through(probe):
    response = probe.get("/_detailed")
    assert response.status_code == 400
    assert response.json()["error"] == {
        "code": "invalid_cursor",
        "message": "Cursor is malformed.",
        "details": {"cursor": "abc"},
    }


def test_validation_error_uses_envelope_without_echoing_input(probe):
    response = probe.post("/_items", json={"name": "", "count": "SECRET-VALUE"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["message"] == "Request validation failed."
    assert {tuple(d["loc"]) for d in error["details"]} == {("body", "name"), ("body", "count")}
    for d in error["details"]:
        assert set(d) == {"loc", "msg", "type"}
    assert "SECRET-VALUE" not in response.text


def test_malformed_json_is_a_validation_error(probe):
    response = probe.post("/_items", content=b'{"name": ', headers={"content-type": "application/json"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"][0]["type"] == "json_invalid"


def test_missing_body_is_a_validation_error(probe):
    response = probe.post("/_items")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_unexpected_exception_is_internal_error_without_leaking(probe):
    response = probe.get("/_boom")
    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "Internal server error.", "details": None}
    }
    assert "secret internals" not in response.text


def test_injected_settings_and_clock_are_on_app_state(app, settings, clock):
    assert app.state.settings is settings
    assert app.state.clock is clock


def test_default_clock_is_utc_now():
    app = create_app(Settings(worker_count=0))
    assert app.state.clock is utc_now


def test_create_app_reads_environment_when_no_settings_given(monkeypatch):
    monkeypatch.setenv("WORKER_COUNT", "0")
    monkeypatch.setenv("FAKE_LLM_SEED", "99")
    app = create_app()
    assert app.state.settings.worker_count == 0
    assert app.state.settings.fake_llm_seed == 99


def test_lifespan_logs_effective_settings(settings, clock, caplog):
    caplog.set_level(logging.INFO, logger="app")
    with TestClient(create_app(settings, clock=clock)):
        pass
    assert any("starting with Settings(" in r.getMessage() for r in caplog.records)


def test_lifespan_creates_schema(settings, clock):
    app = create_app(settings, clock=clock)
    assert not Path(settings.db_path).exists()
    with TestClient(app):
        with closing(sqlite3.connect(settings.db_path)) as conn:
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
            journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert {"tickets", "idx_tickets_claim", "idx_tickets_list"} <= names
    assert journal_mode == "wal"


def test_db_and_repository_are_on_app_state(app, settings):
    assert isinstance(app.state.db, Database)
    assert app.state.db.path == Path(settings.db_path)
    assert isinstance(app.state.repository, Repository)


def test_repository_works_after_startup(client):
    repository = client.app.state.repository
    ticket, created = repository.insert_ticket("t-1001", "Refund", "Please refund me.")
    assert created is True
    assert repository.get_ticket("t-1001") == ticket


@pytest.mark.parametrize(
    "db_path",
    [
        ":memory:",
        "./:memory:",
        pytest.param(".\\:memory:", marks=pytest.mark.skipif(os.name != "nt", reason="Windows path")),
    ],
)
def test_in_memory_db_path_fails_at_create_app(db_path):
    with pytest.raises(ValueError, match="DB_PATH"):
        create_app(Settings(worker_count=0, db_path=db_path))


# Model client and workers.

T1001 = TICKET_PAYLOADS[0]


def test_default_llm_is_the_fake_from_settings(settings, clock):
    settings = replace(settings, fake_llm_seed=7, fake_llm_failure_rate=0.5, fake_llm_gullible=False)
    llm = create_app(settings, clock=clock).state.llm
    assert isinstance(llm, FakeLLM)
    assert (llm.seed, llm.failure_rate, llm.gullible) == (7, 0.5, False)


def test_injected_llm_is_on_app_state(settings, clock):
    fake = FakeLLM(script=[Mode.OK])
    assert create_app(settings, clock=clock, llm=fake).state.llm is fake


@pytest.mark.parametrize("llm", [object(), "fake", FakeLLM.complete])
def test_non_client_llm_is_rejected(settings, clock, llm):
    with pytest.raises(TypeError, match="llm must implement LLMClient"):
        create_app(settings, clock=clock, llm=llm)


def test_workers_disabled_leaves_no_tasks(client, caplog):
    workers = client.app.state.workers
    assert isinstance(workers, WorkerPool)
    assert workers.tasks == [] and len(workers) == 0


def test_lifespan_starts_and_stops_workers(worker_settings, clock, caplog):
    caplog.set_level(logging.INFO, logger="app")
    app = create_app(worker_settings, clock=clock, llm=FakeLLM(failure_rate=0.0, gullible=False))
    with TestClient(app):
        tasks = list(app.state.workers.tasks)
        assert len(tasks) == 2
        assert not any(task.done() for task in tasks)
    assert all(task.done() for task in tasks)
    assert app.state.workers.tasks == []
    messages = [r.getMessage() for r in caplog.records]
    assert "starting 2 workers" in messages
    assert messages[-1] == "workers stopped"


def test_posted_ticket_is_classified_by_the_workers(worker_settings, clock):
    app = create_app(worker_settings, clock=clock, llm=FakeLLM(failure_rate=0.0, gullible=False))
    with TestClient(app) as client:
        assert client.post("/tickets", json=T1001).status_code == 202
        body = wait_for_status(client, "t-1001", "classified")
    assert body["category"] is not None and body["priority"] is not None and body["summary"]
    assert (body["attempts"], body["prompt_version"], body["last_error"]) == (1, "v1", None)


def test_retry_under_the_lifespan(worker_settings, clock):
    app = create_app(worker_settings, clock=clock, llm=FakeLLM(script=[Mode.TIMEOUT, Mode.OK]))
    with TestClient(app) as client:
        client.post("/tickets", json=T1001)
        body = wait_for_status(client, "t-1001", "classified")
    assert (body["attempts"], body["last_error"]) == (2, None)


def test_shutdown_with_an_in_flight_call_keeps_the_claim(worker_settings, clock):
    gate = GateLLM()
    app = create_app(worker_settings, clock=clock, llm=gate)
    with TestClient(app) as client:
        client.post("/tickets", json=T1001)
        wait_until(lambda: gate.parked == 1, "the model call")
        started = time.monotonic()
    assert time.monotonic() - started < 2.0
    stored = app.state.repository.get_ticket("t-1001")
    # What the startup claim reset finds: a pending row still holding its claim.
    assert (stored.status, stored.attempts) == ("pending", 1)
    assert stored.claimed_at is not None
    assert gate.parked == 1


def app_messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name.startswith("app")]


def test_restart_recovers_an_in_flight_ticket_without_waiting_for_the_lease(worker_settings, clock, caplog):
    # A 60 s lease: within the deadline only the startup reset can explain the pickup.
    settings = replace(worker_settings, lease_timeout_s=60.0)
    gate = GateLLM()
    with TestClient(create_app(settings, clock=clock, llm=gate)) as client:
        client.post("/tickets", json=T1001)
        wait_until(lambda: gate.parked == 1, "the model call")
    clock.advance(1)  # a restart takes time; the new claim gets a new token
    caplog.clear()
    caplog.set_level(logging.INFO, logger="app")
    fake = FakeLLM(failure_rate=0.0, gullible=False)
    with TestClient(create_app(settings, clock=clock, llm=fake)) as client:
        body = wait_for_status(client, "t-1001", "classified")
    assert (body["attempts"], body["last_error"], body["prompt_version"]) == (2, None, "v1")
    assert len(fake.calls) == 1
    assert gate.parked == 1
    messages = app_messages(caplog)
    released = messages.index("released 1 claim(s) left by the previous run")
    assert released < messages.index("starting 2 workers")


def test_startup_logs_zero_released_claims_on_a_clean_database(settings, clock, caplog):
    caplog.set_level(logging.INFO, logger="app")
    with TestClient(create_app(settings, clock=clock)):
        pass
    messages = app_messages(caplog)
    released = messages.index("released 0 claim(s) left by the previous run")
    assert released == next(i for i, m in enumerate(messages) if m.startswith("schema ready at ")) + 1


def test_startup_reset_runs_before_the_workers_start(worker_settings, clock, caplog):
    settings = replace(worker_settings, lease_timeout_s=60.0)
    db = Database(settings.db_path)
    db.create_schema()
    repository = Repository(db, clock)
    repository.insert_ticket(T1001["id"], T1001["subject"], T1001["body"])
    with db.connection() as conn:
        conn.execute("UPDATE tickets SET attempts = 1, claimed_at = ? WHERE id = 't-1001'", (format_ts(clock()),))
    clock.advance(1)
    caplog.set_level(logging.INFO, logger="app")
    app = create_app(settings, clock=clock, llm=FakeLLM(failure_rate=0.0, gullible=False))
    with TestClient(app) as client:
        body = wait_for_status(client, "t-1001", "classified")
    assert (body["attempts"], body["last_error"]) == (2, None)
    messages = app_messages(caplog)
    assert messages.index("released 1 claim(s) left by the previous run") < messages.index("starting 2 workers")


def test_health_still_ok_with_workers_running(worker_settings, clock):
    app = create_app(worker_settings, clock=clock, llm=FakeLLM(failure_rate=0.0, gullible=False))
    with TestClient(app) as client:
        assert len(app.state.workers) == 2
        response = client.get("/health")
    assert (response.status_code, response.json()) == (200, {"status": "ok"})
