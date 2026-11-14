import logging
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.api import ApiError
from app.clock import utc_now
from app.config import Settings
from app.db import Database
from app.main import create_app
from app.repository import Repository


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


def test_in_memory_db_path_fails_at_create_app():
    with pytest.raises(ValueError, match="DB_PATH"):
        create_app(Settings(worker_count=0, db_path=":memory:"))
