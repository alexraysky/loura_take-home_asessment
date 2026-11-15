"""POST /tickets/reclassify: the API contract (workers off) and the requeue under real workers."""

import logging
from contextlib import closing
from dataclasses import replace
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app.clock import format_ts
from app.db import Database
from app.llm import FakeLLM, Mode
from app.main import create_app
from app.models import SELECTORS
from app.prompt import PROMPT_VERSION
from app.repository import Repository
from app.validation import FLAG_REASON_ECHO
from tests.sample_data import EXPECTED, SAMPLE_TICKETS, TICKET_PAYLOADS
from tests.workers import GateLLM, wait_for_status, wait_until

NOT_FOUND = {"error": {"code": "ticket_not_found", "message": "Ticket not found.", "details": None}}


def ingest(client, id: str, subject: str = "s", body: str = "b") -> dict:
    response = client.post("/tickets", json={"id": id, "subject": subject, "body": body})
    assert response.status_code == 202
    return response.json()


def set_state(db, id: str, **values) -> None:
    assignments = ", ".join(f"{name} = ?" for name in values)
    with db.connection() as conn:
        with closing(conn.execute(f"UPDATE tickets SET {assignments} WHERE id = ?", (*values.values(), id))):
            pass


def label(db, id: str, prompt_version: str | None, **values) -> None:
    values = {"category": "billing", "priority": "low", "summary": "x", "attempts": 1} | values
    set_state(db, id, status="classified", prompt_version=prompt_version, **values)


def fail(db, id: str, **values) -> None:
    set_state(db, id, **({"status": "failed", "attempts": 3, "last_error": "llm_timeout"} | values))


def reclassify(client, *selectors: str) -> int:
    response = client.post("/tickets/reclassify", json={"select": list(selectors)})
    assert response.status_code == 202, response.text
    body = response.json()
    assert set(body) == {"requeued"}
    return body["requeued"]


def get(client, id: str) -> dict:
    response = client.get(f"/tickets/{id}")
    assert response.status_code == 200
    return response.json()


def assert_reset(ticket: dict, created: dict, now: str) -> None:
    """A requeued ticket reads like a freshly ingested one, except for updated_at."""
    assert ticket == created | {"updated_at": now}
    assert ticket["updated_at"] != created["updated_at"]


def assert_validation_error(response, *expected: tuple[list, str]) -> None:
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["message"] == "Request validation failed."
    for d in error["details"]:
        assert set(d) == {"loc", "msg", "type"}
    if expected:
        assert [(d["loc"], d["type"]) for d in error["details"]] == list(expected)


# Selection (D1-D4), workers off.


def test_failed_selector_alone_requeues_failed_rows(client, db, clock):
    created = ingest(client, "t-failed")
    ingest(client, "t-outdated")
    fail(db, "t-failed", flagged=1, flag_reason=FLAG_REASON_ECHO)
    label(db, "t-outdated", "v0")
    outdated = get(client, "t-outdated")
    clock.advance(5)
    assert reclassify(client, "failed") == 1
    assert_reset(get(client, "t-failed"), created, format_ts(clock()))
    assert get(client, "t-outdated") == outdated


def test_outdated_prompt_selector_alone_requeues_stale_classified_rows(client, db, clock):
    created = ingest(client, "t-outdated")
    ingest(client, "t-failed")
    label(db, "t-outdated", "v0", attempts=2, flagged=1, flag_reason="Model reason.")
    fail(db, "t-failed")
    failed = get(client, "t-failed")
    clock.advance(5)
    assert reclassify(client, "outdated_prompt") == 1
    assert_reset(get(client, "t-outdated"), created, format_ts(clock()))
    assert get(client, "t-failed") == failed


@pytest.mark.parametrize("selectors", [SELECTORS, SELECTORS[::-1]], ids=["in-order", "reversed"])
def test_both_selectors_together(client, db, clock, selectors):
    created = {id: ingest(client, id) for id in ("t-failed", "t-outdated")}
    fail(db, "t-failed")
    label(db, "t-outdated", "v0")
    clock.advance(5)
    assert reclassify(client, *selectors) == 2
    for id, ticket in created.items():
        assert_reset(get(client, id), ticket, format_ts(clock()))


@pytest.mark.parametrize("selectors", [["outdated_prompt"], ["failed"], list(SELECTORS)])
def test_current_version_classified_tickets_are_left_untouched(client, db, clock, selectors):
    ingest(client, "t-current")
    ingest(client, "t-outdated")
    label(db, "t-current", PROMPT_VERSION)
    label(db, "t-outdated", "v0")
    current = get(client, "t-current")
    clock.advance(5)
    assert reclassify(client, *selectors) == (1 if "outdated_prompt" in selectors else 0)
    assert get(client, "t-current") == current


def test_pending_tickets_are_left_untouched(client, db, repository, clock):
    ingest(client, "t-claimed")
    assert repository.claim_next(lease_timeout_s=60).id == "t-claimed"  # in flight
    ingest(client, "t-new")
    ingest(client, "t-backoff")
    later = format_ts(clock() + timedelta(seconds=30))
    set_state(db, "t-backoff", attempts=1, last_error="llm_timeout", next_attempt_at=later)
    before = {id: repository.get_ticket(id) for id in ("t-claimed", "t-new", "t-backoff")}
    clock.advance(5)
    assert reclassify(client, *SELECTORS) == 0
    assert {id: repository.get_ticket(id) for id in before} == before


def test_null_prompt_version_classified_row_counts_as_outdated(client, db, clock):
    created = ingest(client, "t-unversioned")
    label(db, "t-unversioned", None)
    clock.advance(5)
    assert reclassify(client, "outdated_prompt") == 1
    assert_reset(get(client, "t-unversioned"), created, format_ts(clock()))


def test_nothing_to_requeue_is_202_with_zero(client):
    response = client.post("/tickets/reclassify", json={"select": list(SELECTORS)})
    assert (response.status_code, response.json()) == (202, {"requeued": 0})


def test_second_call_returns_zero(client, db):
    ingest(client, "t-failed")
    fail(db, "t-failed")
    assert reclassify(client, *SELECTORS) == 1
    assert reclassify(client, *SELECTORS) == 0


def test_response_is_exactly_the_count(client, db):
    for id in ("t-1", "t-2", "t-3"):
        ingest(client, id)
        fail(db, id)
    response = client.post("/tickets/reclassify", json={"select": ["failed"]})
    assert response.status_code == 202
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"requeued": 3}
    assert "location" not in response.headers


# Validation: the shared envelope, nothing echoed, nothing written.


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param({}, [(["body", "select"], "missing")], id="missing"),
        pytest.param({"select": []}, [(["body", "select"], "too_short")], id="empty"),
        pytest.param({"select": "failed"}, [(["body", "select"], "list_type")], id="string"),
        pytest.param({"select": None}, [(["body", "select"], "list_type")], id="null"),
        pytest.param({"select": {"failed": True}}, [(["body", "select"], "list_type")], id="object"),
        pytest.param({"select": ["bogus"]}, [(["body", "select", 0], "literal_error")], id="unknown"),
        pytest.param({"select": ["FAILED"]}, [(["body", "select", 0], "literal_error")], id="wrong-case"),
        pytest.param({"select": [1]}, [(["body", "select", 0], "literal_error")], id="number"),
        pytest.param({"select": [None]}, [(["body", "select", 0], "literal_error")], id="null-item"),
        pytest.param({"select": ["failed", "failed"]}, [(["body", "select"], "value_error")], id="duplicate"),
        pytest.param(
            {"select": ["failed", "outdated_prompt", "failed"]}, [(["body", "select"], "too_long")], id="too-long"
        ),
        pytest.param({"select": ["failed"], "x": 1}, [(["body", "x"], "extra_forbidden")], id="extra-field"),
        pytest.param(["failed"], [(["body"], "model_attributes_type")], id="non-object"),
        pytest.param("failed", [(["body"], "model_attributes_type")], id="json-string"),
    ],
)
def test_invalid_body_is_validation_error(client, db, body, expected):
    ingest(client, "t-failed")
    fail(db, "t-failed")
    assert_validation_error(client.post("/tickets/reclassify", json=body), *expected)
    assert get(client, "t-failed")["status"] == "failed"


def test_duplicate_selectors_are_rejected_with_a_message(client):
    response = client.post("/tickets/reclassify", json={"select": ["outdated_prompt", "outdated_prompt"]})
    assert_validation_error(response, (["body", "select"], "value_error"))
    assert "selectors must not repeat" in response.json()["error"]["details"][0]["msg"]


def test_missing_body_is_validation_error(client):
    assert_validation_error(client.post("/tickets/reclassify"), (["body"], "missing"))


def test_malformed_json_is_validation_error(client):
    response = client.post(
        "/tickets/reclassify", content=b'{"select": [', headers={"content-type": "application/json"}
    )
    assert_validation_error(response)
    assert response.json()["error"]["details"][0]["type"] == "json_invalid"


@pytest.mark.parametrize("headers", [{}, {"content-type": "text/plain"}], ids=["no-content-type", "text-plain"])
def test_body_not_sent_as_json_is_validation_error(client, db, headers):
    # Without a JSON content type the body is not parsed: the raw bytes are not an object.
    ingest(client, "t-failed")
    fail(db, "t-failed")
    response = client.post("/tickets/reclassify", content=b'{"select": ["failed"]}', headers=headers)
    assert response.request.headers.get("content-type") == headers.get("content-type")
    assert_validation_error(response, (["body"], "model_attributes_type"))
    assert get(client, "t-failed")["status"] == "failed"


def test_undecodable_body_is_http_error(client, db):
    ingest(client, "t-failed")
    fail(db, "t-failed")
    response = client.post(
        "/tickets/reclassify", content=b'{"select": ["\xff"]}', headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "http_error"
    assert get(client, "t-failed")["status"] == "failed"


@pytest.mark.parametrize("query", ["x=1", "select=failed", "x=1&x=2"])
def test_query_parameters_are_rejected(client, db, query):
    ingest(client, "t-failed")
    fail(db, "t-failed")
    response = client.post(f"/tickets/reclassify?{query}", json={"select": ["failed"]})
    key = query.split("=")[0]
    assert_validation_error(response, (["query", key], "extra_forbidden"))
    assert get(client, "t-failed")["status"] == "failed"


def test_validation_error_never_echoes_input(client):
    # Values are never echoed; a key name is part of the loc, as for every route.
    response = client.post("/tickets/reclassify", json={"select": ["SECRET-SELECTOR", "failed"], "x": "SECRET-VALUE"})
    assert_validation_error(response, (["body", "select", 0], "literal_error"), (["body", "x"], "extra_forbidden"))
    assert "SECRET" not in response.text


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH"])
def test_other_methods_are_method_not_allowed(client, method):
    response = client.request(method, "/tickets/reclassify", json={"select": ["failed"]})
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_get_reclassify_is_a_ticket_lookup(client):
    # The action is POST only; GET on the same path is GET /tickets/{id} with id "reclassify".
    response = client.get("/tickets/reclassify")
    assert (response.status_code, response.json()) == (404, NOT_FOUND)


def test_action_wins_over_a_ticket_named_reclassify(client, db, clock):
    created = ingest(client, "reclassify")
    fail(db, "reclassify")
    clock.advance(5)
    assert reclassify(client, "failed") == 1
    assert_reset(get(client, "reclassify"), created, format_ts(clock()))


def test_openapi_documents_reclassify(client):
    openapi = client.get("/openapi.json").json()
    operation = openapi["paths"]["/tickets/reclassify"]["post"]
    refs = {code: r["content"]["application/json"]["schema"]["$ref"] for code, r in operation["responses"].items()}
    assert refs == {
        "202": "#/components/schemas/ReclassifyResponse",
        "422": "#/components/schemas/ErrorEnvelope",
    }
    body = operation["requestBody"]["content"]["application/json"]["schema"]
    assert body == {"$ref": "#/components/schemas/ReclassifyRequest"}
    schemas = openapi["components"]["schemas"]
    select = schemas["ReclassifyRequest"]["properties"]["select"]
    assert (select["type"], select["minItems"], select["maxItems"]) == ("array", 1, 2)
    assert select["items"]["enum"] == list(SELECTORS)
    assert schemas["ReclassifyRequest"]["required"] == ["select"]
    assert schemas["ReclassifyRequest"]["additionalProperties"] is False
    assert schemas["ReclassifyResponse"]["properties"]["requeued"]["minimum"] == 0
    assert "HTTPValidationError" not in schemas


@pytest.mark.parametrize(
    ("selectors", "count", "message"),
    [
        (["failed"], 1, "reclassify: requeued 1 (select=failed)"),
        (["outdated_prompt", "failed"], 2, "reclassify: requeued 2 (select=outdated_prompt,failed)"),
        (["outdated_prompt"], 1, "reclassify: requeued 1 (select=outdated_prompt)"),
    ],
)
def test_reclassify_logs_the_count(client, db, caplog, selectors, count, message):
    for id in ("t-failed", "t-outdated"):
        ingest(client, id, subject="SECRET-SUBJECT", body="SECRET-BODY")
    fail(db, "t-failed")
    label(db, "t-outdated", "v0")
    caplog.set_level(logging.INFO, logger="app")
    assert reclassify(client, *selectors) == count
    records = [r for r in caplog.records if r.name == "app.api"]
    assert [(r.levelno, r.getMessage()) for r in records] == [(logging.INFO, message)]
    assert "SECRET" not in caplog.text


def test_invalid_request_logs_nothing(client, caplog):
    caplog.set_level(logging.INFO, logger="app")
    assert_validation_error(client.post("/tickets/reclassify", json={"select": []}))
    assert [r for r in caplog.records if r.name == "app.api"] == []


# Under real workers (D5).

T1001 = TICKET_PAYLOADS[0]


def plant(settings, clock) -> tuple[Database, Repository]:
    """The database a lifespan will open, with the schema in place, for rows planted before workers start."""
    db = Database(settings.db_path)
    db.create_schema()
    return db, Repository(db, clock)


def test_requeued_outdated_ticket_is_classified_again_by_the_worker(worker_settings, clock):
    fake = FakeLLM(failure_rate=0.0, gullible=False)
    app = create_app(worker_settings, clock=clock, llm=fake)
    with TestClient(app) as client:
        assert client.post("/tickets", json=T1001).status_code == 202
        first = wait_for_status(client, "t-1001", "classified")
        assert (first["attempts"], first["prompt_version"]) == (1, PROMPT_VERSION)
        set_state(app.state.db, "t-1001", prompt_version="v0")
        clock.advance(1)  # the requeue comes after the first finish
        assert reclassify(client, "outdated_prompt") == 1
        body = wait_for_status(client, "t-1001", "classified")
        assert reclassify(client, "outdated_prompt") == 0
    assert (body["attempts"], body["prompt_version"], body["last_error"]) == (1, PROMPT_VERSION, None)
    assert (body["category"], body["priority"], body["summary"]) == (
        first["category"],
        first["priority"],
        first["summary"],
    )
    assert len(fake.calls) == 2


def test_requeued_failed_ticket_is_classified_again_by_the_worker(worker_settings, clock):
    fake = FakeLLM(script=[Mode.TIMEOUT, Mode.TIMEOUT, Mode.TIMEOUT, Mode.OK])
    app = create_app(worker_settings, clock=clock, llm=fake)
    with TestClient(app) as client:
        client.post("/tickets", json=T1001)
        failed = wait_for_status(client, "t-1001", "failed")
        assert (failed["attempts"], failed["last_error"]) == (worker_settings.max_attempts, "llm_timeout")
        clock.advance(1)
        assert reclassify(client, "failed") == 1
        body = wait_for_status(client, "t-1001", "classified")
        assert reclassify(client, *SELECTORS) == 0
    assert (body["attempts"], body["last_error"], body["prompt_version"]) == (1, None, PROMPT_VERSION)
    assert body["category"] is not None and body["priority"] is not None and body["summary"]
    assert len(fake.calls) == worker_settings.max_attempts + 1


def test_in_flight_ticket_is_not_reset_under_the_worker(worker_settings, clock):
    # Timeouts far beyond the test's run: the parked call never times out or loses its lease.
    settings = replace(worker_settings, llm_timeout_s=60.0, lease_timeout_s=120.0)
    gate = GateLLM()
    app = create_app(settings, clock=clock, llm=gate)
    with TestClient(app) as client:
        client.post("/tickets", json=T1001)
        wait_until(lambda: gate.parked == 1, "the model call")
        repository = app.state.repository
        in_flight = repository.get_ticket("t-1001")
        assert (in_flight.status, in_flight.attempts) == ("pending", 1)
        assert in_flight.claimed_at is not None
        # One statement, so the idle worker can never claim the row before it is failed.
        now = format_ts(clock())
        with app.state.db.connection() as conn:
            conn.execute(
                "INSERT INTO tickets (id, subject, body, status, attempts, last_error, created_at, updated_at) "
                "VALUES ('t-failed', 's', 'b', 'failed', 3, 'llm_timeout', ?, ?)",
                (now, now),
            )
        planted = repository.get_ticket("t-failed")
        assert (planted.status, planted.attempts, planted.last_error) == ("failed", 3, "llm_timeout")
        clock.advance(1)
        assert reclassify(client, *SELECTORS) == 1
        assert repository.get_ticket("t-1001") == in_flight
        # The requeued row goes to the idle worker, fresh: attempts 1, no error, a claim of its own.
        # Its claim timestamp is not compared with the clock: the polling worker may have read the
        # clock before it moved.
        wait_until(lambda: gate.parked == 2, "the requeued ticket's model call")
        requeued = repository.get_ticket("t-failed")
        assert (requeued.status, requeued.attempts, requeued.last_error) == ("pending", 1, None)
        assert requeued.claimed_at is not None
        after = repository.get_ticket("t-1001")
        assert (after.status, after.attempts, after.claimed_at) == ("pending", 1, in_flight.claimed_at)
        assert after == in_flight
    assert gate.parked == 2


def test_requeue_converges_with_the_pool(worker_settings, clock):
    db, repository = plant(worker_settings, clock)
    for entry in TICKET_PAYLOADS:
        repository.insert_ticket(entry["id"], entry.get("subject", ""), entry["body"])
        label(db, entry["id"], "v0", category="other", priority="high", attempts=2)
    for id in ("f-1", "f-2"):
        repository.insert_ticket(id, "Refund", "Please refund the second charge on my card.")
        fail(db, id)
    clock.advance(1)
    ids = [entry["id"] for entry in TICKET_PAYLOADS] + ["f-1", "f-2"]
    fake = FakeLLM(failure_rate=0.0, gullible=False)
    with TestClient(create_app(worker_settings, clock=clock, llm=fake)) as client:
        assert reclassify(client, *SELECTORS) == len(ids)

        def all_classified():
            tickets = [get(client, id) for id in ids]
            return tickets if all(t["status"] == "classified" for t in tickets) else None

        tickets = wait_until(all_classified, "every requeued ticket to be classified")
        assert reclassify(client, *SELECTORS) == 0
    for ticket in tickets:
        assert (ticket["attempts"], ticket["prompt_version"], ticket["last_error"]) == (1, PROMPT_VERSION, None)
        if ticket["id"] in EXPECTED:
            assert (ticket["category"], ticket["priority"], ticket["flagged"]) == EXPECTED[ticket["id"]]
    assert len(fake.calls) == len(ids)


def test_flag_is_cleared_and_recomputed_after_requeue(worker_settings, clock):
    # Both rows end failed and flagged with the quote check's reason. After the requeue, the clean
    # ticket is classified unflagged, and the quoted-reply attack t-1005 is caught and flagged again.
    db, repository = plant(worker_settings, clock)
    for id in ("t-1001", "t-1005"):
        repository.insert_ticket(id, *SAMPLE_TICKETS[id])
        fail(db, id, last_error="summary_echoes_ticket", flagged=1, flag_reason=FLAG_REASON_ECHO)
    clock.advance(1)
    # Gullible: t-1005's first call obeys the injection, which the quote check rejects and flags.
    fake = FakeLLM(failure_rate=0.0, gullible=True)
    with TestClient(create_app(worker_settings, clock=clock, llm=fake)) as client:
        assert reclassify(client, "failed") == 2
        clean = wait_for_status(client, "t-1001", "classified")
        attack = wait_for_status(client, "t-1005", "classified")
    assert (clean["flagged"], clean["flag_reason"], clean["attempts"]) == (False, None, 1)
    assert (attack["category"], attack["priority"], attack["flagged"]) == EXPECTED["t-1005"]
    assert (attack["flag_reason"], attack["attempts"], attack["last_error"]) == (FLAG_REASON_ECHO, 2, None)
