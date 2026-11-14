from contextlib import closing
from dataclasses import fields

import pytest

from app.api import TicketOut
from app.clock import format_ts
from app.models import Ticket

HIDDEN = {"claimed_at", "next_attempt_at"}
CONFLICT = {
    "error": {
        "code": "ticket_conflict",
        "message": "Ticket t-1001 already exists with different content.",
        "details": None,
    }
}
CLASSIFIED = {
    "status": "classified",
    "category": "billing",
    "priority": "high",
    "summary": "Customer was charged twice.",
    "attempts": 1,
    "prompt_version": "v1",
}
FAILED = {"status": "failed", "attempts": 3, "last_error": "attempts_exhausted"}


def ingest(client, **overrides):
    payload = {"id": "t-1001", "subject": "Charged twice this month", "body": "Hi, I see two charges..."}
    payload.update(overrides)
    return client.post("/tickets", json=payload)


def row_count(db) -> int:
    with db.connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]


def set_state(db, id: str, **values) -> None:
    assignments = ", ".join(f"{name} = ?" for name in values)
    with db.connection() as conn:
        with closing(conn.execute(f"UPDATE tickets SET {assignments} WHERE id = ?", (*values.values(), id))):
            pass


def assert_validation_error(response) -> list[dict]:
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["message"] == "Request validation failed."
    for d in error["details"]:
        assert set(d) == {"loc", "msg", "type"}
    return error["details"]


# New ticket


def test_new_ticket_is_accepted_with_location(client, clock):
    response = ingest(client)
    assert response.status_code == 202
    assert response.headers["location"] == "/tickets/t-1001"
    now = format_ts(clock())
    assert response.json() == {
        "id": "t-1001",
        "subject": "Charged twice this month",
        "body": "Hi, I see two charges...",
        "status": "pending",
        "category": None,
        "priority": None,
        "summary": None,
        "flagged": False,
        "flag_reason": None,
        "attempts": 0,
        "last_error": None,
        "prompt_version": None,
        "created_at": now,
        "updated_at": now,
    }
    assert not HIDDEN & set(response.json())


def test_response_has_exactly_the_public_fields(client):
    assert set(ingest(client).json()) == set(TicketOut.model_fields)


def test_ticket_out_hides_only_internal_columns():
    assert set(TicketOut.model_fields) == {f.name for f in fields(Ticket)} - HIDDEN


def test_row_is_stored(client, repository):
    body = ingest(client).json()
    stored = repository.get_ticket("t-1001")
    assert stored is not None
    assert (stored.subject, stored.body, stored.status) == (body["subject"], body["body"], body["status"])


# Same payload twice


def test_identical_replay_returns_200_and_existing_ticket(client, clock, db):
    first = ingest(client)
    assert first.status_code == 202
    clock.advance(60)
    second = ingest(client)
    assert second.status_code == 200
    assert second.json() == first.json()
    assert "location" not in second.headers
    assert row_count(db) == 1


def test_omitted_subject_and_empty_subject_are_the_same_payload(client):
    assert ingest(client, subject="").status_code == 202
    response = client.post("/tickets", json={"id": "t-1001", "body": "Hi, I see two charges..."})
    assert response.status_code == 200


def test_replay_of_classified_ticket_returns_200_with_current_state(client, db, repository):
    ingest(client)
    set_state(db, "t-1001", **CLASSIFIED)
    before = repository.get_ticket("t-1001")
    response = ingest(client)
    assert response.status_code == 200
    body = response.json()
    assert (body["status"], body["attempts"], body["category"]) == ("classified", 1, "billing")
    assert repository.get_ticket("t-1001") == before


def test_replay_of_failed_ticket_returns_200_and_does_not_requeue(client, db, repository):
    ingest(client)
    set_state(db, "t-1001", **FAILED)
    before = repository.get_ticket("t-1001")
    response = ingest(client)
    assert response.status_code == 200
    assert (response.json()["status"], response.json()["attempts"]) == ("failed", 3)
    assert repository.get_ticket("t-1001") == before


# Different payload for an existing id


def test_different_body_is_conflict(client, db, repository):
    ingest(client)
    before = repository.get_ticket("t-1001")
    response = ingest(client, body="Something else entirely.")
    assert response.status_code == 409
    assert response.json() == CONFLICT
    assert repository.get_ticket("t-1001") == before
    assert row_count(db) == 1


def test_different_subject_is_conflict(client):
    ingest(client)
    response = ingest(client, subject="Another subject")
    assert response.status_code == 409
    assert response.json() == CONFLICT


@pytest.mark.parametrize("state", [CLASSIFIED, FAILED], ids=["classified", "failed"])
def test_conflict_leaves_processed_row_unchanged(client, db, repository, clock, state):
    ingest(client)
    set_state(db, "t-1001", **state)
    before = repository.get_ticket("t-1001")
    clock.advance(60)
    response = ingest(client, body="Changed body.")
    assert response.status_code == 409
    assert repository.get_ticket("t-1001") == before


@pytest.mark.parametrize(("original", "changed"), [("Hello", "Hello "), ("Hello\n", "Hello\r\n")])
def test_whitespace_difference_is_a_conflict(client, original, changed):
    assert ingest(client, body=original).status_code == 202
    assert ingest(client, body=changed).status_code == 409


def test_conflict_response_never_echoes_content(client):
    ingest(client)
    response = ingest(client, subject="SECRET-SUBJECT", body="SECRET-BODY")
    assert response.status_code == 409
    assert "SECRET-SUBJECT" not in response.text
    assert "SECRET-BODY" not in response.text


# Empty subject and sample-shaped tickets


def test_empty_subject_is_accepted(client, repository):
    response = ingest(client, subject="")
    assert response.status_code == 202
    assert response.json()["subject"] == ""
    assert repository.get_ticket("t-1001").subject == ""


def test_omitted_subject_defaults_to_empty_string(client, repository):
    response = client.post("/tickets", json={"id": "t-1001", "body": "No subject here."})
    assert response.status_code == 202
    assert response.json()["subject"] == ""
    assert repository.get_ticket("t-1001").subject == ""


@pytest.mark.parametrize(
    ("subject", "body"),
    [
        ("", "I cannot log in and nothing works."),
        ("Two things", "My invoice is wrong. Also, the export button crashes the app."),
        ("Re: refund", "Still waiting.\r\n> We have issued a refund.\r\n> Staff\r\nNothing arrived."),
    ],
    ids=["empty-subject", "two-topics", "crlf-quoted"],
)
def test_sample_shaped_tickets_are_accepted(client, repository, subject, body):
    response = ingest(client, subject=subject, body=body)
    assert response.status_code == 202
    assert response.json()["body"] == body
    assert repository.get_ticket("t-1001").body == body


# Exact text preservation


def test_text_is_stored_and_returned_exactly(client, repository):
    subject = "  lead\r\n"
    body = "  Hello,\r\n> quoted line\r\n\tCafé – naïve 日本語 🙂  \n"
    response = ingest(client, subject=subject, body=body)
    assert (response.json()["subject"], response.json()["body"]) == (subject, body)
    stored = repository.get_ticket("t-1001")
    assert (stored.subject, stored.body) == (subject, body)


def test_control_characters_are_preserved(client, repository):
    body = "a\u0000b\tc"
    response = ingest(client, body=body)
    assert response.status_code == 202
    assert response.json()["body"] == body
    assert repository.get_ticket("t-1001").body == body


# Invalid payloads

MISSING = object()


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("id", MISSING, "missing"),
        ("id", "", "string_too_short"),
        ("id", "a" * 129, "string_too_long"),
        ("id", "a/b", "string_pattern_mismatch"),
        ("id", " a", "string_pattern_mismatch"),
        ("id", "t-1\n", "string_pattern_mismatch"),
        ("id", "é", "string_pattern_mismatch"),
        ("id", ".", "string_pattern_mismatch"),
        ("id", "..", "string_pattern_mismatch"),
        ("id", "a.b", "string_pattern_mismatch"),
        ("id", "t-1001.", "string_pattern_mismatch"),
        ("id", 123, "string_type"),
        ("id", None, "string_type"),
        ("subject", "s" * 501, "string_too_long"),
        ("subject", None, "string_type"),
        ("subject", 7, "string_type"),
        ("body", MISSING, "missing"),
        ("body", "", "string_too_short"),
        ("body", "x" * 20_001, "string_too_long"),
        ("body", 5, "string_type"),
        ("status", "classified", "extra_forbidden"),
    ],
)
def test_invalid_field_is_rejected(client, db, field, value, error_type):
    payload = {"id": "t-1001", "subject": "Charged twice", "body": "Hi"}
    if value is MISSING:
        del payload[field]
    else:
        payload[field] = value
    details = assert_validation_error(client.post("/tickets", json=payload))
    assert [(d["loc"], d["type"]) for d in details] == [(["body", field], error_type)]
    assert row_count(db) == 0


def test_boundary_lengths_are_accepted(client):
    response = ingest(client, id="A_z-9" + "a" * 123, subject="s" * 500, body="x" * 20_000)
    assert response.status_code == 202


def test_minimum_lengths_are_accepted(client):
    assert ingest(client, id="a", body="x").status_code == 202


def test_length_limits_count_code_points(client):
    assert ingest(client, body="🙂" * 20_000).status_code == 202
    # "e\u0301" is one grapheme but two code points: 250 copies are 500 code points, 251 are 502.
    assert ingest(client, id="t-1002", subject="e\u0301" * 250).status_code == 202
    details = assert_validation_error(ingest(client, id="t-1003", subject="e\u0301" * 251))
    assert [d["type"] for d in details] == ["string_too_long"]


def test_multiple_errors_are_all_reported(client):
    response = client.post("/tickets", json={"id": "", "subject": "s" * 501, "body": "", "x": 1})
    details = assert_validation_error(response)
    assert {tuple(d["loc"]) for d in details} == {("body", "id"), ("body", "subject"), ("body", "body"), ("body", "x")}


def test_malformed_json_is_validation_error(client):
    response = client.post("/tickets", content=b'{"id": "t-1001", ', headers={"content-type": "application/json"})
    assert assert_validation_error(response)[0]["type"] == "json_invalid"


def test_non_object_json_is_validation_error(client):
    details = assert_validation_error(client.post("/tickets", json=["t-1001", "Hi"]))
    assert [(d["loc"], d["type"]) for d in details] == [(["body"], "model_attributes_type")]


def test_missing_request_body_is_validation_error(client):
    assert_validation_error(client.post("/tickets"))


def test_validation_error_never_echoes_body(client):
    payload = {"id": "t-1001", "body": "", "subject": "SECRET-SUBJECT" * 40, "extra": "SECRET-EXTRA"}
    response = client.post("/tickets", json=payload)
    assert_validation_error(response)
    assert "SECRET-SUBJECT" not in response.text
    assert "SECRET-EXTRA" not in response.text


def test_invalid_payload_for_existing_id_does_not_touch_row(client, repository):
    ingest(client)
    before = repository.get_ticket("t-1001")
    assert_validation_error(client.post("/tickets", json={"id": "t-1001", "body": ""}))
    assert repository.get_ticket("t-1001") == before


def test_openapi_documents_ingest_responses(client):
    openapi = client.get("/openapi.json").json()
    operation = openapi["paths"]["/tickets"]["post"]
    refs = {code: r["content"]["application/json"]["schema"]["$ref"] for code, r in operation["responses"].items()}
    assert refs["200"] == refs["202"] == "#/components/schemas/TicketOut"
    assert refs["409"] == refs["422"] == "#/components/schemas/ErrorEnvelope"
    assert "HTTPValidationError" not in openapi["components"]["schemas"]
