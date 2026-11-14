import base64
from contextlib import closing

import pytest

from app.api import MAX_CURSOR_LENGTH, TicketOut, encode_cursor
from app.clock import format_ts

INVALID_CURSOR = {
    "error": {
        "code": "invalid_cursor",
        "message": "Cursor is not valid; use next_cursor from a previous response.",
        "details": None,
    }
}
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


def label(db, id: str, category: str, priority: str, flagged: bool = False) -> None:
    set_state(
        db,
        id,
        status="classified",
        category=category,
        priority=priority,
        summary="x",
        flagged=int(flagged),
    )


def ids(response) -> list[str]:
    assert response.status_code == 200
    return [item["id"] for item in response.json()["items"]]


def seed(client, clock, n: int, step_s: float = 1) -> list[str]:
    created = []
    for i in range(n):
        created.append(ingest(client, f"t-{i:03d}")["id"])
        clock.advance(step_s)
    return created


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def assert_validation_error(response, *expected: tuple[list[str], str]) -> None:
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["message"] == "Request validation failed."
    for d in error["details"]:
        assert set(d) == {"loc", "msg", "type"}
    assert [(d["loc"], d["type"]) for d in error["details"]] == list(expected)


def assert_invalid_cursor(response) -> None:
    assert response.status_code == 400
    assert response.json() == INVALID_CURSOR


# GET /tickets/{id}


def test_get_returns_pending_ticket_with_public_fields(client):
    created = ingest(client, "t-1001", "Charged twice", "Hi, I see two charges...")
    response = client.get("/tickets/t-1001")
    assert response.status_code == 200
    body = response.json()
    assert body == created
    assert set(body) == set(TicketOut.model_fields)
    assert not {"claimed_at", "next_attempt_at"} & set(body)


def test_get_reflects_current_state(client, db):
    ingest(client, "t-1001")
    label(db, "t-1001", "billing", "high")
    body = client.get("/tickets/t-1001").json()
    assert (body["status"], body["category"], body["priority"], body["summary"]) == (
        "classified",
        "billing",
        "high",
        "x",
    )


def test_unknown_id_is_404(client):
    response = client.get("/tickets/t-missing")
    assert response.status_code == 404
    assert response.json() == NOT_FOUND


# The path is not validated against the ingest id rules: an id that could never be stored is
# just another id that is not stored. (Paths with an encoded or literal "/" never reach the
# route and get the framework's generic 404 "not_found" instead.)
@pytest.mark.parametrize("path_id", ["a.b", "t%201", "%2E%2E", "x" * 5000])
def test_impossible_ids_are_404_not_422(client, path_id):
    response = client.get(f"/tickets/{path_id}")
    assert response.status_code == 404
    assert response.json() == NOT_FOUND


@pytest.mark.parametrize(
    "query, keys",
    [("x=1", ["x"]), ("limit=5", ["limit"]), ("a=1&b=2&a=3", ["a", "b"])],
)
def test_get_rejects_query_parameters(client, query, keys):
    ingest(client, "t-1001")
    response = client.get(f"/tickets/t-1001?{query}")
    assert_validation_error(response, *((["query", key], "extra_forbidden") for key in keys))


# GET /tickets: listing and order


def test_empty_list(client):
    response = client.get("/tickets")
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}


def test_items_use_the_ticket_representation(client):
    created = ingest(client, "t-1001")
    assert client.get("/tickets").json()["items"] == [created]


def test_default_limit_is_20(client, clock):
    created = seed(client, clock, 21)
    body = client.get("/tickets").json()
    assert [item["id"] for item in body["items"]] == created[:20]
    assert body["next_cursor"] is not None


def test_order_is_created_at_then_id(client, clock):
    ingest(client, "b")
    ingest(client, "a")
    clock.advance(1)
    ingest(client, "C")
    clock.advance(1)
    ingest(client, "B")
    # Same timestamp: tiebreak on id in byte order ("B" < "C" < "a" < "b"), then later timestamps.
    assert ids(client.get("/tickets")) == ["a", "b", "C", "B"]


def test_limit_boundaries_are_accepted(client, clock):
    created = seed(client, clock, 3)
    assert ids(client.get("/tickets?limit=1")) == created[:1]
    assert ids(client.get("/tickets?limit=100")) == created


# Filters


@pytest.fixture
def labelled(client, db, clock) -> dict[str, tuple[str, str, bool]]:
    """Seven tickets with labels, one still pending, one failed without labels."""
    labels = {
        "t-1": ("billing", "high", False),
        "t-2": ("technical", "low", True),
        "t-3": ("account", "medium", False),
        "t-4": ("other", "low", False),
        "t-5": ("billing", "low", True),
        "t-6": ("billing", "high", True),
        "t-7": ("technical", "medium", False),
    }
    for id, (category, priority, flagged) in labels.items():
        ingest(client, id)
        label(db, id, category, priority, flagged)
        clock.advance(1)
    ingest(client, "t-pending")
    clock.advance(1)
    ingest(client, "t-failed")
    set_state(db, "t-failed", status="failed", attempts=3, last_error="attempts_exhausted")
    return labels


@pytest.mark.parametrize("category", ["billing", "technical", "account", "other"])
def test_category_filter_alone(client, labelled, category):
    expected = [id for id, (c, _, _) in labelled.items() if c == category]
    assert expected
    assert ids(client.get(f"/tickets?category={category}")) == expected


@pytest.mark.parametrize("priority", ["low", "medium", "high"])
def test_priority_filter_alone(client, labelled, priority):
    expected = [id for id, (_, p, _) in labelled.items() if p == priority]
    assert expected
    assert ids(client.get(f"/tickets?priority={priority}")) == expected


def test_flagged_filter_alone(client, labelled):
    assert ids(client.get("/tickets?flagged=true")) == ["t-2", "t-5", "t-6"]
    # Pending and failed tickets are not flagged, so flagged=false includes them.
    assert ids(client.get("/tickets?flagged=false")) == ["t-1", "t-3", "t-4", "t-7", "t-pending", "t-failed"]


def test_category_filter_excludes_pending_and_unlabelled_failed(client, labelled):
    for category in ("billing", "technical", "account", "other"):
        listed = ids(client.get(f"/tickets?category={category}"))
        assert "t-pending" not in listed
        assert "t-failed" not in listed


def test_category_and_priority_together(client, labelled):
    assert ids(client.get("/tickets?category=billing&priority=high")) == ["t-1", "t-6"]
    assert ids(client.get("/tickets?category=technical&priority=high")) == []


def test_all_three_filters_together(client, labelled):
    assert ids(client.get("/tickets?category=billing&priority=high&flagged=true")) == ["t-6"]
    assert ids(client.get("/tickets?category=billing&priority=high&flagged=false")) == ["t-1"]


# Pagination


def collect_pages(client, url: str) -> list[dict]:
    pages = [client.get(url).json()]
    while pages[-1]["next_cursor"] is not None:
        separator = "&" if "?" in url else "?"
        pages.append(client.get(f"{url}{separator}cursor={pages[-1]['next_cursor']}").json())
        assert len(pages) < 100
    return pages


def test_three_pages_without_repeat_or_skip(client, clock):
    created = seed(client, clock, 7)
    pages = collect_pages(client, "/tickets?limit=3")
    assert [len(page["items"]) for page in pages] == [3, 3, 1]
    assert [page["next_cursor"] is None for page in pages] == [False, False, True]
    seen = [item["id"] for page in pages for item in page["items"]]
    assert seen == created
    assert len(set(seen)) == 7


def test_pages_with_equal_timestamps_use_the_id_tiebreak(client, clock):
    created = sorted(seed(client, clock, 5, step_s=0))
    seen = [item["id"] for page in collect_pages(client, "/tickets?limit=2") for item in page["items"]]
    assert seen == created


def test_exact_multiple_has_no_empty_trailing_page(client, clock):
    seed(client, clock, 6)
    pages = collect_pages(client, "/tickets?limit=3")
    assert [len(page["items"]) for page in pages] == [3, 3]
    assert pages[1]["next_cursor"] is None


def test_insert_between_pages_does_not_disturb_earlier_pages(client, clock):
    created = seed(client, clock, 5)
    first = client.get("/tickets?limit=2").json()
    clock.advance(1)
    ingest(client, "t-new")
    second = client.get(f"/tickets?limit=2&cursor={first['next_cursor']}").json()
    third = client.get(f"/tickets?limit=2&cursor={second['next_cursor']}").json()
    assert client.get("/tickets?limit=2").json() == first
    seen = [item["id"] for page in (first, second, third) for item in page["items"]]
    assert seen == [*created, "t-new"]
    assert third["next_cursor"] is None


def test_cursor_with_filters_continues_the_filtered_sequence(client, labelled):
    pages = collect_pages(client, "/tickets?category=billing&limit=1")
    assert [item["id"] for page in pages for item in page["items"]] == ["t-1", "t-5", "t-6"]


def test_cursor_past_end_is_empty_page(client, clock):
    seed(client, clock, 2)
    last = client.get("/tickets").json()["items"][-1]
    response = client.get(f"/tickets?cursor={encode_cursor(last['created_at'], last['id'])}")
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}


def test_cursor_is_a_position_not_a_reference(client, clock):
    first = ingest(client, "t-a")
    clock.advance(1)
    second = ingest(client, "t-b")
    # "zzz" names no ticket; the cursor still means "after (created_at of t-a, zzz)".
    cursor = encode_cursor(first["created_at"], "zzz")
    assert ids(client.get(f"/tickets?cursor={cursor}")) == [second["id"]]


def test_cursor_is_unpadded_base64url_of_created_at_and_id(client, clock):
    seed(client, clock, 3)
    body = client.get("/tickets?limit=2").json()
    cursor = body["next_cursor"]
    assert "=" not in cursor
    last = body["items"][-1]
    padded = cursor + "=" * (-len(cursor) % 4)
    assert base64.urlsafe_b64decode(padded).decode() == f"{last['created_at']}|{last['id']}"
    # The example cursor from the issue.
    assert (
        encode_cursor("2026-11-14T12:00:00.000000Z", "t-1003")
        == "MjAyNi0xMS0xNFQxMjowMDowMC4wMDAwMDBafHQtMTAwMw"
    )


def test_longest_possible_cursor_fits_the_length_limit(client, clock):
    ingest(client, "x" * 128)
    cursor = encode_cursor(format_ts(clock()), "x" * 128)
    assert len(cursor) == MAX_CURSOR_LENGTH
    assert client.get(f"/tickets?cursor={cursor}").json() == {"items": [], "next_cursor": None}


# Query parameters: FastAPI defaults (unknown keys ignored, last repeated value wins)


def test_unknown_query_parameter_is_ignored(client, clock):
    created = seed(client, clock, 2)
    assert ids(client.get("/tickets?foo=bar")) == created


def test_repeated_query_parameter_takes_the_last_value(client, labelled):
    assert ids(client.get("/tickets?category=technical&category=billing")) == ["t-1", "t-5", "t-6"]
    assert ids(client.get("/tickets?limit=50&limit=2")) == ["t-1", "t-2"]


# Errors


@pytest.mark.parametrize(
    "query, loc, type_",
    [
        ("category=refund", ["query", "category"], "literal_error"),
        ("category=", ["query", "category"], "literal_error"),
        ("category=Billing", ["query", "category"], "literal_error"),
        ("priority=urgent", ["query", "priority"], "literal_error"),
        ("flagged=maybe", ["query", "flagged"], "literal_error"),
        ("flagged=1", ["query", "flagged"], "literal_error"),
        ("flagged=True", ["query", "flagged"], "literal_error"),
        ("flagged=yes", ["query", "flagged"], "literal_error"),
        ("flagged=", ["query", "flagged"], "literal_error"),
        ("limit=0", ["query", "limit"], "greater_than_equal"),
        ("limit=101", ["query", "limit"], "less_than_equal"),
        ("limit=abc", ["query", "limit"], "int_parsing"),
        ("limit=1.5", ["query", "limit"], "int_parsing"),
    ],
)
def test_invalid_filter_value_is_422(client, query, loc, type_):
    assert_validation_error(client.get(f"/tickets?{query}"), (loc, type_))


VALID = encode_cursor("2026-11-14T12:00:00.000000Z", "t-1003")


@pytest.mark.parametrize(
    "cursor",
    [
        pytest.param("", id="empty"),
        pytest.param("abc", id="short-garbage"),
        pytest.param("a", id="length-1-mod-4"),
        pytest.param("!!!!", id="outside-alphabet"),
        # "+" and "/" make the stdlib decoder warn; under warnings-as-errors that would be a 500.
        pytest.param("YWJj+/", id="standard-alphabet"),
        pytest.param("YWJj-_", id="garbage-bytes"),
        pytest.param(VALID + "==", id="padded"),
        pytest.param(VALID + " ", id="trailing-space"),
        pytest.param("é", id="non-ascii"),
        pytest.param(b64("nonsense"), id="no-separator"),
        pytest.param(b64("2026-11-14T12:00:00Z|t-1"), id="non-canonical-timestamp"),
        pytest.param(b64("2026-11-14T12:00:00.000000Z|"), id="empty-id"),
        pytest.param(b64("2026-11-14T12:00:00.000000Z|a b"), id="bad-id"),
        pytest.param(b64("2026-11-14T12:00:00.000000Z|a|b"), id="extra-separator"),
        pytest.param(b64("2026-11-14T12:00:00.000000Z|" + "a" * 129), id="id-too-long"),
        pytest.param(VALID[:-1] + "x", id="non-canonical-trailing-bits"),
        pytest.param("A" * (MAX_CURSOR_LENGTH + 1), id="too-long"),
    ],
)
def test_bad_cursor_is_400_invalid_cursor(client, cursor):
    assert_invalid_cursor(client.get("/tickets", params={"cursor": cursor}))


def test_errors_never_echo_input(client):
    for query in ("category=SECRET", "flagged=SECRET", "limit=SECRET", f"cursor={b64('SECRET')}"):
        response = client.get(f"/tickets?{query}")
        assert response.status_code in (400, 422)
        assert "SECRET" not in response.text
        assert b64("SECRET") not in response.text


def test_openapi_documents_read_responses(client):
    openapi = client.get("/openapi.json").json()

    def refs(operation: dict) -> dict[str, str]:
        return {code: r["content"]["application/json"]["schema"]["$ref"] for code, r in operation["responses"].items()}

    by_id = refs(openapi["paths"]["/tickets/{id}"]["get"])
    listing = refs(openapi["paths"]["/tickets"]["get"])
    assert by_id == {
        "200": "#/components/schemas/TicketOut",
        "404": "#/components/schemas/ErrorEnvelope",
        "422": "#/components/schemas/ErrorEnvelope",
    }
    assert listing == {
        "200": "#/components/schemas/TicketPage",
        "400": "#/components/schemas/ErrorEnvelope",
        "422": "#/components/schemas/ErrorEnvelope",
    }
    assert "HTTPValidationError" not in openapi["components"]["schemas"]
