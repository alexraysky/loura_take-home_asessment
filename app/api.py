"""Routes and request/response schemas.

Holds the shared error contract, the ticket schemas used by every ticket
endpoint, ``POST /tickets``, the read endpoints (``GET /tickets/{id}``,
``GET /tickets`` with its cursor codec) and the operational health probe.
"""

import asyncio
import base64
import re
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field

from app.clock import parse_ts
from app.models import Category, Priority, Status
from app.repository import Repository

router = APIRouter()


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorEnvelope(BaseModel):
    """Body of every error response: ``{"error": {"code", "message", "details"}}``."""

    error: ErrorBody


class ApiError(Exception):
    """Raised by route handlers to return an error in the shared envelope."""

    def __init__(self, status_code: int, code: str, message: str, details: Any | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


# OpenAPI entries for error statuses, so the schema shows the real envelope. Later routes add theirs.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    409: {"model": ErrorEnvelope, "description": "Ticket exists with different content"},
    422: {"model": ErrorEnvelope, "description": "Request validation failed"},
}

# No "." so an id can never be "." or ".." in a path; every allowed character is URL-unreserved.
ID_PATTERN = r"^[A-Za-z0-9_-]+$"


class IngestRequest(BaseModel):
    """Body of ``POST /tickets``. Text is kept exactly as received: no trimming, no normalisation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    subject: str = Field(default="", max_length=500)
    body: str = Field(min_length=1, max_length=20_000)


class TicketOut(BaseModel):
    """Ticket representation used by every endpoint. ``claimed_at`` and ``next_attempt_at`` are internal."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    subject: str
    body: str
    status: Status
    category: Category | None
    priority: Priority | None
    summary: str | None = Field(
        default=None,
        description="Model-generated text describing the request. Untrusted: it grants no approval "
        "and must not trigger an action on its own.",
    )
    flagged: bool
    flag_reason: str | None = Field(
        default=None,
        description="Why the ticket was flagged; model text or the fixed reason set by the quote check.",
    )
    attempts: int
    last_error: str | None
    prompt_version: str | None
    created_at: str
    updated_at: str


class TicketPage(BaseModel):
    """Body of ``GET /tickets``: one page in (created_at, id) order and the cursor for the next one."""

    items: list[TicketOut]
    next_cursor: str | None


def _error_doc(description: str) -> dict[str, Any]:
    return {"model": ErrorEnvelope, "description": description}


# Declaring 422 also keeps FastAPI's own HTTPValidationError schema out of the OpenAPI document.
GET_TICKET_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: _error_doc("No ticket with this id"),
    422: _error_doc("Request validation failed"),
}
LIST_TICKETS_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: _error_doc("Cursor does not decode"),
    422: _error_doc("Request validation failed"),
}

LIMIT_DEFAULT = 20
LIMIT_MAX = 100

# The base64url alphabet with no padding: exactly what encode_cursor emits.
_CURSOR_RE = re.compile(r"[A-Za-z0-9_-]+")
_ID_RE = re.compile(ID_PATTERN)
_MAX_ID_LENGTH = 128
# 27-character timestamp + "|" + 128-character id = 156 bytes -> 208 base64 characters.
MAX_CURSOR_LENGTH = 208


def encode_cursor(created_at: str, id: str) -> str:
    """Opaque position of a ticket: unpadded base64url of ``"{created_at}|{id}"``."""
    raw = f"{created_at}|{id}".encode("ascii")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[str, str]:
    """Inverse of ``encode_cursor``; anything it would not have produced is ``400 invalid_cursor``."""
    # Checking the alphabet first also keeps "+" and "/" away from the decoder, which warns on them.
    if not cursor or len(cursor) > MAX_CURSOR_LENGTH or not _CURSOR_RE.fullmatch(cursor):
        raise _invalid_cursor()
    try:
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        text = raw.decode("ascii")
    except ValueError:  # binascii.Error (bad length) and UnicodeDecodeError are both ValueErrors
        raise _invalid_cursor() from None
    created_at, sep, id = text.partition("|")
    if not sep or len(id) > _MAX_ID_LENGTH or not _ID_RE.fullmatch(id):
        raise _invalid_cursor()
    try:
        parse_ts(created_at)
    except ValueError:
        raise _invalid_cursor() from None
    if encode_cursor(created_at, id) != cursor:  # non-canonical trailing bits, e.g. "YR" for b"a"
        raise _invalid_cursor()
    return created_at, id


def _invalid_cursor() -> ApiError:
    return ApiError(400, "invalid_cursor", "Cursor is not valid; use next_cursor from a previous response.")


def reject_query_params(request: Request) -> None:
    """Dependency for routes that take no query parameters: any key is a 422 validation error."""
    errors = [
        {"loc": ("query", key), "msg": "Unknown query parameter.", "type": "extra_forbidden"}
        for key in dict.fromkeys(key for key, _ in request.query_params.multi_items())
    ]
    if errors:
        raise RequestValidationError(errors)


def get_repository(request: Request) -> Repository:
    return request.app.state.repository


# Kept out of the OpenAPI schema: an ops probe, and GET+HEAD on one route would share an operation id.
@router.api_route("/health", methods=["GET", "HEAD"], include_in_schema=False)
async def health() -> dict[str, str]:
    """Readiness probe; not part of the ticket API."""
    return {"status": "ok"}


@router.post(
    "/tickets",
    status_code=202,
    response_model=TicketOut,
    responses={200: {"model": TicketOut, "description": "Same id and payload already stored"}, **ERROR_RESPONSES},
)
async def ingest_ticket(
    payload: IngestRequest,
    response: Response,
    repository: Annotated[Repository, Depends(get_repository)],
) -> TicketOut:
    """Store a new ticket (202), replay an identical one (200) or reject a changed one (409)."""
    # The repository is synchronous; run it off the event loop so SQLite never blocks other requests.
    ticket, created = await asyncio.to_thread(repository.insert_ticket, payload.id, payload.subject, payload.body)
    if created:
        response.headers["Location"] = f"/tickets/{ticket.id}"
        return TicketOut.model_validate(ticket)
    if (ticket.subject, ticket.body) != (payload.subject, payload.body):
        raise ApiError(409, "ticket_conflict", f"Ticket {ticket.id} already exists with different content.")
    response.status_code = 200
    return TicketOut.model_validate(ticket)


@router.get(
    "/tickets/{id}",
    response_model=TicketOut,
    responses=GET_TICKET_RESPONSES,
    dependencies=[Depends(reject_query_params)],
)
async def get_ticket(id: str, repository: Annotated[Repository, Depends(get_repository)]) -> TicketOut:
    """Return one ticket in its current state, or 404 ``ticket_not_found``."""
    # No pattern check on the path: any id that is not stored is simply not found.
    ticket = await asyncio.to_thread(repository.get_ticket, id)
    if ticket is None:
        raise ApiError(404, "ticket_not_found", "Ticket not found.")
    return TicketOut.model_validate(ticket)


@router.get("/tickets", response_model=TicketPage, responses=LIST_TICKETS_RESPONSES)
async def list_tickets(
    repository: Annotated[Repository, Depends(get_repository)],
    category: Annotated[Category | None, Query()] = None,
    priority: Annotated[Priority | None, Query()] = None,
    flagged: Annotated[Literal["true", "false"] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=LIMIT_MAX)] = LIMIT_DEFAULT,
    cursor: Annotated[str | None, Query()] = None,
) -> TicketPage:
    """One page of tickets in (created_at, id) order; filters combine with AND."""
    after = None if cursor is None else decode_cursor(cursor)
    rows = await asyncio.to_thread(
        repository.list_tickets,
        category=category,
        priority=priority,
        flagged=None if flagged is None else flagged == "true",
        after=after,
        limit=limit + 1,  # one extra row tells us whether a next page exists
    )
    page = rows[:limit]
    next_cursor = encode_cursor(page[-1].created_at, page[-1].id) if len(rows) > limit else None
    return TicketPage(items=[TicketOut.model_validate(t) for t in page], next_cursor=next_cursor)
