"""Routes and request/response schemas.

Holds the shared error contract, the ticket schemas used by every ticket
endpoint, ``POST /tickets`` and the operational health probe.
"""

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, Field

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
    summary: str | None
    flagged: bool
    flag_reason: str | None
    attempts: int
    last_error: str | None
    prompt_version: str | None
    created_at: str
    updated_at: str


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
