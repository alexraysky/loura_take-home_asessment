"""Routes and request/response schemas.

Only the shared error contract and the operational health probe live here so
far; the ticket endpoints are added by later issues.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

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


@router.api_route("/health", methods=["GET", "HEAD"], tags=["ops"])
async def health() -> dict[str, str]:
    """Readiness probe; not part of the ticket API."""
    return {"status": "ok"}
