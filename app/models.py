"""Domain types shared by storage, validation and the API."""

from dataclasses import dataclass
from typing import Literal, get_args

Status = Literal["pending", "classified", "failed"]
Category = Literal["billing", "technical", "account", "other"]
Priority = Literal["low", "medium", "high"]

STATUSES: tuple[Status, ...] = get_args(Status)
CATEGORIES: tuple[Category, ...] = get_args(Category)
PRIORITIES: tuple[Priority, ...] = get_args(Priority)


@dataclass(frozen=True, slots=True)
class Ticket:
    """One row of ``tickets``. Timestamps are canonical ``format_ts`` strings."""

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
    claimed_at: str | None  # lease and claim token; never exposed by the API
    next_attempt_at: str | None  # retry backoff; never exposed by the API
    prompt_version: str | None
    created_at: str
    updated_at: str
