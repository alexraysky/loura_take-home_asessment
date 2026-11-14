"""All SQL lives here. Methods are synchronous; async callers use ``asyncio.to_thread``.

Timestamps are bound as ``format_ts`` strings, never as ``datetime`` objects
(the stdlib datetime adapter is deprecated).
"""

import sqlite3
from contextlib import closing

from app.clock import Clock, format_ts
from app.db import Database
from app.models import Ticket

# Same order as the table and as the Ticket fields; a test keeps the three in sync.
_COLUMNS = (
    "id, subject, body, status, category, priority, summary, flagged, flag_reason, "
    "attempts, last_error, claimed_at, next_attempt_at, prompt_version, created_at, updated_at"
)


def ticket_from_row(row: sqlite3.Row) -> Ticket:
    values = dict(row)
    values["flagged"] = bool(values["flagged"])
    return Ticket(**values)


class Repository:
    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def _now(self) -> str:
        return format_ts(self._clock())

    def insert_ticket(self, id: str, subject: str, body: str) -> tuple[Ticket, bool]:
        """Idempotent insert. Returns the stored row and whether this call created it.

        A duplicate id leaves the existing row untouched (status, attempts and labels
        included); deciding between 200 and 409 is the caller's job (#3).
        """
        now = self._now()
        with self._db.connection() as conn:
            # Close the RETURNING cursor explicitly so the statement is finished (and the
            # autocommit transaction committed) before we decide anything from the result.
            with closing(
                conn.execute(
                    f"INSERT INTO tickets (id, subject, body, created_at, updated_at) "
                    f"VALUES (?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING RETURNING {_COLUMNS}",
                    (id, subject, body, now, now),
                )
            ) as cur:
                row = cur.fetchone()
            if row is not None:
                return ticket_from_row(row), True
            # Rows are never deleted and id/subject/body never change, so this cannot miss.
            existing = conn.execute(f"SELECT {_COLUMNS} FROM tickets WHERE id = ?", (id,)).fetchone()
        return ticket_from_row(existing), False

    def get_ticket(self, id: str) -> Ticket | None:
        with self._db.connection() as conn:
            row = conn.execute(f"SELECT {_COLUMNS} FROM tickets WHERE id = ?", (id,)).fetchone()
        return None if row is None else ticket_from_row(row)
