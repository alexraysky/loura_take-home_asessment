"""All SQL lives here. Methods are synchronous; async callers use ``asyncio.to_thread``.

Timestamps are bound as ``format_ts`` strings, never as ``datetime`` objects
(the stdlib datetime adapter is deprecated).
"""

import sqlite3
from contextlib import closing
from datetime import timedelta

from app.clock import Clock, format_ts
from app.db import Database
from app.models import Category, Priority, Ticket
from app.validation import Accepted

# Same order as the table and as the Ticket fields; a test keeps the three in sync.
_COLUMNS = (
    "id, subject, body, status, category, priority, summary, flagged, flag_reason, "
    "attempts, last_error, claimed_at, next_attempt_at, prompt_version, created_at, updated_at"
)

# One statement, so two workers never take the same row: SQLite takes the write lock before the
# subquery runs. Claiming counts the attempt, so a ticket that crashes its worker cannot loop forever.
_CLAIM_SQL = f"""
UPDATE tickets
SET claimed_at = :now, attempts = attempts + 1, updated_at = :now
WHERE id = (
  SELECT id FROM tickets
  WHERE status = 'pending'
    AND (claimed_at IS NULL OR claimed_at < :lease_cutoff)
    AND (next_attempt_at IS NULL OR next_attempt_at <= :now)
  ORDER BY created_at, id
  LIMIT 1
)
RETURNING {_COLUMNS}"""

# Every finishing write: only the worker holding the live claim may change the row.
_GUARD = " WHERE id = :id AND status = 'pending' AND claimed_at = :token"
# flagged is OR-ed (MAX over 0/1; :flagged is always bound as an int, since MAX(x, NULL) is NULL) and
# the first stored reason is kept, so a code-set flag and its reason survive later attempts.
_FLAG_MERGE = "flagged = MAX(flagged, :flagged), flag_reason = COALESCE(flag_reason, :flag_reason)"

_CLASSIFIED_SQL = (
    "UPDATE tickets SET status = 'classified', category = :category, priority = :priority, "
    f"summary = :summary, {_FLAG_MERGE}, prompt_version = :prompt_version, last_error = NULL, "
    "claimed_at = NULL, next_attempt_at = NULL, updated_at = :now" + _GUARD
)
_RETRY_SQL = (
    "UPDATE tickets SET claimed_at = NULL, last_error = :code, next_attempt_at = :next_attempt_at, "
    f"{_FLAG_MERGE}, updated_at = :now" + _GUARD
)
_FAILED_SQL = (
    "UPDATE tickets SET status = 'failed', claimed_at = NULL, next_attempt_at = NULL, "
    f"last_error = :code, {_FLAG_MERGE}, updated_at = :now" + _GUARD
)


def ticket_from_row(row: sqlite3.Row) -> Ticket:
    values = dict(row)
    values["flagged"] = bool(values["flagged"])
    return Ticket(**values)


def _list_query(
    *,
    category: Category | None,
    priority: Priority | None,
    flagged: bool | None,
    after: tuple[str, str] | None,
    limit: int,
) -> tuple[str, list[object]]:
    """Build the list statement. Pure, so tests can EXPLAIN exactly what runs."""
    clauses: list[str] = []
    params: list[object] = []
    if after is not None:
        # Row-value comparison: one index seek on idx_tickets_list (created_at, id).
        # The expanded "a > ? OR (a = ? AND b > ?)" form degrades to a scan.
        clauses.append("(created_at, id) > (?, ?)")
        params.extend(after)
    if category is not None:
        clauses.append("category = ?")
        params.append(category)
    if priority is not None:
        clauses.append("priority = ?")
        params.append(priority)
    if flagged is not None:
        clauses.append("flagged = ?")
        params.append(int(flagged))
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    return f"SELECT {_COLUMNS} FROM tickets{where} ORDER BY created_at, id LIMIT ?", params


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

    def list_tickets(
        self,
        *,
        category: Category | None = None,
        priority: Priority | None = None,
        flagged: bool | None = None,
        after: tuple[str, str] | None = None,
        limit: int,
    ) -> list[Ticket]:
        """Rows in (created_at, id) order, strictly after ``after`` = (created_at, id), at most ``limit``.

        Filters combine with AND. ``category``/``priority`` compare with ``=``, so rows where the
        column is NULL (pending or failed tickets) never match a label filter. Peeking for a next
        page is the caller's job: pass ``limit + 1``.
        """
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        sql, params = _list_query(category=category, priority=priority, flagged=flagged, after=after, limit=limit)
        with self._db.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [ticket_from_row(row) for row in rows]

    def claim_next(self, *, lease_timeout_s: float) -> Ticket | None:
        """Claim the oldest claimable pending ticket, or return None when there is none.

        Claimable: no live lease (``claimed_at`` NULL or older than ``lease_timeout_s``) and no future
        ``next_attempt_at``. The returned ticket's ``claimed_at`` is the token for the finish write.
        """
        now = self._clock()
        params = {"now": format_ts(now), "lease_cutoff": format_ts(now - timedelta(seconds=lease_timeout_s))}
        with self._db.connection() as conn:
            with closing(conn.execute(_CLAIM_SQL, params)) as cur:
                row = cur.fetchone()
        return None if row is None else ticket_from_row(row)

    def finish_classified(self, id: str, *, token: str, result: Accepted, prompt_version: str) -> bool:
        """Store an accepted classification. False when the claim ``token`` is no longer live."""
        return self._finish(
            _CLASSIFIED_SQL,
            id,
            token,
            flagged=result.flagged,
            flag_reason=result.flag_reason,
            category=result.category,
            priority=result.priority,
            summary=result.summary,
            prompt_version=prompt_version,
        )

    def finish_retry(
        self,
        id: str,
        *,
        token: str,
        code: str,
        delay_s: float,
        sets_flag: bool = False,
        flag_reason: str | None = None,
    ) -> bool:
        """Release the claim with ``last_error = code``; claimable again ``delay_s`` from now."""
        _check_code(code)
        if not delay_s >= 0:
            raise ValueError(f"delay_s must be 0 or greater, got {delay_s!r}")
        next_attempt_at = format_ts(self._clock() + timedelta(seconds=delay_s))
        return self._finish(
            _RETRY_SQL,
            id,
            token,
            flagged=sets_flag,
            flag_reason=flag_reason,
            code=code,
            next_attempt_at=next_attempt_at,
        )

    def finish_failed(
        self,
        id: str,
        *,
        token: str,
        code: str,
        sets_flag: bool = False,
        flag_reason: str | None = None,
    ) -> bool:
        """Make the ticket terminal ``failed`` with ``last_error = code``."""
        _check_code(code)
        return self._finish(_FAILED_SQL, id, token, flagged=sets_flag, flag_reason=flag_reason, code=code)

    def _finish(self, sql: str, id: str, token: str, *, flagged: bool, **params: object) -> bool:
        # int(): the flag merge relies on a 0/1 binding, never NULL.
        params |= {"id": id, "token": token, "flagged": int(bool(flagged)), "now": self._now()}
        with self._db.connection() as conn:
            return conn.execute(sql, params).rowcount == 1


def _check_code(code: str) -> None:
    if not isinstance(code, str) or not code:
        raise ValueError(f"code must be a non-empty string, got {code!r}")
