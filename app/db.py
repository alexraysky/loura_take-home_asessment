"""SQLite connection factory and schema creation.

Every repository operation opens its own connection, configures it, and closes
it before returning. Connections run in autocommit mode: each statement is a
transaction of its own, and nothing is ever held across an ``await``.
Never return a connection or cursor from a repository method.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

# RETURNING (used by the idempotent insert in repository.py for #3 and the claim in #7) needs 3.35.
MIN_SQLITE_VERSION = (3, 35, 0)
# Writers wait this long for the file lock before failing with "database is locked".
BUSY_TIMEOUT_S = 5.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
  id              TEXT PRIMARY KEY,
  subject         TEXT NOT NULL DEFAULT '',
  body            TEXT NOT NULL,
  status          TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'classified', 'failed')),
  category        TEXT CHECK (category IN ('billing', 'technical', 'account', 'other')),
  priority        TEXT CHECK (priority IN ('low', 'medium', 'high')),
  summary         TEXT,
  flagged         INTEGER NOT NULL DEFAULT 0 CHECK (flagged IN (0, 1)),
  flag_reason     TEXT,
  attempts        INTEGER NOT NULL DEFAULT 0,
  last_error      TEXT,
  claimed_at      TEXT,
  next_attempt_at TEXT,
  prompt_version  TEXT,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  CHECK (status <> 'classified'
         OR (category IS NOT NULL AND priority IS NOT NULL AND summary IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS idx_tickets_claim ON tickets (status, next_attempt_at);
CREATE INDEX IF NOT EXISTS idx_tickets_list  ON tickets (created_at, id);
"""


def check_sqlite_version() -> None:
    if sqlite3.sqlite_version_info < MIN_SQLITE_VERSION:
        wanted = ".".join(map(str, MIN_SQLITE_VERSION))
        raise RuntimeError(
            f"SQLite {sqlite3.sqlite_version} is too old; {wanted} or later is required "
            "(the version comes from the Python build, not from Python itself)"
        )


class Database:
    """Connection factory for one SQLite file."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        # Check after normalising: Path("./:memory:") is ":memory:" too, and SQLite would open RAM.
        if str(self.path) == ":memory:":
            raise ValueError(
                "DB_PATH must be a file path: every operation opens its own connection, "
                "so an in-memory database would be empty on each use"
            )

    def connect(self) -> sqlite3.Connection:
        """Open a configured connection. The caller must close it; prefer ``connection()``."""
        conn = sqlite3.connect(self.path, autocommit=True, timeout=BUSY_TIMEOUT_S)
        try:
            conn.row_factory = sqlite3.Row
            # WAL: readers do not block the single writer. Persistent in the file, cheap to repeat.
            conn.execute("PRAGMA journal_mode = WAL")
            # NORMAL is safe against corruption in WAL mode; only a power cut can lose the last commits.
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA foreign_keys = ON")
        except BaseException:
            conn.close()
            raise
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """A configured connection that is closed on exit, whatever happens inside."""
        with closing(self.connect()) as conn:
            yield conn

    def create_schema(self) -> None:
        """Create the tables and indexes if they do not exist. Safe to call on every start."""
        check_sqlite_version()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            if mode != "wal":
                raise RuntimeError(f"could not enable WAL mode for {self.path} (journal_mode is {mode!r})")
            conn.executescript(SCHEMA)
