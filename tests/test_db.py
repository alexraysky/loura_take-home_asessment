import os
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest

from app.db import BUSY_TIMEOUT_S, Database
from app.models import CATEGORIES, PRIORITIES, STATUSES
from app.repository import _CLAIM_SQL, _COLUMNS, _list_query

NOW = "2026-11-14T12:00:00.000000Z"


def insert_row(conn: sqlite3.Connection, id: str = "t-1", **columns) -> None:
    values = {"id": id, "subject": "s", "body": "b", "created_at": NOW, "updated_at": NOW, **columns}
    names = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    conn.execute(f"INSERT INTO tickets ({names}) VALUES ({marks})", tuple(values.values()))


def schema_objects(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'index')")}


def test_create_schema_creates_table_and_indexes(db):
    with db.connection() as conn:
        assert {"tickets", "idx_tickets_claim", "idx_tickets_list"} <= schema_objects(conn)


def test_create_schema_is_idempotent(db):
    with db.connection() as conn:
        insert_row(conn)
    db.create_schema()
    with db.connection() as conn:
        assert [row["id"] for row in conn.execute("SELECT id FROM tickets")] == ["t-1"]


def test_create_schema_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deeper" / "t.db"
    Database(str(path)).create_schema()
    assert path.is_file()


def test_default_relative_path_works_in_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = Database("tickets.db")
    assert db.path.parent == Path(".")
    db.create_schema()
    assert (tmp_path / "tickets.db").is_file()


# All three normalise to ":memory:"; the backslash is only a separator on Windows.
IN_MEMORY_PATHS = [
    ":memory:",
    "./:memory:",
    pytest.param(".\\:memory:", marks=pytest.mark.skipif(os.name != "nt", reason="Windows path")),
]


@pytest.mark.parametrize("path", IN_MEMORY_PATHS)
def test_in_memory_path_is_rejected(path):
    with pytest.raises(ValueError, match="DB_PATH"):
        Database(path)


def test_connection_pragmas(db):
    with db.connection() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == int(BUSY_TIMEOUT_S * 1000)
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.autocommit is True
        assert conn.row_factory is sqlite3.Row


def test_wal_mode_persists_and_is_set_per_connection(db):
    # A plain connection sees the persistent journal mode; synchronous is per connection,
    # so a connection set to FULL does not affect the level a fresh db.connection() gets.
    with closing(sqlite3.connect(db.path, autocommit=True)) as raw:
        assert raw.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        raw.execute("PRAGMA synchronous = FULL")
        assert raw.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
        with db.connection() as conn:
            assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        assert raw.execute("PRAGMA synchronous").fetchone()[0] == 2
    with db.connection() as conn:
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1


def test_connection_is_closed_on_exit(db):
    with db.connection() as conn:
        conn.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_connection_is_closed_when_body_raises(db):
    with pytest.raises(KeyError):
        with db.connection() as conn:
            raise KeyError("boom")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_connect_closes_on_pragma_failure(db, monkeypatch):
    opened: list[sqlite3.Connection] = []

    class FailingPragma(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql.startswith("PRAGMA synchronous"):
                raise sqlite3.OperationalError("injected")
            return super().execute(sql, *args)

    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        conn = real_connect(*args, factory=FailingPragma, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    with pytest.raises(sqlite3.OperationalError, match="injected"):
        db.connect()
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")


def test_old_sqlite_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 1))
    path = tmp_path / "t.db"
    with pytest.raises(RuntimeError, match=r"3\.35\.0"):
        Database(str(path)).create_schema()
    assert not path.exists()


@pytest.mark.parametrize(
    "column, value",
    [("category", "refund"), ("priority", "urgent"), ("status", "processing"), ("flagged", 2)],
)
def test_out_of_set_values_are_rejected(db, column, value):
    with db.connection() as conn:
        insert_row(conn)
        with pytest.raises(sqlite3.IntegrityError) as excinfo:
            conn.execute(f"UPDATE tickets SET {column} = ?", (value,))
    assert excinfo.value.sqlite_errorname == "SQLITE_CONSTRAINT_CHECK"


@pytest.mark.parametrize(
    "labels",
    [
        {},
        {"priority": "low", "summary": "x"},
        {"category": "billing", "summary": "x"},
        {"category": "billing", "priority": "low"},
    ],
    ids=["no-labels", "no-category", "no-priority", "no-summary"],
)
def test_classified_without_labels_is_rejected(db, labels):
    with db.connection() as conn:
        with pytest.raises(sqlite3.IntegrityError) as excinfo:
            insert_row(conn, status="classified", **labels)
        insert_row(conn, **labels)
        with pytest.raises(sqlite3.IntegrityError) as update_excinfo:
            conn.execute("UPDATE tickets SET status = 'classified'")
    assert excinfo.value.sqlite_errorname == "SQLITE_CONSTRAINT_CHECK"
    assert update_excinfo.value.sqlite_errorname == "SQLITE_CONSTRAINT_CHECK"


def test_classified_with_labels_is_accepted(db):
    with db.connection() as conn:
        insert_row(conn)
        conn.execute(
            "UPDATE tickets SET status = 'classified', category = 'billing', priority = 'low', summary = 'x'"
        )
        assert conn.execute("SELECT status FROM tickets").fetchone()[0] == "classified"


def test_every_allowed_value_is_accepted(db):
    # Keeps the CHECK lists in the DDL and the tuples in app.models in sync.
    labels = {"category": "other", "priority": "low", "summary": "x"}
    with db.connection() as conn:
        for status in STATUSES:
            insert_row(conn, f"s-{status}", status=status, **labels)
        for category in CATEGORIES:
            insert_row(conn, f"c-{category}", category=category)
        for priority in PRIORITIES:
            insert_row(conn, f"p-{priority}", priority=priority)
        for flagged in (0, 1):
            insert_row(conn, f"f-{flagged}", flagged=flagged)
        count = conn.execute("SELECT count(*) FROM tickets").fetchone()[0]
    assert count == len(STATUSES) + len(CATEGORIES) + len(PRIORITIES) + 2


def test_body_is_required(db):
    with db.connection() as conn:
        with pytest.raises(sqlite3.IntegrityError) as excinfo:
            insert_row(conn, body=None)
    assert excinfo.value.sqlite_errorname == "SQLITE_CONSTRAINT_NOTNULL"


def test_duplicate_id_is_rejected_without_on_conflict(db):
    with db.connection() as conn:
        insert_row(conn)
        with pytest.raises(sqlite3.IntegrityError) as excinfo:
            insert_row(conn)
    assert excinfo.value.sqlite_errorname == "SQLITE_CONSTRAINT_PRIMARYKEY"


def test_defaults_are_applied(db):
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO tickets (id, body, created_at, updated_at) VALUES ('t-1', 'b', ?, ?)", (NOW, NOW)
        )
        row = conn.execute("SELECT subject, status, flagged, attempts FROM tickets").fetchone()
    assert tuple(row) == ("", "pending", 0, 0)


def test_integrity_check_is_ok(db):
    with db.connection() as conn:
        for i in range(10):
            insert_row(conn, f"t-{i}")
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_claim_query_uses_the_index(db):
    claim = """
        SELECT id FROM tickets
        WHERE status = 'pending'
          AND (claimed_at IS NULL OR claimed_at < :lease_cutoff)
          AND (next_attempt_at IS NULL OR next_attempt_at <= :now)
        ORDER BY created_at, id
        LIMIT 1
    """
    with db.connection() as conn:
        claim_plan = conn.execute(f"EXPLAIN QUERY PLAN {claim}", {"lease_cutoff": NOW, "now": NOW}).fetchall()
    # 3.35 says "SEARCH TABLE tickets USING INDEX ...", 3.36+ drops "TABLE"; match the index part only.
    assert any("USING INDEX idx_tickets_claim" in row["detail"] for row in claim_plan)


def test_claim_statement_uses_the_index(db):
    # The statement the repository actually runs, not a hand-written copy of its subquery.
    with db.connection() as conn:
        rows = conn.execute(f"EXPLAIN QUERY PLAN {_CLAIM_SQL}", {"lease_cutoff": NOW, "now": NOW}).fetchall()
    plan = [row["detail"] for row in rows]
    assert any("USING INDEX idx_tickets_claim" in detail for detail in plan)


@pytest.mark.parametrize("after", [None, (NOW, "t-1")], ids=["first-page", "cursor"])
@pytest.mark.parametrize(
    "filters",
    [{}, {"category": "billing", "priority": "high", "flagged": True}],
    ids=["no-filter", "all-filters"],
)
def test_list_query_uses_the_index(db, after, filters):
    filters = {"category": None, "priority": None, "flagged": None, **filters}
    sql, params = _list_query(after=after, limit=10, **filters)
    with db.connection() as conn:
        plan = [row["detail"] for row in conn.execute(f"EXPLAIN QUERY PLAN {sql}", params)]
    # The index order is the result order: an ordered walk, never a sort.
    assert not any("TEMP B-TREE" in detail for detail in plan)
    assert any("USING INDEX idx_tickets_list" in detail for detail in plan)
    if after is not None:
        # The row-value comparison seeks into the index instead of scanning from the start.
        assert any("USING INDEX idx_tickets_list ((created_at,id)>(?,?))" in detail for detail in plan)


def test_concurrent_writers_wait_for_the_lock(db):
    threads_count = 16
    barrier = threading.Barrier(threads_count, timeout=10)
    errors: list[BaseException] = []

    def write(i: int) -> None:
        try:
            with db.connection() as conn:
                barrier.wait()
                conn.execute("BEGIN IMMEDIATE")
                insert_row(conn, f"t-{i}")
                conn.execute("COMMIT")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    assert errors == []
    with db.connection() as conn:
        assert conn.execute("SELECT count(*) FROM tickets").fetchone()[0] == threads_count
