import threading
from dataclasses import fields

import pytest

from app.clock import format_ts
from app.models import Ticket
from app.repository import _COLUMNS


def test_insert_returns_created_row_with_defaults(repository, clock):
    ticket, created = repository.insert_ticket("t-1001", "Refund", "Please refund me.")
    now = format_ts(clock())
    assert created is True
    assert ticket == Ticket(
        id="t-1001",
        subject="Refund",
        body="Please refund me.",
        status="pending",
        category=None,
        priority=None,
        summary=None,
        flagged=False,
        flag_reason=None,
        attempts=0,
        last_error=None,
        claimed_at=None,
        next_attempt_at=None,
        prompt_version=None,
        created_at=now,
        updated_at=now,
    )
    assert ticket.flagged is False


def test_insert_same_id_returns_existing_row_unchanged(repository, clock):
    first, _ = repository.insert_ticket("t-1001", "Refund", "Please refund me.")
    clock.advance(60)
    second, created = repository.insert_ticket("t-1001", "Other subject", "Other body")
    assert created is False
    assert second == first
    assert repository.get_ticket("t-1001") == first


def test_insert_does_not_touch_classified_row(repository, db, clock):
    repository.insert_ticket("t-1001", "Refund", "Please refund me.")
    with db.connection() as conn:
        conn.execute(
            "UPDATE tickets SET status = 'classified', category = 'billing', priority = 'high', "
            "summary = 'Wants a refund.', attempts = 1, prompt_version = 'v1' WHERE id = 't-1001'"
        )
    classified = repository.get_ticket("t-1001")
    clock.advance(60)
    again, created = repository.insert_ticket("t-1001", "Refund", "Please refund me.")
    assert created is False
    assert again == classified
    assert repository.get_ticket("t-1001") == classified
    assert classified.status == "classified"
    assert classified.attempts == 1


def test_text_is_stored_exactly(repository):
    body = "  Hello,\r\n> quoted line\r\n\tCafé – naïve 日本語 🙂  \n"
    ticket, _ = repository.insert_ticket("t-1008", "", body)
    stored = repository.get_ticket("t-1008")
    assert stored.subject == "" == ticket.subject
    assert stored.body == body == ticket.body


def test_get_ticket_missing_returns_none(repository):
    assert repository.get_ticket("t-missing") is None


def test_get_ticket_returns_row(repository):
    ticket, _ = repository.insert_ticket("t-1001", "Refund", "Please refund me.")
    assert repository.get_ticket("t-1001") == ticket


def test_flagged_maps_to_bool(repository, db):
    repository.insert_ticket("t-1001", "Refund", "Please refund me.")
    with db.connection() as conn:
        conn.execute("UPDATE tickets SET flagged = 1 WHERE id = 't-1001'")
    assert repository.get_ticket("t-1001").flagged is True


def test_columns_match_table_and_dataclass(db):
    with db.connection() as conn:
        table_columns = [row["name"] for row in conn.execute("PRAGMA table_info(tickets)")]
    listed = [name.strip() for name in _COLUMNS.split(",")]
    assert listed == table_columns == [f.name for f in fields(Ticket)]


def test_concurrent_inserts_of_same_id_create_one_row(repository, db):
    threads_count = 16
    barrier = threading.Barrier(threads_count, timeout=10)
    results: list[tuple[Ticket, bool]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def insert(i: int) -> None:
        try:
            barrier.wait()
            result = repository.insert_ticket("t-race", "Subject", f"body-{i}")
            with lock:
                results.append(result)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=insert, args=(i,)) for i in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    assert errors == []
    assert len(results) == threads_count
    assert sum(created for _, created in results) == 1
    stored = repository.get_ticket("t-race")
    assert all(ticket == stored for ticket, _ in results)
    with db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tickets WHERE id = 't-race'").fetchone()[0] == 1


def label(db, id: str, category: str, priority: str, flagged: bool = False) -> None:
    with db.connection() as conn:
        conn.execute(
            "UPDATE tickets SET status = 'classified', category = ?, priority = ?, summary = 'x', flagged = ? "
            "WHERE id = ?",
            (category, priority, int(flagged), id),
        )


def listed(tickets: list[Ticket]) -> list[str]:
    return [t.id for t in tickets]


def test_list_orders_by_created_at_then_id(repository, clock):
    for id in ("b", "a", "B"):
        repository.insert_ticket(id, "", "x")
    clock.advance(1)
    repository.insert_ticket("A", "", "x")
    # Same timestamp: byte order of the id ("B" < "a" < "b"), then the later timestamp.
    assert listed(repository.list_tickets(limit=10)) == ["B", "a", "b", "A"]


def test_list_after_is_exclusive_and_uses_both_columns(repository, clock):
    for id in ("t-1", "t-2", "t-3"):
        repository.insert_ticket(id, "", "x")
    clock.advance(1)
    repository.insert_ticket("t-0", "", "x")
    ts = format_ts(clock().replace(second=0))
    assert listed(repository.list_tickets(after=(ts, "t-2"), limit=10)) == ["t-3", "t-0"]
    assert listed(repository.list_tickets(after=(ts, "t-3"), limit=10)) == ["t-0"]
    assert listed(repository.list_tickets(after=(format_ts(clock()), "t-0"), limit=10)) == []


def test_list_limit_caps_rows(repository, clock):
    for i in range(5):
        repository.insert_ticket(f"t-{i}", "", "x")
        clock.advance(1)
    assert listed(repository.list_tickets(limit=3)) == ["t-0", "t-1", "t-2"]


def test_list_limit_below_one_raises(repository):
    with pytest.raises(ValueError, match="limit"):
        repository.list_tickets(limit=0)


def test_list_filters_combine_with_and(repository, db, clock):
    rows = {
        "t-1": ("billing", "high", True),
        "t-2": ("billing", "low", False),
        "t-3": ("technical", "high", True),
        "t-4": ("billing", "high", False),
    }
    for id, labels in rows.items():
        repository.insert_ticket(id, "", "x")
        label(db, id, *labels)
        clock.advance(1)
    assert listed(repository.list_tickets(category="billing", limit=10)) == ["t-1", "t-2", "t-4"]
    assert listed(repository.list_tickets(priority="high", limit=10)) == ["t-1", "t-3", "t-4"]
    assert listed(repository.list_tickets(flagged=True, limit=10)) == ["t-1", "t-3"]
    assert listed(repository.list_tickets(flagged=False, limit=10)) == ["t-2", "t-4"]
    assert listed(repository.list_tickets(category="billing", priority="high", flagged=True, limit=10)) == ["t-1"]


def test_list_label_filter_excludes_null_category(repository, db):
    repository.insert_ticket("t-pending", "", "x")
    repository.insert_ticket("t-labelled", "", "x")
    label(db, "t-labelled", "other", "low")
    assert listed(repository.list_tickets(category="other", limit=10)) == ["t-labelled"]
    assert listed(repository.list_tickets(priority="low", limit=10)) == ["t-labelled"]
    assert listed(repository.list_tickets(flagged=False, limit=10)) == ["t-labelled", "t-pending"]


def test_list_returns_full_tickets(repository, db, clock):
    for id in ("t-1", "t-2"):
        repository.insert_ticket(id, "Subject", "Body")
        clock.advance(1)
    label(db, "t-2", "account", "medium", flagged=True)
    tickets = repository.list_tickets(limit=10)
    assert tickets == [repository.get_ticket("t-1"), repository.get_ticket("t-2")]
