import threading
from dataclasses import fields

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
