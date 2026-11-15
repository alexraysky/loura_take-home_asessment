import sqlite3
import threading
from dataclasses import fields, replace
from datetime import timedelta

import pytest

from app.clock import format_ts
from app.models import Ticket
from app.repository import _COLUMNS, Repository
from app.validation import FLAG_REASON_ECHO, Accepted


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


LEASE = 60.0
ACCEPTED = Accepted("billing", "low", "Asks where to download invoices.", False, None)


def claim(repository) -> Ticket | None:
    return repository.claim_next(lease_timeout_s=LEASE)


def set_columns(db, id: str, **columns) -> None:
    assignments = ", ".join(f"{name} = ?" for name in columns)
    with db.connection() as conn:
        conn.execute(f"UPDATE tickets SET {assignments} WHERE id = ?", (*columns.values(), id))


def test_claim_next_on_empty_queue_returns_none(repository):
    assert claim(repository) is None


def test_claim_takes_the_oldest_pending_by_created_at_then_id(repository, clock):
    for id in ("b", "a", "B"):
        repository.insert_ticket(id, "", "x")
    clock.advance(1)
    repository.insert_ticket("A", "", "x")
    # The list order: same timestamp by id byte order, then the later timestamp.
    assert [claim(repository).id for _ in range(4)] == ["B", "a", "b", "A"]
    assert claim(repository) is None


def test_claim_sets_attempts_claimed_at_and_updated_at(repository, clock):
    inserted, _ = repository.insert_ticket("t-1001", "Refund", "Please refund me.")
    clock.advance(5)
    now = format_ts(clock())
    claimed = claim(repository)
    assert claimed == replace(inserted, attempts=1, claimed_at=now, updated_at=now)
    assert claimed == repository.get_ticket("t-1001")


def test_leased_row_is_not_claimable_until_the_lease_expires(repository, clock):
    repository.insert_ticket("t-1", "", "x")
    first = claim(repository)
    clock.advance(LEASE)
    # Strict "<": the lease is still live at exactly LEASE seconds.
    assert claim(repository) is None
    clock.advance(1e-6)
    second = claim(repository)
    assert second.id == "t-1"
    assert second.attempts == 2
    assert second.claimed_at != first.claimed_at


def test_next_attempt_at_gate_boundaries(repository, db, clock):
    repository.insert_ticket("t-1", "", "x")
    set_columns(db, "t-1", next_attempt_at=format_ts(clock() + timedelta(seconds=5)))
    clock.advance(4.999999)
    assert claim(repository) is None
    clock.advance(1e-6)
    assert claim(repository).id == "t-1"


def test_classified_and_failed_rows_are_never_claimed(repository, db):
    repository.insert_ticket("t-classified", "", "x")
    repository.insert_ticket("t-failed", "", "x")
    set_columns(db, "t-classified", status="classified", category="other", priority="low", summary="x")
    set_columns(db, "t-failed", status="failed", last_error="not_json")
    assert claim(repository) is None


def run_concurrent_claims(repository, threads_count: int) -> list[Ticket | None]:
    barrier = threading.Barrier(threads_count, timeout=10)
    results: list[Ticket | None] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def take() -> None:
        try:
            barrier.wait()
            ticket = claim(repository)
            with lock:
                results.append(ticket)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=take) for _ in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    assert errors == []
    assert len(results) == threads_count
    return results


@pytest.mark.parametrize("rows", [16, 5])
def test_concurrent_claims_return_distinct_tickets(repository, clock, rows):
    for i in range(rows):
        repository.insert_ticket(f"t-{i:02d}", "", "x")
        clock.advance(1)
    results = run_concurrent_claims(repository, 16)
    claimed = [t for t in results if t is not None]
    assert len(claimed) == rows
    assert results.count(None) == 16 - rows
    assert sorted(t.id for t in claimed) == [f"t-{i:02d}" for i in range(rows)]
    assert all(t.attempts == 1 for t in claimed)
    assert all(repository.get_ticket(t.id).attempts == 1 for t in claimed)


def test_finish_classified_writes_labels_and_clears_the_queue_columns(repository, db, clock):
    repository.insert_ticket("t-1", "", "x")
    set_columns(db, "t-1", last_error="not_json", next_attempt_at=format_ts(clock()))
    ticket = claim(repository)
    clock.advance(2)
    assert repository.finish_classified("t-1", token=ticket.claimed_at, result=ACCEPTED, prompt_version="v1") is True
    assert repository.get_ticket("t-1") == replace(
        ticket,
        status="classified",
        category="billing",
        priority="low",
        summary=ACCEPTED.summary,
        prompt_version="v1",
        last_error=None,
        claimed_at=None,
        next_attempt_at=None,
        updated_at=format_ts(clock()),
    )


@pytest.mark.parametrize("existing", [False, True], ids=["clean-row", "flagged-row"])
@pytest.mark.parametrize("model_flag", [False, True], ids=["model-clean", "model-flag"])
def test_finish_classified_merges_the_flag(repository, db, existing, model_flag):
    repository.insert_ticket("t-1", "", "x")
    if existing:
        set_columns(db, "t-1", flagged=1, flag_reason=FLAG_REASON_ECHO)
    ticket = claim(repository)
    result = replace(ACCEPTED, flagged=model_flag, flag_reason="Model reason." if model_flag else None)
    assert repository.finish_classified("t-1", token=ticket.claimed_at, result=result, prompt_version="v1")
    stored = repository.get_ticket("t-1")
    assert stored.flagged is (existing or model_flag)
    # The first stored reason wins; a missing one is filled from the model.
    expected_reason = FLAG_REASON_ECHO if existing else ("Model reason." if model_flag else None)
    assert stored.flag_reason == expected_reason


def test_finish_retry_sets_backoff_and_releases_the_claim(repository, clock):
    repository.insert_ticket("t-1", "", "x")
    ticket = claim(repository)
    clock.advance(3)
    assert repository.finish_retry("t-1", token=ticket.claimed_at, code="llm_timeout", delay_s=2.5) is True
    stored = repository.get_ticket("t-1")
    assert stored == replace(
        ticket,
        last_error="llm_timeout",
        claimed_at=None,
        next_attempt_at=format_ts(clock() + timedelta(seconds=2.5)),
        updated_at=format_ts(clock()),
    )
    assert (stored.status, stored.attempts) == ("pending", 1)


def test_finish_retry_reads_the_clock_once(db, clock):
    def ticking():
        clock.advance(1)  # every read is a second later
        return clock()

    repository = Repository(db, ticking)
    repository.insert_ticket("t-1", "", "x")
    ticket = claim(repository)
    assert repository.finish_retry("t-1", token=ticket.claimed_at, code="llm_timeout", delay_s=2.5) is True
    stored = repository.get_ticket("t-1")
    assert stored.next_attempt_at == format_ts(clock() + timedelta(seconds=2.5))
    assert stored.updated_at == format_ts(clock())


def test_finish_retry_with_the_echo_flag_flags_the_row(repository):
    repository.insert_ticket("t-1", "", "x")
    ticket = claim(repository)
    assert repository.finish_retry(
        "t-1",
        token=ticket.claimed_at,
        code="summary_echoes_ticket",
        delay_s=0,
        sets_flag=True,
        flag_reason=FLAG_REASON_ECHO,
    )
    stored = repository.get_ticket("t-1")
    assert (stored.flagged, stored.flag_reason) == (True, FLAG_REASON_ECHO)


def test_finish_failed_clears_backoff_and_claim(repository, db, clock):
    repository.insert_ticket("t-1", "", "x")
    set_columns(db, "t-1", next_attempt_at=format_ts(clock()))
    ticket = claim(repository)
    clock.advance(1)
    assert repository.finish_failed("t-1", token=ticket.claimed_at, code="not_json") is True
    assert repository.get_ticket("t-1") == replace(
        ticket,
        status="failed",
        last_error="not_json",
        claimed_at=None,
        next_attempt_at=None,
        updated_at=format_ts(clock()),
    )
    assert claim(repository) is None


FINISHES = {
    "classified": lambda r, token: r.finish_classified("t-1", token=token, result=ACCEPTED, prompt_version="v1"),
    "retry": lambda r, token: r.finish_retry("t-1", token=token, code="not_json", delay_s=1),
    "failed": lambda r, token: r.finish_failed("t-1", token=token, code="not_json"),
}


@pytest.mark.parametrize("finish", FINISHES.values(), ids=FINISHES.keys())
def test_finish_writes_ignore_a_wrong_token(repository, clock, finish):
    repository.insert_ticket("t-1", "", "x")
    ticket = claim(repository)
    clock.advance(1)
    assert finish(repository, format_ts(clock())) is False
    assert repository.get_ticket("t-1") == ticket


@pytest.mark.parametrize("finish", FINISHES.values(), ids=FINISHES.keys())
def test_finish_writes_are_not_replayable(repository, clock, finish):
    repository.insert_ticket("t-1", "", "x")
    ticket = claim(repository)
    assert finish(repository, ticket.claimed_at) is True
    after_first = repository.get_ticket("t-1")
    clock.advance(1)
    assert finish(repository, ticket.claimed_at) is False
    assert repository.get_ticket("t-1") == after_first


def test_finish_after_another_worker_reclaimed_is_ignored(repository, clock):
    repository.insert_ticket("t-1", "", "x")
    first = claim(repository)
    clock.advance(LEASE + 1)
    second = claim(repository)
    assert repository.finish_classified("t-1", token=second.claimed_at, result=ACCEPTED, prompt_version="v1")
    assert repository.finish_failed("t-1", token=first.claimed_at, code="llm_timeout") is False
    stored = repository.get_ticket("t-1")
    assert (stored.status, stored.attempts, stored.last_error) == ("classified", 2, None)


@pytest.mark.parametrize("code", ["", None, 7])
def test_finish_rejects_an_empty_or_non_str_code(repository, code):
    repository.insert_ticket("t-1", "", "x")
    ticket = claim(repository)
    with pytest.raises(ValueError, match="code"):
        repository.finish_retry("t-1", token=ticket.claimed_at, code=code, delay_s=1)
    with pytest.raises(ValueError, match="code"):
        repository.finish_failed("t-1", token=ticket.claimed_at, code=code)
    assert repository.get_ticket("t-1") == ticket


@pytest.mark.parametrize("delay_s", [-1e-6, -1.0, float("nan")])
def test_finish_retry_rejects_a_negative_delay(repository, delay_s):
    repository.insert_ticket("t-1", "", "x")
    ticket = claim(repository)
    with pytest.raises(ValueError, match="delay_s"):
        repository.finish_retry("t-1", token=ticket.claimed_at, code="not_json", delay_s=delay_s)
    assert repository.get_ticket("t-1") == ticket


def test_schema_check_still_guards_finish_classified(repository):
    repository.insert_ticket("t-1", "", "x")
    ticket = claim(repository)
    bogus = Accepted("refund", "low", "x", False, None)
    with pytest.raises(sqlite3.IntegrityError):
        repository.finish_classified("t-1", token=ticket.claimed_at, result=bogus, prompt_version="v1")
    assert repository.get_ticket("t-1") == ticket


@pytest.mark.parametrize("finish", ["retry", "failed"])
def test_flagged_is_bound_as_an_integer(repository, db, finish):
    # Regression: MAX(1, NULL) is NULL in SQLite, so an unflagged attempt must bind 0, never None.
    repository.insert_ticket("t-1", "", "x")
    set_columns(db, "t-1", flagged=1, flag_reason=FLAG_REASON_ECHO)
    ticket = claim(repository)
    assert FINISHES[finish](repository, ticket.claimed_at) is True
    stored = repository.get_ticket("t-1")
    assert (stored.flagged, stored.flag_reason) == (True, FLAG_REASON_ECHO)
    with db.connection() as conn:
        assert tuple(conn.execute("SELECT typeof(flagged), flagged FROM tickets").fetchone()) == ("integer", 1)
