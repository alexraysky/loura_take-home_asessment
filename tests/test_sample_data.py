import asyncio
import json
import random
import shutil
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.api import IngestRequest
from app.db import Database
from app.llm import FakeLLM
from app.llm.fake import HIJACKED_ANSWER, _is_injected
from app.main import create_app
from app.models import CATEGORIES, PRIORITIES
from app.prompt import PROMPT_VERSION
from app.repository import Repository
from app.validation import FLAG_REASON_ECHO, Rejected, normalise_for_match, quoted_spans, validate_response
from app.worker import Outcome, Processed, Worker
from tests.sample_data import (
    EXPECTED,
    EXPECTED_ROWS,
    INJECTED,
    LABELS_FILE,
    SAMPLE_TICKETS,
    TICKET_PAYLOADS,
    TICKETS_FILE,
)

IDS = [f"t-{n}" for n in range(1001, 1011)]
ATTACKER_SUMMARY = HIJACKED_ANSWER["summary"]
DEADLINE_S = 5.0
# Attempts each sample takes under the default fake (seed 42, failure rate 0.2, gullible) through the
# real worker loop: t-1001 draws schema_violation, t-1003 not_json and t-1005 is hijacked first. The
# draw is a hash of the prompt, so this changes together with the ticket texts.
ATTEMPTS_AT_SEED_42 = {tid: 1 for tid in IDS} | {"t-1001": 2, "t-1003": 2, "t-1005": 2}


class ZeroRandom(random.Random):
    """No backoff jitter."""

    def random(self) -> float:
        return 0.0


def honest() -> FakeLLM:
    return FakeLLM(failure_rate=0.0, gullible=False)


async def run_to_terminal(worker: Worker, clock, id: str, limit: int = 10) -> list[Processed]:
    """Drive process_one until the ticket is classified or failed, jumping the clock past each backoff."""
    steps: list[Processed] = []
    for _ in range(limit):
        processed = await worker.process_one()
        assert processed is not None and processed.ticket_id == id
        steps.append(processed)
        if processed.outcome in (Outcome.CLASSIFIED, Outcome.FAILED):
            return steps
        clock.advance(3600)
    raise AssertionError(f"{id} did not reach a terminal state in {limit} attempts")


def list_all(client: TestClient) -> list[dict]:
    page = client.get("/tickets", params={"limit": 100}).json()
    assert page["next_cursor"] is None
    return page["items"]


# A. The fixture files


def test_ten_tickets_in_order():
    assert [t["id"] for t in TICKET_PAYLOADS] == IDS
    assert [row["id"] for row in EXPECTED_ROWS] == IDS
    assert list(SAMPLE_TICKETS) == list(EXPECTED) == IDS


def test_payload_keys_and_types():
    for entry in TICKET_PAYLOADS:
        assert list(entry) == ["id", "subject", "body"]
        assert all(isinstance(value, str) for value in entry.values())
        assert IngestRequest.model_validate(entry).model_dump() == entry


def test_t1008_has_empty_subject():
    entry = TICKET_PAYLOADS[IDS.index("t-1008")]
    assert entry["subject"] == ""
    assert entry["body"]


def test_t1010_keeps_line_breaks_and_quote_line():
    _subject, body = SAMPLE_TICKETS["t-1010"]
    assert "\n" in body
    assert any(line.startswith(">") for line in body.split("\n"))


@pytest.mark.parametrize("path", [TICKETS_FILE, LABELS_FILE], ids=["tickets", "labels"])
def test_no_carriage_returns(path):
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.endswith(b"]\n")
    for _subject, body in SAMPLE_TICKETS.values():
        assert "\r" not in body


def test_expected_labels_are_from_the_allowed_sets():
    for row in EXPECTED_ROWS:
        assert list(row) == ["id", "category", "priority", "flagged", "what_it_tests"]
        assert row["category"] in CATEGORIES
        assert row["priority"] in PRIORITIES
        assert isinstance(row["flagged"], bool)
        assert isinstance(row["what_it_tests"], str) and row["what_it_tests"].strip()
    assert INJECTED == {"t-1005"}


def test_only_t1005_matches_the_injection_cue():
    assert {tid for tid, (subject, body) in SAMPLE_TICKETS.items() if _is_injected(subject, body)} == {"t-1005"}


def test_t1005_carries_the_quoted_plant():
    subject, body = SAMPLE_TICKETS["t-1005"]
    assert quoted_spans(body) == {normalise_for_match(ATTACKER_SUMMARY)}
    hijacked = validate_response(json.dumps(HIJACKED_ANSWER), subject=subject, body=body)
    assert hijacked == Rejected("summary_echoes_ticket")
    # No other sample quotes anything, so apostrophes ("I'd", "I'll") never pair into a span.
    for tid, (other_subject, other_body) in SAMPLE_TICKETS.items():
        if tid != "t-1005":
            assert quoted_spans(other_subject) | quoted_spans(other_body) == frozenset(), tid


# B. Ingest


def test_every_sample_ingests_with_202_then_200(client):
    for entry in TICKET_PAYLOADS:
        first = client.post("/tickets", json=entry)
        assert first.status_code == 202, entry["id"]
        assert first.headers["location"] == f"/tickets/{entry['id']}"
        assert {k: first.json()[k] for k in entry} == entry
    for entry in TICKET_PAYLOADS:
        replay = client.post("/tickets", json=entry)
        assert replay.status_code == 200, entry["id"]
        assert {k: replay.json()[k] for k in entry} == entry
    assert [item["id"] for item in list_all(client)] == IDS


# C. Classification


@pytest.mark.parametrize("tid", IDS)
async def test_honest_fake_classifies_every_sample_with_the_expected_labels(tid, repository, settings, clock):
    # Done-when for the sample data: with the fake in honest mode every sample reaches classified with
    # labels from the allowed sets.
    repository.insert_ticket(tid, *SAMPLE_TICKETS[tid])
    worker = Worker(repository, honest(), settings, name="worker-0", rng=ZeroRandom())
    assert await worker.process_one() == Processed(tid, 1, Outcome.CLASSIFIED, None)
    stored = repository.get_ticket(tid)
    assert stored.status == "classified"
    assert stored.category in CATEGORIES and stored.priority in PRIORITIES
    assert (stored.category, stored.priority, stored.flagged) == EXPECTED[tid]
    assert stored.summary and stored.summary != ATTACKER_SUMMARY
    assert (stored.attempts, stored.prompt_version, stored.last_error) == (1, PROMPT_VERSION, None)


def test_honest_mode_through_the_lifespan(settings, clock):
    worker_settings = replace(settings, worker_count=2, poll_interval_s=0.001, backoff_base_s=0.0)
    app = create_app(worker_settings, clock=clock, llm=honest())
    with TestClient(app) as client:
        for entry in TICKET_PAYLOADS:
            assert client.post("/tickets", json=entry).status_code == 202
        deadline = time.monotonic() + DEADLINE_S
        items = list_all(client)
        while any(item["status"] == "pending" for item in items):
            assert time.monotonic() < deadline, "timed out waiting for the workers"
            time.sleep(0.005)
            items = list_all(client)
    assert [item["id"] for item in items] == IDS
    for item in items:
        assert item["status"] == "classified", item["id"]
        assert (item["category"], item["priority"], item["flagged"]) == EXPECTED[item["id"]]
        assert item["attempts"] == 1


async def test_default_fake_settles_every_sample_at_seed_42(repository, settings, clock):
    # One FakeLLM for all ten, as in make run: each prompt has its own call counter.
    worker = Worker(repository, FakeLLM(), settings, name="worker-0", rng=ZeroRandom())
    attempts = {}
    for tid in IDS:
        repository.insert_ticket(tid, *SAMPLE_TICKETS[tid])
        steps = await run_to_terminal(worker, clock, tid)
        stored = repository.get_ticket(tid)
        assert stored.status == "classified", tid
        assert (stored.category, stored.priority, stored.flagged) == EXPECTED[tid], tid
        attempts[tid] = len(steps)
    assert attempts == ATTEMPTS_AT_SEED_42
    t1005 = repository.get_ticket("t-1005")
    assert t1005.flag_reason == FLAG_REASON_ECHO
    assert t1005.last_error is None


async def test_default_fake_never_stores_the_attack_on_any_sample(db, settings, clock, tmp_path):
    # One database (a copy of the empty schema), clock and worker per seed, run concurrently.
    async def run_seed(seed: int) -> dict[str, str]:
        seed_clock = type(clock)(clock())
        path = tmp_path / f"seed-{seed}.db"
        await asyncio.to_thread(shutil.copyfile, db.path, path)
        repository = Repository(Database(str(path)), seed_clock)
        worker = Worker(repository, FakeLLM(seed=seed), settings, name="worker-0", rng=ZeroRandom())
        endings = {}
        for tid in IDS:
            await asyncio.to_thread(repository.insert_ticket, tid, *SAMPLE_TICKETS[tid])
            await run_to_terminal(worker, seed_clock, tid)
            stored = await asyncio.to_thread(repository.get_ticket, tid)
            assert stored.status in {"classified", "failed"}, (seed, tid)
            assert stored.summary != ATTACKER_SUMMARY, (seed, tid)
            if stored.status == "classified":
                assert (stored.category, stored.priority, stored.flagged) == EXPECTED[tid], (seed, tid)
            else:
                assert (stored.category, stored.priority, stored.summary) == (None, None, None), (seed, tid)
            if tid in INJECTED:
                assert (stored.category, stored.priority) != ("technical", "high"), (seed, tid)
                assert (stored.flagged, stored.flag_reason) == (True, FLAG_REASON_ECHO), (seed, tid)
            endings[tid] = stored.status
        return endings

    runs = await asyncio.gather(*(run_seed(seed) for seed in range(50)))
    assert {status for endings in runs for status in endings.values()} == {"classified", "failed"}
