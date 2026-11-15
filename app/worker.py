"""Classification workers: claim one ticket, run one attempt, write the result, repeat.

The tickets table is the queue. Every repository call runs in a thread (``asyncio.to_thread``)
so SQLite never blocks the event loop. ``process_one`` is the whole loop body, so tests drive it
step by step with a scripted fake and a fake clock.

Log lines carry the worker name, the ticket id, the attempt number and codes; never the subject,
the body, the prompt or the model's text.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from enum import StrEnum

from app.classifier import Attempt, classify_once
from app.config import Settings
from app.llm.base import LLMClient
from app.repository import Repository

log = logging.getLogger("app.worker")

_RNG = random.Random()


class Outcome(StrEnum):
    CLASSIFIED = "classified"
    RETRY = "retry"
    FAILED = "failed"
    STALE = "stale"  # the claim was lost (lease expired, another worker took over); nothing was written


@dataclass(frozen=True, slots=True)
class Processed:
    """What ``process_one`` did with the ticket it claimed."""

    ticket_id: str
    attempt: int  # attempts after the claim, so 1 on the first try
    outcome: Outcome
    code: str | None  # the last_error the attempt produced; None when accepted


def compute_backoff_s(backoff_base_s: float, attempts: int, rng: random.Random = _RNG) -> float:
    """``backoff_base_s * 2^(attempts - 1)`` plus up to ``backoff_base_s`` of jitter."""
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    return backoff_base_s * 2 ** (attempts - 1) + rng.random() * backoff_base_s


class Worker:
    def __init__(
        self,
        repository: Repository,
        llm: LLMClient,
        settings: Settings,
        *,
        name: str = "worker-0",
        rng: random.Random | None = None,
    ) -> None:
        self.name = name
        self._repo = repository
        self._llm = llm
        self._settings = settings
        self._rng = random.Random() if rng is None else rng

    async def process_one(self) -> Processed | None:
        """Claim one ticket and run one attempt on it. None when nothing was claimable.

        A model or validation exception becomes ``unexpected_error`` here; a repository exception
        propagates to ``run``. Cancellation always propagates.
        """
        s = self._settings
        ticket = await asyncio.to_thread(self._repo.claim_next, lease_timeout_s=s.lease_timeout_s)
        if ticket is None:
            log.debug("%s: nothing to claim", self.name)
            return None
        started = time.perf_counter()
        token, n = ticket.claimed_at, ticket.attempts
        if n > s.max_attempts:
            # A previous attempt died holding the last allowed claim: give up without calling the model.
            ok = await asyncio.to_thread(self._repo.finish_failed, ticket.id, token=token, code="attempts_exhausted")
            return self._done(ticket.id, n, Outcome.FAILED, ok, "attempts_exhausted", None, started)

        attempt: Attempt | None
        try:
            attempt = await classify_once(self._llm, ticket.subject, ticket.body, timeout_s=s.llm_timeout_s)
        except Exception:
            log.exception("%s: %s attempt %d/%d raised", self.name, ticket.id, n, s.max_attempts)
            attempt = None

        if attempt is not None and attempt.accepted is not None:
            ok = await asyncio.to_thread(
                self._repo.finish_classified,
                ticket.id,
                token=token,
                result=attempt.accepted,
                prompt_version=attempt.prompt_version,
            )
            return self._done(ticket.id, n, Outcome.CLASSIFIED, ok, None, attempt, started)

        code = "unexpected_error" if attempt is None else attempt.code
        sets_flag = attempt is not None and attempt.sets_flag
        flag_reason = None if attempt is None else attempt.flag_reason
        if n >= s.max_attempts:
            ok = await asyncio.to_thread(
                self._repo.finish_failed,
                ticket.id,
                token=token,
                code=code,
                sets_flag=sets_flag,
                flag_reason=flag_reason,
            )
            return self._done(ticket.id, n, Outcome.FAILED, ok, code, attempt, started)
        ok = await asyncio.to_thread(
            self._repo.finish_retry,
            ticket.id,
            token=token,
            code=code,
            delay_s=compute_backoff_s(s.backoff_base_s, n, self._rng),
            sets_flag=sets_flag,
            flag_reason=flag_reason,
        )
        return self._done(ticket.id, n, Outcome.RETRY, ok, code, attempt, started)

    async def run(self) -> None:
        """Process tickets until cancelled; sleep ``POLL_INTERVAL_S`` whenever the queue is empty."""
        log.info("%s started", self.name)
        try:
            while True:
                try:
                    processed = await self.process_one()
                except Exception:
                    # sqlite3.Error (e.g. "database is locked") and anything else; CancelledError passes.
                    # A ticket whose finish write failed keeps its claim and is recovered by lease expiry.
                    log.exception("%s: iteration failed", self.name)
                    processed = None
                if processed is None:
                    await asyncio.sleep(self._settings.poll_interval_s)
        finally:
            log.info("%s stopped", self.name)

    def _done(
        self,
        ticket_id: str,
        n: int,
        outcome: Outcome,
        written: bool,
        code: str | None,
        attempt: Attempt | None,
        started: float,
    ) -> Processed:
        result = f"{outcome} ({code})" if code else str(outcome)
        if attempt is not None and attempt.timed_out_by is not None:
            result = f"{outcome} ({code}, {attempt.timed_out_by})"
        if not written:
            result = f"stale: claim lost before {result}"
        level = logging.WARNING if not written or code == "attempts_exhausted" else logging.INFO
        log.log(
            level,
            "%s: %s attempt %d/%d -> %s prompt=%s truncated=%s %.3fs",
            self.name,
            ticket_id,
            n,
            self._settings.max_attempts,
            result,
            "-" if attempt is None else attempt.prompt_version,
            "-" if attempt is None else str(attempt.truncated).lower(),
            time.perf_counter() - started,
        )
        return Processed(ticket_id, n, outcome if written else Outcome.STALE, code)


class WorkerPool:
    """``settings.worker_count`` worker tasks sharing one repository, client and jitter source."""

    def __init__(
        self,
        repository: Repository,
        llm: LLMClient,
        settings: Settings,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self._repo = repository
        self._llm = llm
        self._settings = settings
        # One generator for all workers: they run on one event loop, so access is serialised.
        self._rng = random.Random() if rng is None else rng
        self.tasks: list[asyncio.Task[None]] = []

    def __len__(self) -> int:
        return len(self.tasks)

    async def start(self) -> None:
        if self.tasks:
            raise RuntimeError("worker pool is already running")
        count = self._settings.worker_count
        if count == 0:
            log.info("workers disabled (WORKER_COUNT=0)")
            return
        log.info("starting %d workers", count)
        for i in range(count):
            worker = Worker(self._repo, self._llm, self._settings, name=f"worker-{i}", rng=self._rng)
            self.tasks.append(asyncio.create_task(worker.run(), name=worker.name))

    async def stop(self) -> None:
        """Cancel every task and wait for it. In-flight tickets keep their claim. Safe to call twice."""
        if not self.tasks:
            return
        tasks, self.tasks = self.tasks, []
        for task in tasks:
            task.cancel()
        # return_exceptions: a worker that died some other way cannot abort shutdown.
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("workers stopped")
