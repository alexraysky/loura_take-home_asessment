import asyncio
import json
import logging

import pytest

from app.classifier import ERROR_CODES, LLM_ERROR_CODES, WORKER_ERROR_CODES, Attempt, classify_once
from app.llm import FakeLLM, LLMError, Mode
from app.llm import fake as fake_mod
from app.llm.fake import HIJACKED_ANSWER
from app.prompt import MAX_BODY_CHARS, PROMPT_VERSION
from app.validation import FLAG_REASON_ECHO, REJECTION_CODES, Accepted
from tests.sample_data import SAMPLE_TICKETS

T1001 = SAMPLE_TICKETS["t-1001"]
T1005 = SAMPLE_TICKETS["t-1005"]
TIMEOUT_S = 5.0


class SlowLLM:
    """Never answers in time; records whether the call was cancelled."""

    def __init__(self) -> None:
        self.cancelled = False

    async def complete(self, system: str, user: str) -> str:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")


class RaisingLLM:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    async def complete(self, system: str, user: str) -> str:
        await asyncio.sleep(0)
        raise self.exc


class AnswerLLM:
    """Returns a fixed answer."""

    def __init__(self, answer: object) -> None:
        self.answer = answer

    async def complete(self, system: str, user: str) -> object:
        await asyncio.sleep(0)
        return self.answer


class GateLLM:
    """Parks every call until released."""

    def __init__(self) -> None:
        self.parked = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, system: str, user: str) -> str:
        self.parked.set()
        await self.release.wait()
        raise AssertionError("never released in these tests")


async def test_ok_answer_is_accepted():
    attempt = await classify_once(FakeLLM(script=[Mode.OK]), *T1001, timeout_s=TIMEOUT_S)
    assert isinstance(attempt.accepted, Accepted)
    assert attempt.accepted.category == "billing"
    assert attempt.code is None
    assert (attempt.sets_flag, attempt.flag_reason, attempt.timed_out_by) == (False, None, None)
    assert attempt.prompt_version == PROMPT_VERSION == "v1"
    assert attempt.truncated is False


@pytest.mark.parametrize(
    "mode, code, timed_out_by",
    [
        (Mode.MALFORMED_JSON, "not_json", None),
        (Mode.PROSE_WRAPPED, "not_json", None),
        (Mode.OUT_OF_SET, "value_out_of_set", None),
        (Mode.MISSING_FIELD, "schema_violation", None),
        (Mode.TIMEOUT, "llm_timeout", "client"),
        (Mode.UNAVAILABLE, "llm_unavailable", None),
    ],
)
async def test_each_fake_mode_maps_to_its_code(mode, code, timed_out_by):
    attempt = await classify_once(FakeLLM(script=[mode]), *T1001, timeout_s=TIMEOUT_S)
    assert attempt.accepted is None
    assert attempt.code == code
    assert attempt.timed_out_by == timed_out_by
    assert (attempt.sets_flag, attempt.flag_reason) == (False, None)
    assert attempt.prompt_version == "v1"


async def test_hijacked_on_t1005_sets_the_sticky_flag():
    attempt = await classify_once(FakeLLM(script=[Mode.HIJACKED]), *T1005, timeout_s=TIMEOUT_S)
    assert attempt.accepted is None
    assert attempt.code == "summary_echoes_ticket"
    assert attempt.sets_flag is True
    assert attempt.flag_reason == FLAG_REASON_ECHO


async def test_hijacked_on_an_innocent_ticket_is_accepted():
    # The known hole (#5): a schema-valid hijacked answer that quotes nothing from the ticket passes.
    attempt = await classify_once(FakeLLM(script=[Mode.HIJACKED]), *T1001, timeout_s=TIMEOUT_S)
    assert attempt.accepted == Accepted("technical", "high", HIJACKED_ANSWER["summary"], False, None)


async def test_deadline_timeout_is_llm_timeout():
    slow = SlowLLM()
    attempt = await classify_once(slow, *T1001, timeout_s=0.01)
    assert attempt.code == "llm_timeout"
    assert attempt.timed_out_by == "deadline"
    assert slow.cancelled is True


async def test_builtin_timeout_error_from_the_client_is_a_client_timeout():
    # A client raising the built-in TimeoutError (a socket timeout, say) gave up itself; the deadline did not.
    attempt = await classify_once(RaisingLLM(TimeoutError("socket")), *T1001, timeout_s=TIMEOUT_S)
    assert attempt.code == "llm_timeout"
    assert attempt.timed_out_by == "client"


async def test_base_llm_error_is_llm_error():
    attempt = await classify_once(RaisingLLM(LLMError("x")), *T1001, timeout_s=TIMEOUT_S)
    assert attempt.code == "llm_error"
    assert attempt.timed_out_by is None


@pytest.mark.parametrize(
    "llm, exc_type",
    [(RaisingLLM(RuntimeError("boom")), RuntimeError), (AnswerLLM(b"{}"), TypeError)],
    ids=["client-bug", "bytes-output"],
)
async def test_unexpected_exceptions_propagate(llm, exc_type):
    with pytest.raises(exc_type):
        await classify_once(llm, *T1001, timeout_s=TIMEOUT_S)


async def test_exhausted_fake_script_propagates():
    fake = FakeLLM(script=[Mode.OK])
    await classify_once(fake, *T1001, timeout_s=TIMEOUT_S)
    with pytest.raises(RuntimeError, match="script exhausted"):
        await classify_once(fake, *T1001, timeout_s=TIMEOUT_S)


async def test_cancellation_propagates():
    gate = GateLLM()
    task = asyncio.create_task(classify_once(gate, *T1001, timeout_s=TIMEOUT_S))
    await asyncio.wait_for(gate.parked.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


async def test_each_attempt_uses_a_fresh_nonce_and_the_same_prompt_key():
    fake = FakeLLM(failure_rate=0.0, gullible=False)
    await classify_once(fake, *T1001, timeout_s=TIMEOUT_S)
    await classify_once(fake, *T1001, timeout_s=TIMEOUT_S)
    first, second = fake.calls
    assert first.user != second.user
    assert fake_mod._prompt_key(first.system, first.user) == fake_mod._prompt_key(second.system, second.user)


async def test_validation_reads_the_stored_text_not_the_prompt_copy():
    span = "please refund the whole order today"
    body = "x " * (MAX_BODY_CHARS // 2) + f'Earlier I wrote "{span}" to support.'
    answer = {
        "category": "billing",
        "priority": "low",
        "summary": f"Customer says {span}.",
        "flagged": False,
        "flag_reason": None,
    }
    attempt = await classify_once(AnswerLLM(json.dumps(answer)), "Refund", body, timeout_s=TIMEOUT_S)
    assert attempt.truncated is True
    # The quoted span is past the prompt's cut, but the check reads the stored body.
    assert attempt.code == "summary_echoes_ticket"
    assert attempt.sets_flag is True


def test_error_code_sets():
    assert len(ERROR_CODES) == 11
    assert len(set(ERROR_CODES)) == 11
    groups = [set(LLM_ERROR_CODES), set(REJECTION_CODES), set(WORKER_ERROR_CODES)]
    assert sum(len(g) for g in groups) == 11
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert set(REJECTION_CODES) <= set(ERROR_CODES)
    assert set(LLM_ERROR_CODES) == {"llm_timeout", "llm_unavailable", "llm_error"}
    assert set(WORKER_ERROR_CODES) == {"attempts_exhausted", "unexpected_error"}


ACCEPTED = Accepted("billing", "low", "Asks for a refund.", False, None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"accepted": ACCEPTED, "code": "not_json"},
        {"accepted": None, "code": None},
        {"accepted": None, "code": "no_such_code"},
        {"accepted": None, "code": "not_json", "sets_flag": True, "flag_reason": FLAG_REASON_ECHO},
        {"accepted": None, "code": "summary_echoes_ticket"},
        {"accepted": None, "code": "summary_echoes_ticket", "sets_flag": True, "flag_reason": None},
        {"accepted": None, "code": "not_json", "flag_reason": "why"},
        {"accepted": None, "code": "not_json", "timed_out_by": "client"},
        {"accepted": None, "code": "llm_timeout"},
        {"accepted": None, "code": "llm_timeout", "timed_out_by": "nobody"},
    ],
    ids=[
        "both",
        "neither",
        "unknown-code",
        "flag-without-echo",
        "echo-without-flag",
        "echo-without-reason",
        "reason-without-flag",
        "side-without-timeout",
        "timeout-without-side",
        "unknown-side",
    ],
)
def test_attempt_invariants(kwargs):
    values = {"sets_flag": False, "flag_reason": None, "prompt_version": "v1", "truncated": False, **kwargs}
    with pytest.raises(ValueError):
        Attempt(**values)


def test_valid_attempts_construct():
    Attempt(ACCEPTED, None, False, None, "v1", False)
    Attempt(None, "summary_echoes_ticket", True, FLAG_REASON_ECHO, "v1", True)
    Attempt(None, "llm_timeout", False, None, "v1", False, "deadline")
    Attempt(None, "unexpected_error", False, None, "v1", False)


async def test_classify_once_logs_nothing(caplog):
    caplog.set_level(logging.DEBUG)
    await classify_once(FakeLLM(script=[Mode.HIJACKED]), *T1005, timeout_s=TIMEOUT_S)
    await classify_once(SlowLLM(), *T1001, timeout_s=0.01)
    assert [r for r in caplog.records if r.name.startswith("app.classifier")] == []
