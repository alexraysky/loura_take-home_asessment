"""One classification attempt: prompt, model call under a deadline, validation, error code.

No database, no clock, no sleeping, no retry: the worker owns those. Nothing in here keeps ticket
text or model text beyond the call, and nothing is logged.
"""

import asyncio
from dataclasses import dataclass
from typing import Literal

from app.llm.base import LLMClient, LLMError, LLMTimeout, LLMUnavailable
from app.prompt import Prompt, build_prompt
from app.validation import FLAG_REASON_ECHO, REJECTION_CODES, Accepted, Rejected, validate_response

LLM_ERROR_CODES: tuple[str, ...] = ("llm_timeout", "llm_unavailable", "llm_error")
WORKER_ERROR_CODES: tuple[str, ...] = ("attempts_exhausted", "unexpected_error")
# Every value last_error can ever hold.
ERROR_CODES: tuple[str, ...] = LLM_ERROR_CODES + REJECTION_CODES + WORKER_ERROR_CODES

TimedOutBy = Literal["client", "deadline"]


@dataclass(frozen=True, slots=True)
class Attempt:
    """What one call produced. Exactly one of ``accepted`` and ``code`` is set."""

    accepted: Accepted | None
    code: str | None  # one of ERROR_CODES when the attempt was not accepted
    sets_flag: bool  # True only for summary_echoes_ticket
    flag_reason: str | None  # FLAG_REASON_ECHO when sets_flag, else None
    prompt_version: str  # Prompt.version; the worker stores it on acceptance
    truncated: bool  # Prompt.truncated; the worker logs it
    timed_out_by: TimedOutBy | None = None  # which side gave up on an llm_timeout, for the log line only

    def __post_init__(self) -> None:
        if (self.accepted is None) == (self.code is None):
            raise ValueError("exactly one of accepted and code must be set")
        if self.code is not None and self.code not in ERROR_CODES:
            raise ValueError(f"unknown error code {self.code!r}")
        if self.sets_flag != (self.code == "summary_echoes_ticket"):
            raise ValueError("sets_flag goes with summary_echoes_ticket and nothing else")
        if self.flag_reason != (FLAG_REASON_ECHO if self.sets_flag else None):
            raise ValueError("flag_reason is the echo reason exactly when sets_flag")
        if (self.timed_out_by is not None) != (self.code == "llm_timeout"):
            raise ValueError("timed_out_by goes with llm_timeout and nothing else")
        if self.timed_out_by not in (None, "client", "deadline"):
            raise ValueError(f"unknown timeout side {self.timed_out_by!r}")


async def classify_once(llm: LLMClient, subject: str, body: str, *, timeout_s: float) -> Attempt:
    """Build a fresh prompt, call ``llm`` under ``timeout_s`` and validate against the stored text.

    Any exception other than an ``LLMError`` or the deadline (a client bug, non-``str`` output,
    a prompt builder error) propagates, and so does cancellation.
    """
    prompt = build_prompt(subject, body)  # a fresh nonce per attempt
    client_timed_out = False

    async def call() -> str:
        # wait_for raises the same TimeoutError when its deadline expires, so note who raised it.
        nonlocal client_timed_out
        try:
            return await llm.complete(prompt.system, prompt.user)
        except TimeoutError:
            client_timed_out = True
            raise

    try:
        text = await asyncio.wait_for(call(), timeout_s)
    except LLMTimeout:
        return _error("llm_timeout", prompt, timed_out_by="client")
    except TimeoutError:
        return _error("llm_timeout", prompt, timed_out_by="client" if client_timed_out else "deadline")
    except LLMUnavailable:
        return _error("llm_unavailable", prompt)
    except LLMError:
        return _error("llm_error", prompt)
    # The stored subject and body, not the cleaned and cut prompt copy.
    result = validate_response(text, subject=subject, body=body)
    if isinstance(result, Rejected):
        return Attempt(None, result.code, result.sets_flag, result.flag_reason, prompt.version, prompt.truncated)
    return Attempt(result, None, False, None, prompt.version, prompt.truncated)


def _error(code: str, prompt: Prompt, *, timed_out_by: TimedOutBy | None = None) -> Attempt:
    return Attempt(None, code, False, None, prompt.version, prompt.truncated, timed_out_by)
