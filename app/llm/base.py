"""Model boundary: the client protocol and the errors a client may raise.

The client returns raw text on purpose. Turning text into a typed result is
validation's job (app/validation.py), so a client never parses JSON.

Timeouts are enforced on two sides. A client may raise ``LLMTimeout`` when it
gives up on its own deadline (an HTTP read timeout, or the fake simulating
one without sleeping). The worker also wraps every call in
``asyncio.wait_for``, which raises the built-in ``TimeoutError`` and cancels
the call; a client that holds a connection must release it on cancellation.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """One completion call: a constant system prompt plus one user message, returning the model's text."""

    async def complete(self, system: str, user: str) -> str: ...


class LLMError(Exception):
    """Base class for errors raised by an LLM client."""


class LLMTimeout(LLMError):
    """The client gave up waiting for the model.

    Deliberately not a ``TimeoutError``: the worker catches both and can tell
    which side gave up.
    """


class LLMUnavailable(LLMError):
    """The model could not be reached or refused the request (network error, 5xx, rate limit)."""
