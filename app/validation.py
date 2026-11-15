"""Model text to a typed result or a typed rejection.

Rules run in a fixed order and the first failure wins. Every rule is bounded:
the size check runs before anything touches the text, so parsing and the
summary rules only ever see at most 4,096 bytes, and the quote check is linear
in the ticket text. The function is total over ``str``: it never raises for any
model output, and it never logs or returns the model's text on rejection.
"""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal, NoReturn, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.models import Category, Priority

log = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 4096
MAX_SUMMARY_CHARS = 200
MAX_FLAG_REASON_CHARS = 200
MIN_QUOTE_WORDS = 3
FLAG_REASON_ECHO = "summary echoed quoted ticket text"

RejectionCode = Literal[
    "response_too_large",
    "not_json",
    "schema_violation",
    "value_out_of_set",
    "summary_invalid",
    "summary_echoes_ticket",
]
REJECTION_CODES: tuple[RejectionCode, ...] = get_args(RejectionCode)


@dataclass(frozen=True, slots=True)
class Accepted:
    """A classification that passed every rule. ``summary`` is the trimmed text."""

    category: Category
    priority: Priority
    summary: str
    flagged: bool
    flag_reason: str | None


@dataclass(frozen=True, slots=True)
class Rejected:
    """Why the response was refused. ``code`` is what ``last_error`` stores; the text is dropped."""

    code: RejectionCode
    sets_flag: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.code not in REJECTION_CODES:
            raise ValueError(f"unknown rejection code {self.code!r}")
        object.__setattr__(self, "sets_flag", self.code == "summary_echoes_ticket")

    @property
    def flag_reason(self) -> str | None:
        return FLAG_REASON_ECHO if self.sets_flag else None


ValidationResult = Accepted | Rejected


class _ModelOutput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    category: Category
    priority: Priority
    # No length constraint: a bad summary is rule 4's summary_invalid, not a schema violation.
    summary: str
    flagged: bool
    # Required key, nullable.
    flag_reason: str | None = Field(max_length=MAX_FLAG_REASON_CHARS)


# A terminator, optional closing quotes or brackets, whitespace, then more text.
_MULTI_SENTENCE_RE = re.compile(r"[.!?]+[\"'”’)\]]*\s+\S")
_QUOTE_CHARS = "\"“”„«»"
# Any quote pairs with the next quote of any kind; possessive, so an unclosed quote never backtracks.
_QUOTED_RE = re.compile(f"[{_QUOTE_CHARS}]([^{_QUOTE_CHARS}]++)[{_QUOTE_CHARS}]")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\w")
# Stripped from both ends of a normalised span; spaces included, so "x y z ." loses the " .".
_SPAN_EDGE = " .,;:!?'‘’" + _QUOTE_CHARS


def normalise_for_match(text: str) -> str:
    """Whitespace runs collapsed to one space, trimmed and casefolded."""
    return _WS_RE.sub(" ", text).strip().casefold()


def quoted_spans(text: str, *, min_words: int = MIN_QUOTE_WORDS) -> frozenset[str]:
    """Normalised double-quoted spans of at least ``min_words`` words that could fit in a summary."""
    spans = set()
    for m in _QUOTED_RE.finditer(text):
        span = normalise_for_match(m.group(1)).strip(_SPAN_EDGE)
        if len(span) > MAX_SUMMARY_CHARS:
            continue
        if sum(1 for word in span.split(" ") if _WORD_RE.search(word)) >= min_words:
            spans.add(span)
    return frozenset(spans)


def validate_response(text: str, *, subject: str, body: str) -> ValidationResult:
    """Check model ``text`` against the stored ticket ``subject`` and ``body``."""
    if not isinstance(text, str):
        raise TypeError(f"model output must be str, got {type(text).__name__}")
    result = _validate(text, subject, body)
    if isinstance(result, Rejected) and log.isEnabledFor(logging.DEBUG):
        log.debug("model output rejected: %s (%d bytes)", result.code, _utf8_len(text))
    return result


def _validate(text: str, subject: str, body: str) -> ValidationResult:
    # Every character is at least one byte, so a long string is rejected without encoding it.
    if len(text) > MAX_RESPONSE_BYTES or _utf8_len(text) > MAX_RESPONSE_BYTES:
        return Rejected("response_too_large")

    data = _parse_object(text)
    if data is None:
        return Rejected("not_json")

    try:
        out = _ModelOutput.model_validate(data)
    except ValidationError as exc:
        kinds = {e["type"] for e in exc.errors(include_url=False, include_context=False, include_input=False)}
        return Rejected("value_out_of_set" if kinds == {"literal_error"} else "schema_violation")

    flag_reason = None if out.flag_reason is None else out.flag_reason.strip()
    if flag_reason is not None and not _is_clean_line(flag_reason):
        return Rejected("schema_violation")

    summary = out.summary.strip()
    if (
        not 1 <= len(summary) <= MAX_SUMMARY_CHARS
        or not _is_clean_line(summary)
        or _MULTI_SENTENCE_RE.search(summary)
    ):
        return Rejected("summary_invalid")

    haystack = normalise_for_match(summary)
    if any(span in haystack for source in (subject, body) for span in quoted_spans(source)):
        return Rejected("summary_echoes_ticket")

    return Accepted(
        category=out.category,
        priority=out.priority,
        summary=summary,
        flagged=out.flagged,
        flag_reason=flag_reason or None,
    )


def _utf8_len(text: str) -> int:
    # surrogatepass: a lone surrogate is measured, not raised on.
    return len(text.encode("utf-8", "surrogatepass"))


def _parse_object(text: str) -> dict[str, Any] | None:
    """The top-level JSON object, or None when the text is anything else."""
    try:
        data = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_keys)
    except (ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"non-JSON constant {name}")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate key")
    return obj


def _is_clean_line(text: str) -> bool:
    """No line break and no Unicode ``C*`` character (controls, format, surrogates, private use, unassigned)."""
    return len(text.splitlines()) <= 1 and not any(unicodedata.category(ch)[0] == "C" for ch in text)


__all__ = [
    "FLAG_REASON_ECHO",
    "MAX_FLAG_REASON_CHARS",
    "MAX_RESPONSE_BYTES",
    "MAX_SUMMARY_CHARS",
    "MIN_QUOTE_WORDS",
    "REJECTION_CODES",
    "Accepted",
    "Rejected",
    "RejectionCode",
    "ValidationResult",
    "normalise_for_match",
    "quoted_spans",
    "validate_response",
]
