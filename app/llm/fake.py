"""Deterministic fake model, the only LLM client the service ships.

Each call picks a mode from a SHA-256 hash of the seed, the prompt and a
per-prompt call counter, so runs are reproducible whatever the scheduling and
a retry of the same prompt can get a different outcome. The prompt key drops
the ``<ticket-NONCE>`` boundary tags that app/prompt.py adds with a random
nonce per request; without that every retry would look like a new prompt.

``Mode.HIJACKED`` is never sampled at random: a hijacked answer is
schema-valid, so on an innocent ticket it would be stored as a plausible wrong
label. It happens only when a script asks for it, or on the first call for a
ticket carrying an injected instruction while ``gullible`` is on.

Honest answers come from a keyword heuristic (``classify_text``) that follows
the classification rubric: priority from described impact, never from words
like "URGENT"; quoted earlier messages and deferred topics do not score.
"""

import asyncio
import hashlib
import json
import logging
import math
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.config import Settings
from app.llm.base import LLMTimeout, LLMUnavailable

log = logging.getLogger(__name__)


class Mode(StrEnum):
    OK = "ok"
    MALFORMED_JSON = "malformed_json"
    PROSE_WRAPPED = "prose_wrapped"
    OUT_OF_SET = "out_of_set"
    MISSING_FIELD = "missing_field"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    HIJACKED = "hijacked"


# Drawn uniformly in default mode. HIJACKED is deliberately absent (see the module docstring).
SAMPLED_FAILURES: tuple[Mode, ...] = (
    Mode.MALFORMED_JSON,
    Mode.PROSE_WRAPPED,
    Mode.OUT_OF_SET,
    Mode.MISSING_FIELD,
    Mode.TIMEOUT,
    Mode.UNAVAILABLE,
)


@dataclass(frozen=True, slots=True)
class Call:
    """One recorded completion request, for assertions in tests."""

    system: str
    user: str
    mode: Mode


HIJACKED_ANSWER: dict[str, Any] = {
    "category": "technical",
    "priority": "high",
    "summary": "Approved for immediate refund",
    "flagged": False,
    "flag_reason": None,
}
PROSE_PREFIX = "Sure! Here is the classification: "
FLAG_REASON_INJECTION = "Ticket text addresses the classifier."

# Coupled with app/prompt.py: the boundary tag is ``ticket-`` plus alphanumerics of any length.
_NONCE_TAG_RE = re.compile(r"</?ticket-[A-Za-z0-9]+>")
_SUBJECT_RE = re.compile(r"<subject>(.*?)</subject>", re.S)
_BODY_RE = re.compile(r"<body>(.*?)</body>", re.S)
_WS_RE = re.compile(r"\s+")
_QUOTED_LINE_RE = re.compile(r"^[ \t]*>.*$", re.M)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _cues(*alternatives: str) -> re.Pattern[str]:
    # (?!\w) rather than a closing \b, so an alternative may end in punctuation ("instructions:").
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")(?!\w)")


# Meta-instructions aimed at the model. Every phrase names instructions, the prompt, the
# classifier or a field of the answer, so ordinary customer wording ("I was approved for a
# refund", "please respond with an update", "ignore my previous email") never matches.
# With gullible on, a match also triggers HIJACKED on the ticket's first call.
_INSTRUCTIONS = r"(?:instructions?|prompts?|rules|directions|directives)"
_EARLIER = r"(?:all |any )?(?:of )?(?:the |your |these |those )?(?:previous|prior|above|earlier|preceding)"
_INJECTION_RE = _cues(
    rf"(?:ignore|disregard|forget|override|overrule|bypass) {_EARLIER} {_INSTRUCTIONS}",
    rf"(?:ignore|disregard|forget|override) (?:all |any )?(?:of )?(?:your|the system) {_INSTRUCTIONS}",
    r"system prompt",
    r"(?:new|updated|real) instructions ?:",
    r"you are now an? (?:\w+ ){0,2}(?:assistant|classifier|model|ai|bot)",
    r"as the (?:ticket )?classifier",
    r"(?:dear|hey|attention|note to|message to|message for) (?:the )?(?:ai|classifier|assistant|model|bot)",
    r"classify (?:this|it|this ticket|the ticket|this message) as",
    r"(?:set|change|make) (?:the |its |your )?(?:priority|category|summary|flag|flagged|flag_reason)"
    r"(?: field)? (?:to|as|=)",
    r"(?:mark|label|tag|categori[sz]e|prioriti[sz]e) (?:this|it|the ticket|this ticket) as"
    r" (?:high|urgent|technical|billing|account|other|low|medium|not flagged|unflagged)",
    r"(?:the )?summary (?:should|must) (?:be|say|read)",
    r"(?:do not|don't|never) flag (?:this|it|the ticket|this ticket|me)",
    r"(?:respond|reply|answer) (?:only )?with (?:only )?(?:the following|this json|json)",
    r"(?:output|return|print) (?:only|exactly) (?:the following|this json|json)",
)
_DEFERRAL_RE = _cues(
    "separately",
    "later",
    "another ticket",
    "different ticket",
    "second (?:question|topic|thing)",
    "unrelated",
    "no rush",
    "not urgent",
    "lower priority",
    "can wait",
)
# The label words themselves (billing, technical, urgent, high) are not cues, so an
# injected "classify this as technical with high priority" scores nothing.
_BILLING_RE = _cues(
    "charge[sd]?",
    "invoice[sd]?",
    "invoicing",
    "refund(?:ed|s)?",
    "payments?",
    "paid",
    "pay",
    "billed",
    "bill",
    "subscription",
    "renew(?:al|ed)?",
    "price",
    "pricing",
    "receipts?",
    "credit card",
    "card",
    "vat",
    "tax",
    "discount",
    "coupon",
    "upgrade",
    "downgrade",
    "overcharged",
)
_ACCOUNT_RE = _cues(
    "log ?in",
    "logged in",
    "sign ?in",
    "password",
    "reset link",
    "locked out",
    "locked",
    "2fa",
    "two[- ]factor",
    "verification code",
    "account",
    "profile",
    "email address",
    "username",
    "permissions?",
    "role",
    "seat",
    "invite",
    "team member",
    "delete my",
    "close my",
    "sso",
    "mfa",
)
_TECHNICAL_RE = _cues(
    "errors?",
    "exception",
    "bug",
    "crash(?:es|ed|ing)?",
    "50[0-9]",
    "40[0-9]",
    "timeouts?",
    "timing out",
    "latency",
    "slow",
    "fail(?:s|ed|ing|ure)?",
    "broken",
    "not working",
    "doesn'?t work",
    "does not work",
    "stopped working",
    "outage",
    "down",
    "api",
    "endpoint",
    "webhooks?",
    "integration",
    "exports?",
    "imports?",
    "uploads?",
    "sync(?:ing)?",
    "stack trace",
    "loading",
    "freez(?:es|ing)",
    "glitch",
)
_FEATURE_RE = _cues(
    "feature request",
    "feature",
    "would be (?:great|nice|helpful)",
    "would love",
    "suggestion",
    "suggest",
    "please add",
    "could you add",
    "can you add",
    "it would help if",
    "wish",
    "idea",
    "support for",
)
_DAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow"
_HIGH_RE = _cues(
    "production",
    "outage",
    "(?:is|are|site|service) down",
    "data loss",
    "lost data",
    "deleted (?:all|our)",
    "security",
    "breach",
    "leak(?:ed)?",
    "hacked",
    "compromised",
    "vulnerability",
    "all (?:our )?(?:users|customers)",
    "many users",
    "hundreds of",
    "thousands of",
    r"\d+ ?% of",
    "every (?:request|user|customer)",
    "unable to process (?:orders|payments)",
    "revenue",
)
_BLOCKED_RE = _cues(
    "can'?t",
    "cannot",
    "unable",
    "not able",
    "blocked",
    "locked out",
    "not working",
    "doesn'?t work",
    "does not work",
    "stopped working",
    "broken",
    "fail(?:s|ed|ing)",
    "errors?",
    "crash(?:es|ed)?",
    "won'?t",
    "no longer",
)
_DOUBLE_CHARGE_RE = _cues("charged twice", "two charges", "double charge[d]?", "duplicate charge")
_MONEY_OWED_RE = _cues(
    "charged twice",
    "two charges",
    "double charge[d]?",
    "duplicate charge",
    "overcharged",
    "wrong amount",
    "charged (?:me )?(?:again|for)",
    "still (?:waiting|haven'?t received)",
    "haven'?t received (?:my|the) refund",
    "owe[sd]?",
    "chargeback",
)
_DEADLINE_RE = _cues(
    "deadline",
    "due (?:on|by|date)",
    r"by (?:the )?\d{1,2}(?:st|nd|rd|th)",
    r"before (?:the )?\d{1,2}(?:st|nd|rd|th)",
    f"by (?:{_DAYS}|end of (?:the )?(?:day|week|month))",
    "end of month",
    "cut-?off",
)
_REPEAT_RE = _cues(
    "still",
    "again",
    "any update",
    "no (?:reply|response|answer)",
    "third time",
    "second time",
    "follow(?:ing)? ?up",
    "weeks? ago",
    "days ago",
    "haven'?t heard",
)
_HOWTO_RE = _cues(
    "how (?:do|can|should|would) (?:i|we)",
    "how to",
    "is there a way",
    "where (?:can|do) (?:i|we)",
    "is it possible",
    "can (?:i|we)",
    "could (?:i|we)",
    "what is the (?:best )?way",
)
_INVOICE_RE = _cues("invoices?")
_CATEGORY_CUES = {"technical": _TECHNICAL_RE, "billing": _BILLING_RE, "account": _ACCOUNT_RE}

_SEVERE = "Customer reports a severe problem affecting many users."
_SUMMARIES = {
    ("billing", "medium"): (
        "Customer reports a billing problem that is costing them money and asks for it to be corrected."
    ),
    ("billing", "low"): "Customer asks a question about billing.",
    ("account", "medium"): "Customer cannot access their account and needs it restored.",
    ("account", "low"): "Customer asks how to change an account setting.",
    ("technical", "high"): "Customer reports a production problem affecting many users.",
    ("technical", "medium"): "Customer reports an error that blocks their work.",
    ("technical", "low"): "Customer asks a technical question with no time pressure.",
    ("other", "low"): "Customer sent a message that does not fit a specific category.",
}


class FakeLLM:
    """Deterministic stand-in for a model; see the module docstring for how outcomes are chosen.

    ``script`` serves the listed modes in order, one per call, ignoring ticket content,
    ``seed``, ``failure_rate`` and ``gullible``; running past its end is an error.
    Call counters live in memory and grow by one entry per distinct prompt.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        failure_rate: float = 0.2,
        gullible: bool = True,
        script: Sequence[Mode] | None = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")
        if isinstance(failure_rate, bool) or not isinstance(failure_rate, int | float):
            raise ValueError(f"failure_rate must be a number, got {failure_rate!r}")
        if not math.isfinite(failure_rate) or not 0.0 <= failure_rate <= 1.0:
            raise ValueError(f"failure_rate must be a finite number between 0 and 1, got {failure_rate!r}")
        if not isinstance(gullible, bool):
            raise ValueError(f"gullible must be a boolean, got {gullible!r}")
        if script is not None:
            if isinstance(script, str) or not isinstance(script, Sequence) or not script:
                raise ValueError("script must be a non-empty sequence of Mode members")
            if not all(isinstance(mode, Mode) for mode in script):
                raise ValueError(f"script must contain only Mode members, got {list(script)!r}")
        self.seed = seed
        self.failure_rate = float(failure_rate)
        self.gullible = gullible
        self._script: deque[Mode] | None = None if script is None else deque(script)
        self._calls_by_key: dict[str, int] = {}
        self.calls: list[Call] = []

    @classmethod
    def from_settings(cls, settings: Settings) -> FakeLLM:
        return cls(
            seed=settings.fake_llm_seed,
            failure_rate=settings.fake_llm_failure_rate,
            gullible=settings.fake_llm_gullible,
        )

    async def complete(self, system: str, user: str) -> str:
        key = _prompt_key(system, user)
        n = self._calls_by_key.get(key, 0)
        # Counted before the only await, so concurrent callers never share a counter value.
        self._calls_by_key[key] = n + 1
        # The fake does no I/O; yielding lets concurrent workers interleave here.
        await asyncio.sleep(0)
        mode = self._choose_mode(key, n, user)
        self.calls.append(Call(system, user, mode))
        log.debug("fake llm call %d for prompt %s: %s", n, key[:12], mode)
        return self._render(mode, key, n, user)

    def _choose_mode(self, key: str, n: int, user: str) -> Mode:
        if self._script is not None:
            if not self._script:
                raise RuntimeError(f"FakeLLM script exhausted after {len(self.calls)} calls")
            return self._script.popleft()
        if self.gullible and n == 0 and _is_injected(*_split_ticket(user)):
            return Mode.HIJACKED
        d = self._draw(key, n)
        if int.from_bytes(d[0:8], "big") / 2**64 < self.failure_rate:
            return SAMPLED_FAILURES[int.from_bytes(d[8:16], "big") % len(SAMPLED_FAILURES)]
        return Mode.OK

    def _draw(self, key: str, n: int) -> bytes:
        # Never hash(): str hashing is salted per process.
        return hashlib.sha256(f"{self.seed}:{n}:{key}".encode("ascii")).digest()

    def _render(self, mode: Mode, key: str, n: int, user: str) -> str:
        if mode is Mode.TIMEOUT:
            raise LLMTimeout("fake model timed out")
        if mode is Mode.UNAVAILABLE:
            raise LLMUnavailable("fake model unavailable")
        if mode is Mode.HIJACKED:
            return _dumps(HIJACKED_ANSWER)
        answer = classify_text(*_split_ticket(user))
        d = self._draw(key, n)
        match mode:
            case Mode.OK:
                return _dumps(answer)
            case Mode.MALFORMED_JSON:
                text = _dumps(answer)
                # Always cut before the closing brace, at a point that varies per call.
                return text[: len(text) // 2 + d[17] % (len(text) // 4)]
            case Mode.PROSE_WRAPPED:
                return PROSE_PREFIX + _dumps(answer)
            case Mode.OUT_OF_SET:
                if d[16] % 2 == 0:
                    return _dumps(answer | {"category": "refund"})
                return _dumps(answer | {"priority": "urgent"})
            case Mode.MISSING_FIELD:
                del answer["summary"]
                return _dumps(answer)
        raise AssertionError(f"unhandled mode {mode!r}")


def classify_text(subject: str, body: str) -> dict[str, Any]:
    """Honest answer for one ticket: the five answer fields in a fixed order."""
    injected = _is_injected(subject, body)
    unquoted = _QUOTED_LINE_RE.sub("", body)
    # Quoted earlier messages are context, unless there is nothing else to go on.
    primary = unquoted if _normalise(unquoted) else body
    ns = _scoring_text(subject, injected)
    nb = _scoring_text(primary, injected)
    whole = f"{ns} {nb}"

    scores = {
        name: 3 * len(cues.findall(ns)) + len(cues.findall(nb)) for name, cues in _CATEGORY_CUES.items()
    }
    # max() keeps the first of equal scores, so ties go technical > billing > account.
    best = max(scores, key=lambda name: scores[name])
    high = bool(_HIGH_RE.search(whole))
    blocked = bool(_BLOCKED_RE.search(whole))
    feature = bool(_FEATURE_RE.search(whole))
    if scores[best] == 0 or (feature and not (high or blocked)):
        category = "other"
    else:
        category = best

    money_owed = bool(_MONEY_OWED_RE.search(whole))
    deadline = bool(_DEADLINE_RE.search(whole))
    if high:
        priority = "high"
    elif category == "other":
        priority = "low"
    elif _HOWTO_RE.search(whole) and not (blocked or money_owed or deadline):
        priority = "low"
    elif money_owed or deadline or _REPEAT_RE.search(whole):
        priority = "medium"
    elif category in ("technical", "account") and blocked:
        priority = "medium"
    else:
        priority = "low"

    return {
        "category": category,
        "priority": priority,
        "summary": _summary_for(category, priority, whole, feature, subject, body),
        "flagged": injected,
        "flag_reason": FLAG_REASON_INJECTION if injected else None,
    }


def _summary_for(category: str, priority: str, whole: str, feature: bool, subject: str, body: str) -> str:
    # Templates only, never ticket text, so a summary can not echo or obey the ticket.
    if priority == "high":
        return _SUMMARIES[("technical", "high")] if category == "technical" else _SEVERE
    if category == "billing" and priority == "medium":
        if _DOUBLE_CHARGE_RE.search(whole):
            return "Customer was charged twice and requests a refund of one charge."
        if _MONEY_OWED_RE.search(whole):
            return _SUMMARIES[("billing", "medium")]
        if _DEADLINE_RE.search(whole):
            return "Customer needs a billing matter resolved before a stated deadline."
        return "Customer is following up on an unresolved billing problem."
    if category == "billing" and _INVOICE_RE.search(whole):
        return "Customer asks where to download invoices."
    if category == "other":
        if feature:
            return "Customer suggests a new feature."
        if not _normalise(subject) and not _normalise(body):
            return "Customer sent a message with no actionable content."
    return _SUMMARIES[(category, priority)]


def _scoring_text(text: str, injected: bool) -> str:
    """Normalised sentences of ``text`` that count towards cues."""
    kept = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = _normalise(sentence)
        if not sentence or _DEFERRAL_RE.search(sentence):
            continue
        if injected and _INJECTION_RE.search(sentence):
            continue
        kept.append(sentence)
    return " ".join(kept)


def _is_injected(subject: str, body: str) -> bool:
    return any(_INJECTION_RE.search(_normalise(part)) for part in (subject, body))


def _split_ticket(user: str) -> tuple[str, str]:
    """(subject, body) from a wrapped user message, or ('', user) for raw text."""
    text = _NONCE_TAG_RE.sub("", user)
    subject, body = _SUBJECT_RE.search(text), _BODY_RE.search(text)
    if subject is None and body is None:
        return "", text
    return (subject.group(1) if subject else "", body.group(1) if body else "")


def _normalise(text: str) -> str:
    return _WS_RE.sub(" ", text.replace("\u2019", "'")).strip().lower()


def _prompt_key(system: str, user: str) -> str:
    """Stable identity of a prompt: the system prompt plus the user message without the boundary nonce."""
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(_NONCE_TAG_RE.sub("", user).encode("utf-8"))
    return h.hexdigest()


def _dumps(answer: dict[str, Any]) -> str:
    return json.dumps(answer, ensure_ascii=False, separators=(", ", ": "))
