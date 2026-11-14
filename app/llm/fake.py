"""Deterministic fake model, the only LLM client the service ships.

Each call picks a mode from a SHA-256 hash of the seed, the prompt and a
per-prompt call counter, so runs are reproducible whatever the scheduling and
a retry of the same prompt can get a different outcome. The prompt key drops
the outer ``<ticket-NONCE>`` boundary tags that app/prompt.py adds with a random
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
# Only the outermost pair counts, closed with the same nonce; tags inside the ticket are text.
_WRAPPER_RE = re.compile(r"\s*<ticket-([A-Za-z0-9]+)>(.*)</ticket-\1>\s*", re.S)
_FIELDS_RE = re.compile(r"\s*<subject>(.*?)</subject>\s*<body>(.*)</body>\s*", re.S)
_WS_RE = re.compile(r"\s+")
_QUOTED_LINE_RE = re.compile(r"^[ \t]*>.*$", re.M)
# A single line break is a soft wrap, not a sentence end; a blank line is.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


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
# A sentence that puts its topic off to another ticket does not score. Bare "separately" is
# ordinary wording ("charged separately for each seat"), so only topic-deferring uses count.
_DEFERRAL_RE = _cues(
    r"(?:^|(?<=[.!?] ))separately",
    "separately,",
    "(?:also|oh),? separately",
    r"(?:ask|write|email|contact|follow up|raise|open|send|handle|deal with)(?: [\w']+){0,4} separately",
    "(?:on|in) a separate (?:ticket|topic|note|email|message|thread)",
    "another ticket",
    "different ticket",
    "second (?:question|topic|thing)",
    "unrelated",
)
# Low-urgency wording may sit on the only request in the ticket ("No rush, where can I download
# my invoices?"). Bare "later" is mostly temporal ("Two days later I was charged again").
_LOW_URGENCY_RE = _cues(
    "later on",
    "(?:for|until) later",
    "(?:handle|check|look at|do|resolve|revisit|ask about|deal with) (?:this|that|it) later",
    "no rush",
    "not urgent",
    "lower priority",
    "can wait",
)
# "Down" alone is also "down payment", and "our usage is down" is a business metric, so only a
# system that is down counts as an outage.
_DOWN = (
    "(?:(?:site|service|system|app|api|server|db|database|dashboard|endpoint|production|platform"
    "|website)(?: is| are| was| were| has been| have been| went| goes)?(?: completely| totally| still)? down)"
)


def _status(codes: str) -> str:
    # A bare number is a seat count or a price ("520 seats", "$550"); a status code is named as one.
    return (
        rf"(?:(?:https?|status|error|code|returns?|returning|returned|throws?|throwing)"
        rf"(?: an?| the)?(?: http| status| error)?(?: code)? {codes}"
        rf"|(?:got|getting|get) an? {codes}"
        rf"|(?:fail(?:s|ed|ing)?|crash(?:es|ed|ing)?|errors?) with (?:an? )?{codes}"
        rf"|{codes}(?:s|(?: http)?(?: errors?| status| responses?| codes?)))"
    )


# Error tooling ("error logs", "error notifications") is a topic, not a failure.
_ERROR = r"errors?(?! (?:logs?|notifications?|alerts?|reporting|tracking|monitoring|handling|pages?))"
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
    _status(r"[45]\d\d"),
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
    _DOWN,
    "data loss",
    "lost data",
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
    # "Would love help with this." asks for support, not a feature.
    r"would love(?! (?:some |your |any )?(?:help|assistance|support|guidance|advice|to hear))",
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
# A question about a disclosure or bounty programme is not a report of a vulnerability.
_VULN = (
    r"vulnerabilit(?:y|ies)"
    r"(?! (?:disclosure|bounty|program|programme|policy|reporting|scans?|management))"
)
_DAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow"
# Impact that makes a ticket high on its own.
_INCIDENT_RE = _cues(
    "outage",
    _DOWN,
    "production (?:is )?(?:outage|down|incident|failure|crash(?:es|ed)?|broken|issues?|errors?)",
    "data loss",
    "lost data",
    "deleted (?:all|our)",
    "security (?:breach|incident|issue|hole|flaw|problem)",
    "breach(?:ed)?",
    "(?:data|security|credentials?|passwords?) leak(?:ed|s)?",
    "leaked",
    "hacked",
    "compromised",
    "exploited",
    "unauthori[sz]ed access",
    "unable to process (?:orders|payments)",
    "losing (?:revenue|money|sales|orders)",
    "lost (?:revenue|sales|orders)",
    "revenue loss",
)
# Preventive or hypothetical wording turns impact words into a question about them: "How can I
# prevent data loss?", "What happens if all users are locked out?", "We want to avoid an outage."
# Every impact check goes through _states, so the cue applies to incidents, scale, blockers,
# degradation and vulnerabilities alike.
_HYPOTHETICAL_RE = _cues(
    "avoid(?:s|ed|ing)?",
    "prevent(?:s|ed|ing)?",
    "(?:protect|guard) against",
    "prepare for",
    "plan for",
    "in case",
    "in the event",
    "risk of",
    "if",
    "whether",
    "suppose",
    "what (?:happens|would happen)",
)
# The cue counts only in the same clause, before the impact words: "We tried to prevent it, but
# now the API is down." still reports an outage.
_CLAUSE_SPLIT_RE = re.compile(r"[,;:]| but ")
# A reported vulnerability is high, but "Is there a vulnerability I should know about?" and "How do
# I report a vulnerability?" are questions; a question is high only when it names an exploit.
_VULN_RE = _cues(
    rf"(?:security|critical|severe|serious|zero-day|0-day|unpatched) {_VULN}",
    rf"(?:found|discovered|identified|reporting|exploited|exploitable)"
    rf" (?:a |an |the |this )?(?:\w+ ){{0,2}}{_VULN}",
    rf"{_VULN} (?:in|on) (?:your|the|our|prod|production)",
)
_QUESTION_RE = _cues("is there", "are there")
_EXPLOIT_RE = _cues(
    "exploit(?:s|ed|able|ing)?",
    "expos(?:es|ed|ing)",
    "attackers?",
    "proof of concept",
    "disclos(?:e|ed|ing)",
    "being used",
    "leak(?:s|ed|ing)?",
    "unauthori[sz]ed",
    "other (?:users|customers|accounts|workspaces|tenants)",
)
# Wide-scale wording is high only next to a problem in the same sentence: "How do I export data
# for all users?" is a how-to question, "30% of requests are timing out" is an incident.
_SCALE_RE = _cues(
    "all (?:our )?(?:users|customers)",
    "many users",
    "hundreds of",
    "thousands of",
    r"\d+ ?% of",
    "every (?:request|user|customer)",
)
# Production is high only next to a real failure in the same sentence: "I can't find the setting
# in production" and "How can I see error logs in prod?" are how-to questions, "the API returns
# 500 errors in production" is an incident. Only verb forms count, never the noun "error".
_PRODUCTION_RE = _cues("production", "prod")
_FAILURE_RE = _cues(
    "outage",
    _DOWN,
    _status(r"5\d\d"),
    r"(?:throws?|throwing|returns?|returning|returned|getting|got) (?:an? |the )?" + _ERROR,
    "fail(?:s|ed|ing)",
    "crash(?:es|ed|ing)?",
    "broken",
    "timeouts?",
    "timing out",
    "unavailable",
    "not working",
    "stopped working",
)
_PROBLEM_RE = _cues(
    _status(r"5\d\d"),
    "timeouts?",
    "timing out",
    _DOWN,
    "slow",
    "affected",
    "impacted",
)
# Slowness or a named 5xx status makes a technical problem medium even when nothing is blocked;
# "affected" alone does not ("Which accounts are affected by the new SSO policy?").
_DEGRADED_RE = _cues("slow", "timeouts?", "timing out", "latency", _status(r"5\d\d"))
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
    _ERROR,
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
# Repeat cues that report an unresolved follow-up on their own. Bare "still" and "again" are left
# out: "How can I see my invoice again?", "Can I still downgrade?" and "I still love the product"
# are not follow-ups, so a how-to ticket is raised only by these.
_FOLLOW_UP_RE = _cues(
    "any update",
    "no (?:reply|response|answer)",
    "third time",
    "second time",
    "follow(?:ing)? ?up",
    "haven'?t heard",
    "asked (?:twice|again|before|already)",
    "still (?:no|not|waiting|missing|unresolved|broken|failing|unable|can'?t|cannot|have(?:n'?t| not| no))",
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
# A whole sentence of closing courtesy ("Thanks again!", "Can't wait to hear back.") is not a
# request, so its "again" or "can't" must not read as a follow-up or a blocker.
_SIGN_OFF_RE = re.compile(
    r"(?:(?:many |big )?(?:thanks|thank you|thx|cheers)(?: again)?(?: so much| very much| a lot)?"
    r"(?: (?:again )?for (?:all )?(?:your|the) (?:help|time|support|patience|assistance))?"
    r"|(?:can'?t|cannot) wait to hear (?:back|from you)"
    r"|(?:talk|speak) (?:to you )?soon"
    r"|looking forward to (?:hearing from you|hearing back|your reply)"
    r"|(?:have a )?(?:great|nice|good) (?:day|week|weekend))"
    r"[ !.,]*"
)
_INVOICE_RE = _cues("invoices?")
_CATEGORY_CUES = {"technical": _TECHNICAL_RE, "billing": _BILLING_RE, "account": _ACCOUNT_RE}
# Cues that make a sentence a request in its own right. Modifiers such as "again", "can't" or "how
# do I" are left out: "Thanks again!" and "Can't wait to hear back." are sign-offs, not requests.
_REQUEST_CUES = (*_CATEGORY_CUES.values(), _FEATURE_RE, _INCIDENT_RE, _VULN_RE)

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
    ``seed``, ``failure_rate`` and ``gullible``. Running past its end raises ``RuntimeError``;
    that call has already taken its per-prompt call index but is not appended to ``calls``, so
    ``calls`` lists only the calls the script served.
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
    subject_sentences = _scoring_sentences(subject, injected)
    subject_request = _has_request(subject_sentences)
    unquoted = _QUOTED_LINE_RE.sub("", body)
    # Quoted earlier messages are context, unless there is nothing else to go on.
    primary = unquoted if subject_request or _normalise(unquoted) else body
    body_sentences = _scoring_sentences(primary, injected, subject_request)
    sentences = subject_sentences + body_sentences
    ns = " ".join(subject_sentences)
    nb = " ".join(body_sentences)
    whole = f"{ns} {nb}"

    scores = {
        name: 3 * len(cues.findall(ns)) + len(cues.findall(nb)) for name, cues in _CATEGORY_CUES.items()
    }
    # Equal scores go to the category of the sentences that report a blocker, so "I cannot change
    # my email address. How do I export data?" is about the account. max() keeps the first of
    # equal keys, so remaining ties go technical > billing > account.
    reported = " ".join(s for s in sentences if _reports_blocker(s))
    best = max(scores, key=lambda name: (scores[name], len(_CATEGORY_CUES[name].findall(reported))))
    blocked = any(map(_reports_blocker, sentences))
    high = any(map(_reports_high_impact, sentences))
    money_owed = bool(_MONEY_OWED_RE.search(whole))
    deadline = bool(_DEADLINE_RE.search(whole))
    degraded = any(map(_reports_degradation, sentences))
    followed_up = any(map(_reports_follow_up, sentences))
    # Suggestion wording next to a reported problem ("Our export is timing out. Would love help
    # with this.") does not turn the problem into a feature request.
    feature = bool(_FEATURE_RE.search(whole))
    problem = high or blocked or degraded or money_owed or deadline or followed_up
    if scores[best] == 0 or (feature and not problem):
        category = "other"
    else:
        category = best

    # Bare repeat wording ("still", "again") raises a ticket unless it only modifies a how-to
    # question ("How can I see my invoice again?"); a reported follow-up raises it even then.
    repeated = followed_up or bool(_REPEAT_RE.search(whole))
    impaired = blocked or any(_states(_DEGRADED_RE, s) for s in sentences)
    if high:
        priority = "high"
    elif category == "other":
        priority = "low"
    elif _HOWTO_RE.search(whole) and not (blocked or money_owed or deadline or degraded or followed_up):
        priority = "low"
    elif money_owed or deadline or repeated:
        priority = "medium"
    elif category in ("technical", "account") and impaired:
        priority = "medium"
    else:
        priority = "low"

    chasing = repeated and not (impaired or money_owed or deadline)
    return {
        "category": category,
        "priority": priority,
        "summary": _summary_for(category, priority, whole, feature, chasing, subject, body),
        "flagged": injected,
        "flag_reason": FLAG_REASON_INJECTION if injected else None,
    }


def _summary_for(
    category: str, priority: str, whole: str, feature: bool, chasing: bool, subject: str, body: str
) -> str:
    # Templates only, never ticket text, so a summary can not echo or obey the ticket.
    # ``chasing`` says a follow-up is the only reason for a medium priority.
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
    if priority == "medium" and chasing:
        return f"Customer is following up on an unresolved {category} request."
    if category == "billing" and _INVOICE_RE.search(whole):
        return "Customer asks where to download invoices."
    if category == "other":
        if feature:
            return "Customer suggests a new feature."
        if not _normalise(subject) and not _normalise(body):
            return "Customer sent a message with no actionable content."
    return _SUMMARIES[(category, priority)]


def _is_question(sentence: str) -> bool:
    return sentence.endswith("?") or bool(_HOWTO_RE.search(sentence) or _QUESTION_RE.search(sentence))


def _reports_high_impact(sentence: str) -> bool:
    """Whether the sentence reports impact that makes a ticket high on its own."""
    return (
        _reports_incident(sentence)
        or _is_wide_problem(sentence)
        or _is_production_failure(sentence)
        or _is_vulnerability_report(sentence)
    )


def _reports_blocker(sentence: str) -> bool:
    # "I am locked out." reports a blocker, "What happens if all users are locked out?" asks about one.
    return _states(_BLOCKED_RE, sentence)


def _is_production_failure(sentence: str) -> bool:
    if not (_PRODUCTION_RE.search(sentence) and _states(_FAILURE_RE, sentence)):
        return False
    return not _HOWTO_RE.search(sentence) or _reports_blocker(sentence)


def _is_wide_problem(sentence: str) -> bool:
    # Scale counts only in the sentence that reports the problem: "I cannot change my email. How do
    # I export data for all users?" is one blocked user and a how-to question.
    if not _states(_SCALE_RE, sentence):
        return False
    return _reports_blocker(sentence) or _states(_PROBLEM_RE, _reported_part(sentence))


def _reported_part(sentence: str) -> str:
    # Only wording before a how-to question is a report: "The API returns 503. How can I fix it?"
    # and "Dashboard is slow, how can I fix this?" report a problem, "How do I reduce latency?"
    # asks about configuration.
    howto = _HOWTO_RE.search(sentence)
    return sentence[: howto.start()] if howto else sentence


def _reports_incident(sentence: str) -> bool:
    # "Our production API is down. What should we do?" reports an outage; "How do we handle a
    # production outage?", "How do I avoid an outage?" and "What happens if our site goes down?" ask
    # about one.
    return _states(_INCIDENT_RE, _reported_part(sentence))


def _states(pattern: re.Pattern[str], text: str) -> bool:
    """Whether ``pattern`` matches in ``text`` without a hypothetical cue before it in its clause.

    One pass over the matches and clause separators, each clause searched for a cue at most once,
    so a long body is never rescanned per match. Every impact cue starts on a word character after
    a word boundary and every hypothetical cue ends on one, so a cue precedes a match exactly when
    the clause's first cue ends at or before the match starts.
    """
    separators = _CLAUSE_SPLIT_RE.finditer(text)
    separator = next(separators, None)
    clause_start, cue_end = 0, None
    for m in pattern.finditer(text):
        while separator is not None and separator.end() <= m.start():
            clause_start, cue_end = separator.end(), None
            separator = next(separators, None)
        if cue_end is None:
            clause_end = separator.start() if separator is not None else len(text)
            cue = _HYPOTHETICAL_RE.search(text[clause_start:clause_end])
            cue_end = clause_start + cue.end() if cue else len(text) + 1
        if cue_end > m.start():
            return True
    return False


def _reports_degradation(sentence: str) -> bool:
    return _states(_DEGRADED_RE, _reported_part(sentence))


def _reports_follow_up(sentence: str) -> bool:
    # "I have asked twice and still have no response." is a follow-up, "How do I follow up on a
    # refund?" is a how-to question.
    return bool(_FOLLOW_UP_RE.search(_reported_part(sentence)))


def _reports_impact(sentence: str) -> bool:
    """Whether the sentence describes impact that sets priority, whatever its tone."""
    return bool(
        _reports_incident(sentence)
        or _reports_blocker(sentence)
        or _MONEY_OWED_RE.search(sentence)
        or _DEADLINE_RE.search(sentence)
        or _is_wide_problem(sentence)
        or _is_production_failure(sentence)
        or _is_vulnerability_report(sentence)
        or _reports_degradation(sentence)
        or _reports_follow_up(sentence)
    )


def _is_vulnerability_report(sentence: str) -> bool:
    # "We found a vulnerability in your API." is a report, "We want to prevent a security
    # vulnerability in the API." is not.
    if not _states(_VULN_RE, sentence):
        return False
    return not _is_question(sentence) or bool(_EXPLOIT_RE.search(sentence))


def _has_request(sentences: list[str]) -> bool:
    return any(cue.search(sentence) for sentence in sentences for cue in _REQUEST_CUES)


def _scoring_sentences(text: str, injected: bool, request_elsewhere: bool = False) -> list[str]:
    """Normalised sentences of ``text`` that count towards cues.

    ``request_elsewhere`` says the subject already carries a request, so deferred topics and
    low-urgency asides in the body are dropped even when nothing else in the body remains. A
    low-urgency sentence that reports impact is never an aside.
    """
    sentences = [s for s in map(_normalise, _SENTENCE_SPLIT_RE.split(text)) if s]
    # Phrases are matched on the whole normalised text, as _is_injected does, so a phrase broken
    # across a sentence or paragraph boundary marks every sentence it touches.
    joined = " ".join(sentences)
    spans, start = [], 0
    for sentence in sentences:
        spans.append((start, start + len(sentence)))
        start += len(sentence) + 1

    def marked(pattern: re.Pattern[str]) -> list[bool]:
        # Hits and spans are both in text order, so one shared cursor keeps this linear.
        hits = [m.span() for m in pattern.finditer(joined)]
        flags, i = [], 0
        for begin, end in spans:
            while i < len(hits) and hits[i][1] <= begin:
                i += 1
            flags.append(i < len(hits) and hits[i][0] < end)
        return flags

    kept = [i for i, sentence in enumerate(sentences) if not _SIGN_OFF_RE.fullmatch(sentence)]
    if injected:
        injection = marked(_INJECTION_RE)
        kept = [i for i in kept if not injection[i]]
    # A deferred topic or a low-urgency aside is dropped only while another request remains, here
    # or in the subject, so a greeting or sign-off never stands in for the only request. Low-urgency
    # wording is tone, not a topic: "No rush, our API is down and all users are blocked." reports
    # impact and always scores, whereas a deferral moves its topic to another ticket whatever it says.
    low_urgency = marked(_LOW_URGENCY_RE)
    asides = (
        marked(_DEFERRAL_RE),
        [hit and not _reports_impact(sentence) for hit, sentence in zip(low_urgency, sentences, strict=True)],
    )
    for aside in asides:
        rest = [i for i in kept if not aside[i]]
        if request_elsewhere or _has_request([sentences[i] for i in rest]):
            kept = rest
    return [sentences[i] for i in kept]


def _is_injected(subject: str, body: str) -> bool:
    return any(_INJECTION_RE.search(_normalise(part)) for part in (subject, body))


def _unwrap(user: str) -> str:
    """The user message without its outer boundary tags, or unchanged if it has none."""
    wrapper = _WRAPPER_RE.fullmatch(user)
    return wrapper.group(2) if wrapper else user


def _split_ticket(user: str) -> tuple[str, str]:
    """(subject, body) from a wrapped user message, or ('', text) when it has no field structure.

    The ticket text may contain its own field tags, so the fields must span the whole message:
    the body runs to the last ``</body>``. A stray tag can at worst move text from one field to
    the other, never drop it, so _is_injected always sees every word of the ticket.
    """
    text = _unwrap(user)
    fields = _FIELDS_RE.fullmatch(text)
    if fields is None:
        return "", text
    return fields.group(1), fields.group(2)


def _normalise(text: str) -> str:
    return _WS_RE.sub(" ", text.replace("\u2019", "'")).strip().lower()


def _prompt_key(system: str, user: str) -> str:
    """Stable identity of a prompt: the system prompt plus the user message without the boundary tags."""
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(_unwrap(user).encode("utf-8"))
    return h.hexdigest()


def _dumps(answer: dict[str, Any]) -> str:
    return json.dumps(answer, ensure_ascii=False, separators=(", ", ": "))
