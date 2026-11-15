"""Post the sample tickets to a running service through the real HTTP ingest path.

    python -m app.seed [--base-url URL] [--file PATH] [--timeout SECONDS]

Each ticket is sent as is to ``POST <base-url>/tickets``, in file order, and gets one line with its
status code; a summary line follows. Ingest is idempotent, so a second run reports 200 for every
ticket already stored. The loader never prints ticket text or response bodies: only ids, status
codes and the ``error.code`` of an error envelope.

Stdlib only, with no imports from the rest of the app: the loader talks HTTP like any other client,
so it runs under ``uv run --no-dev`` and against a remote deployment.
"""

import argparse
import codecs
import http.client
import json
import math
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, Literal, TextIO

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_tickets.json"
DEFAULT_TIMEOUT_S = 10.0

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

TICKET_KEYS = frozenset({"id", "subject", "body"})
REQUIRED_KEYS = frozenset({"id", "body"})
# Only so much of a response is read: enough for an error envelope, never a whole HTML page.
MAX_ERROR_BYTES = 64 * 1024
# A 200 or 202 echoes the ticket: a legal 20,000-character body is over 80 KB as 4-byte UTF-8
# emoji, or about 240 KB JSON-escaped as \uXXXX pairs. 512 KiB fits it and still bounds the read.
MAX_TICKET_BYTES = 512 * 1024
# An error code is printed only when it looks like one, so a hostile server cannot write to the terminal.
_CODE_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")

Kind = Literal["accepted", "stored", "other"]

EPILOG = """\
exit codes:
  0  every ticket got 202 (accepted) or 200 (already stored)
  1  a ticket got another status (all tickets are still sent), or the server could not be reached
  2  bad arguments or a malformed tickets file; nothing is sent
"""


class UsageError(Exception):
    """A problem with the arguments or the tickets file, found before any request."""


class Unreachable(Exception):
    """The request could not be made or got no HTTP response; ``reason`` names the error type."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class Outcome:
    """One ticket's HTTP result. ``code`` is the error envelope's ``error.code``, if any."""

    id: str
    status: int
    reason: str
    code: str | None = None
    kind: Kind = "other"


def positive_timeout(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a finite number of seconds > 0, got {text!r}")
    return value


def check_base_url(url: str) -> str:
    """The base URL without a trailing slash; ValueError unless it is an http(s) URL with a host.

    ``file:`` is refused because ``urlopen`` would read a local file. A path prefix is kept
    (``http://host/api`` posts to ``/api/tickets``), a query or fragment is refused, and so are
    credentials, which would end up in the output.
    """
    try:
        parts = urllib.parse.urlsplit(url)
        _ = parts.port  # ValueError for a port that is not a number in range
    except ValueError:
        raise ValueError(f"--base-url is not a valid URL: {url!r}") from None
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"--base-url must start with http:// or https://, got {url!r}")
    if not parts.hostname:
        raise ValueError(f"--base-url has no host: {url!r}")
    if parts.username is not None or parts.password is not None:
        raise ValueError("--base-url must not contain credentials")
    if parts.query or parts.fragment:
        raise ValueError(f"--base-url must not have a query or fragment: {url!r}")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.seed",
        description="Post the sample tickets to a running service and print one line per ticket.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        metavar="URL",
        help=f"http:// or https:// URL of the service, optional path prefix (default {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        metavar="PATH",
        help='JSON array of {"id", "subject", "body"} payloads (default data/sample_tickets.json)',
    )
    parser.add_argument(
        "--timeout",
        type=positive_timeout,
        default=DEFAULT_TIMEOUT_S,
        metavar="SECONDS",
        help=f"per-request timeout, > 0 (default {DEFAULT_TIMEOUT_S:g})",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parsed arguments with ``base_url`` normalised; argparse exits with 2 on a usage error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.base_url = check_base_url(args.base_url)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def load_tickets(path: Path) -> list[dict[str, str]]:
    """The tickets in ``path``, checked for shape only; the API's own rules are left to the server.

    Messages name the file and an entry's index, never its content.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise UsageError(f"cannot read {path} ({type(exc).__name__})") from None
    if raw.startswith(codecs.BOM_UTF8):
        raise UsageError(f"{path} starts with a byte order mark; save it as UTF-8 without one")
    try:
        data = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        raise UsageError(f"{path} is not UTF-8") from None
    except json.JSONDecodeError as exc:
        raise UsageError(f"{path} is not valid JSON (line {exc.lineno}, column {exc.colno})") from None
    except RecursionError:
        raise UsageError(f"{path} is nested too deeply") from None
    if not isinstance(data, list):
        raise UsageError(f"{path} must hold a JSON array of tickets")
    if not data:
        raise UsageError(f"{path} holds no tickets")
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            problem = "is not an object"
        elif not entry.keys() <= TICKET_KEYS:
            problem = "has a key other than id, subject and body"
        elif not REQUIRED_KEYS <= entry.keys():
            problem = "lacks id or body"
        elif not all(isinstance(value, str) for value in entry.values()):
            problem = "has a value that is not a string"
        else:
            continue
        raise UsageError(f"entry {index} in {path} {problem}")
    return data


def post_ticket(base_url: str, ticket: dict[str, str], *, timeout: float) -> Outcome:
    """POST one ticket; Unreachable when no HTTP response came back."""
    request = urllib.request.Request(
        f"{base_url}/tickets",
        data=json.dumps(ticket).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                ticket_like = response.status in (HTTPStatus.OK, HTTPStatus.ACCEPTED)
                body = response.read(MAX_TICKET_BYTES if ticket_like else MAX_ERROR_BYTES)
                return _outcome(ticket["id"], response.status, response.reason, body)
        except urllib.error.HTTPError as exc:
            # Every status urlopen does not return; the context manager closes its socket.
            with exc:
                return _outcome(ticket["id"], exc.code, exc.reason, exc.read(MAX_ERROR_BYTES))
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        # On Windows a refused local port surfaces as URLError(TimeoutError), so every
        # transport failure is treated alike.
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        name = type(reason).__name__ if isinstance(reason, BaseException) else type(exc).__name__
        raise Unreachable(name) from None


def _outcome(ticket_id: str, status: int, server_reason: Any, body: bytes) -> Outcome:
    try:
        reason = HTTPStatus(status).phrase
    except ValueError:
        reason = "".join(ch for ch in str(server_reason) if ch.isascii() and ch.isprintable())[:64]
    data = _parse_json(body)
    # A 200 or 202 counts only with the ticket in the body: a wrong base URL can answer 200 with HTML.
    if status in (HTTPStatus.ACCEPTED, HTTPStatus.OK) and isinstance(data, dict) and data.get("id") == ticket_id:
        return Outcome(ticket_id, status, reason, kind="accepted" if status == HTTPStatus.ACCEPTED else "stored")
    return Outcome(ticket_id, status, reason, code=_error_code(data))


def _parse_json(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None


def _error_code(data: Any) -> str | None:
    error = data.get("error") if isinstance(data, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) and _CODE_RE.fullmatch(code) else None


def escape_display(text: str) -> str:
    """``text`` with control, format and line or paragraph separator characters backslash-escaped.

    A ticket id from a custom file goes to the server as is, but a newline or an ANSI escape in it
    must not forge output lines or drive the terminal, so a newline prints as the two characters
    backslash and n. Printable text, non-ASCII letters included, is left as it is.
    """
    return "".join(ch.encode("unicode_escape").decode("ascii") if _is_unsafe(ch) else ch for ch in text)


def _is_unsafe(ch: str) -> bool:
    category = unicodedata.category(ch)
    return category[0] == "C" or category in ("Zl", "Zp")


def format_line(outcome: Outcome) -> str:
    line = f"{escape_display(outcome.id)} {outcome.status} {outcome.reason}".rstrip()
    if outcome.kind == "stored":
        return f"{line} (already stored)"
    if outcome.kind == "other" and outcome.status in (HTTPStatus.ACCEPTED, HTTPStatus.OK):
        return f"{line} (not a ticket response)"
    if outcome.code is not None:
        return f"{line} ({outcome.code})"
    return line


def run(tickets: Sequence[dict[str, str]], base_url: str, *, timeout: float, out: TextIO, err: TextIO) -> int:
    """Post every ticket in order, print a line each and the summary; the exit code."""
    counts: Counter[Kind] = Counter()
    for ticket in tickets:
        try:
            outcome = post_ticket(base_url, ticket, timeout=timeout)
        except Unreachable as exc:
            # A server that is down stays down: stop rather than wait out a timeout per ticket.
            target = f"{escape_display(ticket['id'])} error: could not reach {escape_display(base_url)}"
            print(f"{target} ({exc.reason})", file=out)
            print("seed: is the server running? try: make run", file=err)
            return EXIT_FAILED
        counts[outcome.kind] += 1
        print(format_line(outcome), file=out)
    print(
        f"{len(tickets)} tickets: {counts['accepted']} accepted, {counts['stored']} already stored,"
        f" {counts['other']} other",
        file=out,
    )
    return EXIT_OK if counts["other"] == 0 else EXIT_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        # --help (0) or a usage error (2), already printed by argparse.
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        tickets = load_tickets(args.file)
    except UsageError as exc:
        print(f"seed: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return run(tickets, args.base_url, timeout=args.timeout, out=sys.stdout, err=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
