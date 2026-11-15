import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import uvicorn

from app import seed
from app.config import Settings
from app.llm import FakeLLM
from app.main import create_app
from tests.sample_data import EXPECTED, TICKET_PAYLOADS, TICKETS_FILE

REPO_ROOT = Path(__file__).resolve().parent.parent
IDS = [t["id"] for t in TICKET_PAYLOADS]
DEADLINE_S = 5.0
SENTINEL = "SENTINEL-7d1c"


def envelope(code: str, message: str = "x", details: object = None) -> bytes:
    return json.dumps({"error": {"code": code, "message": message, "details": details}}).encode()


@dataclass(frozen=True)
class Reply:
    """What the scripted server answers to one request. ``drop`` closes the connection unanswered."""

    status: int = 202
    body: bytes | None = None
    content_type: str = "application/json"
    delay_s: float = 0.0
    drop: bool = False


@dataclass
class Received:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


@dataclass
class Script:
    replies: deque[Reply] = field(default_factory=deque)
    received: list[Received] = field(default_factory=list)


class ScriptedHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.server.script.received.append(
            Received("POST", self.path, {k.lower(): v for k, v in self.headers.items()}, body)
        )
        reply = self.server.script.replies.popleft() if self.server.script.replies else Reply()
        if reply.delay_s:
            time.sleep(reply.delay_s)
        if reply.drop:
            self.close_connection = True
            return
        payload = reply.body
        if payload is None:
            # By default the body echoes the ticket, as the real API does.
            payload = body
        self.send_response(reply.status)
        self.send_header("Content-Type", reply.content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        pass


class ScriptedServer(ThreadingHTTPServer):
    # Handler threads are joined on server_close, so none outlives its test.
    daemon_threads = False
    block_on_close = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), ScriptedHandler)
        self.script = Script()
        self.thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.01})

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def reply(self, *replies: Reply) -> None:
        self.script.replies.extend(replies)

    @property
    def received(self) -> list[Received]:
        return self.script.received

    def handle_error(self, request, client_address) -> None:
        # A client that timed out has gone; writing to it fails, which is the point of the test.
        pass

    def stop(self) -> None:
        if self.thread.is_alive():
            self.shutdown()
            self.thread.join()
        self.server_close()


@pytest.fixture
def server():
    srv = ScriptedServer()
    srv.thread.start()
    try:
        yield srv
    finally:
        srv.stop()


def run_seed(capsys, *argv: str) -> tuple[int, str, str]:
    code = seed.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def tickets_file(tmp_path: Path, entries, name: str = "tickets.json") -> Path:
    path = tmp_path / name
    path.write_text(entries if isinstance(entries, str) else json.dumps(entries), encoding="utf-8")
    return path


TWO = [{"id": "t-1", "subject": "", "body": "a"}, {"id": "t-2", "subject": "s", "body": "b"}]


# A. Arguments


def test_defaults():
    args = seed.parse_args([])
    assert args.base_url == seed.DEFAULT_BASE_URL == "http://127.0.0.1:8000"
    assert args.file == seed.DEFAULT_FILE == TICKETS_FILE
    assert args.file.is_file()
    assert args.timeout == seed.DEFAULT_TIMEOUT_S == 10.0


def test_help_exits_zero(capsys):
    code, out, _ = run_seed(capsys, "--help")
    assert code == 0
    for flag in ("--base-url", "--file", "--timeout", "exit codes"):
        assert flag in out


@pytest.mark.parametrize(
    "url",
    [
        "file:///x",
        "file://localhost/etc/hosts",
        "ftp://h",
        "127.0.0.1:8000",
        "http://",
        "http:///tickets",
        "http://h/?x=1",
        "http://h/#f",
        "http://user:pw@h",
        "http://h:notaport",
        "http://h:99999",
        "http://[::1",
        "",
    ],
)
def test_bad_base_url_exits_2(capsys, server, url):
    code, out, err = run_seed(capsys, "--base-url", url)
    assert code == 2
    assert out == ""
    assert "--base-url" in err
    assert server.received == []


@pytest.mark.parametrize(
    ("suffix", "path"),
    [("", "/tickets"), ("/", "/tickets"), ("//", "/tickets"), ("/api", "/api/tickets"), ("/api/", "/api/tickets")],
)
def test_trailing_slash_and_prefix_are_kept(capsys, server, tmp_path, suffix, path):
    code, _, _ = run_seed(capsys, "--base-url", server.base_url + suffix, "--file", str(tickets_file(tmp_path, TWO)))
    assert code == 0
    assert [r.path for r in server.received] == [path, path]


def test_check_base_url_normalises():
    assert seed.check_base_url("HTTP://Example.com:8080/api//") == "http://Example.com:8080/api"
    assert seed.check_base_url("https://h") == "https://h"


@pytest.mark.parametrize("value", ["0", "-1", "abc", "nan", "inf", "-inf", ""])
def test_timeout_must_be_positive_number(capsys, server, value):
    code, out, err = run_seed(capsys, "--base-url", server.base_url, "--timeout", value)
    assert code == 2
    assert out == ""
    assert "--timeout" in err
    assert server.received == []


def test_timeout_accepts_int_and_float():
    assert seed.parse_args(["--timeout", "3"]).timeout == 3.0
    assert seed.parse_args(["--timeout", "0.25"]).timeout == 0.25


# B. Tickets file


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "is not valid JSON"),
        ('{"id": "t-1", "body": "x"}', "must hold a JSON array"),
        ("[]", "holds no tickets"),
        ([{"id": "t-1", "body": "x"}, "t-2"], "entry 1 "),
        ([{"id": "t-1", "body": "x", "priority": "high"}], "entry 0 "),
        ([{"id": "t-1", "subject": "x"}], "entry 0 "),
        ([{"id": "t-1", "body": "x"}, {"id": "t-2", "body": 3}], "entry 1 "),
        ([{"id": 1, "body": "x"}], "entry 0 "),
        ("[" * 100_000 + "]" * 100_000, "nested too deeply"),
    ],
    ids=[
        "not-json",
        "not-an-array",
        "empty-array",
        "entry-not-an-object",
        "unknown-key",
        "without-body",
        "non-string-value",
        "non-string-id",
        "deeply-nested",
    ],
)
def test_malformed_file_exits_2_and_posts_nothing(capsys, server, tmp_path, content, message):
    path = tickets_file(tmp_path, content)
    code, out, err = run_seed(capsys, "--base-url", server.base_url, "--file", str(path))
    assert code == 2
    assert out == ""
    assert err.startswith("seed: ") and message in err
    assert server.received == []


def test_entry_errors_name_the_index_never_the_content(capsys, server, tmp_path):
    path = tickets_file(tmp_path, [{"id": SENTINEL, "body": SENTINEL, "extra": SENTINEL}])
    code, _, err = run_seed(capsys, "--base-url", server.base_url, "--file", str(path))
    assert code == 2
    assert "entry 0" in err
    assert SENTINEL not in err


def test_missing_file_exits_2_and_posts_nothing(capsys, server, tmp_path):
    code, out, err = run_seed(capsys, "--base-url", server.base_url, "--file", str(tmp_path / "missing.json"))
    assert (code, out) == (2, "")
    assert "cannot read" in err and "missing.json" in err
    assert server.received == []


def test_utf8_bom_is_rejected(capsys, server, tmp_path):
    path = tmp_path / "bom.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(TWO).encode())
    code, _, err = run_seed(capsys, "--base-url", server.base_url, "--file", str(path))
    assert code == 2
    assert "byte order mark" in err
    assert server.received == []


def test_non_utf8_file_is_rejected(capsys, server, tmp_path):
    path = tmp_path / "latin1.json"
    path.write_bytes('[{"id": "t-1", "body": "caf\xe9"}]'.encode("latin-1"))
    code, _, err = run_seed(capsys, "--base-url", server.base_url, "--file", str(path))
    assert code == 2
    assert "not UTF-8" in err
    assert server.received == []


def test_load_tickets_returns_the_entries_unchanged():
    assert seed.load_tickets(TICKETS_FILE) == TICKET_PAYLOADS


# C. Request shape


def test_posts_each_entry_unchanged_in_order(capsys, server):
    code, _, _ = run_seed(capsys, "--base-url", server.base_url)
    assert code == 0
    assert [r.method for r in server.received] == ["POST"] * len(TICKET_PAYLOADS)
    assert {r.path for r in server.received} == {"/tickets"}
    assert {r.headers["content-type"] for r in server.received} == {"application/json"}
    assert {r.headers["accept"] for r in server.received} == {"application/json"}
    sent = [json.loads(r.body) for r in server.received]
    assert sent == TICKET_PAYLOADS
    by_id = {entry["id"]: entry for entry in sent}
    assert by_id["t-1008"]["subject"] == ""
    assert "\n> " in by_id["t-1010"]["body"]


# D. Status matrix


def test_202_lines_and_exit_0(capsys, server):
    code, out, err = run_seed(capsys, "--base-url", server.base_url)
    assert code == 0
    assert out.splitlines() == [f"{tid} 202 Accepted" for tid in IDS] + [
        "10 tickets: 10 accepted, 0 already stored, 0 other"
    ]
    assert err == ""


def test_200_is_already_stored_and_exit_0(capsys, server, tmp_path):
    server.reply(Reply(200), Reply(200))
    code, out, _ = run_seed(capsys, "--base-url", server.base_url, "--file", str(tickets_file(tmp_path, TWO)))
    assert code == 0
    assert out.splitlines() == [
        "t-1 200 OK (already stored)",
        "t-2 200 OK (already stored)",
        "2 tickets: 0 accepted, 2 already stored, 0 other",
    ]


def test_mixed_202_and_200_summary(capsys, server, tmp_path):
    server.reply(Reply(200), Reply(202))
    code, out, _ = run_seed(capsys, "--base-url", server.base_url, "--file", str(tickets_file(tmp_path, TWO)))
    assert code == 0
    assert out.splitlines()[-1] == "2 tickets: 1 accepted, 1 already stored, 0 other"


@pytest.mark.parametrize(
    ("reply", "line"),
    [
        (Reply(409, envelope("ticket_conflict")), "t-1 409 Conflict (ticket_conflict)"),
        (
            Reply(422, envelope("validation_error", details=[{"loc": "body"}])),
            "t-1 422 Unprocessable Content (validation_error)",
        ),
        (Reply(400, envelope("http_error")), "t-1 400 Bad Request (http_error)"),
        (Reply(500, envelope("internal_error")), "t-1 500 Internal Server Error (internal_error)"),
        (Reply(500, b"<html>oops</html>", "text/html"), "t-1 500 Internal Server Error"),
        (Reply(502, b"", "text/plain"), "t-1 502 Bad Gateway"),
        (Reply(409, b'{"detail": "conflict"}'), "t-1 409 Conflict"),
        (Reply(409, b'{"error": "conflict"}'), "t-1 409 Conflict"),
        (Reply(409, b'{"error": {"code": 7}}'), "t-1 409 Conflict"),
        (Reply(409, envelope("\x1b[31mred")), "t-1 409 Conflict"),
        (Reply(599, b""), "t-1 599"),
    ],
    ids=[
        "409",
        "422",
        "400",
        "500-envelope",
        "500-html",
        "502-empty",
        "non-envelope-json",
        "error-not-an-object",
        "code-not-a-string",
        "code-with-escape",
        "unknown-status",
    ],
)
def test_other_status_prints_the_code_and_exits_1_but_continues(capsys, server, tmp_path, reply, line):
    server.reply(reply, Reply(202))
    code, out, _ = run_seed(capsys, "--base-url", server.base_url, "--file", str(tickets_file(tmp_path, TWO)))
    assert code == 1
    lines = out.splitlines()
    assert lines[0].startswith(line)
    assert lines[1:] == ["t-2 202 Accepted", "2 tickets: 1 accepted, 0 already stored, 1 other"]
    assert len(server.received) == 2


@pytest.mark.parametrize(
    ("reply", "line"),
    [
        (Reply(201), "t-1 201 Created"),
        (Reply(204, b""), "t-1 204 No Content"),
        (Reply(200, b"<html>docs</html>", "text/html"), "t-1 200 OK (not a ticket response)"),
        (Reply(202, b"{}"), "t-1 202 Accepted (not a ticket response)"),
        (Reply(200, b'{"id": "t-other"}'), "t-1 200 OK (not a ticket response)"),
    ],
    ids=["201", "204", "html-200", "202-without-ticket", "200-other-ticket"],
)
def test_2xx_other_than_200_202_counts_as_other(capsys, server, tmp_path, reply, line):
    server.reply(reply, Reply(202))
    code, out, _ = run_seed(capsys, "--base-url", server.base_url, "--file", str(tickets_file(tmp_path, TWO)))
    assert code == 1
    assert out.splitlines() == [line, "t-2 202 Accepted", "2 tickets: 1 accepted, 0 already stored, 1 other"]


def test_summary_counts(capsys, server, tmp_path):
    entries = [{"id": f"t-{i}", "body": "x"} for i in range(6)]
    server.reply(Reply(202), Reply(200), Reply(409, envelope("ticket_conflict")), Reply(202), Reply(500), Reply(200))
    code, out, _ = run_seed(capsys, "--base-url", server.base_url, "--file", str(tickets_file(tmp_path, entries)))
    assert code == 1
    assert out.splitlines()[-1] == "6 tickets: 2 accepted, 2 already stored, 2 other"


def test_format_line():
    assert seed.format_line(seed.Outcome("t-1", 202, "Accepted", kind="accepted")) == "t-1 202 Accepted"
    assert seed.format_line(seed.Outcome("t-1", 200, "OK", kind="stored")) == "t-1 200 OK (already stored)"
    conflict = seed.Outcome("t-1", 409, "Conflict", "ticket_conflict")
    assert seed.format_line(conflict) == "t-1 409 Conflict (ticket_conflict)"
    assert seed.format_line(seed.Outcome("t-1", 500, "Internal Server Error")) == "t-1 500 Internal Server Error"


# E. Request failures


def closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_connection_refused_aborts_with_exit_1(capsys):
    base = f"http://127.0.0.1:{closed_port()}"
    code, out, err = run_seed(capsys, "--base-url", base, "--timeout", "2")
    assert code == 1
    [line] = out.splitlines()
    assert line.startswith(f"t-1001 error: could not reach {base} (")
    assert err == "seed: is the server running? try: make run\n"


def test_timeout_aborts(capsys, server, tmp_path):
    server.reply(Reply(202, delay_s=0.5))
    path = str(tickets_file(tmp_path, TWO))
    code, out, err = run_seed(capsys, "--base-url", server.base_url, "--file", path, "--timeout", "0.1")
    assert code == 1
    assert out.splitlines() == [f"t-1 error: could not reach {server.base_url} (TimeoutError)"]
    assert "is the server running" in err
    assert len(server.received) == 1


def test_abort_after_partial_progress(capsys, server, tmp_path):
    entries = [{"id": f"t-{i}", "body": "x"} for i in range(4)]
    server.reply(Reply(202), Reply(202), Reply(drop=True))
    code, out, err = run_seed(capsys, "--base-url", server.base_url, "--file", str(tickets_file(tmp_path, entries)))
    assert code == 1
    lines = out.splitlines()
    assert lines[:2] == ["t-0 202 Accepted", "t-1 202 Accepted"]
    assert lines[2].startswith(f"t-2 error: could not reach {server.base_url} (")
    assert len(lines) == 3
    assert "is the server running" in err
    assert len(server.received) == 3


# F. Output hygiene


def test_output_never_contains_ticket_text_or_response_message(capsys, server, tmp_path):
    entries = [{"id": "t-1", "subject": f"subject {SENTINEL}", "body": f"body {SENTINEL}"}] * 2
    server.reply(
        Reply(409, envelope("ticket_conflict", message=SENTINEL, details={"why": SENTINEL})),
        Reply(500, f"<html>{SENTINEL}</html>".encode(), "text/html"),
    )
    code, out, err = run_seed(capsys, "--base-url", server.base_url, "--file", str(tickets_file(tmp_path, entries)))
    assert code == 1
    assert SENTINEL not in out + err
    assert out.splitlines()[:2] == ["t-1 409 Conflict (ticket_conflict)", "t-1 500 Internal Server Error"]


# G. Real round trip and entry point


@pytest.fixture
def live_server(tmp_path):
    """The real app under uvicorn on an ephemeral port, honest fake, two workers."""
    settings = Settings(db_path=str(tmp_path / "tickets.db"), worker_count=2, poll_interval_s=0.01)
    app = create_app(settings, llm=FakeLLM(failure_rate=0.0, gullible=False))
    config = uvicorn.Config(app, log_config=None, access_log=False, lifespan="on", ws="none")
    server = uvicorn.Server(config)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    try:
        deadline = time.monotonic() + DEADLINE_S
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(DEADLINE_S)
        sock.close()
    assert not thread.is_alive()


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=DEADLINE_S) as response:
        return json.load(response)


def test_running_twice_leaves_ten_tickets_and_reports_200(capsys, live_server):
    # Done-when 1: run the loader twice; exactly ten tickets, and the second run reports 200 for all.
    code, out, err = run_seed(capsys, "--base-url", live_server)
    assert (code, err) == (0, "")
    assert out.splitlines() == [f"{tid} 202 Accepted" for tid in IDS] + [
        "10 tickets: 10 accepted, 0 already stored, 0 other"
    ]

    code, out, err = run_seed(capsys, "--base-url", live_server + "/")
    assert (code, err) == (0, "")
    assert out.splitlines() == [f"{tid} 200 OK (already stored)" for tid in IDS] + [
        "10 tickets: 0 accepted, 10 already stored, 0 other"
    ]

    deadline = time.monotonic() + DEADLINE_S
    while True:
        page = get_json(f"{live_server}/tickets?limit=100")
        if all(item["status"] != "pending" for item in page["items"]):
            break
        assert time.monotonic() < deadline, "tickets did not settle"
        time.sleep(0.02)
    assert page["next_cursor"] is None
    assert [item["id"] for item in page["items"]] == IDS
    for item in page["items"]:
        assert item["status"] == "classified", item["id"]
        assert (item["category"], item["priority"], item["flagged"]) == EXPECTED[item["id"]]


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-W", "error", "-m", "app.seed", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: python -m app.seed" in result.stdout


def test_loader_imports_only_the_standard_library():
    # It must run under uv run --no-dev and talk only HTTP: nothing from the app or third parties.
    code = (
        "import sys; before = set(sys.modules); import app.seed; "
        "print(sorted(m for m in set(sys.modules) - before if m.split('.')[0] not in sys.stdlib_module_names))"
    )
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['app', 'app.seed']"
