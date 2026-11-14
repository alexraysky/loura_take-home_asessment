# Ticket classifier

A small service that accepts support tickets over HTTP and classifies them in the background.
It runs on FastAPI with SQLite as the queue, asyncio worker tasks and a deterministic fake LLM, so it needs no API keys.

## Run it

Install [uv](https://docs.astral.sh/uv/) 0.13 or newer. uv downloads Python 3.15 by itself, so no system Python is needed.

```sh
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

An older uv stops with `Required uv version ... does not match`; run `uv self update`.

```sh
make run     # start the API on http://127.0.0.1:8000 (HOST=... PORT=... to override)
make test    # run the test suite
make seed    # load the sample tickets into a running server (arrives with the loader, #10)
```

The first `make run` or `make test` creates `.venv` from the committed `uv.lock`.

`GET /health` (or `HEAD /health`) returns `{"status": "ok"}` once the app has started. It is an operational readiness probe and is not part of the ticket API.

### Without make

The Makefile targets are thin wrappers:

```sh
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
uv run pytest
uv run python -m app.seed --base-url http://127.0.0.1:8000
```

## Configuration

Environment variables with defaults:

```text
DB_PATH=tickets.db
WORKER_COUNT=4            0 disables workers (used by API tests)
MAX_ATTEMPTS=3
LLM_TIMEOUT_S=10
LEASE_TIMEOUT_S=60        must be greater than LLM_TIMEOUT_S; checked at startup
POLL_INTERVAL_S=0.5
BACKOFF_BASE_S=1
FAKE_LLM_SEED=42
FAKE_LLM_FAILURE_RATE=0.2
FAKE_LLM_GULLIBLE=true    the fake obeys injected instructions on the first attempt
```

Unset or blank variables use the default; surrounding whitespace is ignored. Booleans accept `1/0`, `true/false`, `yes/no` and `on/off`, case-insensitive. Integers are plain ASCII digits with an optional sign.

The app refuses to start, naming the variable, when a value is malformed, not finite or out of bounds:

| Variable | Allowed |
|---|---|
| `WORKER_COUNT` | 0 to 100 |
| `MAX_ATTEMPTS` | 1 to 20 |
| `LLM_TIMEOUT_S` | greater than 0, at most 86400 |
| `LEASE_TIMEOUT_S` | greater than `LLM_TIMEOUT_S`, at most 86400 |
| `POLL_INTERVAL_S` | greater than 0, at most 86400 |
| `BACKOFF_BASE_S` | 0 to 86400 |
| `FAKE_LLM_FAILURE_RATE` | 0 to 1 |
| `DB_PATH` | not blank |

## Layout

Files marked `(planned)` arrive with later issues.

```text
app/
  main.py         FastAPI app factory, lifespan, error handlers
  config.py       Settings read from environment variables
  clock.py        Injected UTC clock and the fixed timestamp format
  api.py          Routes, request/response schemas, error envelope
  db.py           Connection factory and schema creation        (planned)
  repository.py   All SQL lives here                            (planned)
  worker.py       Claim, classify, retry loop                   (planned)
  classifier.py   One classification attempt                    (planned)
  prompt.py       System prompt, PROMPT_VERSION, ticket wrapping (planned)
  validation.py   Model text to typed result or typed rejection (planned)
  llm/base.py     LLMClient protocol and error types            (planned)
  llm/fake.py     Deterministic fake                            (planned)
  seed.py         Loads the sample tickets                      (planned)
data/                                                            (planned)
  sample_tickets.json
  expected_labels.json
tests/
.python-version
pyproject.toml
uv.lock
Makefile
README.md
```
