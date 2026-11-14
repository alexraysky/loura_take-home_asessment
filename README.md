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

## API

### `POST /tickets`

```sh
curl -i -X POST http://127.0.0.1:8000/tickets \
  -H 'content-type: application/json' \
  -d '{"id": "t-1001", "subject": "Charged twice this month", "body": "Hi, I see two charges..."}'
```

| Field | Rules |
|---|---|
| `id` | Required string, 1 to 128 characters, matching `^[A-Za-z0-9_-]+$` (letters, digits, `_` and `-`; no `.`) |
| `subject` | Optional string, at most 500 characters, defaults to `""`; `null` is rejected |
| `body` | Required string, 1 to 20,000 characters |

Unknown fields are rejected and values are never coerced (a number is not a string). Limits count Unicode code points. Subject and body are stored and returned exactly as received: no trimming or normalisation.

The id makes the request idempotent:

| Case | Response |
|---|---|
| New id | `202 Accepted`, the ticket with `status: "pending"`, `Location: /tickets/{id}` |
| Existing id, identical `subject` and `body` | `200 OK`, the stored ticket in its current state; nothing is written or requeued |
| Existing id, different `subject` or `body` | `409 Conflict`, code `ticket_conflict`; the stored ticket is unchanged |
| Invalid payload (malformed JSON, or JSON that fails the rules above) | `422`, code `validation_error` |
| Body that cannot be decoded or parsed before validation (for example undecodable UTF-8/16/32 bytes, or JSON nested too deeply) | `400`, code `http_error` |

An omitted `subject` and `"subject": ""` are the same payload. Comparison is exact, so a whitespace or line-ending difference is a conflict.

### `GET /tickets/{id}`

```sh
curl -i http://127.0.0.1:8000/tickets/t-1001
```

`200` with the ticket in its current state, or `404` with code `ticket_not_found`. Any id that is not stored is 404, including ids that `POST /tickets` would reject. This route takes no query parameters: any query parameter (`/tickets/t-1001?x=1`) is `422`, code `validation_error`, type `extra_forbidden`.

### `GET /tickets`

```sh
curl -s 'http://127.0.0.1:8000/tickets?category=billing&limit=2'
```

| Parameter | Rules |
|---|---|
| `category` | One of `billing`, `technical`, `account`, `other` |
| `priority` | One of `low`, `medium`, `high` |
| `flagged` | `true` or `false`, exactly (`1`, `yes` or `True` are rejected) |
| `limit` | 1 to 100, default 20 |
| `cursor` | `next_cursor` from a previous response, passed back unchanged |

An invalid value is `422`, code `validation_error`; a cursor the service did not produce is `400`, code `invalid_cursor`. Unlike `GET /tickets/{id}`, this route keeps the framework defaults for the query string: unknown parameters are ignored, and a repeated parameter takes its last value (`?limit=50&limit=2` means `limit=2`).

Filters combine with AND. Tickets that are still `pending` (or `failed` before classification) have no category or priority, so a category or priority filter excludes them; `flagged=false` does include them.

Tickets are ordered by `created_at` ascending, then `id`, so new tickets land at the end and paging is not disturbed by them. `next_cursor` is `null` on the last page:

```json
{
  "items": [
    {"id": "t-1001", "status": "classified", "category": "billing", "priority": "high", "...": "..."},
    {"id": "t-1003", "status": "classified", "category": "billing", "priority": "low", "...": "..."}
  ],
  "next_cursor": "MjAyNi0xMS0xNFQxMjowMDowMC4wMDAwMDBafHQtMTAwMw"
}
```

The cursor is the position of the last item returned (`created_at` and `id`), not a reference to a ticket: it is accepted after the filters or `limit` change, or if no ticket has that position. A sequence of pages is not a snapshot; a ticket whose labels change while you page can appear on a later page or drop out of a filter.

### Ticket representation

```json
{
  "id": "t-1001",
  "subject": "Charged twice this month",
  "body": "Hi, I see two charges...",
  "status": "pending",
  "category": null,
  "priority": null,
  "summary": null,
  "flagged": false,
  "flag_reason": null,
  "attempts": 0,
  "last_error": null,
  "prompt_version": null,
  "created_at": "2026-11-14T12:00:00.000000Z",
  "updated_at": "2026-11-14T12:00:00.000000Z"
}
```

`claimed_at` and `next_attempt_at` are internal queue columns and are never returned.

### Errors

Every error uses one envelope:

```json
{"error": {"code": "ticket_conflict", "message": "Ticket t-1001 already exists with different content.", "details": null}}
```

| Code | Status | When |
|---|---|---|
| `validation_error` | 422 | The request body or query parameters fail validation |
| `ticket_conflict` | 409 | `POST /tickets` with an existing id and different content |
| `ticket_not_found` | 404 | `GET /tickets/{id}` with an id that is not stored |
| `invalid_cursor` | 400 | `GET /tickets` with a cursor the service did not produce |
| `http_error` | 400 | `POST /tickets` with a body that cannot be decoded or parsed before validation |
| `not_found` | 404 | A path that matches no route |
| `method_not_allowed` | 405 | A method the matched route does not support |
| `internal_error` | 500 | An unexpected server failure |

For `validation_error`, `details` lists one `{"loc", "msg", "type"}` entry per problem. Submitted values are never echoed back.

`invalid_cursor` covers anything that is not exactly a `next_cursor` the service produced: wrong alphabet, padding, length or content.

The sample loader arrives with #10.

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

Unset or blank variables use the default; surrounding whitespace is ignored. Booleans accept `1/0`, `true/false`, `yes/no` and `on/off`, case-insensitive. Numbers are plain ASCII; integers are digits with an optional sign.

The app refuses to start, naming the variable, when a value is malformed, not finite or out of bounds:

| Variable | Allowed |
|---|---|
| `WORKER_COUNT` | 0 to 100 |
| `MAX_ATTEMPTS` | 1 to 20 |
| `LLM_TIMEOUT_S` | greater than 0, at most 86400 |
| `LEASE_TIMEOUT_S` | greater than `LLM_TIMEOUT_S`, at most 86400 |
| `POLL_INTERVAL_S` | 0.001 to 86400 |
| `BACKOFF_BASE_S` | 0 to 86400 |
| `FAKE_LLM_FAILURE_RATE` | 0 to 1 |
| `FAKE_LLM_SEED` | at most 18 digits |
| `DB_PATH` | not blank |

## Layout

Files marked `(planned)` arrive with later issues.

```text
app/
  main.py         FastAPI app factory, lifespan, error handlers
  config.py       Settings read from environment variables
  clock.py        Injected UTC clock and the fixed timestamp format
  api.py          Routes, request/response schemas, error envelope
  models.py       Shared domain types: allowed values and the Ticket row
  db.py           Connection factory and schema creation
  repository.py   All SQL lives here
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
