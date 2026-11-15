"""The sample tickets and their expected labels, loaded once from data/ for every test that uses them."""

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TICKETS_FILE = DATA_DIR / "sample_tickets.json"
LABELS_FILE = DATA_DIR / "expected_labels.json"


def _load(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


# Each entry is a complete POST /tickets payload, in file order.
TICKET_PAYLOADS: list[dict[str, str]] = _load(TICKETS_FILE)
EXPECTED_ROWS: list[dict[str, Any]] = _load(LABELS_FILE)

SAMPLE_TICKETS: dict[str, tuple[str, str]] = {t["id"]: (t.get("subject", ""), t["body"]) for t in TICKET_PAYLOADS}
EXPECTED: dict[str, tuple[str, str, bool]] = {
    row["id"]: (row["category"], row["priority"], row["flagged"]) for row in EXPECTED_ROWS
}
INJECTED: frozenset[str] = frozenset(tid for tid, (_c, _p, flagged) in EXPECTED.items() if flagged)
