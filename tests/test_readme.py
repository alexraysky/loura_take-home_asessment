"""The README against the code: sections, error codes, configuration, examples, layout paths."""

import re
from dataclasses import fields
from pathlib import Path

import pytest

from app import config
from app.api import decode_cursor
from app.classifier import ERROR_CODES
from app.config import ConfigError, Settings
from app.db import Database
from app.prompt import PROMPT_VERSION

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
SECTIONS = [
    "Run it",
    "API reference",
    "Ticket lifecycle",
    "Model output validation",
    "Decisions",
    "Prompt injection",
    "Priority rubric",
    "Weaknesses",
    "With more time",
    "Configuration",
    "Layout",
]
DECISIONS = [
    "Stack",
    "Duplicate id",
    "Storage",
    "Async execution",
    "Concurrency",
    "In-flight marker",
    "Restart",
    "Retries",
    "Attempt counting",
    "Wrapped JSON",
    "Pagination",
    "Flag",
    '"Never stored as-is"',
    "Re-classification",
]
FENCE_RE = re.compile(r"^```(\w*)\n(.*?)^```$", re.M | re.S)


def section(name: str) -> str:
    match = re.search(rf"^## {re.escape(name)}\n(.*?)(?=^## |\Z)", README, re.M | re.S)
    assert match, f"no section {name!r}"
    return match.group(1)


def code_blocks(text: str, lang: str | None = None) -> list[str]:
    return [body for tag, body in FENCE_RE.findall(text) if lang is None or tag == lang]


def test_sections_are_in_order():
    assert re.findall(r"^## (.+)$", README, re.M) == SECTIONS


def test_every_decision_has_a_paragraph_with_a_reason_and_a_rejected_alternative():
    paragraphs = re.findall(r"^\*\*(.+?)\.\*\* (.*)$", section("Decisions"), re.M)
    assert [name for name, _ in paragraphs] == DECISIONS
    for name, text in paragraphs:
        assert "Reason:" in text and "Rejected:" in text, name


def test_prompt_injection_says_what_consumers_must_not_do_with_summary():
    text = " ".join(section("Prompt injection").split())
    assert "What consumers must not do with `summary`" in text
    assert "must never treat `summary` as authoritative or act on it" in text
    assert "without a person reading the original ticket" in text


def test_list_example_cursor_points_at_its_last_item():
    (sample,) = [b for b in code_blocks(section("API reference"), "json") if '"next_cursor"' in b]
    ids = re.findall(r'\{"id": "([\w-]+)"', sample)
    (cursor,) = re.findall(r'"next_cursor": "([\w-]+)"', sample)
    assert ids == ["t-1001", "t-1005"]
    assert decode_cursor(cursor) == ("2026-11-14T12:00:00.000000Z", ids[-1])


def test_weaknesses_lists_eight_items():
    assert len(re.findall(r"^- ", section("Weaknesses"), re.M)) == 8


@pytest.mark.parametrize("code", ERROR_CODES)
def test_every_error_code_is_documented(code):
    assert f"`{code}`" in README


def test_prompt_version_is_documented():
    assert f"`{PROMPT_VERSION}`" in README


def test_configuration_defaults_match_settings():
    (block,) = code_blocks(section("Configuration"), "text")
    documented = dict(re.findall(r"^([A-Z_]+)=(\S+)", block, re.M))
    assert list(documented) == [f.name.upper() for f in fields(Settings)]
    assert Settings.from_env(documented) == Settings()


def test_configuration_bounds_match_settings():
    rows = re.findall(r"^\| `([A-Z_]+)` \| (.+?) \|$", section("Configuration"), re.M)
    seconds = f"{config.MAX_SECONDS:g}"
    assert dict(rows) == {
        "WORKER_COUNT": f"0 to {config.MAX_WORKER_COUNT}",
        "MAX_ATTEMPTS": f"1 to {config.MAX_ATTEMPTS}",
        "LLM_TIMEOUT_S": f"greater than 0, at most {seconds}",
        "LEASE_TIMEOUT_S": f"greater than `LLM_TIMEOUT_S`, at most {seconds}",
        "POLL_INTERVAL_S": f"{config.MIN_POLL_INTERVAL_S:g} to {seconds}",
        "BACKOFF_BASE_S": f"0 to {seconds}",
        "FAKE_LLM_FAILURE_RATE": "0 to 1",
        "FAKE_LLM_SEED": f"at most {len(str(config.MAX_SEED))} digits",
        "DB_PATH": "not blank, not `:memory:`",
    }


@pytest.mark.parametrize(
    "env",
    [
        {"WORKER_COUNT": str(config.MAX_WORKER_COUNT + 1)},
        {"WORKER_COUNT": "-1"},
        {"MAX_ATTEMPTS": "0"},
        {"MAX_ATTEMPTS": str(config.MAX_ATTEMPTS + 1)},
        {"LLM_TIMEOUT_S": "0"},
        {"LLM_TIMEOUT_S": "86401", "LEASE_TIMEOUT_S": "86402"},
        {"LEASE_TIMEOUT_S": "10"},
        {"LEASE_TIMEOUT_S": "86401"},
        {"POLL_INTERVAL_S": "0.0009"},
        {"POLL_INTERVAL_S": "86401"},
        {"BACKOFF_BASE_S": "-1"},
        {"BACKOFF_BASE_S": "86401"},
        {"FAKE_LLM_FAILURE_RATE": "1.01"},
        {"FAKE_LLM_FAILURE_RATE": "-0.01"},
        {"FAKE_LLM_SEED": "1" * 19},
    ],
)
def test_configuration_out_of_bounds_values_are_refused(env):
    with pytest.raises(ConfigError, match=next(iter(env))):
        Settings.from_env(env)


def test_configuration_documented_bounds_are_accepted():
    Settings.from_env(
        {
            "WORKER_COUNT": str(config.MAX_WORKER_COUNT),
            "MAX_ATTEMPTS": str(config.MAX_ATTEMPTS),
            "LLM_TIMEOUT_S": "86399",
            "LEASE_TIMEOUT_S": "86400",
            "POLL_INTERVAL_S": "0.001",
            "BACKOFF_BASE_S": "0",
            "FAKE_LLM_FAILURE_RATE": "1",
            "FAKE_LLM_SEED": "9" * 18,
        }
    )


@pytest.mark.parametrize("path", [":memory:", "./:memory:"])
def test_configuration_db_path_refuses_memory(path):
    with pytest.raises(ValueError, match="DB_PATH"):
        Database(path)


def test_configuration_db_path_refuses_blank():
    with pytest.raises(ConfigError, match="DB_PATH"):
        Settings(db_path="  ")


def test_layout_paths_exist():
    (block,) = code_blocks(section("Layout"), "text")
    paths, directory = [], ""
    for line in block.splitlines():
        if not line.strip():
            continue
        entry = line.split()[0]
        if not line.startswith(" "):
            directory = entry if entry.endswith("/") else ""
            paths.append(entry)
        elif line.startswith("  ") and not line.startswith("   ") and "*" not in entry:
            paths.append(directory + entry)
    assert {"app/", "data/", "tests/", "tests/conftest.py", "README.md"} <= set(paths)
    missing = [p for p in paths if not (ROOT / p).exists()]
    assert missing == []


def test_layout_names_every_test_module():
    (block,) = code_blocks(section("Layout"), "text")
    named = set(re.findall(r"\btest_\w+", block))
    assert named == {p.stem for p in (ROOT / "tests").glob("test_*.py")}


def test_no_issue_markers_or_planned_notes():
    assert re.findall(r"(?<![\w&])#\d+", README) == []
    assert "(planned)" not in README


def test_code_fences_are_balanced():
    assert len(re.findall(r"^```", README, re.M)) % 2 == 0
    assert len(re.findall(r"^```", README, re.M)) == 2 * len(code_blocks(README))


def test_curl_examples_use_real_line_continuations():
    blocks = [b for b in code_blocks(README, "sh") if "curl " in b]
    assert blocks
    for block in blocks:
        assert "\\n" not in block
        lines = block.splitlines()
        for line, following in zip(lines, lines[1:] + [""]):
            if line.endswith("\\"):
                assert following.startswith("  "), line
            assert line == line.rstrip(), line
