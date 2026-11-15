"""The README against the code: sections, error codes, configuration defaults, layout paths."""

import re
from dataclasses import fields
from pathlib import Path

import pytest

from app.classifier import ERROR_CODES
from app.config import Settings
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


def test_every_decision_has_a_paragraph():
    assert re.findall(r"^\*\*(.+?)\.\*\* ", section("Decisions"), re.M) == DECISIONS


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


def test_layout_paths_exist():
    (block,) = code_blocks(section("Layout"), "text")
    paths, directory = [], ""
    for line in block.splitlines():
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
