#!/usr/bin/env python3
"""Structural sanity checks for Python sources.

This dev-only script scans the repository for indicators of structural drift
such as syntax errors, top-level control statements, merge conflict markers,
and unbalanced triple quotes. Findings are printed with a small context window
so regressions can be triaged quickly.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path
from typing import Iterable

CONTEXT_WINDOW = 20
TOP_LEVEL_PATTERN = re.compile(r"^(?: {0,3}|\t{0,1})(return|yield|await|raise)\b")
MERGE_MARKERS = {"<" * 7, "=" * 7, ">" * 7}
TRIPLE_QUOTE_PATTERNS = ['"' * 3, "'" * 3]


def iter_python_files(root: Path) -> Iterable[Path]:
    skip_names = {".git", "venv", ".venv", "__pycache__"}
    for path in root.rglob("*.py"):
        if any(part in skip_names for part in path.parts):
            continue
        yield path


def read_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()


def print_context(lines: list[str], line_no: int) -> None:
    start = max(0, line_no - CONTEXT_WINDOW - 1)
    end = min(len(lines), line_no + CONTEXT_WINDOW)
    for idx in range(start, end):
        prefix = ">" if idx == line_no - 1 else " "
        print(f"{prefix}{idx + 1:5d}: {lines[idx]}")


def check_syntax(path: Path, text: str, summary: dict[str, int]) -> None:
    try:
        ast.parse(text, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - developer aid
        summary.setdefault("syntax_errors", 0)
        summary["syntax_errors"] += 1
        print(f"\n[SyntaxError] {path}:{exc.lineno}:{exc.offset} — {exc.msg}")
        lines = text.splitlines()
        if exc.lineno:
            print_context(lines, exc.lineno)


def check_top_level_statements(path: Path, lines: list[str], summary: dict[str, int]) -> None:
    for idx, line in enumerate(lines, start=1):
        if TOP_LEVEL_PATTERN.search(line):
            summary.setdefault("top_level_statements", 0)
            summary["top_level_statements"] += 1
            print(f"\n[TopLevel] {path}:{idx} — {line.strip()}")
            print_context(lines, idx)


def check_merge_markers(path: Path, lines: list[str], summary: dict[str, int]) -> None:
    for idx, line in enumerate(lines, start=1):
        if any(marker in line for marker in MERGE_MARKERS):
            summary.setdefault("merge_markers", 0)
            summary["merge_markers"] += 1
            print(f"\n[Conflict] {path}:{idx} — {line.strip()}")
            print_context(lines, idx)


def check_triple_quotes(path: Path, text: str, summary: dict[str, int]) -> None:
    counts = {pat: text.count(pat) for pat in TRIPLE_QUOTE_PATTERNS}
    unbalanced = [pat for pat, count in counts.items() if count % 2 != 0]
    if unbalanced:
        summary.setdefault("unbalanced_triple_quotes", 0)
        summary["unbalanced_triple_quotes"] += 1
        print(f"\n[TripleQuote] {path} — unmatched delimiters: {', '.join(unbalanced)}")


def main() -> int:
    root = Path(os.getcwd())
    summary: dict[str, int] = {}
    for py_file in iter_python_files(root):
        text_lines = read_lines(py_file)
        text = "\n".join(text_lines)
        check_syntax(py_file, text, summary)
        check_top_level_statements(py_file, text_lines, summary)
        check_merge_markers(py_file, text_lines, summary)
        check_triple_quotes(py_file, text, summary)

    if not summary:
        print("All clear — no structural anomalies detected.")
    else:
        print("\nSummary:")
        for key, value in sorted(summary.items()):
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
