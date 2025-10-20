"""Structural sanity checks for repository Python files.

This developer-facing utility walks through all ``.py`` files (excluding
virtual environments, git metadata, and other transient directories) and
performs lightweight diagnostics that help catch merge artefacts which often
manifest as ``SyntaxError: 'return' outside function`` at import time.

Usage::

    python scripts/structural_sanity.py

The script emits a report describing:

* Files that fail to ``ast.parse`` along with a context window around the
  offending line.
* Heuristic warnings for potential top-level ``return``/``yield``/``await``/
  ``raise`` statements that are usually indicative of missing wrappers.
* Conflict markers (for example, sequences beginning with seven ``<``
  characters) left over from merges.
* Suspicious triple-quote counts which may hint at unterminated strings.

No changes are made to the filesystem – the script is read-only and meant for
rapid diagnosis before running the full application.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence


EXCLUDE_DIR_NAMES: set[str] = {".git", "venv", ".venv", "__pycache__"}
TOP_LEVEL_PATTERN = re.compile(r"^[ ]{0,3}(return|yield|await|raise)\b")
CONFLICT_MARKERS = ("<" * 7, "=" * 7, ">" * 7)
TRIPLE_SINGLE = "'" * 3
TRIPLE_DOUBLE = '"' * 3


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        parts = set(path.parts)
        if parts & EXCLUDE_DIR_NAMES:
            continue
        yield path


def read_lines(path: Path) -> Sequence[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def context_window(lines: Sequence[str], lineno: int, radius: int = 20) -> str:
    start = max(lineno - radius - 1, 0)
    end = min(lineno + radius, len(lines))
    numbered = []
    for idx in range(start, end):
        prefix = ">>" if idx == lineno - 1 else "  "
        numbered.append(f"{prefix}{idx + 1:5d}: {lines[idx]}")
    return "\n".join(numbered)


def warn_top_level_statements(path: Path, lines: Sequence[str]) -> list[str]:
    warnings: list[str] = []
    for idx, line in enumerate(lines, start=1):
        if TOP_LEVEL_PATTERN.match(line):
            warnings.append(f"{path}:{idx}: suspect top-level statement -> {line.strip()}")
    return warnings


def warn_conflict_markers(path: Path, lines: Sequence[str]) -> list[str]:
    warnings: list[str] = []
    for idx, line in enumerate(lines, start=1):
        if any(marker in line for marker in CONFLICT_MARKERS):
            warnings.append(f"{path}:{idx}: merge conflict marker detected -> {line.strip()}")
    return warnings


def warn_unbalanced_triple_quotes(path: Path, text: str) -> list[str]:
    warnings: list[str] = []
    for triple in (TRIPLE_SINGLE, TRIPLE_DOUBLE):
        count = text.count(triple)
        if count % 2 != 0:
            warnings.append(
                f"{path}: unmatched triple quote ({triple}) count={count}. "
                "Check for unterminated multi-line strings."
            )
    return warnings


def main() -> int:
    root = Path(os.getcwd())
    py_files = sorted(iter_python_files(root))
    parse_failures = 0
    warning_messages: list[str] = []

    for path in py_files:
        lines = read_lines(path)
        text = "\n".join(lines)

        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as exc:  # pragma: no cover - diagnostic path
            parse_failures += 1
            print("=" * 80)
            print(f"SyntaxError in {path}:{exc.lineno}:{exc.offset} -> {exc.msg}")
            print(context_window(lines, exc.lineno or 1))
            print("=" * 80)

        warning_messages.extend(warn_top_level_statements(path, lines))
        warning_messages.extend(warn_conflict_markers(path, lines))
        warning_messages.extend(warn_unbalanced_triple_quotes(path, text))

    if warning_messages:
        print("Warnings:")
        for message in warning_messages:
            print(f"  {message}")

    print("-" * 80)
    print(f"Checked {len(py_files)} Python files. Parse failures: {parse_failures}.")
    print(f"Warnings emitted: {len(warning_messages)}.")
    return 0 if parse_failures == 0 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
