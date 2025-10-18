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
EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__"}
TOP_LEVEL_PATTERN = re.compile(r"^(?: {0,3}|\t{0,1})(return|yield|await|raise)\b")
MERGE_MARKERS = {"<" * 7, "=" * 7, ">" * 7}
TRIPLE_QUOTE_PATTERNS = ['"' * 3, "'" * 3]


def iter_python_files(root: Path, paths: list[str] | None = None) -> Iterable[Path]:
    if paths:
        candidates = [root / Path(p) for p in paths]
    else:
        candidates = [root]

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file() and candidate.suffix == ".py":
            yield candidate
            continue
        if not candidate.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(candidate):
            current = Path(dirpath)
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            if EXCLUDE_DIRS.intersection(current.parts):
                continue
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                yield current / filename


def read_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()


def context_window(lines: list[str], lineno: int, span: int = CONTEXT_WINDOW) -> str:
    start = max(1, lineno - span)
    end = min(len(lines), lineno + span)
    output: list[str] = []
    for idx in range(start, end + 1):
        output.append(f"{idx:5d}: {lines[idx - 1] if 0 <= idx - 1 < len(lines) else ''}")
    return "\n".join(output)


def warn_top_level_statements(path: Path, lines: list[str]) -> list[tuple[str, str]]:
    warnings: list[tuple[str, str]] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent <= 3 and TOP_LEVEL_PATTERN.match(line.strip()):
            warnings.append(
                (
                    f"[TopLevel] {path}:{idx} — {stripped.strip()}",
                    context_window(lines, idx),
                )
            )
    return warnings


def warn_conflict_markers(path: Path, lines: list[str]) -> list[tuple[str, str]]:
    warnings: list[tuple[str, str]] = []
    for idx, line in enumerate(lines, start=1):
        if any(marker in line for marker in MERGE_MARKERS):
            warnings.append(
                (
                    f"[Conflict] {path}:{idx} — {line.strip()}",
                    context_window(lines, idx),
                )
            )
    return warnings


def warn_unbalanced_triple_quotes(path: Path, text: str) -> list[tuple[str, str]]:
    warnings: list[tuple[str, str]] = []
    for marker in TRIPLE_QUOTE_PATTERNS:
        if text.count(marker) % 2 != 0:
            warnings.append((f"[TripleQuote] {path} — unmatched {marker!r}", ""))
    return warnings


def main(fail_on_warnings: bool = False, paths: list[str] | None = None) -> int:
    root = Path(os.getcwd())
    python_files = sorted(iter_python_files(root, paths))
    parse_failures = 0
    warnings: list[tuple[str, str]] = []

    for py_file in python_files:
        lines = read_lines(py_file)
        text = "\n".join(lines)
        try:
            ast.parse(text, filename=str(py_file))
        except SyntaxError as exc:  # pragma: no cover - developer aid
            parse_failures += 1
            lineno = exc.lineno or 1
            print("=" * 80)
            print(
                f"SyntaxError in {py_file}:{lineno}:{exc.offset or 0} — {exc.msg}"
            )
            print(context_window(lines, lineno))
            print("=" * 80)

        warnings.extend(warn_top_level_statements(py_file, lines))
        warnings.extend(warn_conflict_markers(py_file, lines))
        warnings.extend(warn_unbalanced_triple_quotes(py_file, text))

    if warnings:
        print("Warnings:")
        for message, context in warnings:
            print(f"  {message}")
            if context:
                print(context)

    print("-" * 80)
    print(f"Checked {len(python_files)} Python files. Parse failures: {parse_failures}.")
    print(f"Warnings emitted: {len(warnings)}.")

    if parse_failures > 0:
        return 2
    if fail_on_warnings and warnings:
        return 1
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--fail-on-warnings", action="store_true")
    args = parser.parse_args()
    sys.exit(main(fail_on_warnings=args.fail_on_warnings, paths=args.paths))
