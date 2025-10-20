"""Detect suspicious top-level ``return`` statements.

This script scans every Python file in the repository (excluding virtual
environments, git metadata, and cached directories) and emits a diagnostic if a
line begins with the keyword ``return`` at indentation level zero. Such
patterns usually indicate that a helper function lost its wrapper during a
merge, ultimately leading to ``SyntaxError: 'return' outside function`` at
runtime.

Usage::

    python scripts/check_top_level_returns.py

The script exits with a non-zero status when at least one offending line is
found, making it suitable for CI or pre-commit hooks.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable


EXCLUDE_DIR_NAMES: set[str] = {".git", "venv", ".venv", "__pycache__"}
RETURN_PATTERN = re.compile(r"^return\b")


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        parts = set(path.parts)
        if parts & EXCLUDE_DIR_NAMES:
            continue
        yield path


def main() -> int:
    root = Path(os.getcwd())
    offenders: list[str] = []

    for path in iter_python_files(root):
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for lineno, line in enumerate(fh, start=1):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if RETURN_PATTERN.match(line):
                    offenders.append(f"{path}:{lineno}: {line.strip()}")

    if offenders:
        print("Top-level return statements detected:")
        for entry in offenders:
            print(f"  {entry}")
        print("Consider wrapping the relevant code in a function.")
        return 1

    print("No top-level return statements found.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
