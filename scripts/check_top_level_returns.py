#!/usr/bin/env python3
"""Fail fast when a Python file contains a module-level return statement."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Iterable


class ModuleReturnVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[ast.AST] = []
        self.violations: list[ast.Return] = []

    def visit(self, node: ast.AST) -> None:  # type: ignore[override]
        self.stack.append(node)
        super().visit(node)
        self.stack.pop()

    def visit_Return(self, node: ast.Return) -> None:  # noqa: N802 (ast API)
        if any(isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) for parent in self.stack[:-1]):
            return
        if any(isinstance(parent, ast.ClassDef) for parent in self.stack[:-1]):
            return
        self.violations.append(node)
        self.generic_visit(node)


def iter_python_files(root: Path) -> Iterable[Path]:
    skip_names = {".git", "venv", ".venv", "__pycache__"}
    for path in root.rglob("*.py"):
        if any(part in skip_names for part in path.parts):
            continue
        yield path


def check_file(path: Path) -> list[ast.Return]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []  # syntax errors handled elsewhere
    visitor = ModuleReturnVisitor()
    visitor.visit(tree)
    return visitor.violations


def main() -> int:
    root = Path(os.getcwd())
    failures = []
    for file_path in iter_python_files(root):
        violations = check_file(file_path)
        for violation in violations:
            failures.append((file_path, violation.lineno))

    if failures:
        for file_path, lineno in failures:
            print(f"Module-level return detected: {file_path}:{lineno}")
        return 1

    print("No module-level return statements detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
