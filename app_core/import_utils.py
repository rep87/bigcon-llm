"""Utilities for robust imports within the Streamlit app."""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any


def ensure_repo_on_sys_path(anchor_file: str = __file__, levels_up: int = 1) -> str:
    """Ensure the repository root is present on ``sys.path``.

    Parameters
    ----------
    anchor_file:
        File path used to locate the repository root. Defaults to the caller's
        ``__file__``.
    levels_up:
        Number of directory levels to ascend from ``anchor_file`` to reach the
        repository root.

    Returns
    -------
    str
        The resolved repository root that was added or already present on the
        import path.
    """

    here = Path(anchor_file).resolve()
    repo = here
    for _ in range(max(levels_up, 0)):
        repo = repo.parent
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_str


def import_with_debug(mod_name: str) -> Any:
    """Import ``mod_name`` while emitting helpful diagnostics on failure."""

    try:
        return __import__(mod_name, fromlist=["*"])
    except Exception:
        print("[import-debug] cwd     :", os.getcwd())
        print("[import-debug] sys.path:", sys.path[:10])
        print("[import-debug] trying to import:", mod_name)
        traceback.print_exc()
        raise
