"""Lightweight import smoke tests for critical modules."""
from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("[smoke] cwd:", os.getcwd())
try:
    module = importlib.import_module("app_core.panel_extract")
    names = [name for name in dir(module) if not name.startswith("_")]
    print("[smoke] exports:", sorted(names))
    assert (
        hasattr(module, "subset_needed")
        or hasattr(module, "select_needed")
        or hasattr(module, "get_required_subset")
    )
    assert (
        hasattr(module, "NEEDED")
        or hasattr(module, "REQUIRED_COLS")
        or hasattr(module, "NEEDED_COLS")
    )
    print("[smoke] panel_extract OK")
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
