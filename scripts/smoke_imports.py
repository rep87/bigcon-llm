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
print("[smoke] repo_root:", REPO_ROOT)


print("[smoke] cwd:", os.getcwd())
print("[smoke] sys.path[0:5]:", sys.path[0:5])
try:
    module = importlib.import_module("app_core.panel_extract")
    print(
        "[smoke] app_core.panel_extract imported; symbols:",
        hasattr(module, "NEEDED"),
        hasattr(module, "subset_needed"),
    )
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
