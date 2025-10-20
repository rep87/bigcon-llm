import importlib
import traceback
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

print("[smoke] cwd:", os.getcwd())
print("[smoke] repo_root:", ROOT_STR)
try:
    m = importlib.import_module("app_core.panel_extract")
    names = [n for n in dir(m) if not n.startswith("_")]
    print("[smoke] file:", getattr(m, "__file__", "<?>"))
    print("[smoke] exports:", sorted(names))
    assert any(hasattr(m, n) for n in ("subset_needed", "select_needed", "get_required_subset"))
    assert any(hasattr(m, n) for n in ("NEEDED", "REQUIRED_COLS", "NEEDED_COLS"))
    print("[smoke] OK")
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
