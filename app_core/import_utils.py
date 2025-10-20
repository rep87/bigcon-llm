from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def ensure_repo_on_sys_path(anchor_file: str) -> str:
    root = Path(anchor_file).resolve().parent
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root_s


def import_or_raise(mod_name: str):
    try:
        __import__(mod_name)
        return sys.modules[mod_name]
    except Exception:
        print("[import-debug] cwd:", os.getcwd())
        print("[import-debug] sys.path[:5]:", sys.path[:5])
        print("[import-debug] module:", mod_name)
        traceback.print_exc()
        raise
