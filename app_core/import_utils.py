from __future__ import annotations

import importlib
import os
import sys


def ensure_repo_on_sys_path(anchor_file: str) -> str:
    repo_root = os.path.dirname(os.path.abspath(anchor_file))
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)
    return repo_root


def import_or_raise(mod_name: str):
    try:
        return importlib.import_module(mod_name)
    except Exception:
        print(f"[import-utils] cwd={os.getcwd()}")
        print(f"[import-utils] sys.path[:5]={sys.path[:5]}")
        raise
