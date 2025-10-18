"""Configuration helpers for Streamlit flags."""
from __future__ import annotations

import os
from typing import Any


def get_flag(name: str, default: bool) -> bool:
    """Resolve a boolean flag from environment variables or Streamlit secrets."""
    value = os.getenv(name)
    if value is not None:
        return value.lower() in {"1", "true", "yes", "y", "on"}

    try:
        import streamlit as st  # type: ignore

        secret_value: Any = st.secrets.get(name)  # type: ignore[attr-defined]
        if secret_value is None:
            return default
        if isinstance(secret_value, bool):
            return secret_value
        if isinstance(secret_value, (int, float)):
            return bool(secret_value)
        if isinstance(secret_value, str):
            return secret_value.lower() in {"1", "true", "yes", "y", "on"}
    except Exception:
        return default

    return default
