"""Configuration helpers for Streamlit flags and settings."""
from __future__ import annotations

import os
from typing import Any


def get_secret(name: str, default: Any | None = None) -> Any | None:
    """Retrieve a value from ``st.secrets`` if available."""
    try:  # pragma: no cover - Streamlit may be unavailable during tests
        import streamlit as st  # type: ignore

        return st.secrets.get(name, default)  # type: ignore[attr-defined]
    except Exception:
        return default


def get_setting(name: str, default: Any | None = None) -> Any | None:
    """Resolve a configuration value using secrets, env vars, then default."""
    secret_value = get_secret(name, None)
    if secret_value is not None:
        return secret_value

    env_value = os.getenv(name)
    if env_value is not None:
        return env_value

    return default


def get_flag(name: str, default: bool) -> bool:
    """Resolve a boolean flag from configuration sources."""
    value = get_setting(name, None)
    if value is None:
        return default

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}

    return default
