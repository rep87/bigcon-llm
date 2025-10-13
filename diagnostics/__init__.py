"""Diagnostics utilities for cataloging and reporting merchant data."""

from .catalog import load_set1, build_catalog, summarize_catalog
from .report import export_reports

try:  # pragma: no cover - optional modules
    from .resolver_audit import run_resolver_audit, summarize_audit, export_audit
except Exception:  # pragma: no cover - keep API surface predictable
    run_resolver_audit = summarize_audit = export_audit = None

try:  # pragma: no cover - optional modules
    from .access_audit import run_access_audit, summarize_access, export_access
except Exception:  # pragma: no cover - keep API surface predictable
    run_access_audit = summarize_access = export_access = None

__all__ = [
    "load_set1",
    "build_catalog",
    "summarize_catalog",
    "export_reports",
    "run_resolver_audit",
    "summarize_audit",
    "export_audit",
    "run_access_audit",
    "summarize_access",
    "export_access",
]
