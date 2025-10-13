"""Utility helpers for diagnostics report generation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    """Ensure that the given directory exists and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def export_reports(
    cat_df: "pd.DataFrame",
    summary: dict,
    out_dir: str = "diagnostics/results",
) -> dict:
    """Persist catalog diagnostics as CSV/JSON artifacts."""

    ensure_directory(out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    catalog_path = Path(out_dir) / f"catalog_{timestamp}.csv"
    summary_path = Path(out_dir) / f"catalog_summary_{timestamp}.json"

    cat_df.to_csv(catalog_path, index=False)
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    return {"catalog_csv": str(catalog_path), "summary_json": str(summary_path)}
