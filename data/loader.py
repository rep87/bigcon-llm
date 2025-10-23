"""Data catalog loader and profiling helpers for Shinhan card datasets."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "shinhan"
CATALOG_PATH = Path(__file__).resolve().parent.parent / "schemas" / "metrics_catalog.json"
ENCODING = "cp949"
ID_COLUMN = "ENCODED_MCT"
PERIOD_FALLBACK = "TA_YM"
SENTINEL_THRESHOLD = -99990

DATASETS = {
    "set1": "big_data_set1_f.csv",
    "set2": "big_data_set2_f.csv",
    "set3": "big_data_set3_f.csv",
}


@lru_cache(maxsize=1)
def get_catalog() -> Dict[str, Dict[str, Any]]:
    """Return the metrics catalog loaded from JSON."""
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"metrics catalog not found: {CATALOG_PATH}")
    with CATALOG_PATH.open(encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError("metrics catalog must be a JSON object")
    return data


@lru_cache(maxsize=None)
def _load_dataset(source: str) -> pd.DataFrame:
    if source not in DATASETS:
        raise KeyError(f"unknown dataset source: {source}")
    csv_path = DATA_DIR / DATASETS[source]
    if not csv_path.exists():
        raise FileNotFoundError(f"dataset not found: {csv_path}")
    df = pd.read_csv(csv_path, encoding=ENCODING)
    if ID_COLUMN in df.columns:
        df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    period_col = PERIOD_FALLBACK
    if period_col in df.columns:
        df[period_col] = df[period_col].astype(str)
    return df


def _to_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, str) and not value.strip():
            return None
        num = float(value)
        if num <= SENTINEL_THRESHOLD:
            return None
        return num
    except Exception:
        return None


def _format_period(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        year = digits[:4]
        month = digits[4:6]
        if month.isdigit():
            return f"{year}-{month}"
    return text


def _period_series(subset: pd.DataFrame, period_key: str) -> Tuple[pd.Series, pd.Series]:
    raw = subset[period_key].astype(str)
    numeric = pd.to_numeric(raw, errors="coerce")
    return raw, numeric


def _compute_time_span(period_numeric: pd.Series) -> str | None:
    valid = period_numeric.dropna()
    if valid.empty:
        return None
    start = int(valid.min())
    end = int(valid.max())
    if start > end:
        start, end = end, start
    start_text = _format_period(start)
    end_text = _format_period(end)
    if start_text and end_text:
        return f"{start_text}~{end_text}"
    return None


def _compute_recent_coverage(series: pd.Series, period_numeric: pd.Series, window: int = 3) -> float | None:
    if series.empty or period_numeric.empty:
        return None
    valid_periods = period_numeric.dropna().astype(int)
    if valid_periods.empty:
        return None
    unique_periods = sorted(valid_periods.unique())
    tail_periods = unique_periods[-window:]
    if not tail_periods:
        return None
    mask = period_numeric.isin(tail_periods)
    window_values = pd.to_numeric(series[mask], errors="coerce")
    total = mask.sum()
    if total == 0:
        return None
    valid = window_values.notna().sum()
    return float(valid / total)


def profile_columns(merchant_id: str) -> List[Dict[str, Any]]:
    """Return profiling information for all catalog columns for a merchant."""
    catalog = get_catalog()
    results: List[Dict[str, Any]] = []
    merchant_id = str(merchant_id)

    for column_name, meta in catalog.items():
        source = meta.get("source")
        if not source:
            continue
        try:
            dataset = _load_dataset(source)
        except Exception as exc:
            results.append(
                {
                    "source": source,
                    "column": column_name,
                    "label": meta.get("label", column_name),
                    "value_recent": None,
                    "value_prev": None,
                    "coverage": 0.0,
                    "recent_coverage": 0.0,
                    "recency": None,
                    "time_span": None,
                    "stddev": None,
                    "na_rate": 1.0,
                    "sufficiency": "Low",
                    "warning": f"dataset_load_failed: {exc}",
                    "meta": meta,
                }
            )
            continue

        if ID_COLUMN not in dataset.columns:
            results.append(
                {
                    "source": source,
                    "column": column_name,
                    "label": meta.get("label", column_name),
                    "value_recent": None,
                    "value_prev": None,
                    "coverage": 0.0,
                    "recent_coverage": 0.0,
                    "recency": None,
                    "time_span": None,
                    "stddev": None,
                    "na_rate": 1.0,
                    "sufficiency": "Low",
                    "warning": f"missing_id_column:{ID_COLUMN}",
                    "meta": meta,
                }
            )
            continue

        subset = dataset[dataset[ID_COLUMN] == merchant_id].copy()
        if subset.empty or column_name not in subset.columns:
            results.append(
                {
                    "source": source,
                    "column": column_name,
                    "label": meta.get("label", column_name),
                    "value_recent": None,
                    "value_prev": None,
                    "coverage": 0.0,
                    "recent_coverage": 0.0,
                    "recency": None,
                    "time_span": None,
                    "stddev": None,
                    "na_rate": 1.0,
                    "sufficiency": "Low",
                    "warning": "column_missing_or_no_rows",
                    "meta": meta,
                }
            )
            continue

        period_key = meta.get("period_key", PERIOD_FALLBACK)
        has_period = period_key in subset.columns
        recency = None
        value_recent = None
        value_prev = None
        recent_coverage = None
        time_span = None

        series = pd.to_numeric(subset[column_name], errors="coerce")
        series = series.where(series > SENTINEL_THRESHOLD)
        non_null = int(series.notna().sum())
        total = int(len(series)) if len(series) else 0
        coverage = float(non_null / total) if total else 0.0
        na_rate = 1.0 - coverage if total else 1.0
        stddev = float(series.std(ddof=0)) if non_null > 1 else None

        if has_period:
            raw_period, period_numeric = _period_series(subset, period_key)
            valid_subset = subset.assign(_period_numeric=period_numeric)
            valid_subset = valid_subset.dropna(subset=["_period_numeric"])
            valid_subset = valid_subset.sort_values("_period_numeric")
            if not valid_subset.empty:
                last_row = valid_subset.iloc[-1]
                value_recent = _to_float(last_row[column_name])
                recency = _format_period(last_row[period_key])
                if len(valid_subset) > 1:
                    prev_row = valid_subset.iloc[-2]
                    value_prev = _to_float(prev_row[column_name])
            recent_coverage = _compute_recent_coverage(series, period_numeric)
            time_span = _compute_time_span(period_numeric)
        else:
            # static table: treat the latest value as last row
            last_row = subset.iloc[-1]
            value_recent = _to_float(last_row[column_name])
            recency = None
            time_span = None
            recent_coverage = coverage

        null_tolerance = float(meta.get("null_tolerance", 0.3))
        high_threshold = max(0.0, min(1.0, 1.0 - null_tolerance / 2))
        medium_threshold = max(0.0, min(1.0, 1.0 - null_tolerance))
        coverage_to_use = recent_coverage if recent_coverage is not None else coverage
        if coverage_to_use is None:
            sufficiency = "Low"
        elif coverage_to_use >= high_threshold:
            sufficiency = "High"
        elif coverage_to_use >= medium_threshold:
            sufficiency = "Medium"
        else:
            sufficiency = "Low"

        results.append(
            {
                "source": source,
                "column": column_name,
                "label": meta.get("label", column_name),
                "value_recent": value_recent,
                "value_prev": value_prev,
                "coverage": coverage,
                "recent_coverage": recent_coverage if recent_coverage is not None else coverage,
                "recency": recency,
                "time_span": time_span,
                "stddev": stddev,
                "na_rate": na_rate,
                "sufficiency": sufficiency,
                "warning": None,
                "meta": meta,
            }
        )

    return results
