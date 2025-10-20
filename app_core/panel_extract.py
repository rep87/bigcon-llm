from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

SAFE_COLS: Dict[str, List[str]] = {
    "id": ["ENCODED_MCT"],
    "period": ["TA_YM"],
    "age_gender": [
        "M12_MAL_1020_RAT",
        "M12_MAL_30_RAT",
        "M12_MAL_40_RAT",
        "M12_MAL_50_RAT",
        "M12_MAL_60_RAT",
        "M12_FME_1020_RAT",
        "M12_FME_30_RAT",
        "M12_FME_40_RAT",
        "M12_FME_50_RAT",
        "M12_FME_60_RAT",
    ],
    "kpi": [
        "MCT_UE_CLN_REU_RAT",
        "MCT_UE_CLN_NEW_RAT",
    ],
    "flow": [
        "RC_M1_SHC_RSD_UE_CLN_RAT",
        "RC_M1_SHC_WP_UE_CLN_RAT",
        "RC_M1_SHC_FLP_UE_CLN_RAT",
    ],
}

NEEDED: List[str] = (
    SAFE_COLS["id"]
    + SAFE_COLS["period"]
    + SAFE_COLS["age_gender"]
    + SAFE_COLS["kpi"]
    + SAFE_COLS["flow"]
)


@dataclass(frozen=True)
class _AliasSpec:
    code: str
    targets: Sequence[str]


COLUMN_ALIASES: Dict[str, _AliasSpec] = {
    "AGE_1020": _AliasSpec(code="1020", targets=("M12_MAL_1020_RAT", "M12_FME_1020_RAT")),
    "AGE_30": _AliasSpec(code="30", targets=("M12_MAL_30_RAT", "M12_FME_30_RAT")),
    "AGE_40": _AliasSpec(code="40", targets=("M12_MAL_40_RAT", "M12_FME_40_RAT")),
    "AGE_50": _AliasSpec(code="50", targets=("M12_MAL_50_RAT", "M12_FME_50_RAT")),
    "AGE_60": _AliasSpec(code="60", targets=("M12_MAL_60_RAT", "M12_FME_60_RAT")),
}


def _assert_needed_columns(df: pd.DataFrame, needed: Sequence[str]) -> None:
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise KeyError(f"[panel_extract] Missing required columns: {sorted(missing)}")


def _num(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value)
    except Exception:  # pragma: no cover - defensive guard
        return None


def _pct(value: Any) -> float | None:
    val = _num(value)
    if val is None:
        return None
    if 0.0 <= val <= 1.0:
        val *= 100.0
    if val < 0.0 or val > 100.0:
        return None
    return round(val, 1)


def _assert_close(a: float, b: float, eps: float = 1.0) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= eps


def _capture_alias_row(
    df: pd.DataFrame,
    alias_columns: Mapping[str, _AliasSpec],
    index_label: Any,
) -> Dict[str, Any]:
    if not alias_columns:
        return {}
    alias_row: Dict[str, Any] = {}
    for alias_name in alias_columns:
        if alias_name not in df.columns:
            continue
        series = df[alias_name]
        if index_label in series.index:
            alias_row[alias_name] = series.loc[index_label]
    return alias_row


def extract_latest_row(df: pd.DataFrame, mct_id: str) -> pd.Series:
    _assert_needed_columns(df, NEEDED)
    working = df.copy()
    working["ENCODED_MCT"] = working["ENCODED_MCT"].astype(str)
    subset = working[working["ENCODED_MCT"] == str(mct_id)]
    if subset.empty:
        raise ValueError(f"[panel_extract] No rows for ENCODED_MCT={mct_id}")

    period_numeric = pd.to_numeric(subset["TA_YM"], errors="coerce")
    if period_numeric.isna().all():
        raise ValueError("[panel_extract] TA_YM is not numeric-like")
    latest_idx = period_numeric.idxmax()
    return subset.loc[latest_idx]


def build_panel_dict(row: pd.Series, *, alias_row: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    alias_row = alias_row or {}
    alias_totals: Dict[str, float] = {}
    for alias_name, spec in COLUMN_ALIASES.items():
        if alias_name not in alias_row:
            continue
        total = _pct(alias_row[alias_name])
        if total is not None:
            alias_totals[spec.code] = total

    age_pairs = [
        ("1020", "청년(10-20)", "M12_MAL_1020_RAT", "M12_FME_1020_RAT"),
        ("30", "30대", "M12_MAL_30_RAT", "M12_FME_30_RAT"),
        ("40", "40대", "M12_MAL_40_RAT", "M12_FME_40_RAT"),
        ("50", "50대", "M12_MAL_50_RAT", "M12_FME_50_RAT"),
        ("60", "60대+", "M12_MAL_60_RAT", "M12_FME_60_RAT"),
    ]

    age_distribution: List[Dict[str, Any]] = []
    for code, label, male_col, female_col in age_pairs:
        male_val = _pct(row.get(male_col))
        female_val = _pct(row.get(female_col))
        values: List[float] = []
        if male_val is not None:
            values.append(male_val)
        if female_val is not None:
            values.append(female_val)
        total = sum(values) if values else None
        if total is None and code in alias_totals:
            total = alias_totals[code]
        if total is None:
            continue
        age_distribution.append({
            "code": code,
            "label": label,
            "value": round(total, 1),
        })
    age_distribution.sort(key=lambda item: item["value"], reverse=True)

    female_total = sum((_pct(row.get(col)) or 0.0) for col in SAFE_COLS["age_gender"][5:])
    male_total = sum((_pct(row.get(col)) or 0.0) for col in SAFE_COLS["age_gender"][:5])

    flow_map = {
        "residential": _pct(row.get("RC_M1_SHC_RSD_UE_CLN_RAT")),
        "workplace": _pct(row.get("RC_M1_SHC_WP_UE_CLN_RAT")),
        "floating": _pct(row.get("RC_M1_SHC_FLP_UE_CLN_RAT")),
    }

    kpi_map = {
        "revisit_rate": _pct(row.get("MCT_UE_CLN_REU_RAT")),
        "new_rate": _pct(row.get("MCT_UE_CLN_NEW_RAT")),
    }

    ta_ym = _num(row.get("TA_YM"))
    if ta_ym is None:
        raise ValueError("[panel_extract] TA_YM missing or invalid for panel row")

    warnings: List[str] = []
    age_sum = sum(item["value"] for item in age_distribution)
    gender_sum = female_total + male_total
    if not _assert_close(age_sum, 100.0):
        warnings.append(f"age_sum={age_sum:.2f}")
    if not _assert_close(gender_sum, 100.0):
        warnings.append(f"gender_sum={gender_sum:.2f}")

    return {
        "ta_ym": int(ta_ym),
        "age_distribution": age_distribution,
        "gender_share": {
            "female": round(female_total, 1),
            "male": round(male_total, 1),
        },
        "kpis": kpi_map,
        "flow": flow_map,
        "warnings": warnings,
    }


def extract_panel_for(
    df: pd.DataFrame,
    mct_id: str,
    *,
    allow_alias: bool = False,
) -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("[panel_extract] df must be a pandas.DataFrame")

    alias_candidates: Dict[str, _AliasSpec] = {}
    if allow_alias:
        alias_candidates = {
            name: spec for name, spec in COLUMN_ALIASES.items() if name in df.columns
        }

    working = df[[col for col in df.columns if col in NEEDED]].copy()
    _assert_needed_columns(working, NEEDED)

    latest_row = extract_latest_row(working, mct_id)
    alias_row = (
        _capture_alias_row(df, alias_candidates, latest_row.name) if alias_candidates else {}
    )
    return build_panel_dict(latest_row, alias_row=alias_row)


__all__ = [
    "SAFE_COLS",
    "NEEDED",
    "COLUMN_ALIASES",
    "extract_latest_row",
    "build_panel_dict",
    "extract_panel_for",
]
