"""Exact-column panel extraction helpers for Agent-1 summaries."""

from __future__ import annotations

from typing import Any, Dict, List

import math

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


def _assert_needed_columns(df: pd.DataFrame, needed: List[str]) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"[panel_extract] Missing required columns: {missing}")


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


def extract_latest_row(df: pd.DataFrame, mct_id: str) -> pd.Series:
    _assert_needed_columns(df, NEEDED)
    if "ENCODED_MCT" not in df.columns:
        raise KeyError("[panel_extract] ENCODED_MCT column missing after validation")

    working = df.copy()
    working["ENCODED_MCT"] = working["ENCODED_MCT"].astype(str)
    sub = working[working["ENCODED_MCT"] == str(mct_id)]
    if sub.empty:
        raise ValueError(f"[panel_extract] No rows for ENCODED_MCT={mct_id}")

    working_period = pd.to_numeric(sub["TA_YM"], errors="coerce")
    if working_period.isna().all():
        raise ValueError("[panel_extract] TA_YM is not numeric-like")
    latest_idx = working_period.idxmax()
    return sub.loc[latest_idx]


def build_panel_dict(row: pd.Series) -> Dict[str, Any]:
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
        if male_val is None and female_val is None:
            continue
        total = (male_val or 0.0) + (female_val or 0.0)
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


def extract_panel_for(df: pd.DataFrame, mct_id: str) -> Dict[str, Any]:
    _assert_needed_columns(df, NEEDED)
    latest_row = extract_latest_row(df, mct_id)
    return build_panel_dict(latest_row)


__all__ = [
    "SAFE_COLS",
    "NEEDED",
    "extract_latest_row",
    "build_panel_dict",
    "extract_panel_for",
]
