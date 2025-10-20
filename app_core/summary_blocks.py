from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .panel_extract import _pct, latest_valid_row, subset_needed


def _age_buckets_from_row(row: pd.Series) -> list[dict]:
    def _age(code: str, label: str, male: str, female: str) -> dict:
        value = (_pct(row[male]) or 0.0) + (_pct(row[female]) or 0.0)
        return {"code": code, "label": label, "value": round(value, 1)}

    return [
        _age("1020", "10–20대", "M12_MAL_1020_RAT", "M12_FME_1020_RAT"),
        _age("30", "30대", "M12_MAL_30_RAT", "M12_FME_30_RAT"),
        _age("40", "40대", "M12_MAL_40_RAT", "M12_FME_40_RAT"),
        _age("50", "50대", "M12_MAL_50_RAT", "M12_FME_50_RAT"),
        _age("60", "60대+", "M12_MAL_60_RAT", "M12_FME_60_RAT"),
    ]


def _flow_from_row(row: pd.Series) -> dict:
    return {
        "거주": float(_pct(row["RC_M1_SHC_RSD_UE_CLN_RAT"])) if _pct(row["RC_M1_SHC_RSD_UE_CLN_RAT"]) is not None else None,
        "직장": float(_pct(row["RC_M1_SHC_WP_UE_CLN_RAT"])) if _pct(row["RC_M1_SHC_WP_UE_CLN_RAT"]) is not None else None,
        "유동": float(_pct(row["RC_M1_SHC_FLP_UE_CLN_RAT"])) if _pct(row["RC_M1_SHC_FLP_UE_CLN_RAT"]) is not None else None,
    }


def _kpis_from_row(row: pd.Series) -> dict:
    return {
        "new_rate": float(_pct(row["MCT_UE_CLN_NEW_RAT"])) if _pct(row["MCT_UE_CLN_NEW_RAT"]) is not None else None,
        "revisit_rate": float(_pct(row["MCT_UE_CLN_REU_RAT"])) if _pct(row["MCT_UE_CLN_REU_RAT"]) is not None else None,
    }


def pick_latest_baseline_trend_yoy(df: pd.DataFrame, mct_id: str) -> Dict[str, Any]:
    working = subset_needed(df)
    working = working.copy()
    working["__TA_YM_INT__"] = pd.to_numeric(working["TA_YM"], errors="coerce").astype("Int64")
    working = working.dropna(subset=["__TA_YM_INT__"])

    latest_row, latest_ym = latest_valid_row(working, mct_id)

    def _row_for(ym: int) -> Optional[pd.Series]:
        match = working[(working["ENCODED_MCT"] == mct_id) & (working["__TA_YM_INT__"] == ym)]
        if match.empty:
            return None
        return match.iloc[0]

    prev1 = _row_for(latest_ym - 1)
    prev2 = _row_for(latest_ym - 2)
    yoy_row = _row_for(latest_ym - 100)

    ages = sorted(_age_buckets_from_row(latest_row), key=lambda x: -x["value"])
    flow = _flow_from_row(latest_row)
    kpi = _kpis_from_row(latest_row)

    def _delta(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
        if curr is None or prev is None:
            return None
        try:
            return round(float(curr) - float(prev), 1)
        except Exception:  # pragma: no cover - defensive guard
            return None

    trend: Dict[str, Any] = {}
    if prev2 is not None:
        prev2_k = _kpis_from_row(prev2)
        trend["revisit_pp_vs_2m"] = _delta(kpi["revisit_rate"], prev2_k["revisit_rate"])
        trend["new_pp_vs_2m"] = _delta(kpi["new_rate"], prev2_k["new_rate"])
    elif prev1 is not None:
        prev1_k = _kpis_from_row(prev1)
        trend["revisit_pp_vs_1m"] = _delta(kpi["revisit_rate"], prev1_k["revisit_rate"])
        trend["new_pp_vs_1m"] = _delta(kpi["new_rate"], prev1_k["new_rate"])

    yoy: Optional[Dict[str, Any]] = None
    if yoy_row is not None:
        yoy_k = _kpis_from_row(yoy_row)
        yoy = {
            "revisit_pp_yoy": _delta(kpi["revisit_rate"], yoy_k["revisit_rate"]),
            "new_pp_yoy": _delta(kpi["new_rate"], yoy_k["new_rate"]),
        }

    return {
        "latest_ym": int(latest_ym),
        "age_top3": ages[:3],
        "age_all": ages,
        "flow": {k: (round(v, 1) if v is not None else None) for k, v in flow.items()},
        "kpi": {k: (round(v, 1) if v is not None else None) for k, v in kpi.items()},
        "trend": trend,
        "yoy": yoy,
        "guard_fallback": bool(latest_row.attrs.get("guard_fallback")),
    }


__all__ = [
    "pick_latest_baseline_trend_yoy",
]
