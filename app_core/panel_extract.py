from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

# Explicit whitelist of columns that may be accessed from the panel CSV.
SAFE_COLS: dict[str, list[str]] = {
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

# Flattened whitelist to enforce exact column access.
NEEDED: list[str] = (
    SAFE_COLS["id"]
    + SAFE_COLS["period"]
    + SAFE_COLS["age_gender"]
    + SAFE_COLS["kpi"]
    + SAFE_COLS["flow"]
)


def subset_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy containing only the required columns.

    Raises
    ------
    KeyError
        If any required column is missing from ``df``.
    """

    missing = [col for col in NEEDED if col not in df.columns]
    if missing:
        raise KeyError(f"[panel_extract] Missing required columns: {sorted(missing)}")
    return df[NEEDED].copy()


# Backward compatibility aliases
select_needed = subset_needed
get_required_subset = subset_needed
NEEDED_COLS = NEEDED
REQUIRED_COLS = NEEDED


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:  # pragma: no cover - defensive guard
        return None


def _pct(value: Any) -> float | None:
    val = _num(value)
    if val is None:
        return None
    # Dataset encodes rates on a 0-100 scale. Values outside are dropped.
    if val < 0 or val > 100:
        return None
    return float(val)


def _age_bucket_sum(row: pd.Series, columns: Iterable[str]) -> float:
    values = [v for v in (_pct(row.get(col)) for col in columns) if v is not None]
    return float(sum(values)) if values else 0.0


def latest_valid_row(df: pd.DataFrame, mct_id: str, eps: float = 0.5) -> tuple[pd.Series, int]:
    """Return the most recent row that passes guard checks.

    If no row satisfies the guard conditions the latest available row is
    returned and marked with ``row.attrs["guard_fallback"] = True``.
    """

    sub = df[df["ENCODED_MCT"].astype(str) == str(mct_id)].copy()
    if sub.empty:
        raise ValueError(f"[panel_extract] No rows for ENCODED_MCT={mct_id}")

    sub["__TA_YM_INT__"] = pd.to_numeric(sub["TA_YM"], errors="coerce").astype("Int64")
    sub = sub.dropna(subset=["__TA_YM_INT__"])
    if sub.empty:
        raise ValueError("[panel_extract] TA_YM not parseable")

    def _row_ok(r: pd.Series) -> bool:
        age_1020 = _age_bucket_sum(r, ["M12_MAL_1020_RAT", "M12_FME_1020_RAT"])
        age_30 = _age_bucket_sum(r, ["M12_MAL_30_RAT", "M12_FME_30_RAT"])
        age_40 = _age_bucket_sum(r, ["M12_MAL_40_RAT", "M12_FME_40_RAT"])
        age_50 = _age_bucket_sum(r, ["M12_MAL_50_RAT", "M12_FME_50_RAT"])
        age_60 = _age_bucket_sum(r, ["M12_MAL_60_RAT", "M12_FME_60_RAT"])
        age_sum = age_1020 + age_30 + age_40 + age_50 + age_60

        female = _age_bucket_sum(
            r,
            [
                "M12_FME_1020_RAT",
                "M12_FME_30_RAT",
                "M12_FME_40_RAT",
                "M12_FME_50_RAT",
                "M12_FME_60_RAT",
            ],
        )
        male = _age_bucket_sum(
            r,
            [
                "M12_MAL_1020_RAT",
                "M12_MAL_30_RAT",
                "M12_MAL_40_RAT",
                "M12_MAL_50_RAT",
                "M12_MAL_60_RAT",
            ],
        )
        gender_sum = female + male

        flow_sum = _age_bucket_sum(
            r,
            [
                "RC_M1_SHC_RSD_UE_CLN_RAT",
                "RC_M1_SHC_WP_UE_CLN_RAT",
                "RC_M1_SHC_FLP_UE_CLN_RAT",
            ],
        )

        def _ok(total: float) -> bool:
            return abs(total - 100.0) <= eps

        return _ok(age_sum) and _ok(gender_sum) and _ok(flow_sum)

    sub = sub.sort_values("__TA_YM_INT__", ascending=False)
    for _, row in sub.iterrows():
        if _row_ok(row):
            row = row.copy()
            row.attrs["guard_fallback"] = False
            return row, int(row["__TA_YM_INT__"])

    fallback = sub.iloc[0].copy()
    fallback.attrs["guard_fallback"] = True
    return fallback, int(fallback["__TA_YM_INT__"])


def _age_buckets(row: pd.Series) -> list[dict[str, Any]]:
    specs = [
        ("1020", "10–20대", "M12_MAL_1020_RAT", "M12_FME_1020_RAT"),
        ("30", "30대", "M12_MAL_30_RAT", "M12_FME_30_RAT"),
        ("40", "40대", "M12_MAL_40_RAT", "M12_FME_40_RAT"),
        ("50", "50대", "M12_MAL_50_RAT", "M12_FME_50_RAT"),
        ("60", "60대+", "M12_MAL_60_RAT", "M12_FME_60_RAT"),
    ]
    buckets: list[dict[str, Any]] = []
    for code, label, male_col, female_col in specs:
        male = _pct(row.get(male_col)) or 0.0
        female = _pct(row.get(female_col)) or 0.0
        total = male + female
        buckets.append({"code": code, "label": label, "value": round(total, 1)})
    buckets.sort(key=lambda item: item["value"], reverse=True)
    return buckets


def _flow_breakdown(row: pd.Series) -> dict[str, float | None]:
    return {
        "residential": _pct(row.get("RC_M1_SHC_RSD_UE_CLN_RAT")),
        "workplace": _pct(row.get("RC_M1_SHC_WP_UE_CLN_RAT")),
        "floating": _pct(row.get("RC_M1_SHC_FLP_UE_CLN_RAT")),
    }


def _gender_share(row: pd.Series) -> dict[str, float]:
    female = _age_bucket_sum(
        row,
        [
            "M12_FME_1020_RAT",
            "M12_FME_30_RAT",
            "M12_FME_40_RAT",
            "M12_FME_50_RAT",
            "M12_FME_60_RAT",
        ],
    )
    male = _age_bucket_sum(
        row,
        [
            "M12_MAL_1020_RAT",
            "M12_MAL_30_RAT",
            "M12_MAL_40_RAT",
            "M12_MAL_50_RAT",
            "M12_MAL_60_RAT",
        ],
    )
    return {"female": round(female, 1), "male": round(male, 1)}


def build_panel_dict(row: pd.Series, latest_ym: int) -> dict[str, Any]:
    buckets = _age_buckets(row)
    kpis = {
        "revisit_rate": _pct(row.get("MCT_UE_CLN_REU_RAT")),
        "new_rate": _pct(row.get("MCT_UE_CLN_NEW_RAT")),
    }
    flow = _flow_breakdown(row)
    gender = _gender_share(row)

    warnings: list[str] = []
    if row.attrs.get("guard_fallback"):
        warnings.append("[guard fallback] used latest available month; sums outside 100±0.5")

    return {
        "ta_ym": latest_ym,
        "age_distribution": buckets,
        "gender_share": gender,
        "kpis": kpis,
        "flow": flow,
        "warnings": warnings,
        "guard_fallback": bool(row.attrs.get("guard_fallback")),
    }


def extract_panel_for(df: pd.DataFrame, mct_id: str, *, allow_alias: bool = False) -> dict[str, Any]:
    """Extract sanitized panel metrics for a single merchant.

    Parameters
    ----------
    df:
        Source dataframe.
    mct_id:
        Merchant identifier.
    allow_alias:
        Ignored. Present for backwards compatibility with older callers.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("[panel_extract] df must be a pandas.DataFrame")

    working = subset_needed(df)
    row, latest_ym = latest_valid_row(working, mct_id)
    return build_panel_dict(row, latest_ym)


__all__ = [
    "SAFE_COLS",
    "NEEDED",
    "subset_needed",
    "select_needed",
    "get_required_subset",
    "NEEDED_COLS",
    "REQUIRED_COLS",
]


if __name__ == "__main__":  # pragma: no cover - developer self-check
    import pandas as _pd

    _df = _pd.DataFrame(columns=NEEDED)
    assert list(_df.columns) == NEEDED
    print("[panel_extract] self-check OK")

