"""Access audit helpers for validating resolver coverage across Shinhan datasets."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import json

import pandas as pd

from bigcon_2agent_mvp_v3 import read_csv_smart, resolve_merchant, tick, to_ms

from .report import ensure_directory


SET1_COLS = [
    "ENCODED_MCT",
    "MCT_NM",
    "SIGUNGU",
    "CATEGORY",
    "ADDR_BASE",
    "MCT_BRD_NUM",
    "HPSN_MCT_ZCD_NM",
    "HPSN_MCT_BZN_CD_NM",
    "MCT_ME_D",
]

SET2_COLS = [
    "ENCODED_MCT",
    "TA_YM",
    "MCT_OPE_MS_CN",
    "RC_M1_SAA",
    "RC_M1_TO_UE_CT",
    "RC_M1_UE_CUS_CN",
    "RC_M1_AV_NP_AT",
    "APV_CE_RAT",
    "DLV_SAA_RAT",
    "M1_SME_RY_SAA_RAT",
    "M1_SME_RY_CNT_RAT",
    "M12_SME_RY_SAA_PCE_RT",
    "M12_SME_BZN_SAA_PCE_RT",
    "M12_SME_RY_ME_MCT_RAT",
    "M12_SME_BZN_ME_MCT_RAT",
]

SET3_COLS = [
    "ENCODED_MCT",
    "TA_YM",
    "MCT_UE_CLN_REU_RAT",
    "MCT_UE_CLN_NEW_RAT",
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
    "RC_M1_SHC_FLP_UE_CLN_RAT",
    "RC_M1_SHC_RSD_UE_CLN_RAT",
    "RC_M1_SHC_WP_UE_CLN_RAT",
]

_SET1_ALIASES = {
    "ENCODED_MCT": ["ENCODED_MCT", "MCT_ID"],
    "MCT_NM": ["MCT_NM", "STORE_NAME"],
    "SIGUNGU": ["SIGUNGU", "MCT_SIGUNGU_NM"],
    "CATEGORY": ["CATEGORY", "ARE_D", "MCT_ME_D"],
    "ADDR_BASE": ["ADDR_BASE", "MCT_BSE_AR"],
    "MCT_BRD_NUM": ["MCT_BRD_NUM"],
}


def _normalize(text: str | None) -> str:
    return str(text).strip() if text is not None else ""


def _find_column(columns: Sequence[str], candidates: Iterable[str]) -> str | None:
    lowered = {str(col).lower(): col for col in columns}
    for cand in candidates:
        if not cand:
            continue
        cand_low = cand.lower()
        if cand_low in lowered:
            return lowered[cand_low]
    for cand in candidates:
        cand_low = cand.lower()
        for col in columns:
            if cand_low in str(col).lower():
                return col
    return None


def _resolve_columns(df: pd.DataFrame, aliases: dict[str, Iterable[str]]) -> dict[str, str | None]:
    columns = list(df.columns)
    return {key: _find_column(columns, values) for key, values in aliases.items()}


def _prepare_set1(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str | None]]:
    raw = read_csv_smart(path)
    mapping = _resolve_columns(raw, _SET1_ALIASES)
    work = pd.DataFrame()
    for key, source in mapping.items():
        if source is None:
            work[key] = pd.Series(pd.NA, index=raw.index, dtype="string")
        else:
            work[key] = raw[source].astype("string")
    if mapping.get("CATEGORY") is None:
        fallback = _find_column(raw.columns, ["ARE_D", "MCT_ME_D"])
        if fallback is not None:
            work["CATEGORY"] = raw[fallback].astype("string")
    return raw, work, mapping


def _prepare_set(path: Path, expected_cols: Sequence[str]) -> pd.DataFrame:
    df = read_csv_smart(path)
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _select_latest(row_df: pd.DataFrame, expected_cols: Sequence[str]) -> pd.Series | None:
    if row_df is None or row_df.empty:
        return None
    work = row_df.copy()
    if "TA_YM" in work.columns:
        try:
            work["TA_YM"] = work["TA_YM"].astype(str)
        except Exception:
            work["TA_YM"] = work["TA_YM"].astype("string")
        work = work.sort_values("TA_YM")
    row = work.iloc[-1]
    return row.reindex(expected_cols, fill_value=pd.NA)


def _nan_ratio(row: pd.Series | None) -> float:
    if row is None or row.empty:
        return 1.0
    total = len(row)
    if total == 0:
        return 0.0
    return float(row.isna().sum()) / float(total)


def _out_of_range(row: pd.Series | None, columns: Sequence[str]) -> list[str]:
    flags: list[str] = []
    if row is None or row.empty:
        return flags
    for col in columns:
        if col not in row.index:
            continue
        try:
            value = float(row[col]) if pd.notna(row[col]) else None
        except (TypeError, ValueError):
            value = None
        if value is None:
            continue
        if value < -10 or value > 110:
            flags.append(col)
    return flags


def _mask_prefix(name: str, length: int) -> tuple[str, str]:
    clean = _normalize(name)
    if not clean:
        return "", ""
    prefix = clean[: max(length, 0)]
    masked = f"{prefix}{'***'}"
    return masked, prefix


def _infer_truth(
    set1_df: pd.DataFrame, sigungu: str | None, prefix: str | None
) -> tuple[str | None, str | None, str | None]:
    if not prefix:
        return None, None, None
    work = set1_df.copy()
    if sigungu:
        work = work[work["SIGUNGU"].str.strip().str.casefold() == sigungu.strip().casefold()]
    work = work[work["MCT_NM"].str.startswith(prefix, na=False)]
    if work.empty:
        return None, None, None
    unique_ids = [v for v in work["ENCODED_MCT"].dropna().astype(str).unique() if v]
    unique_brands = [v for v in work["MCT_BRD_NUM"].dropna().astype(str).unique() if v]
    truth_id = unique_ids[0] if len(unique_ids) == 1 else None
    truth_brand = unique_brands[0] if len(unique_brands) == 1 else None
    truth_name = work.iloc[0]["MCT_NM"] if truth_id else None
    if truth_name is not None:
        truth_name = str(truth_name)
    return truth_id, truth_name, truth_brand


def _build_query(sigungu: str, masked: str) -> str:
    sig = sigungu.strip() if sigungu else ""
    inner = masked.strip() if masked else ""
    if inner and not inner.startswith("{"):
        inner = f"{{{inner}}}"
    return f"{sig} {inner}".strip()


def run_access_audit(
    mode: str,
    sigungu: str = "성동구",
    n: int = 10,
    search_terms: list[str] | None = None,
    mask_prefix_len: int = 2,
    brand_match: bool = True,
    seed: int = 42,
) -> "pd.DataFrame":
    """Run an access audit over Shinhan Set1/2/3 without invoking LLM helpers."""

    mode = (mode or "").lower()
    if mode not in {"random", "search"}:
        raise ValueError("mode must be 'random' or 'search'")

    sigungu = sigungu or ""
    base_dir = Path("data") / "shinhan"
    set1_raw, set1_std, mapping = _prepare_set1(base_dir / "big_data_set1_f.csv")
    set2_df = _prepare_set(base_dir / "big_data_set2_f.csv", SET2_COLS)
    set3_df = _prepare_set(base_dir / "big_data_set3_f.csv", SET3_COLS)

    merchants_df = pd.DataFrame({})
    for col in ["ENCODED_MCT", "MCT_NM", "SIGUNGU", "CATEGORY", "ADDR_BASE"]:
        source = mapping.get(col)
        if source is None:
            merchants_df[col] = pd.Series(pd.NA, index=set1_raw.index, dtype="string")
        else:
            merchants_df[col] = set1_raw[source].astype("string")
    merchants_df = merchants_df.fillna("")

    set1_std = set1_std.fillna("")
    set1_std["SIGUNGU"] = set1_std["SIGUNGU"].astype(str)
    set1_std["MCT_NM"] = set1_std["MCT_NM"].astype(str)
    set1_std["ENCODED_MCT"] = set1_std["ENCODED_MCT"].astype(str)
    set1_std["MCT_BRD_NUM"] = set1_std["MCT_BRD_NUM"].astype(str)

    if "SIGUNGU" not in set1_std.columns:
        set1_std["SIGUNGU"] = ""

    if mode == "random":
        candidates = set1_std.copy()
        if sigungu:
            candidates = candidates[
                candidates["SIGUNGU"].str.strip().str.casefold()
                == sigungu.strip().casefold()
            ]
        if candidates.empty:
            samples = pd.DataFrame()
        else:
            unique_candidates = candidates.drop_duplicates(subset=["ENCODED_MCT"])
            sample_size = min(int(n), len(unique_candidates))
            samples = unique_candidates.sample(sample_size, random_state=seed)
        query_rows = []
        for _, row in samples.iterrows():
            masked, prefix = _mask_prefix(row.get("MCT_NM", ""), mask_prefix_len)
            query_rows.append(
                {
                    "input_text": _build_query(sigungu, masked),
                    "masked": masked,
                    "prefix": prefix,
                    "truth_id": _normalize(row.get("ENCODED_MCT")),
                    "truth_name": _normalize(row.get("MCT_NM")),
                    "truth_brand": _normalize(row.get("MCT_BRD_NUM")),
                }
            )
    else:
        query_rows = []
        terms = search_terms or []
        for term in terms:
            stripped = str(term or "").strip()
            if not stripped:
                continue
            inner = stripped
            if "{" in inner and "}" in inner:
                start = inner.find("{")
                end = inner.find("}", start + 1)
                if start != -1 and end != -1:
                    inner = inner[start + 1 : end]
            prefix = inner.split("*")[0]
            masked = inner if inner else stripped
            truth_id, truth_name, truth_brand = _infer_truth(set1_std, sigungu, prefix)
            query_rows.append(
                {
                    "input_text": _build_query(sigungu, masked),
                    "masked": inner,
                    "prefix": prefix,
                    "truth_id": truth_id,
                    "truth_name": truth_name,
                    "truth_brand": truth_brand,
                }
            )

    records: list[dict] = []
    ratio_columns_set2 = [col for col in SET2_COLS if col.endswith("RAT") or col.endswith("RT")]
    ratio_columns_set3 = [col for col in SET3_COLS if col.endswith("RAT") or col.endswith("RT")]

    present_counts = {
        "s1": sum(1 for col in SET1_COLS if _find_column(set1_raw.columns, [col]) is not None),
        "s2": sum(1 for col in SET2_COLS if _find_column(set2_df.columns, [col]) is not None),
        "s3": sum(1 for col in SET3_COLS if _find_column(set3_df.columns, [col]) is not None),
    }

    encoded_col = mapping.get("ENCODED_MCT") or "ENCODED_MCT"

    id_to_brand = {
        _normalize(row.get("ENCODED_MCT")): _normalize(row.get("MCT_BRD_NUM"))
        for _, row in set1_std.iterrows()
    }

    for row in query_rows:
        masked_name = row.get("masked")
        prefix = row.get("prefix")
        t0 = tick()
        resolved, debug_info = resolve_merchant(
            masked_name,
            prefix,
            sigungu,
            merchants_df,
            original_question=row.get("input_text"),
            allow_llm=False,
        )
        elapsed_ms = to_ms(t0)
        resolved_id = resolved.get("encoded_mct") if resolved else None
        resolved_id = _normalize(resolved_id) if resolved_id else None
        path = (debug_info or {}).get("path") if isinstance(debug_info, dict) else None

        if resolved_id and encoded_col in set1_raw.columns:
            s1_subset = set1_raw[set1_raw[encoded_col].astype(str) == resolved_id]
        else:
            s1_subset = pd.DataFrame()
        s2_subset = set2_df[set2_df["ENCODED_MCT"].astype(str) == resolved_id] if resolved_id else pd.DataFrame()
        s3_subset = set3_df[set3_df["ENCODED_MCT"].astype(str) == resolved_id] if resolved_id else pd.DataFrame()

        s1_row = _select_latest(s1_subset, SET1_COLS)
        s2_row = _select_latest(s2_subset, SET2_COLS)
        s3_row = _select_latest(s3_subset, SET3_COLS)

        out_of_range = []
        out_of_range.extend(_out_of_range(s2_row, ratio_columns_set2))
        out_of_range.extend(_out_of_range(s3_row, ratio_columns_set3))

        truth_id = row.get("truth_id") or None
        truth_brand = row.get("truth_brand") or None
        resolved_brand = id_to_brand.get(resolved_id or "") if resolved_id else None
        is_correct = False
        if resolved_id and truth_id and resolved_id == truth_id:
            is_correct = True
        elif brand_match and truth_brand and resolved_brand and truth_brand == resolved_brand:
            is_correct = True

        records.append(
            {
                "input_text": row.get("input_text"),
                "resolved_id": resolved_id,
                "truth_id": truth_id,
                "truth_name": row.get("truth_name"),
                "truth_brand": truth_brand,
                "s1_found": bool(resolved_id and not s1_subset.empty),
                "s1_cols_present": present_counts["s1"],
                "s1_cols_expected": len(SET1_COLS),
                "s1_nan_ratio": _nan_ratio(s1_row),
                "s2_found": bool(resolved_id and not s2_subset.empty),
                "s2_cols_present": present_counts["s2"],
                "s2_cols_expected": len(SET2_COLS),
                "s2_nan_ratio": _nan_ratio(s2_row),
                "s3_found": bool(resolved_id and not s3_subset.empty),
                "s3_cols_present": present_counts["s3"],
                "s3_cols_expected": len(SET3_COLS),
                "s3_nan_ratio": _nan_ratio(s3_row),
                "out_of_range_flags": ";".join(out_of_range),
                "path": path,
                "elapsed_ms": elapsed_ms,
                "is_correct": bool(is_correct),
            }
        )

    return pd.DataFrame(records)


def summarize_access(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "resolved_rate": 0.0,
            "s1_access_rate": 0.0,
            "s2_access_rate": 0.0,
            "s3_access_rate": 0.0,
            "s1_coverage": 0.0,
            "s2_coverage": 0.0,
            "s3_coverage": 0.0,
            "s1_nan_median": 0.0,
            "s2_nan_median": 0.0,
            "s3_nan_median": 0.0,
            "out_of_range_count": 0,
            "accuracy": 0.0,
        }

    total = len(df)
    resolved_rate = (df["resolved_id"].notna() & (df["resolved_id"] != "")).mean() * 100
    s1_access_rate = df["s1_found"].mean() * 100 if "s1_found" in df else 0.0
    s2_access_rate = df["s2_found"].mean() * 100 if "s2_found" in df else 0.0
    s3_access_rate = df["s3_found"].mean() * 100 if "s3_found" in df else 0.0

    def _coverage(col_present: str, col_expected: str) -> float:
        if col_present not in df or col_expected not in df:
            return 0.0
        ratios = df[col_present] / df[col_expected]
        return float(ratios.mean() * 100)

    coverage = {
        "s1": _coverage("s1_cols_present", "s1_cols_expected"),
        "s2": _coverage("s2_cols_present", "s2_cols_expected"),
        "s3": _coverage("s3_cols_present", "s3_cols_expected"),
    }

    def _median(column: str) -> float:
        if column not in df or df[column].empty:
            return 0.0
        return float(df[column].median() * 100)

    out_of_range_count = int((df["out_of_range_flags"].fillna("") != "").sum())
    accuracy = df["is_correct"].mean() * 100 if "is_correct" in df else 0.0

    return {
        "resolved_rate": float(resolved_rate),
        "s1_access_rate": float(s1_access_rate),
        "s2_access_rate": float(s2_access_rate),
        "s3_access_rate": float(s3_access_rate),
        "s1_coverage": coverage["s1"],
        "s2_coverage": coverage["s2"],
        "s3_coverage": coverage["s3"],
        "s1_nan_median": _median("s1_nan_ratio"),
        "s2_nan_median": _median("s2_nan_ratio"),
        "s3_nan_median": _median("s3_nan_ratio"),
        "out_of_range_count": out_of_range_count,
        "accuracy": float(accuracy),
        "total": total,
    }


def export_access(
    df: pd.DataFrame,
    summary: dict,
    out_dir: str = "diagnostics/results",
) -> dict:
    ensure_directory(out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(out_dir) / f"access_audit_{timestamp}.csv"
    json_path = Path(out_dir) / f"access_audit_summary_{timestamp}.json"

    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    return {"csv": str(csv_path), "summary": str(json_path)}

