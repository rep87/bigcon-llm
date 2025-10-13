"""Utilities for building a merchant catalog diagnostic from Shinhan Set1."""


from __future__ import annotations

from typing import Iterable

import pandas as pd


_REQUIRED_COLS = {
    "ENCODED_MCT": ["ENCODED_MCT", "ENCODED-MCT", "MCT_ID", "STORE_ID"],
    "MCT_NM": ["MCT_NM", "MERCHANT_NAME", "STORE_NAME", "MCTNAME"],
    "SIGUNGU": ["SIGUNGU", "SIGUNGU_NM", "MCT_SIGUNGU", "MCT_SIGUNGU_NM"],
    "CATEGORY": ["CATEGORY", "MCT_CATEGORY", "MCT_CAT", "MCT_ME_D", "ARE_D", "MCT_ME"],
    "MCT_BRD_NUM": ["MCT_BRD_NUM", "BRD_NUM", "BRAND_ID", "MCTBRDNUM"],
}


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {col: _normalize(col) for col in columns}
    for cand in candidates:
        norm_cand = _normalize(cand)
        for col, norm_col in normalized.items():
            if not norm_cand:
                continue
            if norm_cand in norm_col or norm_col in norm_cand:
                return col
    return None


def _clean_value(value) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if trimmed else pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def load_set1(path: str = "data/shinhan/big_data_set1_f.csv") -> "pd.DataFrame":
    """필수 컬럼만 표준화해서 반환.
    반환 컬럼: ['ENCODED_MCT','MCT_NM','SIGUNGU','CATEGORY','MCT_BRD_NUM']
    - 인코딩 스마트 로딩(utf-8/utf-8-sig/cp949/euc-kr/latin-1 시도)
    - 컬럼명 대소문자/한글 혼합 대비: 포함/부분일치로 매핑
    - 결측은 빈 문자열 또는 NaN으로 통일
    """

    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]
    last_error: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_error = exc
    else:
        raise last_error or FileNotFoundError(path)

    columns = list(df.columns)
    result = pd.DataFrame(index=df.index)
    for target, candidates in _REQUIRED_COLS.items():
        source = _find_column(columns, candidates)
        if source is None:
            result[target] = pd.Series(pd.NA, index=df.index, dtype="string")
            continue
        series = df[source]
        cleaned = series.map(_clean_value)
        result[target] = cleaned.astype("string")

    return result


def _unique_non_empty(values: Iterable) -> list[str]:
    seen: list[str] = []
    for val in values:
        if pd.isna(val):
            continue
        text = str(val).strip()
        if not text or text in seen:
            continue
        seen.append(text)
    return seen


def build_catalog(df: "pd.DataFrame", sigungu_filter: str | None = None) -> "pd.DataFrame":
    """상호 카탈로그 생성.
    반환 컬럼(예시):
    ['SIGUNGU','MCT_NM','MCT_BRD_NUM','n_locations','encoded_mct_list','has_encoded_mct','note']
    규칙:
      - SIGUNGU로 선택 필터(없으면 전체)
      - (SIGUNGU, MCT_NM, MCT_BRD_NUM) 단위로 groupby
      - encoded_mct_list = 해당 그룹의 ENCODED_MCT 유니크 리스트
      - n_locations = len(encoded_mct_list)
      - has_encoded_mct = n_locations > 0
      - note:
          - n_locations == 0 → 'missing_id'
          - n_locations == 1 → 'ok'
          - n_locations >= 2 → 'duplicate_name' (동일 상호 다점포)
    """

    work = df.copy()
    expected_cols = ["SIGUNGU", "MCT_NM", "MCT_BRD_NUM", "ENCODED_MCT"]
    for col in expected_cols:
        if col not in work.columns:
            work[col] = pd.Series(pd.NA, index=work.index, dtype="string")
        else:
            work[col] = work[col].map(_clean_value).astype("string")
    if sigungu_filter:
        target = sigungu_filter.strip().casefold()
        work = work[work["SIGUNGU"].fillna("").str.casefold() == target]

    records: list[dict] = []
    group_cols = ["SIGUNGU", "MCT_NM", "MCT_BRD_NUM"]
    if work.empty:
        return pd.DataFrame(columns=group_cols + ["n_locations", "encoded_mct_list", "has_encoded_mct", "note"])

    grouped = work.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        sigungu, name, brand = (keys + (None,) * 3)[:3]
        sigungu_val = _clean_value(sigungu)
        name_val = _clean_value(name)
        brand_val = _clean_value(brand)
        encoded_list = _unique_non_empty(group["ENCODED_MCT"]) if "ENCODED_MCT" in group else []
        n_locations = len(encoded_list)
        has_encoded = n_locations > 0
        if n_locations == 0:
            note = "missing_id"
        elif n_locations == 1:
            note = "ok"
        else:
            note = "duplicate_name"
        records.append(
            {
                "SIGUNGU": sigungu_val,
                "MCT_NM": name_val,
                "MCT_BRD_NUM": brand_val,
                "n_locations": n_locations,
                "encoded_mct_list": encoded_list,
                "has_encoded_mct": has_encoded,
                "note": note,
            }
        )

    catalog_df = pd.DataFrame(records)
    return catalog_df.sort_values(["n_locations", "MCT_NM"], ascending=[False, True], ignore_index=True)


def summarize_catalog(cat_df: "pd.DataFrame") -> dict:
    """간단 요약 메트릭 반환.
    - total_rows
    - unique_names
    - pct_missing_id
    - pct_duplicate_name
    - top_duplicated (상위 10개: name/brand/locations)
    """

    if cat_df is None or cat_df.empty:
        return {
            "total_rows": 0,
            "unique_names": 0,
            "pct_missing_id": 0.0,
            "pct_duplicate_name": 0.0,
            "top_duplicated": [],
        }

    total_rows = int(len(cat_df))
    unique_names = int(cat_df["MCT_NM"].dropna().nunique()) if "MCT_NM" in cat_df.columns else 0
    missing_count = int((cat_df["n_locations"] == 0).sum()) if "n_locations" in cat_df.columns else 0
    duplicate_count = int((cat_df["n_locations"] >= 2).sum()) if "n_locations" in cat_df.columns else 0
    pct_missing = round((missing_count / total_rows) * 100, 2) if total_rows else 0.0
    pct_duplicate = round((duplicate_count / total_rows) * 100, 2) if total_rows else 0.0

    top = []
    if "note" in cat_df.columns:
        dup_df = cat_df[cat_df["note"] == "duplicate_name"].copy()
        if not dup_df.empty:
            dup_df = dup_df.sort_values("n_locations", ascending=False)
            for _, row in dup_df.head(10).iterrows():
                top.append(
                    {
                        "name": row.get("MCT_NM"),
                        "brand": row.get("MCT_BRD_NUM"),
                        "locations": int(row.get("n_locations", 0) or 0),
                    }
                )

    return {
        "total_rows": total_rows,
        "unique_names": unique_names,
        "pct_missing_id": pct_missing,
        "pct_duplicate_name": pct_duplicate,
        "top_duplicated": top,
    }
