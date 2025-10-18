# -*- coding: utf-8 -*-
# BIGCON 2-Agent MVP (Colab, Gemini API) — v3 (fits actual 3-dataset structure)
# %pip -q install google-generativeai pandas openpyxl

import ast
import datetime
import json
import os
import random
import re
from time import perf_counter
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd
import numpy as np
from jsonschema import Draft7Validator

from app_core.formatters import merge_age_buckets, to_float_pct

__all__ = [
    "agent1_pipeline",
    "build_agent2_prompt",
    "build_agent2_prompt_overhauled",
    "call_gemini_agent2",
    "call_gemini_agent2_overhauled",
    "infer_question_type",
    "load_actioncard_schema",
    "load_actioncard_schema_current",
    "AGENT2_PROMPT_TRACE",
    "AGENT2_RESPONSE_TRACE",
    "llm_json_safe_parse",
]

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / 'data'
SHINHAN_DIR = DATA_DIR / 'shinhan'
EXTERNAL_DIR = DATA_DIR / 'external'
OUTPUT_DIR = DATA_DIR / 'outputs'
SCHEMA_PATH = APP_ROOT / 'schemas' / 'actioncard.schema.json'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MONTHS = 8
SEED = 42
random.seed(SEED); np.random.seed(SEED)

_SCHEMA_CACHE = None
_SCHEMA_VALIDATOR = None
AGENT2_PROMPT_TRACE: dict = {}
AGENT2_RESPONSE_TRACE: dict = {}


def tick():
    return perf_counter()


def to_ms(t0):
    return int((perf_counter() - t0) * 1000)


def _env_flag(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped.strip()


def _json_error_window(text: str, exc: Exception, radius: int = 120) -> str:
    pos = getattr(exc, "pos", None)
    if pos is None and isinstance(exc, json.JSONDecodeError):
        pos = exc.pos
    if pos is None:
        return ""
    start = max(pos - radius, 0)
    end = min(len(text), pos + radius)
    return text[start:end]


def _extract_json_candidate(text: str) -> tuple[str | None, int | None]:
    if not text:
        return None, None
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = cleaned[start : idx + 1]
                    return snippet, start
        start = cleaned.find("{", start + 1)
    return None, None


def _apply_json_repairs(text: str) -> list[str]:
    candidates: list[str] = []
    if not text:
        return candidates
    base = text.strip()
    seen = set()

    def _push(value: str):
        if value and value not in seen:
            seen.add(value)
            candidates.append(value)

    _push(base)
    repaired = re.sub(r",\s*(\]|\})", r"\1", base)
    _push(repaired)
    repaired = re.sub(r"'", '"', repaired)
    repaired = re.sub(r"\bNaN\b", "null", repaired)
    repaired = re.sub(r"\bInfinity\b", "null", repaired)
    repaired = re.sub(r"\b-?inf\b", "null", repaired, flags=re.IGNORECASE)
    _push(repaired)
    return candidates


def _attempt_literal_eval(text: str):
    try:
        value = ast.literal_eval(text)
    except Exception:
        return None
    try:
        return json.loads(json.dumps(value))
    except Exception:
        return None


def llm_json_safe_parse(text: str, schema_validator: Draft7Validator | None = None):
    logs: list[dict] = []
    if text is None:
        logs.append({"pass": "strict", "success": False, "error": "empty_response"})
        return None, logs

    cleaned = text.strip()
    if not cleaned:
        logs.append({"pass": "strict", "success": False, "error": "empty_response"})
        return None, logs

    def _validate(obj):
        if schema_validator is not None:
            schema_validator.validate(obj)

    # Strict pass ---------------------------------------------------------
    try:
        parsed = json.loads(cleaned)
        _validate(parsed)
        logs.append({"pass": "strict", "success": True})
        return parsed, logs
    except Exception as exc:
        entry = {"pass": "strict", "success": False, "error": str(exc)}
        if isinstance(exc, json.JSONDecodeError):
            entry.update({"line": exc.lineno, "col": exc.colno, "excerpt": _json_error_window(cleaned, exc)})
        logs.append(entry)

    # Extract pass --------------------------------------------------------
    snippet, offset = _extract_json_candidate(cleaned)
    if snippet:
        try:
            parsed = json.loads(snippet)
            _validate(parsed)
            logs.append({"pass": "extract", "success": True, "offset": offset})
            return parsed, logs
        except Exception as exc:
            entry = {"pass": "extract", "success": False, "error": str(exc), "offset": offset}
            if isinstance(exc, json.JSONDecodeError):
                entry.update({"line": exc.lineno, "col": exc.colno, "excerpt": _json_error_window(snippet, exc)})
            logs.append(entry)
    else:
        logs.append({"pass": "extract", "success": False, "error": "candidate_not_found"})

    candidate = snippet or cleaned

    # Repair pass ---------------------------------------------------------
    for idx, variant in enumerate(_apply_json_repairs(candidate)):
        try:
            parsed = json.loads(variant)
            _validate(parsed)
            logs.append({"pass": "repair", "success": True, "variant": idx})
            return parsed, logs
        except Exception as exc:
            entry = {"pass": "repair", "success": False, "variant": idx, "error": str(exc)}
            if isinstance(exc, json.JSONDecodeError):
                entry.update({"line": exc.lineno, "col": exc.colno, "excerpt": _json_error_window(variant, exc)})
            logs.append(entry)

    literal_candidate = _attempt_literal_eval(candidate)
    if literal_candidate is not None:
        try:
            _validate(literal_candidate)
            logs.append({"pass": "literal_eval", "success": True})
            return literal_candidate, logs
        except Exception as exc:
            logs.append({"pass": "literal_eval", "success": False, "error": str(exc)})

    return None, logs


def _structured_value(agent1_json: dict | None, key: str) -> tuple[float | None, str | None]:
    agent1_json = agent1_json or {}
    debug = agent1_json.get("debug") or {}
    snapshot = debug.get("snapshot") or {}
    sanitized = snapshot.get("sanitized") or {}
    kpis = agent1_json.get("kpis") or {}
    raw_value = sanitized.get(key, kpis.get(key))
    value, _ = to_float_pct(raw_value)
    return value, sanitized.get("latest_ta_ym") or debug.get("panel", {}).get("latest_ta_ym")


def _structured_evidence_entry(
    key: str,
    value: float | None,
    *,
    period: str | None,
    snippet: str | None = None,
) -> dict:
    if value is None:
        return {
            "source": "STRUCTURED",
            "key": key or "근거 없음",
            "value": "—",
            "period": period or "—",
            "snippet": snippet or "근거 없음",
            "doc_id": None,
            "chunk_id": None,
            "score": None,
        }
    display_value = f"{value:.1f}%" if isinstance(value, (int, float)) else value
    return {
        "source": "STRUCTURED",
        "key": key,
        "value": display_value,
        "period": period or "recent_8m",
        "snippet": snippet or display_value,
        "doc_id": None,
        "chunk_id": None,
        "score": None,
    }


def _structured_fallback_cards(agent1_json: dict | None, question_type: str | None) -> dict:
    question_type = question_type or "GENERIC"
    age_records = merge_age_buckets(agent1_json or {})
    top_age = age_records[0] if age_records else None
    top_age_label = top_age.get("label") if top_age else "핵심 고객층"
    top_age_value = top_age.get("value") if top_age else None
    top_age_key = top_age.get("key") if top_age else "age_distribution"

    flow_pct, period_hint = _structured_value(agent1_json, "flow_pct")
    resident_pct, _ = _structured_value(agent1_json, "resident_pct")
    work_pct, _ = _structured_value(agent1_json, "work_pct")
    new_pct, _ = _structured_value(agent1_json, "new_pct")
    revisit_pct, _ = _structured_value(agent1_json, "revisit_pct")

    if period_hint is None:
        panel = ((agent1_json or {}).get("debug") or {}).get("panel") or {}
        period_hint = panel.get("latest_ta_ym") or "recent_8m"

    def _make_card(
        idea_title: str,
        audience: str,
        channels: list[str],
        execution: list[str],
        copy_samples: list[str],
        measurement: list[str],
        evidence_specs: list[dict],
    ) -> dict:
        evidence = [item for item in evidence_specs if item]
        if not evidence:
            evidence = [
                _structured_evidence_entry(
                    "근거 없음",
                    None,
                    period=period_hint,
                    snippet="근거 없음",
                )
            ]
        return {
            "idea_title": idea_title,
            "audience": audience,
            "channels": channels,
            "execution": execution,
            "copy_samples": copy_samples,
            "measurement": measurement,
            "evidence": evidence,
        }

    age_evidence = _structured_evidence_entry(
        f"age_distribution.{top_age_key}",
        top_age_value,
        period=period_hint,
        snippet=f"{top_age_label} {top_age_value:.1f}%" if isinstance(top_age_value, (int, float)) else None,
    ) if top_age_value is not None else None
    flow_evidence = _structured_evidence_entry(
        "customer_mix_detail.유동",
        flow_pct,
        period=period_hint,
        snippet=f"유동 {flow_pct:.1f}%" if isinstance(flow_pct, (int, float)) else None,
    ) if flow_pct is not None else None
    resident_evidence = _structured_evidence_entry(
        "customer_mix_detail.거주",
        resident_pct,
        period=period_hint,
        snippet=f"거주 {resident_pct:.1f}%" if isinstance(resident_pct, (int, float)) else None,
    ) if resident_pct is not None else None
    work_evidence = _structured_evidence_entry(
        "customer_mix_detail.직장",
        work_pct,
        period=period_hint,
        snippet=f"직장 {work_pct:.1f}%" if isinstance(work_pct, (int, float)) else None,
    ) if work_pct is not None else None
    new_evidence = _structured_evidence_entry(
        "new_pct",
        new_pct,
        period=period_hint,
        snippet=f"신규 {new_pct:.1f}%" if isinstance(new_pct, (int, float)) else None,
    ) if new_pct is not None else None
    revisit_evidence = _structured_evidence_entry(
        "revisit_pct",
        revisit_pct,
        period=period_hint,
        snippet=f"재방문 {revisit_pct:.1f}%" if isinstance(revisit_pct, (int, float)) else None,
    ) if revisit_pct is not None else None

    cards: list[dict] = []

    if question_type == "Q2_LOW_RETENTION":
        cards.append(
            _make_card(
                "단골 회수 메시지 캠페인",
                top_age_label,
                ["카카오톡 채널", "SMS"],
                [
                    "재방문 고객에게 이탈 기간별 리마인드 메시지를 발송합니다.",
                    "멤버십 쿠폰을 재발급해 재방문 유인을 만듭니다.",
                ],
                [
                    f"다시 찾아주시면 {top_age_label} 전용 혜택을 준비했어요!",
                    "이번 주 안에 방문 시 추가 리필을 드립니다.",
                ],
                ["재방문 고객 수", "쿠폰 재사용률"],
                [revisit_evidence or new_evidence, age_evidence],
            )
        )
        cards.append(
            _make_card(
                "스탬프 적립 프로모션",
                "재방문 유도 고객",
                ["현장 POP", "멤버십 앱"],
                [
                    "현장 스탬프판을 비치하고 3회 방문 시 무료 메뉴를 제공합니다.",
                    "앱에서 실시간 적립 현황을 보여줍니다.",
                ],
                [
                    "스탬프 3개 모으면 아메리카노 무료!",
                    "이번 주 멤버십 적립률 TOP 고객 공지",
                ],
                ["스탬프 적립률", "혜택 교환 건수"],
                [new_evidence, revisit_evidence],
            )
        )
        cards.append(
            _make_card(
                "현장 체류 경험 강화",
                "점심/퇴근 고객",
                ["오프라인 이벤트", "지역 커뮤니티"],
                [
                    "점심시간 한정 시식/시연 이벤트를 기획합니다.",
                    "지역 커뮤니티에 리뷰 인증 시 혜택을 제공합니다.",
                ],
                [
                    "점심 인증샷 업로드 시 디저트 증정",
                    "퇴근길에 들르면 마감 할인 제공",
                ],
                ["이벤트 참여자 수", "리뷰 증가량"],
                [flow_evidence, resident_evidence or work_evidence],
            )
        )
    elif question_type == "Q3_FOOD_ISSUE":
        cards.append(
            _make_card(
                "시그니처 메뉴 집중 홍보",
                top_age_label,
                ["SNS", "지도 리뷰"],
                [
                    "대표 메뉴 조리 과정을 짧은 영상으로 제작합니다.",
                    "지도 리뷰에 맛/위생 키워드 응답을 상시 관리합니다.",
                ],
                [
                    f"{top_age_label} 손님이 좋아한 시그니처 레시피 공개",
                    "오늘 만든 신선한 원두/재료 강조",
                ],
                ["영상 조회수", "긍정 리뷰 비중"],
                [age_evidence, flow_evidence or resident_evidence],
            )
        )
        cards.append(
            _make_card(
                "피크타임 회전율 개선",
                "점심 피크 고객",
                ["현장 안내", "테이블 오더"],
                [
                    "피크 시간 사전 주문/픽업 라인을 운영합니다.",
                    "혼잡도를 현장 안내판으로 공지합니다.",
                ],
                [
                    "점심 전 미리 주문하면 대기 없이 픽업!",
                    "테이블에서 QR 주문하고 바로 받아가세요",
                ],
                ["테이블 회전 시간", "픽업 사전 주문 비중"],
                [work_evidence, flow_evidence],
            )
        )
        cards.append(
            _make_card(
                "리뷰 기반 품질 보완",
                "재방문 잠재 고객",
                ["리뷰 DM", "멤버십"],
                [
                    "리뷰에서 제기된 맛/서비스 이슈를 정리하고 개선 공지를 보냅니다.",
                    "개선 후 재방문 고객에게 보상 쿠폰을 제공합니다.",
                ],
                [
                    "리뷰 남겨주신 의견을 반영해 레시피를 새로 손봤습니다.",
                    "재방문 시 무료 토핑을 드립니다.",
                ],
                ["리뷰 응답률", "재방문 고객 수"],
                [revisit_evidence or new_evidence, resident_evidence],
            )
        )
    else:  # Q1_CAFE_CHANNELS or generic
        cards.append(
            _make_card(
                "핵심 연령대 SNS 집중 광고",
                top_age_label,
                ["인스타그램", "네이버 플레이스"],
                [
                    "연령대 관심사에 맞춘 숏폼 콘텐츠를 주 3회 게시합니다.",
                    "플레이스 리뷰 상단에 시즌 한정 메뉴를 노출합니다.",
                ],
                [
                    f"#{top_age_label} 취향 저격 신메뉴 공개",
                    "방문 인증 시 음료 1+1",
                ],
                ["SNS 참여율", "플레이스 클릭수"],
                [age_evidence, flow_evidence],
            )
        )
        cards.append(
            _make_card(
                "오피스 타겟 쿠폰 제휴",
                "직장인 고객",
                ["회사 제휴", "배달앱 광고"],
                [
                    "근처 오피스와 제휴해 점심 시간 할인 쿠폰을 제공합니다.",
                    "배달앱 홈 배너에 오피스 타겟 전용 메뉴를 노출합니다.",
                ],
                [
                    "점심 11-14시 전용 2,000원 할인 쿠폰",
                    "회사 이메일 인증 시 추가 적립",
                ],
                ["쿠폰 사용률", "점심 매출"],
                [work_evidence, resident_evidence or flow_evidence],
            )
        )
        cards.append(
            _make_card(
                "단골 확보 멤버십 메시지",
                "재방문 잠재 고객",
                ["멤버십 앱", "카카오톡 채널"],
                [
                    "재방문율이 낮은 고객에게 주말 한정 혜택을 보냅니다.",
                    "멤버십 포인트로 시즌 굿즈 교환 이벤트를 알립니다.",
                ],
                [
                    "이번 주말 재방문 시 포인트 2배 적립",
                    "굿즈 한정 수량 예약 링크",
                ],
                ["재방문 고객 수", "멤버십 활성화율"],
                [revisit_evidence, new_evidence],
            )
        )

    return {"answers": cards[:4]}

USE_LLM = _env_flag("AGENT1_USE_LLM", "true").lower() not in {"0", "false", "no"}
DEBUG_MAX_PREVIEW = int(_env_flag("DEBUG_MAX_PREVIEW", "200") or 200)
DEBUG_SHOW_RAW = _env_flag("DEBUG_SHOW_RAW", "true").lower() in {"1", "true", "yes"}


def _mask_debug_preview(text: str | None, limit: int = DEBUG_MAX_PREVIEW) -> str:
    if not text:
        return ""
    masked = re.sub(r"\{[^{}]*\}", "{***}", str(text))
    masked = re.sub(r"([A-Za-z0-9]{4})[A-Za-z0-9]{4,}", r"\1***", masked)
    return masked[:limit]


def _normalize_str(value: str) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", "", text)


def _normalize_compare(value: str | None) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", "", text)
    return text.upper()


def _wildcard_to_regex(masked: str | None) -> re.Pattern | None:
    if not masked:
        return None
    normalized = _normalize_str(masked)
    if not normalized:
        return None
    pattern = "".join(".*" if ch == "*" else re.escape(ch) for ch in normalized)
    try:
        return re.compile(f"^{pattern}")
    except re.error:
        return None

def load_actioncard_schema_current():
    global _SCHEMA_CACHE, _SCHEMA_VALIDATOR
    if _SCHEMA_CACHE is not None and _SCHEMA_VALIDATOR is not None:
        return _SCHEMA_CACHE, _SCHEMA_VALIDATOR
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"스키마 파일을 찾을 수 없습니다: {SCHEMA_PATH}")
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        _SCHEMA_CACHE = json.load(f)

    if isinstance(_SCHEMA_CACHE, dict) and "schemas" in _SCHEMA_CACHE:
        validators: dict[str, Draft7Validator] = {}
        for key, schema_obj in _SCHEMA_CACHE.get("schemas", {}).items():
            try:
                validators[key] = Draft7Validator(schema_obj)
            except Exception:
                continue
        _SCHEMA_VALIDATOR = validators
    else:
        _SCHEMA_VALIDATOR = Draft7Validator(_SCHEMA_CACHE)
    return _SCHEMA_CACHE, _SCHEMA_VALIDATOR

def read_csv_smart(path):
    for enc in ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1']:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding='utf-8', errors='replace')


def normalize_rate_series(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return series
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("−", "-", regex=False)
    )
    cleaned = cleaned.str.replace(r"[^0-9\-.]", "", regex=True)
    numeric = pd.to_numeric(cleaned, errors='coerce')
    if numeric is None:
        return None
    scaled = numeric.copy()
    mask = scaled.notna() & (scaled.abs() <= 1)
    scaled.loc[mask] = scaled.loc[mask] * 100
    scaled.loc[(scaled < 0) | (scaled > 100)] = np.nan
    return scaled

def ym_to_date(ym_series):
    s = pd.to_datetime(ym_series.astype(str) + '01', format='%Y%m%d', errors='coerce')
    return s

def load_set1(shinhan_dir):
    p = shinhan_dir / 'big_data_set1_f.csv'
    df = read_csv_smart(p)
    ren = {}
    for c in df.columns:
        cu = str(c).upper()
        if cu == 'ENCODED_MCT': ren[c] = 'ENCODED_MCT'
        elif 'SIGUNGU' in cu:   ren[c] = 'SIGUNGU'
        elif 'BSE_AR' in cu:    ren[c] = 'ADDR_BASE'
        elif ('ZCD' in cu) or ('BZN' in cu) or ('업종' in cu): ren[c] = 'CATEGORY'
        elif cu == 'MCT_NM':    ren[c] = 'MCT_NM'
    df = df.rename(columns=ren)
    df = df.loc[:, ~df.columns.duplicated()]
    keep = ['ENCODED_MCT','MCT_NM','ADDR_BASE','SIGUNGU','CATEGORY']
    for k in keep:
        if k not in df.columns: df[k] = np.nan
    df = df[keep].drop_duplicates('ENCODED_MCT')
    if 'ENCODED_MCT' in df.columns:
        df['ENCODED_MCT'] = df['ENCODED_MCT'].apply(lambda v: str(v).strip() if pd.notna(v) else '')
    return df

def load_set2(shinhan_dir):
    p = shinhan_dir / 'big_data_set2_f.csv'
    df = read_csv_smart(p)
    df['TA_YM'] = df['TA_YM'].astype(str)
    df['_date'] = ym_to_date(df['TA_YM'])
    return df

def load_set3(shinhan_dir):
    p = shinhan_dir / 'big_data_set3_f.csv'
    df = read_csv_smart(p)
    df['TA_YM'] = df['TA_YM'].astype(str)
    df['_date'] = ym_to_date(df['TA_YM'])
    keep_cols = [
        'ENCODED_MCT','TA_YM','_date',
        'M12_MAL_1020_RAT','M12_MAL_30_RAT','M12_MAL_40_RAT','M12_MAL_50_RAT','M12_MAL_60_RAT',
        'M12_FME_1020_RAT','M12_FME_30_RAT','M12_FME_40_RAT','M12_FME_50_RAT','M12_FME_60_RAT',
        'MCT_UE_CLN_REU_RAT','MCT_UE_CLN_NEW_RAT',
        'RC_M1_SHC_RSD_UE_CLN_RAT','RC_M1_SHC_WP_UE_CLN_RAT','RC_M1_SHC_FLP_UE_CLN_RAT',
        'APV_CE_RAT'
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep_cols]
    rate_cols = [
        'M12_MAL_1020_RAT','M12_MAL_30_RAT','M12_MAL_40_RAT','M12_MAL_50_RAT','M12_MAL_60_RAT',
        'M12_FME_1020_RAT','M12_FME_30_RAT','M12_FME_40_RAT','M12_FME_50_RAT','M12_FME_60_RAT',
        'MCT_UE_CLN_REU_RAT','MCT_UE_CLN_NEW_RAT',
        'RC_M1_SHC_RSD_UE_CLN_RAT','RC_M1_SHC_WP_UE_CLN_RAT','RC_M1_SHC_FLP_UE_CLN_RAT'
    ]
    for col in rate_cols:
        if col in df.columns:
            original = df[col].copy()
            df[f'{col}_raw'] = original
            df[col] = normalize_rate_series(original)
    return df

def load_weather_monthly(external_dir):
    f = None
    for e in ('.csv','.parquet','.parq','.feather'):
        cand = list(external_dir.glob(f'**/*{e}'))
        if cand:
            f = cand[0]; break
    if not f:
        print('⚠️ 외부(날씨) 데이터가 없습니다. 날씨 분석은 제한됩니다.')
        return None
    if f.suffix.lower() == '.csv':
        wx = read_csv_smart(f)
    elif f.suffix.lower() in ('.parquet','.parq'):
        wx = pd.read_parquet(f)
    elif f.suffix.lower() == '.feather':
        wx = pd.read_feather(f)
    else:
        return None

    c_dt = None
    for c in wx.columns:
        cl = str(c).lower()
        if any(k in cl for k in ['date','ymd','dt','일자','날짜','yyyymm']):
            c_dt = c; break
    if c_dt is None:
        raise ValueError('날씨 데이터에 날짜(또는 YYYYMM) 컬럼을 찾지 못했습니다.')
    dt = pd.to_datetime(wx[c_dt].astype(str), errors='coerce')
    wx['_ym'] = dt.dt.strftime('%Y%m')
    c_rain = None
    for c in wx.columns:
        cl = c.lower()
        if any(k in cl for k in ['rain','precip','rn_mm','rainfall','rr','강수','강수량','비']):
            c_rain = c; break
    if c_rain is None:
        wx['_rain_val'] = 0.0
    else:
        wx['_rain_val'] = pd.to_numeric(wx[c_rain], errors='coerce').fillna(0.0)

    monthly = wx.groupby('_ym', as_index=False)['_rain_val'].sum().rename(columns={'_ym':'TA_YM','_rain_val':'RAIN_SUM'})
    monthly['TA_YM'] = monthly['TA_YM'].astype(str)
    monthly['_date'] = pd.to_datetime(monthly['TA_YM'] + '01', format='%Y%m%d', errors='coerce')
    return monthly[['TA_YM','_date','RAIN_SUM']]


def _format_percent_debug(value):
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num < 0 or num > 100:
        return None
    return round(num, 2)


def _format_percent_text(value):
    pct = _format_percent_debug(value)
    if pct is None:
        return '—'
    return f"{pct:.1f}%"


def _format_customer_mix_debug(detail):
    if not isinstance(detail, dict):
        return '—'
    ordered_labels = ['유동', '거주', '직장']
    parts = []
    for label in ordered_labels:
        pct = _format_percent_text(detail.get(label))
        if pct != '—':
            parts.append(f"{label} {pct}")
    for label, value in detail.items():
        if label in ordered_labels:
            continue
        pct = _format_percent_text(value)
        if pct != '—':
            parts.append(f"{label} {pct}")
    return ', '.join(parts[:3]) if parts else '—'


def _format_age_segments_debug(segments):
    if not isinstance(segments, (list, tuple)):
        return '—'
    formatted = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        label = seg.get('label') or seg.get('code')
        value = _format_percent_text(seg.get('value'))
        if label and value != '—':
            formatted.append(f"{label} {value}")
    return ', '.join(formatted[:3]) if formatted else '—'


def _build_debug_table(qinfo, merchant_match, sanitized_snapshot):
    industry_candidate = None
    if merchant_match:
        industry_candidate = merchant_match.get('category')
    if not industry_candidate:
        industry_candidate = qinfo.get('merchant_industry_label') or qinfo.get('industry')
    industry_labels = {
        'cafe': '카페',
        'restaurant': '음식점',
        'retail': '소매',
    }
    industry = industry_labels.get(industry_candidate, industry_candidate or '—')

    address = '—'
    if merchant_match:
        addr = merchant_match.get('address')
        if isinstance(addr, (list, tuple)):
            addr = ' / '.join([str(v) for v in addr if v])
        if addr:
            address = str(addr)

    revisit = _format_percent_text((sanitized_snapshot or {}).get('revisit_pct'))
    new = _format_percent_text((sanitized_snapshot or {}).get('new_pct'))
    revisit_block = '—'
    if revisit != '—' or new != '—':
        revisit_block = f"신규 {new} / 재방문 {revisit}"

    table = {
        '업종': industry,
        '주소': address,
        '주요 고객층': _format_age_segments_debug((sanitized_snapshot or {}).get('age_top_segments')),
        '고객 유형': _format_customer_mix_debug((sanitized_snapshot or {}).get('customer_mix_detail')),
        '신규/재방문': revisit_block,
        '객단가 구간': (sanitized_snapshot or {}).get('avg_ticket_band_label') or '—',
    }
    return table

def build_panel(shinhan_dir, merchants_df=None, target_id=None):
    s1 = merchants_df if merchants_df is not None else load_set1(shinhan_dir)
    s2_all = load_set2(shinhan_dir)
    s3_all = load_set3(shinhan_dir)

    stats = {
        'set2_merchants_before': int(s2_all['ENCODED_MCT'].astype(str).nunique()) if 'ENCODED_MCT' in s2_all.columns else 0,
        'set3_merchants_before': int(s3_all['ENCODED_MCT'].astype(str).nunique()) if 'ENCODED_MCT' in s3_all.columns else 0,
    }

    s2 = s2_all
    s3 = s3_all
    if target_id:
        tid = str(target_id)
        s2 = s2_all[s2_all['ENCODED_MCT'].astype(str) == tid]
        s3 = s3_all[s3_all['ENCODED_MCT'].astype(str) == tid]

    stats['set2_rows_after'] = int(len(s2))
    stats['set3_rows_after'] = int(len(s3))

    m23 = pd.merge(s2, s3, on=['ENCODED_MCT','TA_YM','_date'], how='outer', suffixes=('_s2',''))
    panel = pd.merge(m23, s1, on='ENCODED_MCT', how='left')
    def nz(x):
        try:
            return pd.to_numeric(x, errors='coerce').fillna(0.0)
        except Exception:
            return pd.Series([0.0] * len(x))
    if 'M12_MAL_1020_RAT' in panel.columns and 'M12_FME_1020_RAT' in panel.columns:
        panel['YOUTH_SHARE'] = nz(panel['M12_MAL_1020_RAT']) + nz(panel['M12_FME_1020_RAT'])
    else:
        panel['YOUTH_SHARE'] = np.nan
    panel['REVISIT_RATE'] = pd.to_numeric(panel.get('MCT_UE_CLN_REU_RAT', np.nan), errors='coerce')
    panel['NEW_RATE'] = pd.to_numeric(panel.get('MCT_UE_CLN_NEW_RAT', np.nan), errors='coerce')
    panel.rename(columns={'ENCODED_MCT':'_merchant_id'}, inplace=True)
    if '_merchant_id' in panel.columns:
        panel['_merchant_id'] = panel['_merchant_id'].astype(str)
    stats['merchants_after'] = int(panel['_merchant_id'].nunique()) if '_merchant_id' in panel.columns else 0
    stats['set2_merchants_after'] = int(s2['ENCODED_MCT'].astype(str).nunique()) if 'ENCODED_MCT' in s2.columns else 0
    stats['set3_merchants_after'] = int(s3['ENCODED_MCT'].astype(str).nunique()) if 'ENCODED_MCT' in s3.columns else 0
    return panel, stats


def call_llm_for_mask(original_question: str | None, merchant_mask: str | None, sigungu: str | None):
    meta = {
        'used': False,
        'model': 'models/gemini-2.5-flash',
        'prompt_preview': '',
        'resp_bytes': 0,
        'safety_blocked': False,
        'elapsed_ms': 0,
        'error': None,
    }

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print('⚠️ GEMINI_API_KEY 미설정으로 LLM 보조 매칭을 건너뜁니다.')
        meta['error'] = 'missing_api_key'
        return None, meta
    try:
        import google.generativeai as genai
    except ImportError:
        print('⚠️ google-generativeai 미설치로 LLM 보조 매칭을 건너뜁니다.')
        meta['error'] = 'missing_dependency'
        return None, meta

    genai.configure(api_key=api_key)
    model_name = meta['model']
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            'temperature': 0.1,
            'top_p': 0.8,
            'max_output_tokens': 128,
        },
    )

    prompt = f"""당신은 텍스트에서 마스킹된 가맹점 단서를 정리하는 한국어 어시스턴트입니다.
질문에서 확인 가능한 상호 마스크와 시군구만 JSON으로 다시 써주세요.
추정이나 생성은 금지하며, 정보가 없으면 null을 넣습니다.

질문: {original_question}
현재 추출: merchant_mask={merchant_mask}, sigungu={sigungu}

JSON 형식:
{{"merchant_mask":"문자열 또는 null","sigungu":"문자열 또는 null","notes":"간단 메모"}}
"""

    meta['prompt_preview'] = _mask_debug_preview(prompt)
    t0 = tick()

    try:
        response = model.generate_content(prompt)
    except Exception as exc:
        meta['elapsed_ms'] = to_ms(t0)
        meta['error'] = str(exc)
        print('⚠️ LLM 보조 매칭 호출 실패:', exc)
        return None, meta

    def _response_text(resp):
        parts: list[str] = []

        def _append_text(value):
            if value:
                parts.append(str(value))

        for part in getattr(resp, 'parts', []) or []:
            _append_text(getattr(part, 'text', None))

        # google.generativeai 응답은 candidates[*].content.parts 에도 텍스트가 담길 수 있다.
        for candidate in getattr(resp, 'candidates', []) or []:
            content = getattr(candidate, 'content', None)
            if content is None:
                continue
            for part in getattr(content, 'parts', []) or []:
                _append_text(getattr(part, 'text', None))

        if hasattr(resp, 'text'):
            try:
                quick_text = resp.text
            except ValueError:
                quick_text = None
            _append_text(quick_text)

        return '\n'.join([p for p in parts if p])

    text = _response_text(response)
    meta['elapsed_ms'] = to_ms(t0)

    prompt_feedback = getattr(response, 'prompt_feedback', None)
    block_reason = None
    if prompt_feedback is not None:
        block_reason = getattr(prompt_feedback, 'block_reason', None)
    safety_blocked = bool(block_reason and str(block_reason).lower() != 'block_none')

    if not safety_blocked:
        # 후보의 finish_reason 이 안전 차단을 나타내면 안전 차단으로 간주한다.
        for candidate in getattr(response, 'candidates', []) or []:
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason is None:
                continue
            fr_text = str(finish_reason).lower()
            if 'safety' in fr_text or 'blocked' in fr_text or fr_text in {'block_safety', '2'}:
                safety_blocked = True
                break

    meta['safety_blocked'] = safety_blocked

    if not text:
        print('⚠️ LLM 보조 매칭 응답이 비었습니다.')
        meta['used'] = True
        meta['error'] = 'empty_text'
        return None, meta

    meta['used'] = True
    meta['resp_bytes'] = len(text.encode('utf-8'))

    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        print('⚠️ LLM 보조 매칭에서 JSON을 찾지 못했습니다.')
        meta['error'] = 'json_not_found'
        return None, meta

    try:
        data = json.loads(match.group(0))
    except Exception as exc:
        print('⚠️ LLM 보조 매칭 JSON 파싱 실패:', exc)
        meta['error'] = f'json_parse_error: {exc}'
        return None, meta

    return (data if isinstance(data, dict) else None), meta


def resolve_merchant(
    masked_name: str | None,
    mask_prefix: str | None,
    sigungu: str | None,
    merchants_df: pd.DataFrame | None,
    original_question: str | None = None,
    allow_llm: bool = True,
):
    debug_info = {
        'candidates': [],
        'path': None,
        'notes': None,
        'suggestions': None,
        'llm': None,
    }

    if merchants_df is None or merchants_df.empty:
        return None, debug_info

    if not masked_name:
        debug_info['notes'] = '마스킹 상호 미제공'
        print("⚠️ resolve_merchant: 입력된 마스킹 상호가 없어 매칭을 건너뜁니다.")
        return None, debug_info

    df = merchants_df.copy()
    df['_norm_name'] = df['MCT_NM'].apply(_normalize_compare)
    df['_norm_sigungu'] = df['SIGUNGU'].apply(_normalize_compare)
    df['_norm_category'] = df['CATEGORY'].apply(_normalize_compare)

    norm_sigungu = _normalize_compare(sigungu)
    prefix_norm = _normalize_compare(mask_prefix) if mask_prefix else ''

    base = df
    if norm_sigungu:
        exact = base[base['_norm_sigungu'] == norm_sigungu]
        if exact.empty:
            base = base[base['_norm_sigungu'].str.contains(norm_sigungu, na=False)]
        else:
            base = exact
    sigungu_filter_count = int(len(base))

    def _preview_candidates(frame: pd.DataFrame) -> list[dict]:
        preview = []
        for _, row in frame.head(3).iterrows():
            preview.append({
                'ENCODED_MCT': row['ENCODED_MCT'],
                'MCT_NM': row['MCT_NM'],
                'SIGUNGU': row['SIGUNGU'],
                'CATEGORY': row['CATEGORY'],
                'score': float(row.get('__score')) if '__score' in row else None,
            })
        return preview

    if base.empty:
        debug_payload = {
            'input': {'masked_name': masked_name, 'mask_prefix': mask_prefix, 'sigungu': sigungu},
            'sigungu_filter_count': sigungu_filter_count,
            'rule': 'rule1',
            'candidates': [],
        }
        print("🧭 resolve_phase:", json.dumps(debug_payload, ensure_ascii=False))
        print(f"⚠️ 가맹점 미일치 – {masked_name}·{sigungu}를 확인해 주세요.")
        debug_info['notes'] = 'sigungu_filter_empty'
        return None, debug_info

    # Rule-1 strict: startswith
    if prefix_norm:
        rule1 = base[base['_norm_name'].str.startswith(prefix_norm, na=False)]
    else:
        rule1 = base.iloc[0:0]
    rule1_count = int(len(rule1))

    if rule1_count == 1:
        row = rule1.iloc[0]
        resolved = {
            'encoded_mct': str(row['ENCODED_MCT']),
            'masked_name': row.get('MCT_NM'),
            'address': row.get('ADDR_BASE'),
            'sigungu': row.get('SIGUNGU'),
            'category': row.get('CATEGORY'),
            'score': 1.0,
        }
        debug_info['candidates'] = _preview_candidates(rule1.assign(__score=1.0))
        debug_info['path'] = 'rule1'
        print(
            "🧭 resolve_phase:",
            json.dumps({
                'input': {'masked_name': masked_name, 'mask_prefix': mask_prefix, 'sigungu': sigungu},
                'rule': 'rule1',
                'sigungu_filter_count': sigungu_filter_count,
                'rule1_count': rule1_count,
                'candidates': debug_info['candidates'],
            }, ensure_ascii=False),
        )
        print("✅ resolved_merchant_id:", resolved['encoded_mct'])
        return resolved, debug_info

    def _score_rows(frame: pd.DataFrame) -> pd.DataFrame:
        scored = frame.copy()
        scores = []
        for _, r in scored.iterrows():
            name_norm = r['_norm_name'] or ''
            base_score = 0.0
            if prefix_norm:
                if name_norm.startswith(prefix_norm):
                    base_score = 1.0
                elif prefix_norm in name_norm:
                    base_score = 0.8
                else:
                    base_score = SequenceMatcher(None, prefix_norm, name_norm).ratio()
            fuzzy = SequenceMatcher(None, prefix_norm, name_norm).ratio() if prefix_norm else 0.0
            base_val = max(base_score, fuzzy)
            length_bonus = 0.05 if prefix_norm and len(name_norm) > len(prefix_norm) else 0.0
            scores.append(round(min(base_val + length_bonus, 1.05), 4))
        scored['__score'] = scores
        scored['__name_len'] = scored['_norm_name'].str.len().fillna(0)
        return scored

    # Rule-2 fallback if strict fails
    rule2_base = base if rule1_count == 0 else rule1
    rule2 = _score_rows(rule2_base)
    top = rule2.sort_values(['__score', '__name_len', 'ENCODED_MCT'], ascending=[False, False, True])
    debug_candidates = _preview_candidates(top)
    debug_info['candidates'] = debug_candidates

    chosen = None
    path = 'rule2' if rule1_count == 0 else 'rule1'
    if not top.empty and float(top.iloc[0]['__score']) >= 0.85:
        chosen = top.iloc[0]
        debug_info['path'] = path
    elif not top.empty and rule1_count > 1:
        chosen = top.iloc[0]
        debug_info['path'] = 'rule1'

    print(
        "🧭 resolve_phase:",
        json.dumps({
            'input': {'masked_name': masked_name, 'mask_prefix': mask_prefix, 'sigungu': sigungu},
            'rule': path,
            'sigungu_filter_count': sigungu_filter_count,
            'rule1_count': rule1_count,
            'candidates': debug_candidates,
        }, ensure_ascii=False),
    )

    if chosen is not None:
        resolved = {
            'encoded_mct': str(chosen['ENCODED_MCT']),
            'masked_name': chosen.get('MCT_NM'),
            'address': chosen.get('ADDR_BASE'),
            'sigungu': chosen.get('SIGUNGU'),
            'category': chosen.get('CATEGORY'),
            'score': float(chosen.get('__score')) if pd.notna(chosen.get('__score')) else None,
        }
        print("✅ resolved_merchant_id:", resolved['encoded_mct'])
        return resolved, debug_info

    # Rule-2 failed → optional LLM assist
    if allow_llm:
        llm_result, llm_meta = call_llm_for_mask(original_question, masked_name, sigungu)
        debug_info['notes'] = 'llm_invoked'
        if llm_meta:
            debug_info['llm'] = {'parsed': llm_result, **llm_meta}
        if llm_result:
            new_mask = llm_result.get('merchant_mask') or masked_name
            new_prefix = (new_mask.split('*', 1)[0].strip() if new_mask else mask_prefix)
            new_sigungu = llm_result.get('sigungu') or sigungu
            if (new_mask, new_sigungu) != (masked_name, sigungu):
                match, nested_debug = resolve_merchant(
                    new_mask,
                    new_prefix,
                    new_sigungu,
                    merchants_df,
                    original_question=original_question,
                    allow_llm=False,
                )
                if isinstance(nested_debug, dict):
                    if llm_meta:
                        nested_debug.setdefault('llm', {'parsed': llm_result, **llm_meta})
                    nested_debug['notes'] = nested_debug.get('notes') or 'llm_invoked'
                    if not nested_debug.get('path'):
                        nested_debug['path'] = 'llm'
                return match, nested_debug

    # No match → surface suggestions
    base_scores = []
    if prefix_norm:
        for _, row in base.iterrows():
            ratio = SequenceMatcher(None, prefix_norm, row['_norm_name'] or '').ratio()
            base_scores.append((ratio, row))
        base_scores.sort(key=lambda x: x[0], reverse=True)
        suggestions = [
            {
                'ENCODED_MCT': r['ENCODED_MCT'],
                'MCT_NM': r['MCT_NM'],
                'SIGUNGU': r['SIGUNGU'],
                'CATEGORY': r['CATEGORY'],
            }
            for ratio, r in base_scores[:3]
            if ratio > 0
        ]
    else:
        suggestions = []

    debug_info['suggestions'] = suggestions
    print(f"⚠️ 가맹점 미일치 – {masked_name}·{sigungu}를 확인해 주세요.")
    if suggestions:
        print("🔍 유사 후보:", json.dumps(suggestions, ensure_ascii=False))
    return None, debug_info

def parse_question(q):
    original = q or ''
    normalized = unicodedata.normalize('NFKC', original)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    lower_q = normalized.lower()
    age_cond = None
    if '10대' in original or 'teen' in lower_q or re.search(r'\b1[0-9]\b', lower_q):
        age_cond = ('<=', 19)
    if '20대' in original or '20s' in lower_q or 'twenties' in lower_q:
        if ('이하' in original) or ('under' in lower_q) or ('<=' in lower_q):
            age_cond = ('<=', 20)
        else:
            age_cond = ('range', (20,29))
    if '청소년' in original:
        age_cond = ('<=', 19)

    weather = None
    if ('비' in original) or ('우천' in original) or ('rain' in lower_q):
        weather = 'rain'
    elif ('맑' in original) or ('sunny' in lower_q) or ('clear' in lower_q):
        weather = 'clear'
    elif ('눈' in original) or ('snow' in lower_q):
        weather = 'snow'

    months = DEFAULT_MONTHS
    weeks_requested = None
    week_match = re.search(r'(\d+)\s*주', original)
    if week_match:
        try:
            weeks_requested = int(week_match.group(1))
        except ValueError:
            weeks_requested = None
        if weeks_requested and weeks_requested > 0:
            months = max(1, round(weeks_requested / 4))
    if '이번달' in original or 'this month' in lower_q:
        months = 1
    elif ('한달' in original) or ('1달' in original) or ('month' in lower_q):
        months = 1
    elif '분기' in original or 'quarter' in lower_q:
        months = 3

    industry = None
    if ('카페' in original) or ('커피' in original):
        industry = 'cafe'
    elif ('요식' in original) or ('restaurant' in lower_q) or ('식당' in original):
        industry = 'restaurant'

    merchant_mask = None
    pattern_used = 'none'
    brace_match = re.search(r'\{([^{}]+)\}', normalized)
    if brace_match:
        merchant_mask = brace_match.group(1).strip()
        pattern_used = 'curly_brace'

    mask_prefix = None
    if merchant_mask:
        mask_prefix = merchant_mask.split('*', 1)[0].strip()

    sigungu_pattern = 'hangul_gu_regex'
    sigungu_match = re.search(r'(?P<sigungu>[가-힣]{2,}구)', normalized)
    if sigungu_match:
        merchant_sigungu = sigungu_match.group('sigungu')
    else:
        merchant_sigungu = '성동구'
        sigungu_pattern = 'default_sigungu'

    merchant_info = {
        'masked_name': merchant_mask,
        'mask_prefix': mask_prefix,
        'sigungu': merchant_sigungu,
        'industry_label': None,
    }

    explicit_id = None
    trimmed = original.strip()
    if re.fullmatch(r'[A-Z0-9]{10,12}', trimmed):
        explicit_id = trimmed

    return {
        'original_question': original,
        'age_cond': age_cond,
        'weather': weather,
        'months': months,
        'weeks_requested': weeks_requested,
        'industry': industry,
        'normalized_question': normalized,
        'merchant_masked_name': merchant_info['masked_name'],
        'merchant_mask_prefix': merchant_info['mask_prefix'],
        'merchant_sigungu': merchant_info['sigungu'],
        'merchant_industry_label': merchant_info['industry_label'],
        'merchant_explicit_id': explicit_id,
        'merchant_pattern_used': pattern_used,
        'merchant_sigungu_pattern': sigungu_pattern,
    }

def subset_period(panel, months=DEFAULT_MONTHS):
    if panel['_date'].isna().all():
        return panel.iloc[0:0]
    maxd = panel['_date'].max()
    thr = maxd - pd.Timedelta(days=31*months)
    return panel[panel['_date'] >= thr]

def kpi_summary(panel_sub):
    if panel_sub.empty:
        return {}, {'latest_raw_snapshot': None, 'sanitized_snapshot': None}
    latest_idx = panel_sub.groupby('_merchant_id')['_date'].idxmax()
    snap = panel_sub.loc[latest_idx].sort_values('_date', ascending=False)

    def _safe_float(val):
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        return f

    def _clean_pct(val):
        if val is None or pd.isna(val):
            return None
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        if f < 0 or f > 100:
            return None
        return round(f, 1)

    detail_row = snap.iloc[0]
    youth_latest = _safe_float(detail_row.get('YOUTH_SHARE'))
    revisit_latest = _safe_float(detail_row.get('REVISIT_RATE'))
    new_latest = _safe_float(detail_row.get('NEW_RATE'))

    age_labels = {
        '1020': '청년(10-20)',
        '30': '30대',
        '40': '40대',
        '50': '50대',
        '60': '60대',
    }
    age_distribution = []
    age_allowlist = []
    for code, label in age_labels.items():
        cols = [f'M12_MAL_{code}_RAT', f'M12_FME_{code}_RAT']
        raw_vals = pd.Series([detail_row.get(c) for c in cols])
        numeric_vals = pd.to_numeric(raw_vals, errors='coerce')
        if numeric_vals.notna().sum() == 0:
            continue
        total = numeric_vals.sum(skipna=True)
        cleaned = _clean_pct(total)
        if cleaned is not None:
            age_distribution.append({'code': code, 'label': label, 'value': cleaned})
            age_allowlist.append(code)
    age_distribution.sort(key=lambda x: x['value'], reverse=True)

    customer_mix_map = [
        ('유동', 'RC_M1_SHC_FLP_UE_CLN_RAT'),
        ('거주', 'RC_M1_SHC_RSD_UE_CLN_RAT'),
        ('직장', 'RC_M1_SHC_WP_UE_CLN_RAT'),
    ]
    customer_mix_detail = {}
    for label, col in customer_mix_map:
        customer_mix_detail[label] = _clean_pct(_safe_float(detail_row.get(col)))

    ticket_band_raw = detail_row.get('APV_CE_RAT')
    ticket_band = None
    if isinstance(ticket_band_raw, str):
        parts = ticket_band_raw.split('_', 1)
        ticket_band = parts[-1].strip() if parts else ticket_band_raw.strip()
    elif pd.notna(ticket_band_raw):
        ticket_band = str(ticket_band_raw)

    sanitized = {
        'youth_share_avg': _clean_pct(youth_latest),
        'revisit_rate_avg': _clean_pct(revisit_latest),
        'new_rate_avg': _clean_pct(new_latest),
        'age_distribution': age_distribution,
        'age_top_segments': age_distribution[:3],
        'age_allowlist': age_allowlist,
        'customer_mix_detail': customer_mix_detail,
        'avg_ticket_band_label': ticket_band,
        'n_merchants': int(snap['_merchant_id'].nunique()),
    }

    raw_snapshot = {
        'TA_YM': detail_row.get('TA_YM'),
        '_date': str(detail_row.get('_date')),
        'MCT_UE_CLN_REU_RAT_raw': detail_row.get('MCT_UE_CLN_REU_RAT_raw'),
        'MCT_UE_CLN_NEW_RAT_raw': detail_row.get('MCT_UE_CLN_NEW_RAT_raw'),
        'M12_MAL_1020_RAT_raw': detail_row.get('M12_MAL_1020_RAT_raw'),
        'M12_FME_1020_RAT_raw': detail_row.get('M12_FME_1020_RAT_raw'),
        'RC_M1_SHC_FLP_UE_CLN_RAT_raw': detail_row.get('RC_M1_SHC_FLP_UE_CLN_RAT_raw'),
        'RC_M1_SHC_RSD_UE_CLN_RAT_raw': detail_row.get('RC_M1_SHC_RSD_UE_CLN_RAT_raw'),
        'RC_M1_SHC_WP_UE_CLN_RAT_raw': detail_row.get('RC_M1_SHC_WP_UE_CLN_RAT_raw'),
    }

    sanitized_snapshot = {
        'revisit_pct': sanitized['revisit_rate_avg'],
        'new_pct': sanitized['new_rate_avg'],
        'youth_pct': sanitized['youth_share_avg'],
        'customer_mix_detail': sanitized['customer_mix_detail'],
        'age_top_segments': sanitized['age_top_segments'],
        'age_allowlist': sanitized['age_allowlist'],
        'avg_ticket_band_label': sanitized['avg_ticket_band_label'],
    }

    print("🗂 KPI raw snapshot:", json.dumps(raw_snapshot, ensure_ascii=False))
    print("✅ KPI sanitized:", json.dumps(sanitized_snapshot, ensure_ascii=False))

    return sanitized, {'latest_raw_snapshot': raw_snapshot, 'sanitized_snapshot': sanitized_snapshot}


    sanitized_snapshot = {
        'revisit_pct': sanitized['revisit_rate_avg'],
        'new_pct': sanitized['new_rate_avg'],
        'youth_pct': sanitized['youth_share_avg'],
        'customer_mix_detail': sanitized['customer_mix_detail'],
        'age_top_segments': sanitized['age_top_segments'],
        'avg_ticket_band_label': sanitized['avg_ticket_band_label'],
    }

    print("🗂 KPI raw snapshot:", json.dumps(raw_snapshot, ensure_ascii=False))
    print("✅ KPI sanitized:", json.dumps(sanitized_snapshot, ensure_ascii=False))

    return sanitized, {'latest_raw_snapshot': raw_snapshot, 'sanitized_snapshot': sanitized_snapshot}


def weather_effect(panel_sub, wx_monthly):
    if (wx_monthly is None) or panel_sub.empty or ('REVISIT_RATE' not in panel_sub):
        return {'metric':'REVISIT_RATE','effect':None,'ci':[None,None],'note':'날씨/표본 부족'}
    m = panel_sub.groupby('TA_YM', as_index=False)['REVISIT_RATE'].mean()
    m = m.merge(wx_monthly[['TA_YM','RAIN_SUM']], on='TA_YM', how='inner')
    if m.empty or m['RAIN_SUM'].nunique() < 2:
        return {'metric':'REVISIT_RATE','effect':None,'ci':[None,None],'note':'상관 추정 불가'}
    corr = m['REVISIT_RATE'].corr(m['RAIN_SUM'])
    return {'metric':'REVISIT_RATE','effect':float(corr), 'ci':[None,None], 'note':'피어슨 상관(월단위)'}

def agent1_pipeline(question, shinhan_dir=SHINHAN_DIR, external_dir=EXTERNAL_DIR):
    debug_block = {
        'input': {
            'original': _mask_debug_preview(question, limit=120),
            'flags': {
                'USE_LLM': USE_LLM,
                'DEBUG_MAX_PREVIEW': DEBUG_MAX_PREVIEW,
                'DEBUG_SHOW_RAW': DEBUG_SHOW_RAW,
            },
        },
        'errors': [],
    }

    merchants_df = load_set1(shinhan_dir)

    parse_t0 = tick()
    try:
        qinfo = parse_question(question)
    except Exception as exc:
        debug_block['errors'].append({'stage': 'parse', 'msg': str(exc)})
        debug_block['parse'] = {'elapsed_ms': to_ms(parse_t0)}
        raise
    parse_elapsed = to_ms(parse_t0)
    debug_block['parse'] = {
        'merchant_mask': qinfo.get('merchant_masked_name'),
        'mask_prefix': qinfo.get('merchant_mask_prefix'),
        'sigungu': qinfo.get('merchant_sigungu'),
        'pattern_used': qinfo.get('merchant_pattern_used'),
        'elapsed_ms': parse_elapsed,
    }

    run_id = datetime.datetime.utcnow().isoformat()
    parse_log = {
        'original': qinfo.get('original_question'),
        'merchant_mask': qinfo.get('merchant_masked_name'),
        'mask_prefix': qinfo.get('merchant_mask_prefix'),
        'sigungu': qinfo.get('merchant_sigungu'),
        'explicit_id': qinfo.get('merchant_explicit_id'),
    }
    print("🆔 agent1_run:", run_id)
    print("🧾 question_fields:", json.dumps(parse_log, ensure_ascii=False))

    merchant_match = None
    resolve_meta = {
        'candidates': [],
        'path': None,
        'notes': None,
        'suggestions': None,
        'llm': None,
    }

    resolve_stage = {
        'path': 'none',
        'candidates_top3': [],
        'resolved_merchant_id': None,
    }

    resolve_t0 = tick()
    try:
        explicit_id = qinfo.get('merchant_explicit_id')
        if explicit_id:
            lookup = merchants_df[merchants_df['ENCODED_MCT'] == explicit_id]
            print(
                "🏷 explicit_id_lookup:",
                json.dumps({'explicit_id': explicit_id, 'row_count': int(len(lookup))}, ensure_ascii=False),
            )
            if not lookup.empty:
                row = lookup.iloc[0]
                merchant_match = {
                    'encoded_mct': str(row['ENCODED_MCT']),
                    'masked_name': row.get('MCT_NM'),
                    'address': row.get('ADDR_BASE'),
                    'sigungu': row.get('SIGUNGU'),
                    'category': row.get('CATEGORY'),
                    'score': None,
                }
                resolve_meta['path'] = 'user'

        if merchant_match is None:
            merchant_match, resolve_meta = resolve_merchant(
                qinfo.get('merchant_masked_name'),
                qinfo.get('merchant_mask_prefix'),
                qinfo.get('merchant_sigungu'),
                merchants_df,
                original_question=qinfo.get('normalized_question') or question,
                allow_llm=USE_LLM,
            )
    except Exception as exc:
        debug_block['errors'].append({'stage': 'resolve', 'msg': str(exc)})
        raise
    finally:
        resolve_stage['elapsed_ms'] = to_ms(resolve_t0)

    target_id = None
    if merchant_match and merchant_match.get('encoded_mct') is not None:
        target_id = str(merchant_match['encoded_mct'])
        merchant_match['encoded_mct'] = target_id

    if resolve_meta.get('path') is None:
        resolve_meta['path'] = 'llm' if resolve_meta.get('llm') else 'none'
    resolve_stage['path'] = resolve_meta.get('path') or 'none'
    resolve_stage['resolved_merchant_id'] = target_id

    candidate_payload = []
    for cand in resolve_meta.get('candidates') or []:
        cid = cand.get('ENCODED_MCT') or cand.get('encoded_mct')
        try:
            score_val = cand.get('score')
            score = round(float(score_val), 4) if score_val is not None else None
        except (TypeError, ValueError):
            score = None
        candidate_payload.append({
            'id': str(cid) if cid is not None else None,
            'name': cand.get('MCT_NM') or cand.get('masked_name'),
            'sigungu': cand.get('SIGUNGU') or cand.get('sigungu'),
            'score': score,
        })
    resolve_stage['candidates_top3'] = candidate_payload[:3]
    debug_block['resolve'] = resolve_stage

    llm_meta = resolve_meta.get('llm') or {}
    agent1_llm = {
        'used': bool(llm_meta.get('used')),
        'model': llm_meta.get('model'),
        'prompt_preview': llm_meta.get('prompt_preview', ''),
        'resp_bytes': llm_meta.get('resp_bytes'),
        'safety_blocked': bool(llm_meta.get('safety_blocked')),
        'elapsed_ms': llm_meta.get('elapsed_ms'),
    }
    debug_block['agent1_llm'] = agent1_llm

    panel_stage = {}
    panel_t0 = tick()
    try:
        panel, panel_stats = build_panel(shinhan_dir, merchants_df=merchants_df, target_id=target_id)
    except Exception as exc:
        panel_stage['elapsed_ms'] = to_ms(panel_t0)
        debug_block['errors'].append({'stage': 'panel', 'msg': str(exc)})
        debug_block['panel'] = panel_stage
        raise
    panel_elapsed = to_ms(panel_t0)
    sub = subset_period(panel, months=qinfo['months'])
    panel_stage.update({
        'rows_before': int(len(panel)),
        'rows_after': int(len(sub)),
        'latest_ta_ym': str(sub['TA_YM'].max()) if not sub.empty and 'TA_YM' in sub.columns else None,
        'elapsed_ms': panel_elapsed,
        'stats': panel_stats,
    })
    debug_block['panel'] = panel_stage

    wxm = None
    try:
        wxm = load_weather_monthly(external_dir)
    except Exception as exc:
        debug_block['errors'].append({'stage': 'weather', 'msg': str(exc)})
        wxm = None

    snapshot_t0 = tick()
    kpis, kpi_debug = kpi_summary(sub)
    snapshot_elapsed = to_ms(snapshot_t0)

    raw_snapshot = {}
    raw_source = (kpi_debug or {}).get('latest_raw_snapshot') or {}
    for key, value in raw_source.items():
        if key.endswith('_raw'):
            raw_snapshot[key[:-4]] = value
        else:
            raw_snapshot[key] = value
    sanitized_snapshot = (kpi_debug or {}).get('sanitized_snapshot') or {}
    debug_block['snapshot'] = {
        'raw': raw_snapshot,
        'sanitized': sanitized_snapshot,
        'elapsed_ms': snapshot_elapsed,
    }

    render_table = _build_debug_table(qinfo, merchant_match, sanitized_snapshot)
    debug_block['render'] = {
        'table_dict': render_table,
    }

    wfx = weather_effect(sub, wxm)

    notes = []
    quality = 'normal'
    if sub.empty:
        notes.append('질문 조건 표본 부족 또는 기간 데이터 없음')
        quality = 'low'
    if wxm is None and qinfo['weather'] is not None:
        notes.append('날씨 데이터 부재: 날씨 관련 효과는 추정하지 못했습니다.')
        quality = 'low'
    if qinfo.get('merchant_masked_name') is None:
        notes.append('{상호} 형태의 입력이 없어 가맹점 식별을 진행하지 못했습니다.')
        quality = 'low'
    if merchant_match is None:
        notes.append('질문과 일치하는 가맹점을 찾지 못해 전체 표본을 사용했습니다.')
        quality = 'low'

    merchant_query = {
        'masked_name': qinfo.get('merchant_masked_name'),
        'mask_prefix': qinfo.get('merchant_mask_prefix'),
        'sigungu': qinfo.get('merchant_sigungu'),
        'industry_label': qinfo.get('merchant_industry_label'),
    }

    merchants_covered = int(sub['_merchant_id'].nunique()) if not sub.empty else 0

    out = {
        'context': {
            'intent': question,
            'parsed': qinfo,
            'merchant_query': merchant_query,
            'run_id': run_id,
            'panel_stats': panel_stage.get('stats', {}),
            'merchant_candidates': resolve_meta.get('candidates'),
            'merchant_resolution_path': resolve_stage['path'],
        },
        'kpis': kpis,
        'weather_effect': wfx,
        'limits': notes,
        'quality': quality,
        'period': {
            'max_date': str(panel['_date'].max() if '_date' in panel.columns else None),
            'months': qinfo['months'],
            'weeks_requested': qinfo.get('weeks_requested'),
        },
        'sample': {
            'merchants_covered': merchants_covered
        },
        'debug': debug_block,
    }

    if merchant_match:
        out['context']['merchant'] = merchant_match
        out['context']['merchant_masked_name'] = merchant_match.get('masked_name')

    out_path = OUTPUT_DIR / 'agent1_output.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('✅ Agent-1 JSON 저장:', out_path)
    return out

QUESTION_TYPE_INFO = {
    "Q1_CAFE_CHANNELS": {
        "label": "주요 방문 고객 특성에 따른 채널 추천 및 홍보안",
        "instructions": [
            "연령/성별·유동/거주 구성 비중을 활용해 채널과 메시지를 제시합니다.",
            "온·오프라인 3~4개 홍보 아이디어를 간결하게 작성합니다.",
            "각 아이디어는 고객군 → 채널 → 실행 요약을 포함합니다.",
        ],
    },
    "Q2_LOW_RETENTION": {
        "label": "재방문률 30% 이하 개선 아이디어",
        "instructions": [
            "재방문·신규 비중을 근거로 재방문 촉진 액션을 제시합니다.",
            "3~4개의 프로모션/멤버십/CRM 아이디어를 제공합니다.",
            "각 아이디어는 타깃 고객과 실행 단계를 분명히 합니다.",
        ],
    },
    "Q3_FOOD_ISSUE": {
        "label": "요식업의 가장 큰 문제 가설 + 보완 아이디어",
        "instructions": [
            "식음업 특성(방문 고객/시간대/유동)을 근거로 문제 가설을 세웁니다.",
            "3~4개의 개선 아이디어를 제시하고 실행 단계를 나열합니다.",
            "문제 가설과 해결 아이디어를 한 세트로 서술합니다.",
        ],
    },
    "GENERIC": {
        "label": "일반 컨설팅 질문",
        "instructions": [
            "핵심 고객·성과 데이터를 근거로 3~4개의 실행 아이디어를 제공합니다.",
            "각 아이디어는 대상, 채널, 실행 단계, 측정 지표를 포함합니다.",
        ],
    },
}


ORGANIZER_QUESTION_TYPES = {
    "Q1_CAFE_CHANNELS",
    "Q2_LOW_RETENTION",
    "Q3_FOOD_ISSUE",
}


def _schema_key_for_question(question_type: str | None) -> str:
    if question_type in ORGANIZER_QUESTION_TYPES:
        return "organizer"
    return "legacy"


def _resolve_schema_for_question(
    question_type: str | None,
) -> tuple[dict, Draft7Validator | None, str]:
    schema_bundle, validator_bundle = load_actioncard_schema_current()
    key = _schema_key_for_question(question_type)
    if isinstance(schema_bundle, dict) and "schemas" in schema_bundle:
        schema_obj = schema_bundle.get("schemas", {}).get(key) or {}
    else:
        schema_obj = schema_bundle if isinstance(schema_bundle, dict) else {}
        key = "legacy"

    validator = None
    if isinstance(validator_bundle, dict):
        validator = validator_bundle.get(key)
    else:
        validator = validator_bundle
    return schema_obj, validator, key


def infer_question_type(question_text: str | None) -> str:
    text = (question_text or "").lower()
    if not text:
        return "GENERIC"
    if any(keyword in text for keyword in ["채널", "홍보", "sns", "캠페인"]):
        return "Q1_CAFE_CHANNELS"
    if any(keyword in text for keyword in ["재방문", "retention", "재구매", "단골"]):
        return "Q2_LOW_RETENTION"
    if any(keyword in text for keyword in ["요식", "식당", "food", "맛집"]):
        return "Q3_FOOD_ISSUE"
    return "GENERIC"


def _summarise_rag_context(rag_context: dict | None) -> tuple[str, str]:
    if not isinstance(rag_context, dict):
        return ("", "RAG 비활성화: 컨텍스트가 전달되지 않았습니다.")

    enabled = bool(rag_context.get("enabled"))
    threshold = rag_context.get("threshold")
    max_score = rag_context.get("max_score")
    raw_hits = rag_context.get("hits")
    chunks_for_hits = rag_context.get("chunks") if isinstance(rag_context.get("chunks"), list) else []
    hits = int(raw_hits if raw_hits is not None else len(chunks_for_hits))
    selected_docs = rag_context.get("selected_doc_ids") or []
    mode = rag_context.get("mode") or "auto"
    reason_lines: list[str] = []

    if not enabled:
        if rag_context.get("selection_missing"):
            reason_lines.append("RAG 요청됨이나 선택된 문서가 없습니다.")
        elif rag_context.get("requested") and rag_context.get("error"):
            reason_lines.append(f"오류: {rag_context['error']}")
        else:
            reason_lines.append("UI 토글 또는 모드로 인해 비활성화되었습니다.")
    else:
        reason_lines.append(f"모드={mode}, 선택 문서={selected_docs or '없음'}")
        if max_score is None:
            reason_lines.append("최고 점수를 계산하지 못했습니다.")
        elif threshold is not None and max_score < threshold and mode != "always":
            reason_lines.append(f"최고 점수 {max_score:.2f} < 임계값 {threshold:.2f}")

    include_rag = bool(
        enabled
        and hits > 0
        and (mode == "always" or threshold is None or (max_score is not None and max_score >= threshold))
    )

    prompt_block = ""
    if include_rag:
        snippets = []
        for chunk in (rag_context.get("chunks") or [])[: hits or 5]:
            text = str(chunk.get("text") or "").strip()
            if len(text) > 220:
                text = text[:220].rstrip() + "…"
            snippets.append(
                {
                    "doc_id": chunk.get("doc_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "score": float(chunk.get("score") or 0.0),
                    "snippet": text,
                }
            )
        rag_payload = json.dumps(snippets, ensure_ascii=False, indent=2)
        summary = f"RAG 포함: hits={hits}, max_score={max_score}, threshold={threshold}, mode={mode}"
        prompt_block = f"{summary}\n{rag_payload}"
        rag_context['prompt_note'] = summary
    else:
        reason = " ; ".join(reason_lines) if reason_lines else "근거 없음"
        rag_context['prompt_note'] = f"RAG 제외: {reason}"

    return prompt_block, "\n- ".join(reason_lines)


def build_agent2_prompt_overhauled(
    agent1_json,
    *,
    question_text: str | None = None,
    question_type: str | None = None,
    rag_context: dict | None = None,
):
    inferred_type = question_type or infer_question_type(question_text)
    info = QUESTION_TYPE_INFO.get(inferred_type, QUESTION_TYPE_INFO["GENERIC"])
    schema_obj = {}
    try:
        schema_obj, _, _ = _resolve_schema_for_question(inferred_type)
    except Exception as exc:
        schema_obj = {"schema_error": str(exc)}
    schema_text = json.dumps(schema_obj, ensure_ascii=False, indent=2)

    snapshot = (agent1_json or {}).get("debug", {}).get("snapshot", {})
    if isinstance(snapshot, dict):
        sanitized_snapshot = snapshot.get("sanitized") or {}
    else:
        sanitized_snapshot = {}
    age_allowlist = []
    if isinstance(sanitized_snapshot, dict):
        allowlist_candidate = sanitized_snapshot.get("age_allowlist")
        if isinstance(allowlist_candidate, list):
            age_allowlist = [str(code) for code in allowlist_candidate if str(code)]
        elif isinstance(allowlist_candidate, (tuple, set)):
            age_allowlist = [str(code) for code in allowlist_candidate if str(code)]
        if not age_allowlist:
            distribution = sanitized_snapshot.get("age_distribution") or []
            if isinstance(distribution, list):
                for item in distribution:
                    code = str(item.get("code")) if isinstance(item, dict) else None
                    if code:
                        age_allowlist.append(code)
    age_allowlist = list(dict.fromkeys(age_allowlist))

    base_rules = [
        "질문에 직접 답하십시오. 목표치나 구간을 임의로 추정하지 마십시오(제공된 경우에만 사용).",
        "주요 근거는 Agent-1 JSON이며, RAG가 활성화되고 유효할 때만 RAG 스니펫을 근거로 포함합니다.",
        "모든 아이디어에 최소 1개 이상의 근거를 붙이고, 없으면 '근거 없음'으로 명시합니다.",
        "근거에는 출처(STRUCTURED/RAG)와 핵심 수치 또는 스니펫을 함께 제시합니다.",
        "상호명은 항상 마스킹 상태를 유지합니다.",
        "answers 배열에 3~4개의 간결한 아이디어를 작성합니다.",
        "age cohort 규칙: Agent-1 JSON에 존재하는 연령대만 사용하고, 0~100% 범위를 벗어나면 '—'로 표기합니다.",
    ]
    if age_allowlist:
        base_rules.append(f"사용 가능한 연령대 코드: {', '.join(age_allowlist)}")

    type_rules = info.get("instructions", [])

    rag_block, rag_reason = _summarise_rag_context(rag_context)

    sections = [
        "당신은 한국어 소상공인 컨설턴트입니다.",
        f"question_type={inferred_type}",
        f"질문 유형 설명: {info['label']}",
        f"질문 원문: {question_text or '—'}",
        "[출력 규칙]",
        "- " + "\n- ".join(base_rules),
    ]

    if type_rules:
        sections.append("[질문 유형별 지침]")
        sections.append("- " + "\n- ".join(type_rules))

    sections.append("[출력 스키마(JSON)]")
    sections.append(schema_text)
    sections.append("[데이터(JSON)]")
    sections.append(json.dumps(agent1_json, ensure_ascii=False, indent=2))

    if rag_block:
        sections.append("[RAG_CONTEXT]")
        sections.append(rag_block)
    elif rag_reason:
        sections.append(f"[RAG 참고 메모]\n- {rag_reason}")

    sections.append("[응답 형식]")
    sections.append("JSON만 출력하세요. 마크다운/설명/코드블럭 금지. 문자열 내 줄바꿈은 \\n으로 표기하세요.")

    guide = "\n\n".join(sections)
    schema_keys = []
    if isinstance(schema_obj, dict) and isinstance(schema_obj.get("properties"), dict):
        schema_keys = sorted(schema_obj["properties"].keys())
    rag_doc_ids = []
    rag_threshold = None
    rag_max_score = None
    rag_mode = None
    if isinstance(rag_context, dict):
        rag_doc_ids = list(rag_context.get("selected_doc_ids") or [])
        rag_threshold = rag_context.get("threshold")
        rag_max_score = rag_context.get("max_score")
        rag_mode = rag_context.get("mode")
    global AGENT2_PROMPT_TRACE
    AGENT2_PROMPT_TRACE = {
        "question_type": inferred_type,
        "organizer_mode": inferred_type in ORGANIZER_QUESTION_TYPES,
        "schema_keys": schema_keys,
        "rag_included": bool(rag_block),
        "rag_reason": rag_reason,
        "rag_context_doc_ids": rag_doc_ids,
        "rag_threshold": rag_threshold,
        "rag_max_score": rag_max_score,
        "rag_mode": rag_mode,
    }
    return guide

def call_gemini_agent2_overhauled(
    prompt_text,
    model_name: str = 'models/gemini-2.5-flash',
    *,
    question_type: str | None = None,
    agent1_json: dict | None = None,
    **kwargs,
):
    """Call Gemini for Agent-2 with robust JSON parsing and fallback cards."""

    import google.generativeai as genai
    import datetime

    if kwargs:
        _ = ", ".join(sorted(kwargs.keys()))  # noqa: F841 - reserved for debugging

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY가 설정되지 않았습니다.')
    genai.configure(api_key=api_key)

    candidates: list[str] = []
    try:
        available = []
        for model in genai.list_models():
            if "generateContent" in getattr(model, "supported_generation_methods", []):
                available.append(model.name)
        for name in available:
            tail = name.split('/')[-1]
            if '2.5' in tail and 'flash' in tail:
                candidates.append(name)
    except Exception:
        pass

    if not candidates:
        candidates = [
            "models/gemini-2.5-flash",
            "models/gemini-2.5-flash-latest",
            "models/gemini-2.5-flash-001",
            "gemini-2.5-flash",
            "gemini-2.5-flash-latest",
            "gemini-2.5-flash-001",
        ]

    if model_name and model_name not in candidates:
        candidates.insert(0, model_name)

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.1,
        "top_k": 32,
        "max_output_tokens": 2048,
    }

    prompt_payload = (prompt_text or "").rstrip()
    if "JSON만 출력하세요" not in prompt_payload:
        prompt_payload = (
            prompt_payload
            + "\n\nJSON만 출력하세요. 마크다운/설명/코드블럭 금지. 문자열 내 줄바꿈은 \\n으로 표기하세요."
        )

    try:
        _, schema_validator, schema_key = _resolve_schema_for_question(question_type)
        schema_error = None
    except Exception as exc:
        schema_validator = None
        schema_key = "unknown"
        schema_error = str(exc)

    global AGENT2_RESPONSE_TRACE
    AGENT2_RESPONSE_TRACE = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "question_type": question_type,
        "requested_model": model_name,
        "model_candidates": list(dict.fromkeys(candidates)),
        "generation_config": dict(generation_config),
        "schema_key": schema_key,
        "schema_error": schema_error,
        "prompt_length": len(prompt_payload),
        "attempts": [],
    }

    def _extract_text(resp) -> str:
        text_value = ''
        try:
            payload = getattr(resp, 'text', None)
            if payload:
                text_value = payload
        except Exception:
            pass
        if (not text_value) and getattr(resp, 'candidates', None):
            try:
                cand0 = resp.candidates[0]
                if cand0 and getattr(cand0, 'content', None) and cand0.content.parts:
                    for part in cand0.content.parts:
                        piece = getattr(part, 'text', '')
                        if piece:
                            text_value += piece
            except Exception:
                pass
        return (text_value or '').strip()

    def _blocked_info(resp):
        info = {}
        try:
            info['prompt_feedback'] = getattr(resp, 'prompt_feedback', None)
        except Exception:
            pass
        try:
            if getattr(resp, 'candidates', None):
                info['candidate_safety'] = getattr(resp.candidates[0], 'safety_ratings', None)
                info['finish_reason'] = getattr(resp.candidates[0], 'finish_reason', None)
        except Exception:
            pass
        return info

    def _run_once(name: str, prompt_body: str):
        model = genai.GenerativeModel(
            model_name=name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        response = model.generate_content(
            prompt_body,
            response_mime_type="application/json",
        )
        text_value = _extract_text(response)
        info = _blocked_info(response)
        return text_value, info, name

    def _repair_with_llm(name: str, failing_text: str):
        snippet = _strip_code_fences(failing_text or "").strip()
        if len(snippet) > 4000:
            snippet = snippet[:4000]
        repair_prompt = (
            "아래 JSON을 스키마에 맞게 고쳐 JSON만 반환하세요. 고칠 수 없으면 {\"answers\": []}를 반환하세요."
            "\n\n=== JSON 후보 ===\n" + snippet
        )
        repair_config = dict(generation_config)
        repair_config.update({"temperature": 0.1, "top_p": 0.05, "max_output_tokens": 1024})
        try:
            model = genai.GenerativeModel(
                model_name=name,
                generation_config=repair_config,
                safety_settings=safety_settings,
            )
            response = model.generate_content(
                repair_prompt,
                response_mime_type="application/json",
            )
            repaired_text = _extract_text(response)
            info = _blocked_info(response)
            return repaired_text, {"meta": info}
        except Exception as exc:  # pragma: no cover - network guard
            return None, {"error": str(exc)}

    chosen_payload = None
    chosen_model = None
    last_error = schema_error or "unknown"

    for attempt in range(2):
        for model_candidate in candidates:
            attempt_record = {
                "attempt": attempt + 1,
                "model": model_candidate,
            }
            try:
                text_value, info, used_model = _run_once(model_candidate, prompt_payload)
            except Exception as exc:  # pragma: no cover - network guard
                attempt_record["error"] = str(exc)
                last_error = str(exc)
                AGENT2_RESPONSE_TRACE["attempts"].append(attempt_record)
                continue

            attempt_record["raw_preview"] = (text_value or "")[:600]
            attempt_record["raw_length"] = len(text_value or "")
            attempt_record["meta"] = info

            parsed, parse_logs = llm_json_safe_parse(text_value, schema_validator)
            attempt_record["parse_logs"] = parse_logs

            if parsed is not None:
                attempt_record["status"] = "parsed"
                chosen_payload = parsed
                chosen_model = used_model
                AGENT2_RESPONSE_TRACE["attempts"].append(attempt_record)
                break

            attempt_record["status"] = "parse_failed"
            if parse_logs:
                last_error = parse_logs[-1].get("error") or last_error

            repaired_text, repair_meta = _repair_with_llm(model_candidate, text_value)
            attempt_record["repair_preview"] = (repaired_text or "")[:600]
            attempt_record["repair_meta"] = repair_meta
            if repaired_text:
                repaired, repair_logs = llm_json_safe_parse(repaired_text, schema_validator)
                attempt_record["repair_logs"] = repair_logs
                if repaired is not None:
                    attempt_record["status"] = "repaired"
                    chosen_payload = repaired
                    chosen_model = model_candidate
                    AGENT2_RESPONSE_TRACE["attempts"].append(attempt_record)
                    break
                if repair_logs:
                    last_error = repair_logs[-1].get("error") or last_error

            AGENT2_RESPONSE_TRACE["attempts"].append(attempt_record)
        if chosen_payload is not None:
            break

    if chosen_payload is not None:
        AGENT2_RESPONSE_TRACE.update({"status": "success", "chosen_model": chosen_model})
        try:
            debug_payload = {
                "ts": datetime.datetime.utcnow().isoformat(),
                "chosen_model": chosen_model,
                "attempts": AGENT2_RESPONSE_TRACE["attempts"],
                "schema_key": schema_key,
            }
            (OUTPUT_DIR / 'gemini_debug.json').write_text(
                json.dumps(debug_payload, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
        except Exception:
            pass
        (OUTPUT_DIR / 'agent2_result.json').write_text(
            json.dumps(chosen_payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        return chosen_payload

    AGENT2_RESPONSE_TRACE.update({
        "status": "fallback",
        "last_error": last_error,
    })
    fallback = _structured_fallback_cards(agent1_json, question_type)
    AGENT2_RESPONSE_TRACE["fallback_answers"] = len(fallback.get("answers", []))
    try:
        if schema_validator is not None:
            schema_validator.validate(fallback)
    except Exception:
        pass
    (OUTPUT_DIR / 'agent2_result.json').write_text(
        json.dumps(fallback, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return fallback


# ---------------------------------------------------------------------------
# Backward-compatibility shims
# ---------------------------------------------------------------------------

def load_actioncard_schema(*args, **kwargs):
    """Backward-compatible wrapper for legacy imports."""
    return load_actioncard_schema_current(*args, **kwargs)


def build_agent2_prompt(*args, **kwargs):
    """Backward-compatible wrapper for the overhauled Agent-2 prompt builder."""
    return build_agent2_prompt_overhauled(*args, **kwargs)


def call_gemini_agent2(*args, **kwargs):
    """Backward-compatible wrapper that delegates to the updated Gemini caller."""
    return call_gemini_agent2_overhauled(*args, **kwargs)


def main():
    import argparse
    a1 = None; prompt_text = ''; a2 = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', type=str, default=None)
    parser.add_argument('--model', type=str, default='gemini-2.5-flash')
    args, _ = parser.parse_known_args()
    q = args.question or os.getenv('QUESTION')
    if not q:
        try:
            q = input('질문을 입력하세요: ').strip()
        except Exception:
            q = None
    if not q:
        q = '성동구 {고향***} 기준으로, 재방문율을 4주 안에 높일 실행카드 제시해줘.'
        print('ℹ️ 질문이 없어 기본 예시를 사용합니다:', q)

    try:
        a1 = agent1_pipeline(q, SHINHAN_DIR, EXTERNAL_DIR)
        prompt_text = build_agent2_prompt_overhauled(a1, question_text=q)
        print('\n==== Gemini Prompt Preview (앞부분) ====')
        print(prompt_text[:800] + ('\n... (생략)' if len(prompt_text)>800 else ''))
        a2 = call_gemini_agent2_overhauled(
            prompt_text,
            model_name=args.model,
            agent1_json=a1,
            question_type=infer_question_type(q),
        )
        print('\n==== Agent-2 결과 (앞부분) ====')
        print(json.dumps(a2, ensure_ascii=False, indent=2)[:800] + '\n...')
    except FileNotFoundError as e:
        print('⚠️ 데이터 파일을 찾지 못했습니다:', e)
        print('예) /content/bigcon/shinhan/big_data_set1_f.csv, big_data_set2_f.csv, big_data_set3_f.csv')
        print('   /content/bigcon/external/weather.csv (선택)')
    except Exception as e:
        print('⚠️ 실행 중 오류:', e)

if __name__ == '__main__':
    main()
