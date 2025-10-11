# -*- coding: utf-8 -*-
# BIGCON 2-Agent MVP (Colab, Gemini API) — v3 (fits actual 3-dataset structure)
# %pip -q install google-generativeai pandas openpyxl

import os, json, re, random, sys, glob, datetime
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd
import numpy as np
from jsonschema import Draft7Validator, ValidationError

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

def load_actioncard_schema():
    global _SCHEMA_CACHE, _SCHEMA_VALIDATOR
    if _SCHEMA_CACHE is not None and _SCHEMA_VALIDATOR is not None:
        return _SCHEMA_CACHE, _SCHEMA_VALIDATOR
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"스키마 파일을 찾을 수 없습니다: {SCHEMA_PATH}")
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        _SCHEMA_CACHE = json.load(f)
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
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print('⚠️ GEMINI_API_KEY 미설정으로 LLM 보조 매칭을 건너뜁니다.')
        return None
    try:
        import google.generativeai as genai
    except ImportError:
        print('⚠️ google-generativeai 미설치로 LLM 보조 매칭을 건너뜁니다.')
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name='models/gemini-2.5-flash',
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

    try:
        response = model.generate_content(prompt)
    except Exception as exc:
        print('⚠️ LLM 보조 매칭 호출 실패:', exc)
        return None

    def _response_text(resp):
        parts = []
        for part in getattr(resp, 'parts', []) or []:
            text = getattr(part, 'text', None)
            if text:
                parts.append(text)
        if hasattr(resp, 'text') and resp.text:
            parts.append(resp.text)
        return '\n'.join(parts)

    text = _response_text(response)
    if not text:
        print('⚠️ LLM 보조 매칭 응답이 비었습니다.')
        return None

    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        print('⚠️ LLM 보조 매칭에서 JSON을 찾지 못했습니다.')
        return None

    try:
        data = json.loads(match.group(0))
    except Exception as exc:
        print('⚠️ LLM 보조 매칭 JSON 파싱 실패:', exc)
        return None

    return data if isinstance(data, dict) else None


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
        llm_result = call_llm_for_mask(original_question, masked_name, sigungu)
        debug_info['notes'] = 'llm_invoked'
        if llm_result:
            debug_info['llm'] = llm_result
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
                    nested_debug.setdefault('llm', llm_result)
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
    brace_match = re.search(r'\{([^{}]+)\}', normalized)
    if brace_match:
        merchant_mask = brace_match.group(1).strip()

    mask_prefix = None
    if merchant_mask:
        mask_prefix = merchant_mask.split('*', 1)[0].strip()

    sigungu_match = re.search(r'(?P<sigungu>[가-힣]{2,}구)', normalized)
    merchant_sigungu = sigungu_match.group('sigungu') if sigungu_match else '성동구'

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
    for code, label in age_labels.items():
        cols = [f'M12_MAL_{code}_RAT', f'M12_FME_{code}_RAT']
        vals = pd.to_numeric(pd.Series([detail_row.get(c) for c in cols]), errors='coerce')
        total = vals.sum(skipna=True)
        if pd.notna(total):
            cleaned = _clean_pct(total)
            if cleaned is not None:
                age_distribution.append({'code': code, 'label': label, 'value': cleaned})
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
    merchants_df = load_set1(shinhan_dir)
    qinfo = parse_question(question)

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

    print(
        "🧪 parse_debug:",
        json.dumps(
            {
                'original': qinfo.get('original_question'),
                'normalized': qinfo.get('normalized_question'),
                'merchant_mask': qinfo.get('merchant_masked_name'),
                'mask_prefix': qinfo.get('merchant_mask_prefix'),
                'sigungu': qinfo.get('merchant_sigungu'),
            },
            ensure_ascii=False,
        ),
    )

    merchant_match = None
    resolve_meta = {
        'candidates': [],
        'path': None,
        'notes': None,
        'suggestions': None,
        'llm': None,
    }
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
            resolve_meta['path'] = 'explicit_id'

    if merchant_match is None:
        merchant_match, resolve_meta = resolve_merchant(
            qinfo.get('merchant_masked_name'),
            qinfo.get('merchant_mask_prefix'),
            qinfo.get('merchant_sigungu'),
            merchants_df,
            original_question=qinfo.get('normalized_question') or question,
            allow_llm=True,
        )

    target_id = None
    if merchant_match and merchant_match.get('encoded_mct') is not None:
        target_id = str(merchant_match['encoded_mct'])
        merchant_match['encoded_mct'] = target_id

    panel, panel_stats = build_panel(shinhan_dir, merchants_df=merchants_df, target_id=target_id)
    panel_focus = panel
    print(
        "📦 panel_filter:",
        json.dumps({
            'target_id': target_id,
            **panel_stats,
        }, ensure_ascii=False),
    )
    print("🏁 merchant_match:", json.dumps(merchant_match, ensure_ascii=False))
    print(
        "🧭 resolve_summary:",
        json.dumps(
            {
                'path': resolve_meta.get('path'),
                'notes': resolve_meta.get('notes'),
                'candidates': resolve_meta.get('candidates'),
            },
            ensure_ascii=False,
            default=str,
        ),
    )

    sub = subset_period(panel_focus, months=qinfo['months'])

    wxm = None
    try:
        wxm = load_weather_monthly(external_dir)
    except Exception:
        wxm = None

    kpis, kpi_debug = kpi_summary(sub)
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
            'panel_stats': panel_stats,
            'merchant_candidates': resolve_meta.get('candidates'),
            'merchant_resolution_path': resolve_meta.get('path'),
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
        'debug': {
            'parsed': parse_log,
            'resolved_merchant_id': target_id,
            'resolve_path': resolve_meta.get('path'),
            'resolve_candidates': resolve_meta.get('candidates'),
            'resolve_notes': resolve_meta.get('notes'),
            'resolve_suggestions': resolve_meta.get('suggestions'),
            'llm_result': resolve_meta.get('llm'),
            'latest_raw_snapshot': kpi_debug.get('latest_raw_snapshot'),
            'sanitized_snapshot': kpi_debug.get('sanitized_snapshot'),
            'panel_stats': panel_stats,
            'merchants_covered': merchants_covered,
        },
    }

    if merchant_match:
        out['context']['merchant'] = merchant_match
        out['context']['merchant_masked_name'] = merchant_match.get('masked_name')

    out_path = OUTPUT_DIR / 'agent1_output.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('✅ Agent-1 JSON 저장:', out_path)
    return out



def build_agent2_prompt(agent1_json):
    try:
        schema, _ = load_actioncard_schema()
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    except Exception as e:
        schema_text = json.dumps({"schema_error": str(e)}, ensure_ascii=False, indent=2)

    rules = [
        "Agent-1 JSON만 근거로 활용하고 외부 추정은 금지합니다.",
        "모든 카드에 타겟 → 채널 → 방법 → 카피(2개 이상) → KPI → 리스크/완화 → 근거를 채웁니다.",
        "근거 문장은 반드시 숫자+컬럼명+기간 형식이며 정보가 없으면 null 또는 '—'로 둡니다.",
        "품질이 낮거나 데이터가 부족하면 마지막 카드에 '데이터 보강 제안'을 추가합니다.",
        "상호명은 항상 마스킹된 형태로 유지합니다.",
        "KPI.expected_uplift와 range는 근거가 있을 때만 값을 넣고, 없으면 null을 유지합니다."
    ]

    rules_text = "- " + "\n- ".join(rules)

    guide = f"""당신은 한국어 소상공인 컨설턴트입니다. 아래 Agent-1 JSON만 근거로 사용하여,
반드시 액션카드 스키마(JSON)로만 답하세요. 불확실한 수치는 null 또는 '—'로 남겨두세요.

[출력 규칙]
{rules_text}

[액션카드 스키마(JSON)]
{schema_text}

[데이터(JSON)]
{json.dumps(agent1_json, ensure_ascii=False, indent=2)}
"""
    return guide

def call_gemini_agent2(prompt_text, model_name='models/gemini-2.5-flash'):
    """
    Gemini 2.5 Flash 전용 호출:
    - google-generativeai==0.8.3 기준
    - 2.5 flash 가용성 자동 확인(list_models) 후 2.5 flash 계열만 시도
    - response_mime_type 강제 해제(빈 응답 회피), 텍스트→JSON 추출
    - 세이프티/빈응답/가용성 이슈 시 폴백 카드 반환 (앱은 계속 동작)
    - 디버그 로그: outputs/gemini_debug.json
    """
    import google.generativeai as genai
    import re, json, datetime

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY가 설정되지 않았습니다.')
    genai.configure(api_key=api_key)

    # 1) 계정에서 실제 가능한 2.5-flash 계열 모델만 수집
    candidates = []
    try:
        avail = []
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                avail.append(m.name)  # 예: 'models/gemini-2.5-flash'
        # 2.5 flash 계열만 필터
        for name in avail:
            tail = name.split("/")[-1]
            if "2.5" in tail and "flash" in tail:
                candidates.append(name)
    except Exception:
        pass

    # 가용성 조회 실패/비어있으면 합리적 후보(2.5 flash 계열)만 시도
    if not candidates:
        candidates = [
            "models/gemini-2.5-flash",
            "models/gemini-2.5-flash-latest",
            "models/gemini-2.5-flash-001",
            "gemini-2.5-flash",
            "gemini-2.5-flash-latest",
            "gemini-2.5-flash-001",
        ]

    # 사용자가 인자를 줬다면 맨 앞에 둠(역시 2.5 flash 계열이어야 함)
    if model_name and model_name not in candidates:
        candidates.insert(0, model_name)

    # 세이프티(완화) + 토큰 보수적
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 32,
        "max_output_tokens": 900,
        # "response_mime_type": "application/json",  # 강제하지 않음(빈 응답 회피)
    }

    def _extract_text(resp):
        text = ""
        try:
            t = getattr(resp, "text", None)
            if t: text = t
        except Exception:
            pass
        if (not text) and getattr(resp, "candidates", None):
            try:
                cand0 = resp.candidates[0]
                if cand0 and getattr(cand0, "content", None) and cand0.content.parts:
                    for p in cand0.content.parts:
                        pt = getattr(p, "text", "")
                        if pt: text += pt
            except Exception:
                pass
        return (text or "").strip()

    def _blocked_info(resp):
        info = {}
        try: info["prompt_feedback"] = getattr(resp, "prompt_feedback", None)
        except Exception: pass
        try:
            if getattr(resp, "candidates", None):
                info["candidate_safety"] = getattr(resp.candidates[0], "safety_ratings", None)
                info["finish_reason"] = getattr(resp.candidates[0], "finish_reason", None)
        except Exception: pass
        return info

    def _run_once(name: str):
        model = genai.GenerativeModel(
            model_name=name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        resp = model.generate_content(prompt_text)
        text = _extract_text(resp)
        info = _blocked_info(resp)
        return text, info, name

    try:
        _, schema_validator = load_actioncard_schema()
        schema_error = None
    except Exception as e:
        schema_validator = None
        schema_error = str(e)

    all_attempt_logs = []
    last_error = schema_error

    for attempt in range(2):
        attempt_logs = []
        for mname in candidates:
            try:
                text, info, used = _run_once(mname)
                log_entry = {"model": used, "has_text": bool(text), "meta": info}
                if not text:
                    log_entry["error"] = "empty_text"
                    attempt_logs.append(log_entry)
                    last_error = "LLM 응답이 비어 있습니다."
                    continue

                core = None
                m = re.search(r"\{[\s\S]*\}", text)
                if m:
                    try:
                        core = json.loads(m.group(0))
                    except Exception as je:
                        log_entry["error"] = f"json_parse_error: {je}"
                        attempt_logs.append(log_entry)
                        last_error = f"JSON 파싱 실패: {je}"
                        continue
                else:
                    log_entry["error"] = "json_not_found"
                    attempt_logs.append(log_entry)
                    last_error = "응답에서 JSON 블록을 찾지 못했습니다."
                    continue

                if not isinstance(core, dict):
                    log_entry["error"] = "json_not_object"
                    attempt_logs.append(log_entry)
                    last_error = "추출된 JSON이 객체 형식이 아닙니다."
                    continue

                validation_ok = True
                if schema_validator is not None:
                    try:
                        schema_validator.validate(core)
                    except ValidationError as ve:
                        validation_ok = False
                        log_entry["validation_error"] = ve.message
                        last_error = f"스키마 검증 실패: {ve.message}"

                if validation_ok:
                    attempt_logs.append(log_entry)
                    all_attempt_logs.append({"attempt": attempt + 1, "logs": attempt_logs})
                    # 디버그 기록
                    try:
                        debug = {
                            "ts": datetime.datetime.utcnow().isoformat(),
                            "chosen_model": used,
                            "attempts": all_attempt_logs,
                            "preview_text": text[:400],
                            "schema_error": schema_error
                        }
                        with open(OUTPUT_DIR / "gemini_debug.json", "w", encoding="utf-8") as f:
                            json.dump(debug, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

                    outp = OUTPUT_DIR / 'agent2_result.json'
                    with open(outp, 'w', encoding='utf-8') as f:
                        json.dump(core, f, ensure_ascii=False, indent=2)
                    print('✅ Agent-2 결과 저장:', outp)
                    return core

                attempt_logs.append(log_entry)
            except Exception as e:
                attempt_logs.append({"model": mname, "error": str(e)})
                last_error = str(e)

        all_attempt_logs.append({"attempt": attempt + 1, "logs": attempt_logs})
        if schema_validator is None:
            break

    # 모두 실패 → 폴백(앱 다운 방지)
    fallback = {
        "recommendations": [{
            "title": "데이터 보강 제안",
            "what": "모델 가용성/세이프티/쿼터로 카드 생성을 보류합니다.",
            "when": "환경 확인 후 재시도",
            "where": ["대시보드"],
            "how": ["2.5 flash 가용성 확인", "API 키/쿼터 확인", "프롬프트 길이 축소"],
            "copy": ["데이터를 조금만 더 주세요!"],
            "kpi": {"target": "revisit_rate", "expected_uplift": None, "range": [None, None]},
            "risks": ["LLM 안전 필터/쿼터/모델 가용성"],
            "checklist": ["App secrets 확인", "list_models() 결과에서 2.5 flash 검색"],
            "evidence": [
                "gemini_debug.json 로그 참조",
                f"사유: {last_error or '원인 미상'}"
            ]
        }]
    }
    try:
        debug = {
            "ts": datetime.datetime.utcnow().isoformat(),
            "attempts": all_attempt_logs,
            "schema_error": schema_error,
            "last_error": last_error
        }
        with open(OUTPUT_DIR / "gemini_debug.json", "w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    outp = OUTPUT_DIR / 'agent2_result.json'
    with open(outp, 'w', encoding='utf-8') as f:
        json.dump(fallback, f, ensure_ascii=False, indent=2)
    print('⚠️ Agent-2: 2.5 flash 가용성/세이프티 문제 → 폴백 카드 반환')
    return fallback

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
        prompt_text = build_agent2_prompt(a1)
        print('\n==== Gemini Prompt Preview (앞부분) ====')
        print(prompt_text[:800] + ('\n... (생략)' if len(prompt_text)>800 else ''))
        a2 = call_gemini_agent2(prompt_text, model_name=args.model)
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
