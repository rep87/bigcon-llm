# -*- coding: utf-8 -*-
# BIGCON 2-Agent MVP (Colab, Gemini API) — v3 (fits actual 3-dataset structure)
# %pip -q install google-generativeai pandas openpyxl

import os, json, re, random, sys, glob
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path('/content/bigcon')
SHINHAN_DIR = DATA_DIR / 'shinhan'
EXTERNAL_DIR = DATA_DIR / 'external'
OUTPUT_DIR = DATA_DIR / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MONTHS = 8
SEED = 42
random.seed(SEED); np.random.seed(SEED)

def read_csv_smart(path):
    for enc in ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1']:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding='utf-8', errors='replace')

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
    keep = ['ENCODED_MCT','MCT_NM','ADDR_BASE','SIGUNGU','CATEGORY']
    for k in keep:
        if k not in df.columns: df[k] = np.nan
    return df[keep].drop_duplicates('ENCODED_MCT')

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
        'RC_M1_SHC_RSD_UE_CLN_RAT','RC_M1_SHC_WP_UE_CLN_RAT','RC_M1_SHC_FLP_UE_CLN_RAT'
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[keep_cols]

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

def build_panel(shinhan_dir):
    s1 = load_set1(shinhan_dir)
    s2 = load_set2(shinhan_dir)
    s3 = load_set3(shinhan_dir)
    m23 = pd.merge(s2, s3, on=['ENCODED_MCT','TA_YM','_date'], how='outer', suffixes=('_s2',''))
    panel = pd.merge(m23, s1, on='ENCODED_MCT', how='left')
    def nz(x):
        try: return pd.to_numeric(x, errors='coerce').fillna(0.0)
        except: return pd.Series([0.0]*len(x))
    if 'M12_MAL_1020_RAT' in panel.columns and 'M12_FME_1020_RAT' in panel.columns:
        panel['YOUTH_SHARE'] = nz(panel['M12_MAL_1020_RAT']) + nz(panel['M12_FME_1020_RAT'])
    else:
        panel['YOUTH_SHARE'] = np.nan
    panel['REVISIT_RATE'] = pd.to_numeric(panel.get('MCT_UE_CLN_REU_RAT', np.nan), errors='coerce')
    panel['NEW_RATE'] = pd.to_numeric(panel.get('MCT_UE_CLN_NEW_RAT', np.nan), errors='coerce')
    panel.rename(columns={'ENCODED_MCT':'_merchant_id', 'CATEGORY':'_category'}, inplace=True)
    return panel

def parse_question(q):
    q = (q or '').lower()
    age_cond = None
    if '10대' in q or 'teen' in q or re.search(r'\b1[0-9]\b', q):
        age_cond = ('<=', 19)
    if '20대' in q or '20s' in q or 'twenties' in q:
        if ('이하' in q) or ('under' in q) or ('<=' in q):
            age_cond = ('<=', 20)
        else:
            age_cond = ('range', (20,29))
    if '청소년' in q:
        age_cond = ('<=', 19)

    weather = None
    if ('비' in q) or ('우천' in q) or ('rain' in q):
        weather = 'rain'
    elif ('맑' in q) or ('sunny' in q) or ('clear' in q):
        weather = 'clear'
    elif ('눈' in q) or ('snow' in q):
        weather = 'snow'

    months = DEFAULT_MONTHS
    if '이번달' in q or 'this month' in q:
        months = 1
    elif ('한달' in q) or ('1달' in q) or ('month' in q):
        months = 1
    elif '분기' in q or 'quarter' in q:
        months = 3

    industry = None
    if ('카페' in q) or ('커피' in q):
        industry = 'cafe'
    elif ('요식' in q) or ('restaurant' in q) or ('식당' in q):
        industry = 'restaurant'

    return {'age_cond': age_cond, 'weather': weather, 'months': months, 'industry': industry}

def subset_period(panel, months=DEFAULT_MONTHS):
    if panel['_date'].isna().all():
        return panel.iloc[0:0]
    maxd = panel['_date'].max()
    thr = maxd - pd.Timedelta(days=31*months)
    return panel[panel['_date'] >= thr]

def kpi_summary(panel_sub):
    if panel_sub.empty:
        return {}
    latest_idx = panel_sub.groupby('_merchant_id')['_date'].idxmax()
    snap = panel_sub.loc[latest_idx]
    youth_mean = snap['YOUTH_SHARE'].mean(skipna=True)
    revisit_mean = snap['REVISIT_RATE'].mean(skipna=True)
    new_mean = snap['NEW_RATE'].mean(skipna=True)
    age_cols = [c for c in snap.columns if re.match(r'M12_(MAL|FME)_(1020|30|40|50|60)_RAT', c)]
    top_age = None
    if age_cols:
        means = snap[age_cols].mean(numeric_only=True).sort_values(ascending=False)
        if len(means)>0:
            top_age = means.index[0]
    return {
        'youth_share_avg': float(youth_mean) if pd.notna(youth_mean) else None,
        'revisit_rate_avg': float(revisit_mean) if pd.notna(revisit_mean) else None,
        'new_rate_avg': float(new_mean) if pd.notna(new_mean) else None,
        'top_age_segment': top_age,
        'n_merchants': int(snap['_merchant_id'].nunique())
    }

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
    panel = build_panel(shinhan_dir)
    qinfo = parse_question(question)
    sub = subset_period(panel, months=qinfo['months'])

    wxm = None
    try:
        wxm = load_weather_monthly(external_dir)
    except Exception:
        wxm = None

    kpis = kpi_summary(sub)
    wfx  = weather_effect(sub, wxm)

    notes = []
    quality = 'normal'
    if sub.empty:
        notes.append('질문 조건 표본 부족 또는 기간 데이터 없음')
        quality = 'low'
    if wxm is None and qinfo['weather'] is not None:
        notes.append('날씨 데이터 부재: 날씨 관련 효과는 추정하지 못했습니다.')
        quality = 'low'

    out = {
        'context': {'intent': question, 'parsed': qinfo},
        'kpis': kpis,
        'weather_effect': wfx,
        'limits': notes,
        'quality': quality,
        'period': {
            'max_date': str(panel['_date'].max() if '_date' in panel.columns else None),
            'months': qinfo['months']
        },
        'sample': {
            'merchants_covered': int(sub['_merchant_id'].nunique()) if not sub.empty else 0
        }
    }
    out_path = OUTPUT_DIR / 'agent1_output.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('✅ Agent-1 JSON 저장:', out_path)
    return out

def build_agent2_prompt(agent1_json):
    schema = {
      'recommendations': [{
        'title':'string',
        'what':'string',
        'when':'string',
        'where':['string'],
        'how':['string'],
        'copy':['string','string'],
        'kpi':{'target':'string','expected_uplift':'float|null','range':['float|null','float|null']},
        'risks':['string'],
        'checklist':['string'],
        'evidence':['string']
      }]
    }
    guide = f"""당신은 소상공인 컨설턴트입니다. 아래 **데이터(JSON)**만 근거로 사용하여,
반드시 **액션카드 스키마(JSON)**로만 답하세요. 외부 지식/새로운 수치 추정 금지.

[액션카드 스키마(JSON)]
{json.dumps(schema, ensure_ascii=False, indent=2)}

[데이터(JSON)]
{json.dumps(agent1_json, ensure_ascii=False, indent=2)}

[출력 규칙]
- What/When/Where/How/Copy/KPI/risks/checklist/evidence 모두 채우기.
- KPI.expected_uplift와 range는 입력 JSON의 weather_effect나 kpis에서 유도 가능할 때만 사용(없으면 null).
- evidence에는 지표명/값/기간 등 근거를 1~3줄로 요약.
- quality=='low' 또는 limits 존재 시, '데이터 보강 제안' 카드를 맨 마지막에 추가.
"""
    return guide

def call_gemini_agent2(prompt_text, model_name='gemini-2.5-flash'):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        try:
            api_key = input('Enter GEMINI_API_KEY: ').strip()
        except Exception:
            api_key = None
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY가 설정되지 않았습니다.')
    os.environ['GEMINI_API_KEY'] = api_key

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    generation_config = {
        'temperature': 0.4,
        'top_p': 0.9,
        'top_k': 32,
        'max_output_tokens': 1400,
        'response_mime_type': 'application/json',
    }
    model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
    resp = model.generate_content(prompt_text)
    text = ''
    try:
        text = resp.text
    except Exception:
        if hasattr(resp, 'candidates') and resp.candidates:
            parts = resp.candidates[0].content.parts
            if parts:
                text = getattr(parts[0], 'text', '')
    if not text:
        raise RuntimeError('Gemini 응답이 비어 있습니다.')
    try:
        data = json.loads(text)
    except Exception:
        data = {'raw': text}
    outp = OUTPUT_DIR / 'agent2_result.json'
    with open(outp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print('✅ Agent-2 결과 저장:', outp)
    return data

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
        q = '20대 이하 고객 비중이 높은 매장을 대상으로 이번달 마케팅 채널과 메시지를 추천해줘.'
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
