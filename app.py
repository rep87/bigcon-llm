import os, json, traceback, re
from pathlib import Path
import streamlit as st
import pandas as pd

# ===== 페이지 기본 =====
st.set_page_config(page_title="성동구 소상공인 비밀상담사 (MVP)", page_icon="💬", layout="wide")
st.title("성동구 소상공인 비밀상담사 (MVP)")
st.caption("Agent-1: 데이터 집계/요약 → Agent-2: 실행카드(JSON) 생성")


def _mask_name(raw: str) -> str:
    if not raw:
        return "—"
    name = str(raw)
    if "*" in name:
        return name
    trimmed = name.strip()
    if len(trimmed) <= 1:
        return trimmed or "—"
    if len(trimmed) == 2:
        return trimmed[0] + "*"
    return trimmed[:2] + ("*" * max(4, len(trimmed) - 2))


def _extract_merchant_name(agent1_json: dict) -> str:
    context = (agent1_json or {}).get("context", {})
    keys = [
        "merchant_masked_name",
        "masked_name",
        "merchant_name_masked",
        "merchant_name",
        "store_name"
    ]
    for key in keys:
        val = context.get(key)
        if val:
            return _mask_name(val)
    return "—"


def _format_percent(value) -> str:
    if value is None:
        return "—"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    if abs(num) <= 1:
        num *= 100
    return f"{num:.1f}%"


def _humanize_age_segment(code: str) -> str:
    if not code or not isinstance(code, str):
        return None
    parts = code.split("_")
    if len(parts) < 3:
        return code
    gender_code = parts[1]
    age_code = parts[2]
    gender_map = {"MAL": "남성", "FME": "여성"}
    age_map = {
        "1020": "10-20대",
        "30": "30대",
        "40": "40대",
        "50": "50대",
        "60": "60대"
    }
    gender = gender_map.get(gender_code, "")
    age = age_map.get(age_code, age_code)
    label = " ".join([v for v in [gender, age] if v])
    return label or code


def _collect_major_customers(agent1_json: dict) -> str:
    kpis = (agent1_json or {}).get("kpis", {})
    segments = []
    top_seg = kpis.get("top_age_segment")
    human = _humanize_age_segment(top_seg)
    if human:
        segments.append(human)
    youth_share = kpis.get("youth_share_avg")
    if youth_share is not None:
        segments.append(f"청년 {_format_percent(youth_share)}")
    unique = []
    for item in segments:
        if item and item not in unique:
            unique.append(item)
    return " · ".join(unique[:3]) if unique else "—"


def _collect_overview_row(agent1_json: dict) -> pd.DataFrame:
    context = (agent1_json or {}).get("context", {})
    parsed = context.get("parsed", {})
    industry_code = parsed.get("industry") or context.get("industry")
    industry_labels = {
        "cafe": "카페",
        "restaurant": "음식점",
        "retail": "소매"
    }
    industry = industry_labels.get(industry_code, industry_code or "—")
    addr = context.get("address_masked") or context.get("address") or context.get("addr_base")
    if isinstance(addr, (list, tuple)):
        addr = " / ".join([str(v) for v in addr if v])
    address = addr if addr else "—"

    kpis = (agent1_json or {}).get("kpis", {})
    new_rate = kpis.get("new_rate_avg")
    revisit_rate = kpis.get("revisit_rate_avg")
    if new_rate is not None or revisit_rate is not None:
        new_text = _format_percent(new_rate)
        revisit_text = _format_percent(revisit_rate)
        new_revisit = f"신규 {new_text} / 재방문 {revisit_text}"
    else:
        new_revisit = "—"

    customer_type = context.get("customer_mix") or kpis.get("customer_mix") or "—"
    spend_band = kpis.get("avg_ticket_band") or context.get("avg_ticket_band") or "—"

    data = {
        "업종": industry,
        "주소": address,
        "주요 고객층": _collect_major_customers(agent1_json),
        "고객 유형": customer_type if customer_type else "—",
        "신규/재방문": new_revisit,
        "객단가 구간": spend_band if spend_band else "—"
    }
    return pd.DataFrame([data])


def _build_diagnosis(agent1_json: dict) -> str:
    kpis = (agent1_json or {}).get("kpis", {})
    weather = (agent1_json or {}).get("weather_effect", {})
    parts = []
    youth = kpis.get("youth_share_avg")
    if youth is not None:
        parts.append(f"청년 비중 {_format_percent(youth)}")
    new_rate = kpis.get("new_rate_avg")
    revisit_rate = kpis.get("revisit_rate_avg")
    if new_rate is not None and revisit_rate is not None:
        parts.append(f"신규 {_format_percent(new_rate)} · 재방문 {_format_percent(revisit_rate)}")
    elif new_rate is not None:
        parts.append(f"신규 {_format_percent(new_rate)}")
    elif revisit_rate is not None:
        parts.append(f"재방문 {_format_percent(revisit_rate)}")
    effect = weather.get("effect")
    if effect is not None:
        try:
            eff_val = float(effect)
            trend = "우천 시 감소" if eff_val < 0 else "우천 시 증가"
            parts.append(f"월별 비·재방문 상관 {eff_val:.2f} → {trend} 경향")
        except (TypeError, ValueError):
            pass
    if not parts:
        return "—"
    first_sentence = " · ".join(parts[:2]) + "."
    if len(parts) <= 2:
        return first_sentence
    return first_sentence + " " + " ".join(parts[2:]) + "."


def _build_goal_lines(agent1_json: dict) -> tuple[str, list[str]]:
    period = (agent1_json or {}).get("period", {})
    months = period.get("months")
    if months:
        try:
            weeks = int(float(months) * 4)
        except (TypeError, ValueError):
            weeks = None
        if weeks and weeks > 0:
            period_text = f"최근 {months}개월 (약 {weeks}주)"
        else:
            period_text = f"최근 {months}개월"
    else:
        period_text = "기간 정보 —"

    kpis = (agent1_json or {}).get("kpis", {})
    mapping = [
        ("revisit_rate_avg", "재방문율"),
        ("new_rate_avg", "신규 고객 비중"),
        ("youth_share_avg", "청년 고객 비중")
    ]
    lines = []
    for key, label in mapping:
        value = kpis.get(key)
        if value is not None:
            lines.append(f"{label}: 현황 {_format_percent(value)} → 목표 구간 —")
    if not lines:
        lines.append("KPI 목표 구간 —")
    return period_text, lines[:3]


def _format_list(values) -> str:
    if not values:
        return "—"
    if isinstance(values, (list, tuple)):
        items = [str(v) for v in values if v]
        return " · ".join(items) if items else "—"
    return str(values)


def _format_kpi(kpi_obj: dict) -> str:
    if not isinstance(kpi_obj, dict):
        return "—"
    target = kpi_obj.get("target") or "—"
    uplift = kpi_obj.get("expected_uplift")
    rng = kpi_obj.get("range")
    parts = [f"타깃: {target}"]
    if uplift is not None:
        parts.append(f"기대 상승 {uplift}")
    if isinstance(rng, (list, tuple)) and len(rng) == 2 and any(r is not None for r in rng):
        parts.append(f"목표 구간 {rng[0]}~{rng[1]}")
    else:
        parts.append("목표 구간 —")
    return " · ".join(parts)


def _split_cards(cards: list[dict]) -> tuple[list[dict], list[dict]]:
    if not cards:
        return [], []
    main, booster = [], []
    for card in cards:
        title = str(card.get("title", ""))
        if "보강" in title or "데이터" in title:
            booster.append(card)
        else:
            main.append(card)
    return main, booster


def render_summary_view(agent1_json: dict, agent2_json: dict) -> None:
    merchant_title = _extract_merchant_name(agent1_json)
    st.header(f"📊 {merchant_title} 가맹점 방문 고객 현황 분석")

    overview_df = _collect_overview_row(agent1_json)
    st.subheader("현황 표")
    st.table(overview_df)

    st.subheader("한 줄 진단")
    st.markdown(f"- {_build_diagnosis(agent1_json)}")

    period_text, goal_lines = _build_goal_lines(agent1_json)
    st.subheader("목표")
    st.markdown(f"- 기간 가정: {period_text}")
    for line in goal_lines:
        st.markdown(f"- {line}")

    cards = (agent2_json or {}).get("recommendations", [])
    main_cards, booster_cards = _split_cards(cards)
    display_cards = main_cards[:2]
    if booster_cards:
        display_cards.extend(booster_cards[:1])

    st.subheader("실행 카드")
    if not display_cards:
        st.info("실행 카드가 제공되지 않았습니다.")
    for card in display_cards:
        with st.container():
            st.markdown(f"**{card.get('title', '—')}**")
            st.markdown(f"- 타겟: {card.get('what', '—')}")
            st.markdown(f"- 채널: {_format_list(card.get('where'))}")
            st.markdown(f"- 방법: {_format_list(card.get('how'))}")
            st.markdown(f"- 카피: {_format_list(card.get('copy'))}")
            st.markdown(f"- KPI: {_format_kpi(card.get('kpi'))}")
            st.markdown(f"- 리스크/완화: {_format_list(card.get('risks'))}")
            st.markdown(f"- 근거: {_format_list(card.get('evidence'))}")

    limits = (agent1_json or {}).get("limits", [])
    st.subheader("한계/데이터 보강")
    st.markdown("**현재 한계**")
    if limits:
        for item in limits[:5]:
            st.markdown(f"- {item}")
    else:
        st.markdown("- 한계 정보가 제공되지 않았습니다.")

    improvement_suggestions = []
    for item in limits:
        text = str(item)
        if "날씨" in text:
            improvement_suggestions.append("날씨 데이터 연계를 통해 우천 가설을 검증합니다.")
        elif "표본" in text or "데이터" in text:
            improvement_suggestions.append("누락 구간을 점검해 표본을 보강합니다.")
    if not improvement_suggestions:
        improvement_suggestions.append("다음 스프린트에서 결합 데이터 소스를 재점검합니다.")

    st.markdown("**다음 스프린트 보강 계획**")
    for suggestion in improvement_suggestions[:3]:
        st.markdown(f"- {suggestion}")

# ===== 경로 & 키 =====
DATA_DIR = Path("data")
SHINHAN_DIR = DATA_DIR / "shinhan"
EXTERNAL_DIR = DATA_DIR / "external"

API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
if not API_KEY:
    st.warning("GEMINI_API_KEY가 설정되지 않았습니다. (앱 설정의 App secrets에 등록하세요)")

# ===== 사이드바: 데이터 상태 =====
st.sidebar.header("데이터 상태")
st.sidebar.write(f"📁 SHINHAN_DIR 존재: {SHINHAN_DIR.exists()}")
st.sidebar.write(f"📁 EXTERNAL_DIR 존재: {EXTERNAL_DIR.exists()}")

# ===== 질문 입력 =====
default_q = "비 오는 날 20대 이하 카페 홍보 전략 제시해줘"
question = st.text_input("질문을 입력하세요", value=default_q)

# ===== 실행 버튼 =====
if st.button("분석 실행", type="primary"):
    # 지연 로딩 임포트 (배포 런타임 문제 회피)
    from bigcon_2agent_mvp_v3 import agent1_pipeline, build_agent2_prompt, call_gemini_agent2

    # Agent-1
    with st.spinner("Agent-1: 데이터 집계/요약 중..."):
        try:
            a1 = agent1_pipeline(question, SHINHAN_DIR, EXTERNAL_DIR)
            st.success("Agent-1 JSON 생성 완료")
            with st.expander("🔎 Agent-1 출력(JSON) 보기", expanded=False):
                st.json(a1)
        except Exception:
            st.error("Agent-1 실행 오류")
            st.code(traceback.format_exc())
            st.stop()

    # Agent-2
    with st.spinner("Agent-2: 카드 생성 중..."):
        try:
            os.environ["GEMINI_API_KEY"] = API_KEY  # 내부 함수가 env 읽도록 주입
            prompt_text = build_agent2_prompt(a1)
            result = call_gemini_agent2(prompt_text)
            st.success("Agent-2 카드 생성 완료")
        except Exception:
            st.error("Agent-2 실행 오류")
            st.code(traceback.format_exc())
            st.stop()

    # 출력
    try:
        render_summary_view(a1, result)
    except Exception:
        st.error("요약 뷰를 렌더링하는 중 오류가 발생했습니다.")
        st.code(traceback.format_exc())

    with st.expander("🧾 Agent-2 출력(JSON) 보기", expanded=False):
        st.json(result)

# 최초 안내
if not st.session_state.get("_intro_shown"):
    st.info("✅ 업로드 성공! 이제 질문 입력 후 [분석 실행]을 눌러 카드 결과를 확인해보세요.")
    st.session_state["_intro_shown"] = True
