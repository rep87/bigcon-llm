import os, json, traceback, re
from pathlib import Path
import streamlit as st
import pandas as pd

# ===== 페이지 기본 =====
st.set_page_config(page_title="성동구 소상공인 비밀상담사 (MVP)", page_icon="💬", layout="wide")
st.title("성동구 소상공인 비밀상담사 (MVP)")
st.caption("Agent-1: 데이터 집계/요약 → Agent-2: 실행카드(JSON) 생성")
show_debug = st.checkbox("🔍 디버그 보기", value=True)


def _env_flag(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


DEBUG_SHOW_RAW = _env_flag("DEBUG_SHOW_RAW", "true").lower() in {"1", "true", "yes"}


def _get_debug_section(agent1_json: dict | None) -> dict:
    debug = (agent1_json or {}).get("debug")
    return debug if isinstance(debug, dict) else {}


def _get_debug_snapshot(agent1_json: dict | None) -> dict:
    debug = _get_debug_section(agent1_json)
    snap = debug.get("snapshot")
    if isinstance(snap, dict):
        sanitized = snap.get("sanitized")
        if isinstance(sanitized, dict):
            return sanitized
    legacy = debug.get("sanitized_snapshot")
    return legacy if isinstance(legacy, dict) else {}


def _get_debug_raw_snapshot(agent1_json: dict | None) -> dict:
    debug = _get_debug_section(agent1_json)
    snap = debug.get("snapshot")
    if isinstance(snap, dict):
        raw = snap.get("raw")
        if isinstance(raw, dict):
            return raw
    legacy = debug.get("latest_raw_snapshot")
    return legacy if isinstance(legacy, dict) else {}


def render_debug_view(agent1_json: dict | None, show_raw: bool = DEBUG_SHOW_RAW) -> None:
    debug = _get_debug_section(agent1_json)
    if not debug:
        st.info("디버그 정보가 없습니다.")
        return

    def _flatten_rows(obj: dict) -> pd.DataFrame:
        rows = []
        for key, value in (obj or {}).items():
            if isinstance(value, (dict, list)):
                try:
                    text = json.dumps(value, ensure_ascii=False)
                except TypeError:
                    text = str(value)
                rows.append({"항목": key, "값": text})
            else:
                rows.append({"항목": key, "값": value})
        return pd.DataFrame(rows)

    def _numeric_values(data):
        if isinstance(data, dict):
            for val in data.values():
                yield from _numeric_values(val)
        elif isinstance(data, (list, tuple)):
            for item in data:
                yield from _numeric_values(item)
        elif isinstance(data, (int, float)):
            yield data

    errors = debug.get("errors", [])
    resolve_info = debug.get("resolve", {}) or {}
    panel_info = debug.get("panel", {}) or {}
    snapshot_info = (debug.get("snapshot", {}) or {})
    sanitized_snapshot = snapshot_info.get("sanitized") or {}
    agent1_llm = debug.get("agent1_llm", {}) or {}

    warnings = []
    if resolve_info.get("resolved_merchant_id") is None:
        warnings.append("가맹점 미확정: 전표본 요약으로 떨어질 위험")
    rows_after = panel_info.get("rows_after")
    if rows_after is not None and rows_after != 1:
        warnings.append("단일 상점 패널 아님")
    for num in _numeric_values(sanitized_snapshot):
        try:
            val = float(num)
        except (TypeError, ValueError):
            continue
        if val < 0 or val > 100:
            warnings.append("정규화 실패 의심")
            break
    if agent1_llm.get("safety_blocked"):
        warnings.append("LLM 안전성 차단")

    for err in errors:
        stage = err.get('stage', 'unknown')
        msg = err.get('msg', '')
        st.error(f"[{stage}] {msg}")
    for warn in warnings:
        st.error(warn)

    input_info = debug.get("input", {}) or {}
    st.markdown("#### 입력/플래그")
    st.write("원문:", input_info.get("original") or "—")
    flags = input_info.get("flags") or {}
    if flags:
        st.write({k: flags.get(k) for k in sorted(flags)})

    parse_info = debug.get("parse", {}) or {}
    st.markdown("#### 파싱 결과")
    st.write({
        "merchant_mask": parse_info.get("merchant_mask"),
        "mask_prefix": parse_info.get("mask_prefix"),
        "sigungu": parse_info.get("sigungu"),
        "pattern_used": parse_info.get("pattern_used"),
        "elapsed_ms": parse_info.get("elapsed_ms"),
    })

    st.markdown("#### 가맹점 매칭")
    st.write({
        "path": resolve_info.get("path"),
        "resolved_merchant_id": resolve_info.get("resolved_merchant_id"),
        "elapsed_ms": resolve_info.get("elapsed_ms"),
    })
    candidates = resolve_info.get("candidates_top3") or []
    if candidates:
        st.table(pd.DataFrame(candidates))
    else:
        st.write("후보 없음")

    st.markdown("#### 패널 필터")
    st.write({
        "rows_before": panel_info.get("rows_before"),
        "rows_after": panel_info.get("rows_after"),
        "latest_ta_ym": panel_info.get("latest_ta_ym"),
        "elapsed_ms": panel_info.get("elapsed_ms"),
    })

    st.markdown("#### 스냅샷")
    if show_raw:
        raw_df = _flatten_rows(snapshot_info.get("raw") or {})
        if not raw_df.empty:
            st.caption("원본(raw)")
            st.table(raw_df)
    sanitized_df = _flatten_rows(sanitized_snapshot)
    if not sanitized_df.empty:
        st.caption("정규화(sanitized)")
        st.table(sanitized_df)

    render_info = debug.get("render", {}) or {}
    table_dict = render_info.get("table_dict")
    if isinstance(table_dict, dict) and table_dict:
        st.markdown("#### 렌더 테이블")
        st.table(pd.DataFrame([table_dict]))

    st.markdown("#### Agent-1 LLM")
    st.write({
        "used": agent1_llm.get("used"),
        "model": agent1_llm.get("model"),
        "resp_bytes": agent1_llm.get("resp_bytes"),
        "safety_blocked": agent1_llm.get("safety_blocked"),
        "elapsed_ms": agent1_llm.get("elapsed_ms"),
    })
    preview = agent1_llm.get("prompt_preview")
    if preview:
        st.caption("프롬프트 프리뷰")
        st.code(preview)


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
    if num < 0 or num > 100:
        return "—"
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
    debug_snapshot = _get_debug_snapshot(agent1_json)
    kpis = (agent1_json or {}).get("kpis", {})
    segments = []
    age_segments = debug_snapshot.get("age_top_segments") or kpis.get("age_top_segments") or []
    for item in age_segments[:3]:
        label = item.get("label") if isinstance(item, dict) else None
        value = item.get("value") if isinstance(item, dict) else None
        if label and value is not None:
            segments.append(f"{label} {_format_percent(value)}")
    if not segments:
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
    return ", ".join(unique[:3]) if unique else "—"


def _format_customer_mix(detail: dict | None) -> str:
    if not detail or not isinstance(detail, dict):
        return "—"
    ordered_labels = ["유동", "거주", "직장"]
    parts = []
    for label in ordered_labels:
        value = detail.get(label)
        if value is not None:
            percent = _format_percent(value)
            if percent != "—":
                parts.append(f"{label} {percent}")
    for label, value in detail.items():
        if label in ordered_labels:
            continue
        if value is None:
            continue
        percent = _format_percent(value)
        if percent != "—":
            parts.append(f"{label} {percent}")
    return ", ".join(parts[:3]) if parts else "—"


def _collect_overview_row(agent1_json: dict) -> tuple[pd.DataFrame, dict]:
    context = (agent1_json or {}).get("context", {})
    parsed = context.get("parsed", {})
    merchant = context.get("merchant", {})
    industry_candidate = merchant.get("category") or parsed.get("merchant_industry_label") or parsed.get("industry")
    industry_labels = {
        "cafe": "카페",
        "restaurant": "음식점",
        "retail": "소매"
    }
    industry = industry_labels.get(industry_candidate, industry_candidate or "—")
    addr = merchant.get("address") or context.get("address_masked") or context.get("address") or context.get("addr_base")
    if isinstance(addr, (list, tuple)):
        addr = " / ".join([str(v) for v in addr if v])
    address = addr if addr else "—"

    debug_snapshot = _get_debug_snapshot(agent1_json)
    kpis = (agent1_json or {}).get("kpis", {})
    new_rate = debug_snapshot.get("new_pct", kpis.get("new_rate_avg"))
    revisit_rate = debug_snapshot.get("revisit_pct", kpis.get("revisit_rate_avg"))
    new_text = _format_percent(new_rate)
    revisit_text = _format_percent(revisit_rate)
    if new_text == "—" and revisit_text == "—":
        new_revisit = "—"
    else:
        new_revisit = f"신규 {new_text} / 재방문 {revisit_text}"

    customer_mix_detail = debug_snapshot.get("customer_mix_detail") or kpis.get("customer_mix_detail")
    customer_type = _format_customer_mix(customer_mix_detail)
    spend_band = (
        debug_snapshot.get("avg_ticket_band_label")
        or kpis.get("avg_ticket_band_label")
        or context.get("avg_ticket_band")
        or "—"
    )
    if isinstance(spend_band, str):
        spend_band = spend_band.strip()
        spend_band = re.sub(r"(상위)(\d)", r"\1 \2", spend_band)
    elif spend_band is None:
        spend_band = "—"

    data = {
        "업종": industry,
        "주소": address,
        "주요 고객층": _collect_major_customers(agent1_json),
        "고객 유형": customer_type if customer_type else "—",
        "신규/재방문": new_revisit,
        "객단가 구간": spend_band if spend_band else "—"
    }
    return pd.DataFrame([data]), data


def _build_diagnosis(agent1_json: dict) -> str:
    debug_snapshot = _get_debug_snapshot(agent1_json)
    kpis = (agent1_json or {}).get("kpis", {})
    sentences = []

    mix_detail = debug_snapshot.get("customer_mix_detail") or kpis.get("customer_mix_detail")
    mix_items = []
    if isinstance(mix_detail, dict):
        sorted_mix = sorted(
            [(label, val) for label, val in mix_detail.items() if val is not None],
            key=lambda x: x[1],
            reverse=True,
        )
        for label, value in sorted_mix[:2]:
            percent = _format_percent(value)
            if percent != "—":
                mix_items.append(f"{label} {percent}")
    if mix_items:
        sentences.append(" · ".join(mix_items) + " 고객 구성입니다.")

    rate_parts = []
    new_text = _format_percent(debug_snapshot.get("new_pct", kpis.get("new_rate_avg")))
    revisit_text = _format_percent(debug_snapshot.get("revisit_pct", kpis.get("revisit_rate_avg")))
    youth_text = _format_percent(debug_snapshot.get("youth_pct", kpis.get("youth_share_avg")))
    if new_text != "—":
        rate_parts.append(f"신규 {new_text}")
    if revisit_text != "—":
        rate_parts.append(f"재방문 {revisit_text}")
    if youth_text != "—":
        rate_parts.append(f"청년 고객 {youth_text}")
    if rate_parts:
        sentences.append(" · ".join(rate_parts) + "입니다.")

    if not sentences:
        return "—"
    return " ".join(sentences[:2])


def _build_goal_lines(agent1_json: dict) -> tuple[str, list[str]]:
    period = (agent1_json or {}).get("period", {})
    months = period.get("months")
    weeks_requested = period.get("weeks_requested")
    if weeks_requested:
        try:
            weeks_val = int(weeks_requested)
        except (TypeError, ValueError):
            weeks_val = None
        if weeks_val and months:
            period_text = f"향후 {weeks_val}주 (약 {months}개월)"
        elif weeks_val:
            period_text = f"향후 {weeks_val}주"
        else:
            period_text = "기간 정보 —"
    elif months:
        period_text = f"최근 {months}개월"
    else:
        period_text = "기간 정보 —"

    debug_snapshot = _get_debug_snapshot(agent1_json)
    kpis = (agent1_json or {}).get("kpis", {})
    mapping = [
        ("revisit_rate_avg", "재방문율"),
        ("new_rate_avg", "신규 고객 비중"),
        ("youth_share_avg", "청년 고객 비중")
    ]
    lines = []
    for key, label in mapping:
        sanitized_key = {
            "revisit_rate_avg": "revisit_pct",
            "new_rate_avg": "new_pct",
            "youth_share_avg": "youth_pct",
        }.get(key)
        value = debug_snapshot.get(sanitized_key, kpis.get(key))
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


def render_summary_view(
    agent1_json: dict,
    agent2_json: dict,
    overview_df: pd.DataFrame | None = None,
    table_dict: dict | None = None,
) -> None:
    merchant_title = _extract_merchant_name(agent1_json)
    st.header(f"📊 {merchant_title} 가맹점 방문 고객 현황 분석")

    context = (agent1_json or {}).get("context", {})
    if context and not context.get("merchant"):
        st.warning("질문과 정확히 일치하는 가맹점을 찾지 못해 표본 전체 요약을 보여드립니다.")

    debug_info = _get_debug_section(agent1_json)
    if overview_df is None or table_dict is None:
        render_info = debug_info.get("render") if isinstance(debug_info, dict) else None
        if isinstance(render_info, dict) and isinstance(render_info.get("table_dict"), dict):
            table_dict = render_info.get("table_dict")
            overview_df = pd.DataFrame([table_dict])
        else:
            overview_df, table_dict = _collect_overview_row(agent1_json)
            if isinstance(debug_info, dict):
                debug_info.setdefault("render", {})["table_dict"] = table_dict
    if overview_df is None:
        if table_dict:
            overview_df = pd.DataFrame([table_dict])
        else:
            overview_df = pd.DataFrame()
    try:
        print("📊 overview_table:", json.dumps(overview_df.to_dict(orient="records"), ensure_ascii=False))
    except Exception:
        pass
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
default_q = "성동구 {고향***} 기준으로, 재방문율 4주 플랜 작성해줘."
question = st.text_input("질문을 입력하세요", value=default_q)
st.caption("상호는 반드시 {}로 감싸 주세요. 예) 성동구 {동대******}")

# ===== 실행 버튼 =====
if st.button("분석 실행", type="primary"):
    # 지연 로딩 임포트 (배포 런타임 문제 회피)
    from bigcon_2agent_mvp_v3 import agent1_pipeline, build_agent2_prompt, call_gemini_agent2

    # Agent-1
    with st.spinner("Agent-1: 데이터 집계/요약 중..."):
        try:
            a1 = agent1_pipeline(question, SHINHAN_DIR, EXTERNAL_DIR)
            try:
                overview_df, table_dict = _collect_overview_row(a1)
            except Exception:
                overview_df, table_dict = pd.DataFrame(), {}
            if isinstance(a1, dict):
                dbg = _get_debug_section(a1)
                dbg.setdefault('render', {})['table_dict'] = table_dict
                a1['debug'] = dbg
            st.session_state['_latest_overview'] = (overview_df, table_dict)
            st.session_state['_latest_agent1'] = a1
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
        overview_cached = st.session_state.get('_latest_overview', (None, None))
        render_summary_view(a1, result, overview_df=overview_cached[0], table_dict=overview_cached[1])
    except Exception:
        st.error("요약 뷰를 렌더링하는 중 오류가 발생했습니다.")
        st.code(traceback.format_exc())

    with st.expander("🧾 Agent-2 출력(JSON) 보기", expanded=False):
        st.json(result)

# ===== 디버그 뷰 =====
if show_debug:
    latest_agent1 = st.session_state.get('_latest_agent1')
    with st.expander("🔍 디버그 상세", expanded=True):
        render_debug_view(latest_agent1, show_raw=DEBUG_SHOW_RAW)

# 최초 안내
if not st.session_state.get("_intro_shown"):
    st.info("✅ 업로드 성공! 이제 질문 입력 후 [분석 실행]을 눌러 카드 결과를 확인해보세요.")
    st.session_state["_intro_shown"] = True
