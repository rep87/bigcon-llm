import os, json, traceback, re, hashlib
from pathlib import Path
import streamlit as st
import pandas as pd
from diagnostics import (
    build_catalog,
    export_reports,
    export_access,
    load_set1,
    run_access_audit,
    summarize_access,
    summarize_catalog,
)

# ===== 페이지 기본 =====
st.set_page_config(page_title="성동구 소상공인 비밀상담사 (MVP)", page_icon="💬", layout="wide")
st.title("성동구 소상공인 비밀상담사 (MVP)")
st.caption("Agent-1: 데이터 집계/요약 → Agent-2: 실행카드(JSON) 생성")
show_debug = st.checkbox("🔍 디버그 보기", value=True)


def _env_flag(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


DEBUG_SHOW_RAW = _env_flag("DEBUG_SHOW_RAW", "true").lower() in {"1", "true", "yes"}


@st.cache_data(show_spinner=False)
def _load_set1_cached(path: str = "data/shinhan/big_data_set1_f.csv") -> pd.DataFrame:
    """Cached helper that wraps :func:`diagnostics.load_set1`."""

    return load_set1(path)


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Create a stable hash signature for a dataframe's current content."""

    if df is None or df.empty:
        return "empty"
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.md5(csv_bytes).hexdigest()


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

# ===== 탭 구성 =====
analysis_tab, diagnostics_tab = st.tabs(["📈 분석", "🧪 진단"])

with analysis_tab:
    default_q = "성동구 {고향***} 기준으로, 재방문율 4주 플랜 작성해줘."
    question = st.text_input("질문을 입력하세요", value=default_q)
    st.caption("상호는 반드시 {}로 감싸 주세요. 예) 성동구 {동대******}")

    if st.button("분석 실행", type="primary"):
        from bigcon_2agent_mvp_v3 import agent1_pipeline, build_agent2_prompt, call_gemini_agent2

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

        with st.spinner("Agent-2: 카드 생성 중..."):
            try:
                os.environ["GEMINI_API_KEY"] = API_KEY
                prompt_text = build_agent2_prompt(a1)
                result = call_gemini_agent2(prompt_text)
                st.success("Agent-2 카드 생성 완료")
            except Exception:
                st.error("Agent-2 실행 오류")
                st.code(traceback.format_exc())
                st.stop()

        try:
            overview_cached = st.session_state.get('_latest_overview', (None, None))
            render_summary_view(a1, result, overview_df=overview_cached[0], table_dict=overview_cached[1])
        except Exception:
            st.error("요약 뷰를 렌더링하는 중 오류가 발생했습니다.")
            st.code(traceback.format_exc())

        with st.expander("🧾 Agent-2 출력(JSON) 보기", expanded=False):
            st.json(result)

    if show_debug:
        latest_agent1 = st.session_state.get('_latest_agent1')
        with st.expander("🔍 디버그 상세", expanded=True):
            render_debug_view(latest_agent1, show_raw=DEBUG_SHOW_RAW)

    if not st.session_state.get("_intro_shown"):
        st.info("✅ 업로드 성공! 이제 질문 입력 후 [분석 실행]을 눌러 카드 결과를 확인해보세요.")
        st.session_state["_intro_shown"] = True

with diagnostics_tab:
    st.subheader("상호 카탈로그")

    try:
        base_df = _load_set1_cached()
    except FileNotFoundError:
        st.error("Set1 CSV 파일을 찾을 수 없습니다. (data/shinhan/big_data_set1_f.csv)")
        base_df = pd.DataFrame(columns=["ENCODED_MCT", "MCT_NM", "SIGUNGU", "CATEGORY", "MCT_BRD_NUM"])
        st.session_state.pop('diagnostics_catalog', None)
    except Exception as exc:  # pragma: no cover - defensive guard
        st.error(f"Set1 CSV를 불러오는 중 오류가 발생했습니다: {exc}")
        base_df = pd.DataFrame(columns=["ENCODED_MCT", "MCT_NM", "SIGUNGU", "CATEGORY", "MCT_BRD_NUM"])
        st.session_state.pop('diagnostics_catalog', None)

    unique_sigungu = sorted({str(v) for v in base_df.get("SIGUNGU", pd.Series(dtype="string")).dropna().unique() if str(v).strip()})
    sigungu_options = ["전체"] + unique_sigungu if unique_sigungu else ["전체"]
    default_sigungu = "성동구" if "성동구" in sigungu_options else sigungu_options[0]
    selected_sigungu = st.selectbox("시군구 선택", options=sigungu_options, index=sigungu_options.index(default_sigungu))

    max_rows = int(st.number_input("최대 표기 행수", min_value=10, max_value=5000, value=100, step=10))
    search_query = st.text_input("상호 검색", value="", placeholder="상호 일부를 입력하세요")
    note_filter = st.multiselect("상태 필터(note)", options=["missing_id", "ok", "duplicate_name"])
    sort_option = st.selectbox("정렬", options=["점포수 내림차순", "상호명 오름차순"])

    if st.button("상호 카탈로그 생성", key="btn_catalog"):
        if base_df.empty:
            st.warning("카탈로그를 생성할 데이터가 없습니다.")
            st.session_state.pop('diagnostics_catalog', None)
        else:
            with st.spinner("상호 카탈로그를 준비하는 중입니다..."):
                sigungu_filter = None if selected_sigungu == "전체" else selected_sigungu
                catalog_df = build_catalog(base_df, sigungu_filter=sigungu_filter)
                catalog_df = catalog_df.reset_index(drop=True)
            st.session_state['diagnostics_catalog'] = {
                'source': catalog_df,
                'sigungu': sigungu_filter,
                'signature': None,
                'exports': None,
            }
            st.success("상호 카탈로그가 생성되었습니다.")

    diag_state = st.session_state.get('diagnostics_catalog')
    if not diag_state or diag_state.get('source') is None:
        st.info("상단의 [상호 카탈로그 생성] 버튼을 눌러 진단을 실행하세요.")
    else:
        catalog_df = diag_state['source']
        display_df = catalog_df.copy()

        if search_query.strip():
            display_df = display_df[display_df['MCT_NM'].fillna("").str.contains(search_query.strip(), case=False, na=False)]
        if note_filter:
            display_df = display_df[display_df['note'].isin(note_filter)]

        if sort_option == "점포수 내림차순" and 'n_locations' in display_df:
            display_df = display_df.sort_values('n_locations', ascending=False)
        elif sort_option == "상호명 오름차순" and 'MCT_NM' in display_df:
            display_df = display_df.sort_values('MCT_NM', na_position='last')

        display_df = display_df.reset_index(drop=True)
        summary = summarize_catalog(display_df)

        signature = _hash_dataframe(display_df)
        if signature != diag_state.get('signature'):
            exports = export_reports(display_df, summary)
            diag_state['signature'] = signature
            diag_state['exports'] = exports
            st.session_state['diagnostics_catalog'] = diag_state

        summary_metrics = [
            ("총 상호", f"{summary.get('total_rows', 0):,}"),
            ("고유 상호", f"{summary.get('unique_names', 0):,}"),
            ("ID 누락 비율", f"{summary.get('pct_missing_id', 0.0):.2f}%"),
            ("동명 다점포 비율", f"{summary.get('pct_duplicate_name', 0.0):.2f}%"),
        ]
        metric_cols = st.columns(len(summary_metrics))
        for col, (label, value) in zip(metric_cols, summary_metrics):
            col.metric(label, value)

        top_dups = summary.get('top_duplicated') or []
        if top_dups:
            st.markdown("**다점포 상위 10개**")
            st.table(pd.DataFrame(top_dups))

        display_columns = ["SIGUNGU", "MCT_NM", "MCT_BRD_NUM", "n_locations", "has_encoded_mct", "note"]
        for col in display_columns:
            if col not in display_df.columns:
                display_df[col] = pd.NA
        st.markdown("**상호 목록 (요약)**")
        st.dataframe(display_df[display_columns].head(max_rows))

        if 'encoded_mct_list' in display_df.columns:
            with st.expander("encoded_mct_list 보기", expanded=False):
                st.dataframe(display_df[["SIGUNGU", "MCT_NM", "encoded_mct_list"]].head(max_rows))

        exports = diag_state.get('exports') or {}
        csv_path = exports.get('catalog_csv')
        json_path = exports.get('summary_json')
        dl_cols = st.columns(2)
        if csv_path and Path(csv_path).exists():
            with open(csv_path, "rb") as fp:
                dl_cols[0].download_button(
                    "카탈로그 CSV 다운로드",
                    data=fp.read(),
                    file_name=Path(csv_path).name,
                    mime="text/csv",
                )
        else:
            dl_cols[0].write("CSV 파일이 아직 생성되지 않았습니다.")
        if json_path and Path(json_path).exists():
            with open(json_path, "rb") as fp:
                dl_cols[1].download_button(
                    "요약 JSON 다운로드",
                    data=fp.read(),
                    file_name=Path(json_path).name,
                    mime="application/json",
                )
        else:
            dl_cols[1].write("JSON 파일이 아직 생성되지 않았습니다.")

    st.markdown("---")
    st.subheader("접근 감사 (Access Audit)")

    audit_default_index = sigungu_options.index(default_sigungu)
    audit_sigungu = st.selectbox(
        "시군구 선택 (접근 감사)",
        options=sigungu_options,
        index=audit_default_index,
        key="audit_sigungu",
    )

    audit_mode = st.radio(
        "샘플 모드",
        options=["Random", "Search"],
        key="audit_mode",
        horizontal=True,
    )

    audit_terms: list[str] | None = None
    if audit_mode == "Random":
        audit_n = int(
            st.number_input(
                "샘플 수",
                min_value=1,
                max_value=200,
                value=10,
                step=1,
                key="audit_sample_count",
            )
        )
        audit_prefix = int(
            st.number_input(
                "마스킹 접두 길이",
                min_value=1,
                max_value=4,
                value=2,
                step=1,
                key="audit_prefix_len",
            )
        )
    else:
        search_input = st.text_area(
            "상호 검색어 (줄당 1개, {고향***} 형식)",
            value="",
            key="audit_search_terms",
            height=120,
        )
        audit_terms = [line.strip() for line in search_input.splitlines() if line.strip()]
        audit_n = 0
        audit_prefix = 2

    audit_brand_match = st.checkbox(
        "동일 브랜드를 정답으로 인정", value=True, key="audit_brand_match"
    )

    if st.button("접근 감사 실행", key="btn_access_audit"):
        if audit_mode == "Search" and not audit_terms:
            st.warning("검색 모드에서는 최소 1개의 검색어를 입력해 주세요.")
        else:
            try:
                with st.spinner("접근 감사를 수행하는 중입니다..."):
                    progress = st.progress(0)
                    audit_df = run_access_audit(
                        mode=audit_mode.lower(),
                        sigungu=None if audit_sigungu == "전체" else audit_sigungu,
                        n=audit_n or 10,
                        search_terms=audit_terms,
                        mask_prefix_len=audit_prefix,
                        brand_match=audit_brand_match,
                        seed=42,
                    )
                    progress.progress(100)
                summary = summarize_access(audit_df)
                signature = _hash_dataframe(audit_df)
                audit_state = {
                    "df": audit_df,
                    "summary": summary,
                    "sigungu": audit_sigungu,
                    "mode": audit_mode,
                    "signature": signature,
                    "exports": None,
                }
                st.session_state['diagnostics_access_audit'] = audit_state
                st.success("접근 감사가 완료되었습니다.")
            except FileNotFoundError:
                st.error("Shinhan CSV 파일을 찾을 수 없습니다. (Set1/2/3)")
            except Exception:
                st.error("접근 감사 실행 중 오류가 발생했습니다.")
                st.code(traceback.format_exc())

    audit_state = st.session_state.get('diagnostics_access_audit')
    if not audit_state or audit_state.get('df') is None:
        st.info("상단의 [접근 감사 실행] 버튼을 눌러 결과를 생성하세요.")
    else:
        audit_df = audit_state['df']
        summary = audit_state.get('summary', {})
        signature = _hash_dataframe(audit_df)
        if signature != audit_state.get('signature'):
            audit_state['signature'] = signature
            audit_state['exports'] = None
            st.session_state['diagnostics_access_audit'] = audit_state

        if audit_df is not None and not audit_df.empty and not audit_state.get('exports'):
            audit_state['exports'] = export_access(audit_df, summary)
            st.session_state['diagnostics_access_audit'] = audit_state

        st.markdown(
            f"총 {len(audit_df):,}건 샘플 · Resolver 성공률 {summary.get('resolved_rate', 0.0):.1f}%"
        )

        metric_rows = [
            [
                ("Resolver 성공률", f"{summary.get('resolved_rate', 0.0):.1f}%"),
                ("정확도", f"{summary.get('accuracy', 0.0):.1f}%"),
                ("Out-of-range 건수", f"{summary.get('out_of_range_count', 0)}"),
            ],
            [
                ("S1 접근률", f"{summary.get('s1_access_rate', 0.0):.1f}%"),
                ("S2 접근률", f"{summary.get('s2_access_rate', 0.0):.1f}%"),
                ("S3 접근률", f"{summary.get('s3_access_rate', 0.0):.1f}%"),
            ],
            [
                ("S1 컬럼 커버리지", f"{summary.get('s1_coverage', 0.0):.1f}%"),
                ("S2 컬럼 커버리지", f"{summary.get('s2_coverage', 0.0):.1f}%"),
                ("S3 컬럼 커버리지", f"{summary.get('s3_coverage', 0.0):.1f}%"),
            ],
            [
                ("S1 NaN 중간값", f"{summary.get('s1_nan_median', 0.0):.1f}%"),
                ("S2 NaN 중간값", f"{summary.get('s2_nan_median', 0.0):.1f}%"),
                ("S3 NaN 중간값", f"{summary.get('s3_nan_median', 0.0):.1f}%"),
            ],
        ]

        for items in metric_rows:
            cols = st.columns(len(items))
            for col, (label, value) in zip(cols, items):
                col.metric(label, value)

        warn_conditions = []
        if 'is_correct' in audit_df.columns:
            warn_conditions.append(~audit_df['is_correct'].astype(bool))
        if 's1_found' in audit_df.columns:
            warn_conditions.append(~audit_df['s1_found'].astype(bool))
        if 's2_found' in audit_df.columns:
            warn_conditions.append(~audit_df['s2_found'].astype(bool))
        if 's3_found' in audit_df.columns:
            warn_conditions.append(~audit_df['s3_found'].astype(bool))
        if 'out_of_range_flags' in audit_df.columns:
            warn_conditions.append(audit_df['out_of_range_flags'].fillna("") != "")

        warn_mask = None
        for cond in warn_conditions:
            warn_mask = cond if warn_mask is None else (warn_mask | cond)

        if warn_mask is None:
            warn_df = audit_df.iloc[0:0]
        else:
            warn_df = audit_df[warn_mask]

        warn_cols = [
            "input_text",
            "resolved_id",
            "truth_id",
            "truth_brand",
            "is_correct",
            "s1_found",
            "s2_found",
            "s3_found",
            "s1_cols_present",
            "s1_cols_expected",
            "s2_cols_present",
            "s2_cols_expected",
            "s3_cols_present",
            "s3_cols_expected",
            "out_of_range_flags",
            "path",
        ]
        if warn_df.empty:
            st.success("경고 항목이 없습니다. 모든 샘플이 정상 접근되었습니다.")
        else:
            for col in warn_cols:
                if col not in warn_df.columns:
                    warn_df[col] = pd.NA
            st.markdown("**실패/경고 샘플 (상위 100건)**")
            st.dataframe(warn_df[warn_cols].head(100))

        exports = audit_state.get('exports') or {}
        dl_cols = st.columns(2)
        csv_path = exports.get('csv') if isinstance(exports, dict) else None
        json_path = exports.get('summary') if isinstance(exports, dict) else None
        if csv_path and Path(csv_path).exists():
            with open(csv_path, "rb") as fp:
                dl_cols[0].download_button(
                    "접근 감사 CSV 다운로드",
                    data=fp.read(),
                    file_name=Path(csv_path).name,
                    mime="text/csv",
                )
        else:
            dl_cols[0].write("CSV 파일이 아직 생성되지 않았습니다.")
        if json_path and Path(json_path).exists():
            with open(json_path, "rb") as fp:
                dl_cols[1].download_button(
                    "접근 감사 요약 JSON 다운로드",
                    data=fp.read(),
                    file_name=Path(json_path).name,
                    mime="application/json",
                )
        else:
            dl_cols[1].write("JSON 파일이 아직 생성되지 않았습니다.")
