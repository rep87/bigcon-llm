import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app_core.failsoft import (
    compose_fail_soft_answer,
    external_adapter,
    rag_adapter,
    structured_adapter,
    weather_adapter,
)
from app_core.formatters import (
    get_age_bucket_details,
    merge_age_buckets,
    three_line_diagnosis,
    to_float_pct,
)

import pandas as pd
import streamlit as st

try:
    from rag import RetrievalTool
except Exception:  # pragma: no cover - optional dependency path
    RetrievalTool = None

def _get_debug_snapshot(agent1_json: dict | None) -> dict:
    debug = _get_debug_section(agent1_json)
    snap = debug.get("snapshot")
    if isinstance(snap, dict):
        sanitized = snap.get("sanitized")
        if isinstance(sanitized, dict):
            return sanitized
    legacy = debug.get("sanitized_snapshot")
    return legacy if isinstance(legacy, dict) else {}


def _compute_rag_info(question_text: str, flags_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    selected_docs = [
        str(doc_id)
        for doc_id in (flags_snapshot.get("rag_selected_ids") or [])
        if str(doc_id)
    ]
    rag_requested = bool(flags_snapshot.get("use_rag", False))
    rag_mode = str(flags_snapshot.get("rag_mode", "auto"))
    rag_threshold = float(flags_snapshot.get("rag_threshold", 0.35))
    rag_top_k = int(flags_snapshot.get("rag_top_k", 5))
    return rag_adapter(
        question_text,
        RETRIEVAL_TOOL,
        enabled=rag_requested and bool(selected_docs),
        top_k=rag_top_k,
        threshold=rag_threshold,
        mode=rag_mode,
        requested=rag_requested,
        doc_ids=selected_docs,
    )


def _prepare_rag_prompt_context(rag_info: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(rag_info, dict):
        return None
    payload = rag_info.get("payload") or {}
    chunks = payload.get("chunks") or []
    evidence = payload.get("evidence") or []
    context = {
        "enabled": bool(rag_info.get("enabled")),
        "requested": bool(rag_info.get("requested")),
        "selection_missing": bool(rag_info.get("selection_missing")),
        "selected_doc_ids": list(rag_info.get("selected_doc_ids") or []),
        "threshold": rag_info.get("threshold"),
        "mode": rag_info.get("mode"),
        "max_score": rag_info.get("max_score"),
        "hits": len(chunks),
        "chunks": [dict(chunk) for chunk in chunks],
        "evidence": [dict(item) for item in evidence],
        "error": rag_info.get("error"),
        "catalog_size": rag_info.get("catalog_size"),
        "top_scores": list(payload.get("top_scores") or []),
    }
    return context


def _shorten_snippet(text: Any, limit: int = 140) -> str:
    snippet = str(text or "").strip()
    if len(snippet) > limit:
        snippet = snippet[:limit].rstrip() + "…"
    return snippet


def _format_evidence_line(entry: Dict[str, Any]) -> str:
    source = str(entry.get("source") or "NONE").upper()
    key = entry.get("key") or "—"
    value = entry.get("value")
    if value is None:
        value_text = "—"
    else:
        value_text = str(value)
    period = entry.get("period")
    snippet = entry.get("snippet")
    parts = [f"[{source}] {key}: {value_text}"]
    if period:
        parts.append(f"({period})")
    line = " ".join(parts)
    if snippet:
        line += f" — {_shorten_snippet(snippet)}"
    return f"- {line}"


def _iter_debug_distribution(data: Any) -> List[Tuple[str, Any]]:
    entries: List[Tuple[str, Any]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            entries.append((str(key), value))
    elif isinstance(data, Sequence):
        for item in data:
            if not isinstance(item, dict):
                continue
            code = item.get("code") or item.get("key") or item.get("id") or item.get("label")
            if code is None:
                continue
            value = item.get("value")
            if value is None:
                for alt in ("percent", "ratio", "pct", "share"):
                    if alt in item:
                        value = item[alt]
                        break
            entries.append((str(code), value))
    return entries


_DEBUG_F_KEYS = ["F", "f", "FME", "female", "여", "여성"]
_DEBUG_M_KEYS = ["M", "m", "MAL", "male", "남", "남성"]


def _format_debug_pct(value: Any) -> str:
    pct, hint = to_float_pct(value)
    if pct is None:
        return f"{value!r} ({hint})"
    return f"{pct:.1f}% ({hint})"


def _lookup_gender_value(mapping: Dict[str, Any], aliases: Sequence[str]) -> Any:
    for alias in aliases:
        if alias in mapping:
            return mapping.get(alias)
    return None


def _build_debug_report_markdown(
    agent1_json: dict | None,
    *,
    question_type: str | None,
    rag_info: Dict[str, Any] | None,
    rag_prompt_context: Dict[str, Any] | None,
    prompt_trace: Dict[str, Any] | None,
    response_trace: Dict[str, Any] | None,
) -> str:
    if not isinstance(agent1_json, dict):
        return ""

    kpis = (agent1_json.get("kpis") or {}) if isinstance(agent1_json, dict) else {}
    sanitized_snapshot = _get_debug_snapshot(agent1_json)
    debug_section = _get_debug_section(agent1_json)
    raw_snapshot = ((debug_section.get("snapshot") or {}).get("raw") or {}) if isinstance(debug_section, dict) else {}

    lines: List[str] = ["### Debug Report"]
    lines.append("**Agent-1 Snapshot**")

    age_entries = _iter_debug_distribution(
        sanitized_snapshot.get("age_distribution") or kpis.get("age_distribution")
    )
    if age_entries:
        lines.append("- age_distribution:")
        for key, raw_value in age_entries:
            lines.append(f"    - {key}: {_format_debug_pct(raw_value)}")
    else:
        lines.append("- age_distribution: 없음")

    gender_data = (
        sanitized_snapshot.get("age_by_gender")
        or sanitized_snapshot.get("age_gender")
        or kpis.get("age_by_gender")
        or kpis.get("age_gender")
        or {}
    )
    if isinstance(gender_data, dict) and gender_data:
        lines.append("- age_by_gender:")
        for bucket, mapping in gender_data.items():
            if not isinstance(mapping, dict):
                continue
            female_raw = _lookup_gender_value(mapping, _DEBUG_F_KEYS)
            male_raw = _lookup_gender_value(mapping, _DEBUG_M_KEYS)
            lines.append(
                "    - "
                + f"{bucket}: F {_format_debug_pct(female_raw)}, M {_format_debug_pct(male_raw)}"
            )
    elif raw_snapshot:
        raw_keys = [key for key in raw_snapshot.keys() if "M12_" in str(key)]
        if raw_keys:
            lines.append(f"- raw age keys detected: {', '.join(raw_keys[:6])}")

    mix_detail = sanitized_snapshot.get("customer_mix_detail") or kpis.get("customer_mix_detail")
    if isinstance(mix_detail, dict) and mix_detail:
        lines.append("- customer_mix_detail:")
        for key, value in mix_detail.items():
            lines.append(f"    - {key}: {_format_debug_pct(value)}")

    new_raw = sanitized_snapshot.get("new_pct") or kpis.get("new_rate_avg")
    revisit_raw = sanitized_snapshot.get("revisit_pct") or kpis.get("revisit_rate_avg")
    lines.append(f"- 신규 비중: {_format_debug_pct(new_raw)}")
    lines.append(f"- 재방문 비중: {_format_debug_pct(revisit_raw)}")

    normalization_notes = [
        f"{key}={value}"
        for key, value in sanitized_snapshot.items()
        if isinstance(key, str) and "normalize" in key.lower()
    ]
    if normalization_notes:
        lines.append("- normalization flags:")
        for note in normalization_notes:
            lines.append(f"    - {note}")

    lines.append("\n**Age Merge Decision**")
    age_details = get_age_bucket_details(agent1_json or {})
    if age_details:
        for item in age_details:
            final_val = item.get("final_value")
            final_text = f"{final_val:.1f}%" if isinstance(final_val, (int, float)) else "—"
            status = "included" if item.get("included") else "skipped"
            lines.append(
                f"- {item.get('label')} ({item.get('key')}): {final_text} ← {item.get('source')} ({status})"
            )
            if item.get("notes"):
                lines.append(f"    - {item['notes']}")
    else:
        lines.append("- no age buckets detected")

    prompt_trace = prompt_trace or {}
    lines.append("\n**Agent-2 Prompt**")
    lines.append(f"- question_type: {question_type or prompt_trace.get('question_type')}")
    lines.append(f"- organizer_mode: {prompt_trace.get('organizer_mode')}")
    lines.append(f"- schema_keys: {prompt_trace.get('schema_keys')}")
    lines.append(
        f"- rag_context_included: {prompt_trace.get('rag_included')}"
        + (f" (reason: {prompt_trace.get('rag_reason')})" if prompt_trace.get("rag_reason") else "")
    )
    if prompt_trace.get("rag_context_doc_ids"):
        lines.append(f"- rag doc_ids: {prompt_trace.get('rag_context_doc_ids')}")

    response_trace = response_trace or {}
    lines.append("\n**Agent-2 Response Trace**")
    if not response_trace:
        lines.append("- response trace unavailable")
    else:
        status = response_trace.get("status")
        lines.append(f"- status: {status}")
        if response_trace.get("generation_config"):
            lines.append(f"- generation_config: {response_trace.get('generation_config')}")
        if response_trace.get("prompt_length") is not None:
            lines.append(f"- prompt_length: {response_trace.get('prompt_length')}")
        if response_trace.get("chosen_model"):
            lines.append(f"- chosen_model: {response_trace.get('chosen_model')}")
        if response_trace.get("last_error"):
            lines.append(f"- last_error: {response_trace.get('last_error')}")
        if response_trace.get("fallback_answers") is not None:
            lines.append(f"- fallback_answers: {response_trace.get('fallback_answers')}")
        attempts = response_trace.get("attempts") or []
        if attempts:
            lines.append("- attempts:")
            for attempt in attempts:
                label = f"attempt {attempt.get('attempt')} / {attempt.get('model')}"
                lines.append(f"    - {label}: {attempt.get('status')}")
                parse_logs = attempt.get("parse_logs") or []
                excerpt = ""
                for entry in parse_logs:
                    if not entry.get("success") and entry.get("excerpt"):
                        excerpt = entry["excerpt"].replace('\n', ' ')[:160]
                        break
                if excerpt:
                    lines.append(f"        · excerpt: {excerpt}")
                if attempt.get("raw_preview"):
                    preview = attempt.get("raw_preview").replace('\n', ' ')
                    lines.append(f"        · raw_preview: {preview[:160]}")
                if attempt.get("repair_logs"):
                    rep_logs = attempt["repair_logs"]
                    for entry in rep_logs:
                        if not entry.get("success") and entry.get("excerpt"):
                            snippet = entry["excerpt"].replace('\n', ' ')[:160]
                            lines.append(f"        · repair excerpt: {snippet}")
                            break

    rag_info = rag_info or {}
    lines.append("\n**RAG Retrieval**")
    if not rag_info:
        lines.append("- RAG info unavailable")
    else:
        lines.append(
            f"- requested={rag_info.get('requested')} enabled={rag_info.get('enabled')} selection_missing={rag_info.get('selection_missing')}"
        )
        lines.append(
            f"- selected_doc_ids: {rag_info.get('selected_doc_ids')} catalog_size={rag_info.get('catalog_size')}"
        )
        lines.append(
            f"- mode={rag_info.get('mode')} threshold={rag_info.get('threshold')} top_k={rag_info.get('top_k')}"
        )
        lines.append(
            f"- max_score={rag_info.get('max_score')} include_evidence={rag_info.get('include_evidence')} error={rag_info.get('error')}"
        )
        payload = rag_info.get("payload") or {}
        top_scores = rag_prompt_context.get("top_scores") if isinstance(rag_prompt_context, dict) else []
        if not top_scores:
            top_scores = payload.get("top_scores") or []
        if top_scores:
            rounded = [round(float(score), 3) for score in top_scores[:5]]
            lines.append(f"- top_scores: {rounded}")
        if not rag_info.get("include_evidence"):
            reason = ""
            max_score = rag_info.get("max_score")
            threshold = rag_info.get("threshold")
            if rag_info.get("error"):
                reason = rag_info.get("error")
            elif max_score is None:
                reason = "no hits"
            elif threshold is not None and max_score < threshold:
                reason = f"max_score {max_score:.3f} < threshold {threshold:.2f}"
            else:
                reason = "evidence suppressed"
            lines.append(f"- evidence omitted reason: {reason}")

    return "\n".join(lines)


def _match_evidence_chunk(
    entry: Dict[str, Any],
    retrieval_payload: Dict[str, Any] | None,
) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    if not isinstance(entry, dict) or not retrieval_payload:
        return None, None
    doc_id = entry.get("doc_id")
    if not doc_id:
        return None, None
    chunk_id = entry.get("chunk_id")
    chunks = retrieval_payload.get("chunks") or []
    evidence_list = retrieval_payload.get("evidence") or []
    target_chunk = None
    target_meta = None
    chunk_id_str = str(chunk_id) if chunk_id is not None else None

    for chunk in chunks:
        if str(chunk.get("doc_id")) != str(doc_id):
            continue
        current_chunk_id = chunk.get("chunk_id")
        if chunk_id_str is not None and str(current_chunk_id) != chunk_id_str:
            continue
        target_chunk = chunk
        break

    if target_chunk is None:
        for chunk in chunks:
            if str(chunk.get("doc_id")) == str(doc_id):
                target_chunk = chunk
                break

    if target_chunk is not None:
        for meta in evidence_list:
            if str(meta.get("doc_id")) != str(doc_id):
                continue
            if chunk_id_str is not None:
                if str(meta.get("chunk_id")) == chunk_id_str:
                    target_meta = meta
                    break
            else:
                target_meta = meta
                break

    return target_chunk, target_meta


def _get_debug_raw_snapshot(agent1_json: dict | None) -> dict:
    debug = _get_debug_section(agent1_json)
    snap = debug.get("snapshot")
    if isinstance(snap, dict):
        raw = snap.get("raw")
        if isinstance(raw, dict):
            return raw
    legacy = debug.get("latest_raw_snapshot")
    return legacy if isinstance(legacy, dict) else {}


def _render_main_views(
    question_text: str,
    agent1_payload: dict | None,
    agent2_payload: dict | None,
    *,
    rag_info_override: Dict[str, Any] | None = None,
) -> None:
    flags_snapshot = st.session_state.get("_data_flags", {}).copy()
    structured_payload = structured_adapter(agent1_payload)
    weather_payload = weather_adapter(question_text, enabled=flags_snapshot.get("use_weather", False))
    external_payload = external_adapter(question_text, enabled=flags_snapshot.get("use_external", False))
    if rag_info_override is not None:
        rag_info = rag_info_override
    else:
        rag_info = _compute_rag_info(question_text, flags_snapshot)

    retrieval_payload = rag_info.get("payload") if isinstance(rag_info, dict) else None
    if rag_info.get("error"):
        st.warning(f"RetrievalTool 오류: {rag_info['error']}")
    if retrieval_payload is not None:
        st.session_state['_latest_retrieval'] = retrieval_payload
    else:
        st.session_state['_latest_retrieval'] = None
    st.session_state['_latest_rag_info'] = rag_info
    st.session_state['_latest_rag_prompt_context'] = _prepare_rag_prompt_context(rag_info)

    fail_soft_payload = compose_fail_soft_answer(
        question_text,
        structured_payload,
        weather_payload,
        external_payload,
        rag_info,
        flags_snapshot,
    )
    st.session_state['_latest_failsoft'] = fail_soft_payload

    try:
        overview_cached = st.session_state.get('_latest_overview', (None, None))
        if isinstance(agent2_payload, dict):
            if retrieval_payload:
                agent2_payload.setdefault("evidence", retrieval_payload.get("evidence", []))
                agent2_payload.setdefault("retrieval_chunks", retrieval_payload.get("chunks", []))
            agent2_payload.setdefault("used_data", fail_soft_payload.get("used_data"))
            if APP_MODE == "debug":
                agent2_payload.setdefault("caveats", fail_soft_payload.get("caveats"))

        if APP_MODE == "debug" and show_debug:
            render_fail_soft_answer(fail_soft_payload, rag_info=rag_info)
        render_summary_view(
            agent1_payload,
            agent2_payload or {},
            overview_df=overview_cached[0],
            table_dict=overview_cached[1],
            retrieval_payload=retrieval_payload,
        )
    except Exception:
        st.error("요약 뷰를 렌더링하는 중 오류가 발생했습니다.")
        st.code(traceback.format_exc())

def render_debug_view(agent1_json: dict | None, show_raw: bool = DEBUG_SHOW_RAW) -> None:
    debug = _get_debug_section(agent1_json)
    if not debug:
        st.info("디버그 정보가 없습니다.")
        return

    question_type = st.session_state.get("_latest_question_type")
    rag_info_state = st.session_state.get("_latest_rag_info")
    rag_prompt_state = st.session_state.get("_latest_rag_prompt_context")
    prompt_trace = st.session_state.get("_latest_prompt_trace")
    response_trace = st.session_state.get("_latest_response_trace")
    report_markdown = _build_debug_report_markdown(
        agent1_json,
        question_type=question_type,
        rag_info=rag_info_state,
        rag_prompt_context=rag_prompt_state,
        prompt_trace=prompt_trace,
        response_trace=response_trace,
    )
    if report_markdown:
        st.markdown(report_markdown)

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
    age_details = get_age_bucket_details(agent1_json or {})
    if age_details:
        st.caption("연령 분포 결정")
        st.table(pd.DataFrame(age_details))

    render_info = debug.get("render", {}) or {}
    table_dict = render_info.get("table_dict")
    if isinstance(table_dict, dict) and table_dict:
        st.markdown("#### 렌더 테이블")
        st.table(pd.DataFrame([table_dict]))

    rag_info_state = st.session_state.get("_latest_rag_info")
    rag_prompt_state = st.session_state.get("_latest_rag_prompt_context") or {}
    st.markdown("#### RAG 상태")
    if isinstance(rag_info_state, dict):
        summary = {
            "requested": rag_info_state.get("requested"),
            "enabled": rag_info_state.get("enabled"),
            "selected_doc_ids": rag_info_state.get("selected_doc_ids"),
            "threshold": rag_info_state.get("threshold"),
            "max_score": rag_info_state.get("max_score"),
            "mode": rag_info_state.get("mode"),
            "error": rag_info_state.get("error"),
        }
        st.write(summary)
        payload = rag_info_state.get("payload") or {}
        chunk_rows = []
        for chunk in (payload.get("chunks") or [])[:5]:
            chunk_rows.append(
                {
                    "doc_id": chunk.get("doc_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "score": chunk.get("score"),
                }
            )
        if chunk_rows:
            st.table(pd.DataFrame(chunk_rows))
        else:
            st.write("근거 없음 또는 임계값 미달")
        prompt_note = rag_prompt_state.get("prompt_note")
        if prompt_note:
            st.caption(_shorten_snippet(prompt_note, limit=200))
    else:
        st.write("RAG 정보가 없습니다.")

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


def _render_sources_footer(used_data: Dict[str, Any]) -> None:
    if not used_data:
        return

    st.caption("Sources Used")
    cols = st.columns(4)
    structured_text = "✅ Structured" if used_data.get("structured") else "⚪️ Structured"
    weather_text = "✅ Weather" if used_data.get("weather") else "⚪️ Weather"
    external_text = "✅ External" if used_data.get("external") else "⚪️ External"
    rag_info = used_data.get("rag") or {}
    rag_enabled = bool(rag_info.get("enabled"))
    rag_hits = rag_info.get("hits") or 0
    rag_label = "✅ RAG" if rag_enabled and rag_hits else ("⚪️ RAG" if rag_enabled else "🚫 RAG")
    rag_details = []
    if rag_hits:
        rag_details.append(f"hits={rag_hits}")
    max_score = rag_info.get("max_score")
    if isinstance(max_score, (int, float)):
        rag_details.append(f"max={max_score:.2f}")
    threshold = rag_info.get("threshold")
    if isinstance(threshold, (int, float)):
        rag_details.append(f"θ={threshold:.2f}")
    mode = rag_info.get("mode")
    if mode:
        rag_details.append(str(mode))
    selected_docs = rag_info.get("selected_docs") or []
    if selected_docs:
        rag_details.append(f"docs={len(selected_docs)}")
    elif rag_info.get("requested") and not rag_enabled:
        rag_details.append("docs=0")
    rag_text = rag_label + (" (" + ", ".join(rag_details) + ")" if rag_details else "")

    cols[0].markdown(structured_text)
    cols[1].markdown(weather_text)
    cols[2].markdown(external_text)
    cols[3].markdown(rag_text)


def render_fail_soft_answer(
    payload: Optional[Dict[str, Any]],
    *,
    rag_info: Optional[Dict[str, Any]] = None,
) -> None:
    st.subheader("Fail-soft 응답")
    if not payload:
        st.info("응답이 생성되지 않았습니다.")
        return

    segments = payload.get("segments") or []
    if not segments:
        st.info("응답이 생성되지 않았습니다.")
    rag_error = (rag_info or {}).get("error") if rag_info else None

    for segment in segments:
        cols = st.columns([12, 1])
        with cols[0]:
            st.markdown(f"- {segment.get('text')}")
        with cols[1]:
            evidence_chunk = segment.get("evidence")
            evidence_meta = segment.get("evidence_meta")
            source = segment.get("source")
            if evidence_chunk:
                _render_evidence_badge(evidence_chunk, evidence_meta)
            elif source == "rag":
                rag_enabled = bool((rag_info or {}).get("enabled"))
                if rag_enabled:
                    _render_evidence_badge(None, None, tooltip="관련 근거 없음")
                elif rag_error:
                    _render_evidence_badge(None, None, tooltip=f"RAG 오류: {rag_error}")
                else:
                    _render_evidence_badge(None, None, disabled=True, tooltip="RAG 비활성화")

    caveats = payload.get("caveats") or []
    if caveats:
        unique_caveats = []
        for item in caveats:
            if item and item not in unique_caveats:
                unique_caveats.append(item)
        if unique_caveats:
            st.caption("주의 사항")
            for item in unique_caveats:
                st.markdown(f"- {item}")

    if rag_error and all("RAG 오류" not in item for item in caveats):
        st.error(f"RAG 오류: {rag_error}")

    _render_sources_footer(payload.get("used_data") or {})

def _render_evidence_badge(
    chunk: dict | None,
    evidence_meta: dict | None,
    *,
    disabled: bool = False,
    tooltip: str | None = None,
) -> None:
    if disabled:
        label = "📎 (OFF)"
        if tooltip:
            label += f" — {tooltip}"
        st.caption(label)
        return

    if not chunk:
        if tooltip:
            st.caption(f"📎 {tooltip}")
        else:
            st.write("")
        return

    popover_fn = getattr(st, "popover", None)
    score = chunk.get("score")
    score_text = f"{float(score):.3f}" if score is not None else "—"
    doc_id = chunk.get("doc_id") or "—"
    chunk_id = chunk.get("chunk_id") or "—"
    title = evidence_meta.get("title") if isinstance(evidence_meta, dict) else None
    uri = evidence_meta.get("uri") if isinstance(evidence_meta, dict) else None

    def _short_label(value: str) -> str:
        text = str(value or "")
        if len(text) <= 8:
            return text
        return text[:5] + "…"

    display_token = chunk.get("chunk_id") or chunk.get("doc_id") or "근거"
    badge_label = f"📎 {_short_label(display_token)}·{score_text}"

    body_lines = [f"**문서 제목:** {title or doc_id}"]
    body_lines.append(f"**문서 ID:** {doc_id}")
    body_lines.append(f"**Chunk ID:** {chunk_id}")
    body_lines.append(f"**유사도:** {score_text}")
    text = chunk.get("text") or "—"
    body_lines.append("\n**내용 발췌**\n")
    body_lines.append(text)
    if uri:
        body_lines.append(f"\n[원본 열기]({uri})")

    if callable(popover_fn):
        with popover_fn(badge_label):
            for line in body_lines:
                st.markdown(line)
    else:  # pragma: no cover - fallback for older Streamlit
        with st.expander(badge_label, expanded=False):
            for line in body_lines:
                st.markdown(line)


def _env_flag(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


DEBUG_SHOW_RAW = _env_flag("DEBUG_SHOW_RAW", "true").lower() in {"1", "true", "yes"}


def _secret_value(name: str, default: str | None = None) -> str | None:
    try:
        return st.secrets.get(name, default)
    except Exception:  # pragma: no cover - Streamlit secrets unavailable
        return default


DEFAULT_RAG_ROOT = "data/rag"
RAG_ROOT = _secret_value("RAG_ROOT") or os.getenv("RAG_ROOT") or DEFAULT_RAG_ROOT
RAG_EMBED_VERSION = os.getenv("RAG_EMBED_VERSION", "embed_v1")
_DEFAULT_APP_MODE = (_secret_value("APP_MODE") or os.getenv("APP_MODE") or "public").lower()
if _DEFAULT_APP_MODE not in {"public", "debug"}:
    _DEFAULT_APP_MODE = "public"
if "_app_mode" not in st.session_state:
    st.session_state["_app_mode"] = _DEFAULT_APP_MODE
APP_MODE = st.session_state.get("_app_mode", "public")

if APP_MODE == "debug":
    show_debug = st.checkbox(
        "🔍 디버그 보기",
        value=st.session_state.get("show_debug_checkbox", True),
        key="show_debug_checkbox",
    )
else:
    show_debug = False
RAG_ROOT_PATH = Path(RAG_ROOT).expanduser()
RETRIEVAL_INIT_ERROR: str | None = None
RETRIEVAL_TOOL: object | None = None
RAG_CATALOG: list[Dict[str, Any]] = []
RAG_CATALOG_ERROR: str | None = None

if RetrievalTool is not None:
    try:
        RETRIEVAL_TOOL = RetrievalTool(root=RAG_ROOT, embed_version=RAG_EMBED_VERSION)
    except Exception as exc:  # pragma: no cover - defensive guard for UI
        RETRIEVAL_INIT_ERROR = str(exc)
else:  # pragma: no cover - module missing
    RETRIEVAL_INIT_ERROR = "rag.RetrievalTool 모듈을 불러오지 못했습니다."

if RETRIEVAL_TOOL is not None and RETRIEVAL_INIT_ERROR is None:
    if not RAG_ROOT_PATH.exists():
        RAG_CATALOG_ERROR = f"RAG_ROOT 경로({RAG_ROOT_PATH})가 존재하지 않습니다."
    else:
        try:
            catalog_entries = RETRIEVAL_TOOL.load_catalog()
            for entry in catalog_entries:
                origin_path = entry.origin_path
                origin_uri = origin_path
                if origin_path:
                    path_obj = Path(origin_path)
                    if not path_obj.is_absolute():
                        origin_uri = (RAG_ROOT_PATH / path_obj).as_posix()
                    else:
                        origin_uri = path_obj.as_posix()
                RAG_CATALOG.append(
                    {
                        "document_id": entry.doc_id,
                        "title": entry.title,
                        "num_chunks": entry.num_chunks,
                        "embedding_model": entry.embedding_model,
                        "created_at": entry.created_at,
                        "origin_path": origin_uri,
                        "tags": list(entry.tags or []),
                        "year": entry.year,
                    }
                )
        except Exception as exc:  # pragma: no cover - defensive guard
            RAG_CATALOG_ERROR = str(exc)


def _get_debug_section(agent1_json: dict | None) -> dict:
    debug = (agent1_json or {}).get("debug")
    return debug if isinstance(debug, dict) else {}


if "_data_flags" not in st.session_state:
    st.session_state["_data_flags"] = {
        "use_weather": False,
        "use_external": False,
        "use_rag": True,
        "rag_threshold": 0.35,
        "rag_top_k": 5,
        "rag_mode": "auto",
        "rag_filter": "",
        "rag_selected_ids": [],
    }


def _get_debug_snapshot(agent1_json: dict | None) -> dict:
    debug = _get_debug_section(agent1_json)
    snap = debug.get("snapshot")
    if isinstance(snap, dict):
        sanitized = snap.get("sanitized")
        if isinstance(sanitized, dict):
            return sanitized
    legacy = debug.get("sanitized_snapshot")
    return legacy if isinstance(legacy, dict) else {}


def _compute_rag_info(question_text: str, flags_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    selected_docs = [
        str(doc_id)
        for doc_id in (flags_snapshot.get("rag_selected_ids") or [])
        if str(doc_id)
    ]
    rag_requested = bool(flags_snapshot.get("use_rag", False))
    rag_mode = str(flags_snapshot.get("rag_mode", "auto"))
    rag_threshold = float(flags_snapshot.get("rag_threshold", 0.35))
    rag_top_k = int(flags_snapshot.get("rag_top_k", 5))
    return rag_adapter(
        question_text,
        RETRIEVAL_TOOL,
        enabled=rag_requested and bool(selected_docs),
        top_k=rag_top_k,
        threshold=rag_threshold,
        mode=rag_mode,
        requested=rag_requested,
        doc_ids=selected_docs,
    )


def _prepare_rag_prompt_context(rag_info: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(rag_info, dict):
        return None
    payload = rag_info.get("payload") or {}
    chunks = payload.get("chunks") or []
    evidence = payload.get("evidence") or []
    context = {
        "enabled": bool(rag_info.get("enabled")),
        "requested": bool(rag_info.get("requested")),
        "selection_missing": bool(rag_info.get("selection_missing")),
        "selected_doc_ids": list(rag_info.get("selected_doc_ids") or []),
        "threshold": rag_info.get("threshold"),
        "mode": rag_info.get("mode"),
        "max_score": rag_info.get("max_score"),
        "hits": len(chunks),
        "chunks": [dict(chunk) for chunk in chunks],
        "evidence": [dict(item) for item in evidence],
        "error": rag_info.get("error"),
        "catalog_size": rag_info.get("catalog_size"),
        "top_scores": list(payload.get("top_scores") or []),
    }
    return context


def _shorten_snippet(text: Any, limit: int = 140) -> str:
    snippet = str(text or "").strip()
    if len(snippet) > limit:
        snippet = snippet[:limit].rstrip() + "…"
    return snippet


def _format_evidence_line(entry: Dict[str, Any]) -> str:
    source = str(entry.get("source") or "NONE").upper()
    key = entry.get("key") or "—"
    value = entry.get("value")
    if value is None:
        value_text = "—"
    else:
        value_text = str(value)
    period = entry.get("period")
    snippet = entry.get("snippet")
    parts = [f"[{source}] {key}: {value_text}"]
    if period:
        parts.append(f"({period})")
    line = " ".join(parts)
    if snippet:
        line += f" — {_shorten_snippet(snippet)}"
    return f"- {line}"


def _iter_debug_distribution(data: Any) -> List[Tuple[str, Any]]:
    entries: List[Tuple[str, Any]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            entries.append((str(key), value))
    elif isinstance(data, Sequence):
        for item in data:
            if not isinstance(item, dict):
                continue
            code = item.get("code") or item.get("key") or item.get("id") or item.get("label")
            if code is None:
                continue
            value = item.get("value")
            if value is None:
                for alt in ("percent", "ratio", "pct", "share"):
                    if alt in item:
                        value = item[alt]
                        break
            entries.append((str(code), value))
    return entries


_DEBUG_F_KEYS = ["F", "f", "FME", "female", "여", "여성"]
_DEBUG_M_KEYS = ["M", "m", "MAL", "male", "남", "남성"]


def _format_debug_pct(value: Any) -> str:
    pct, hint = to_float_pct(value)
    if pct is None:
        return f"{value!r} ({hint})"
    return f"{pct:.1f}% ({hint})"


def _lookup_gender_value(mapping: Dict[str, Any], aliases: Sequence[str]) -> Any:
    for alias in aliases:
        if alias in mapping:
            return mapping.get(alias)
    return None


def _build_debug_report_markdown(
    agent1_json: dict | None,
    *,
    question_type: str | None,
    rag_info: Dict[str, Any] | None,
    rag_prompt_context: Dict[str, Any] | None,
    prompt_trace: Dict[str, Any] | None,
    response_trace: Dict[str, Any] | None,
) -> str:
    if not isinstance(agent1_json, dict):
        return ""

    kpis = (agent1_json.get("kpis") or {}) if isinstance(agent1_json, dict) else {}
    sanitized_snapshot = _get_debug_snapshot(agent1_json)
    debug_section = _get_debug_section(agent1_json)
    raw_snapshot = ((debug_section.get("snapshot") or {}).get("raw") or {}) if isinstance(debug_section, dict) else {}

    lines: List[str] = ["### Debug Report"]
    lines.append("**Agent-1 Snapshot**")

    age_entries = _iter_debug_distribution(
        sanitized_snapshot.get("age_distribution") or kpis.get("age_distribution")
    )
    if age_entries:
        lines.append("- age_distribution:")
        for key, raw_value in age_entries:
            lines.append(f"    - {key}: {_format_debug_pct(raw_value)}")
    else:
        lines.append("- age_distribution: 없음")

    gender_data = (
        sanitized_snapshot.get("age_by_gender")
        or sanitized_snapshot.get("age_gender")
        or kpis.get("age_by_gender")
        or kpis.get("age_gender")
        or {}
    )
    if isinstance(gender_data, dict) and gender_data:
        lines.append("- age_by_gender:")
        for bucket, mapping in gender_data.items():
            if not isinstance(mapping, dict):
                continue
            female_raw = _lookup_gender_value(mapping, _DEBUG_F_KEYS)
            male_raw = _lookup_gender_value(mapping, _DEBUG_M_KEYS)
            lines.append(
                "    - "
                + f"{bucket}: F {_format_debug_pct(female_raw)}, M {_format_debug_pct(male_raw)}"
            )
    elif raw_snapshot:
        raw_keys = [key for key in raw_snapshot.keys() if "M12_" in str(key)]
        if raw_keys:
            lines.append(f"- raw age keys detected: {', '.join(raw_keys[:6])}")

    mix_detail = sanitized_snapshot.get("customer_mix_detail") or kpis.get("customer_mix_detail")
    if isinstance(mix_detail, dict) and mix_detail:
        lines.append("- customer_mix_detail:")
        for key, value in mix_detail.items():
            lines.append(f"    - {key}: {_format_debug_pct(value)}")

    new_raw = sanitized_snapshot.get("new_pct") or kpis.get("new_rate_avg")
    revisit_raw = sanitized_snapshot.get("revisit_pct") or kpis.get("revisit_rate_avg")
    lines.append(f"- 신규 비중: {_format_debug_pct(new_raw)}")
    lines.append(f"- 재방문 비중: {_format_debug_pct(revisit_raw)}")

    normalization_notes = [
        f"{key}={value}"
        for key, value in sanitized_snapshot.items()
        if isinstance(key, str) and "normalize" in key.lower()
    ]
    if normalization_notes:
        lines.append("- normalization flags:")
        for note in normalization_notes:
            lines.append(f"    - {note}")

    lines.append("\n**Age Merge Decision**")
    age_details = get_age_bucket_details(agent1_json or {})
    if age_details:
        for item in age_details:
            final_val = item.get("final_value")
            final_text = f"{final_val:.1f}%" if isinstance(final_val, (int, float)) else "—"
            status = "included" if item.get("included") else "skipped"
            lines.append(
                f"- {item.get('label')} ({item.get('key')}): {final_text} ← {item.get('source')} ({status})"
            )
            if item.get("notes"):
                lines.append(f"    - {item['notes']}")
    else:
        lines.append("- no age buckets detected")

    prompt_trace = prompt_trace or {}
    lines.append("\n**Agent-2 Prompt**")
    lines.append(f"- question_type: {question_type or prompt_trace.get('question_type')}")
    lines.append(f"- organizer_mode: {prompt_trace.get('organizer_mode')}")
    lines.append(f"- schema_keys: {prompt_trace.get('schema_keys')}")
    lines.append(
        f"- rag_context_included: {prompt_trace.get('rag_included')}"
        + (f" (reason: {prompt_trace.get('rag_reason')})" if prompt_trace.get("rag_reason") else "")
    )
    if prompt_trace.get("rag_context_doc_ids"):
        lines.append(f"- rag doc_ids: {prompt_trace.get('rag_context_doc_ids')}")

    response_trace = response_trace or {}
    lines.append("\n**Agent-2 Response Trace**")
    if not response_trace:
        lines.append("- response trace unavailable")
    else:
        status = response_trace.get("status")
        lines.append(f"- status: {status}")
        if response_trace.get("generation_config"):
            lines.append(f"- generation_config: {response_trace.get('generation_config')}")
        if response_trace.get("prompt_length") is not None:
            lines.append(f"- prompt_length: {response_trace.get('prompt_length')}")
        if response_trace.get("chosen_model"):
            lines.append(f"- chosen_model: {response_trace.get('chosen_model')}")
        if response_trace.get("last_error"):
            lines.append(f"- last_error: {response_trace.get('last_error')}")
        if response_trace.get("fallback_answers") is not None:
            lines.append(f"- fallback_answers: {response_trace.get('fallback_answers')}")
        attempts = response_trace.get("attempts") or []
        if attempts:
            lines.append("- attempts:")
            for attempt in attempts:
                label = f"attempt {attempt.get('attempt')} / {attempt.get('model')}"
                lines.append(f"    - {label}: {attempt.get('status')}")
                parse_logs = attempt.get("parse_logs") or []
                excerpt = ""
                for entry in parse_logs:
                    if not entry.get("success") and entry.get("excerpt"):
                        excerpt = entry["excerpt"].replace('\n', ' ')[:160]
                        break
                if excerpt:
                    lines.append(f"        · excerpt: {excerpt}")
                if attempt.get("raw_preview"):
                    preview = attempt.get("raw_preview").replace('\n', ' ')
                    lines.append(f"        · raw_preview: {preview[:160]}")
                if attempt.get("repair_logs"):
                    rep_logs = attempt["repair_logs"]
                    for entry in rep_logs:
                        if not entry.get("success") and entry.get("excerpt"):
                            snippet = entry["excerpt"].replace('\n', ' ')[:160]
                            lines.append(f"        · repair excerpt: {snippet}")
                            break

    rag_info = rag_info or {}
    lines.append("\n**RAG Retrieval**")
    if not rag_info:
        lines.append("- RAG info unavailable")
    else:
        lines.append(
            f"- requested={rag_info.get('requested')} enabled={rag_info.get('enabled')} selection_missing={rag_info.get('selection_missing')}"
        )
        lines.append(
            f"- selected_doc_ids: {rag_info.get('selected_doc_ids')} catalog_size={rag_info.get('catalog_size')}"
        )
        lines.append(
            f"- mode={rag_info.get('mode')} threshold={rag_info.get('threshold')} top_k={rag_info.get('top_k')}"
        )
        lines.append(
            f"- max_score={rag_info.get('max_score')} include_evidence={rag_info.get('include_evidence')} error={rag_info.get('error')}"
        )
        payload = rag_info.get("payload") or {}
        top_scores = rag_prompt_context.get("top_scores") if isinstance(rag_prompt_context, dict) else []
        if not top_scores:
            top_scores = payload.get("top_scores") or []
        if top_scores:
            rounded = [round(float(score), 3) for score in top_scores[:5]]
            lines.append(f"- top_scores: {rounded}")
        if not rag_info.get("include_evidence"):
            reason = ""
            max_score = rag_info.get("max_score")
            threshold = rag_info.get("threshold")
            if rag_info.get("error"):
                reason = rag_info.get("error")
            elif max_score is None:
                reason = "no hits"
            elif threshold is not None and max_score < threshold:
                reason = f"max_score {max_score:.3f} < threshold {threshold:.2f}"
            else:
                reason = "evidence suppressed"
            lines.append(f"- evidence omitted reason: {reason}")

    return "\n".join(lines)


def _match_evidence_chunk(
    entry: Dict[str, Any],
    retrieval_payload: Dict[str, Any] | None,
) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    if not isinstance(entry, dict) or not retrieval_payload:
        return None, None
    doc_id = entry.get("doc_id")
    if not doc_id:
        return None, None
    chunk_id = entry.get("chunk_id")
    chunks = retrieval_payload.get("chunks") or []
    evidence_list = retrieval_payload.get("evidence") or []
    target_chunk = None
    target_meta = None
    chunk_id_str = str(chunk_id) if chunk_id is not None else None

    for chunk in chunks:
        if str(chunk.get("doc_id")) != str(doc_id):
            continue
        current_chunk_id = chunk.get("chunk_id")
        if chunk_id_str is not None and str(current_chunk_id) != chunk_id_str:
            continue
        target_chunk = chunk
        break

    if target_chunk is None:
        for chunk in chunks:
            if str(chunk.get("doc_id")) == str(doc_id):
                target_chunk = chunk
                break

    if target_chunk is not None:
        for meta in evidence_list:
            if str(meta.get("doc_id")) != str(doc_id):
                continue
            if chunk_id_str is not None:
                if str(meta.get("chunk_id")) == chunk_id_str:
                    target_meta = meta
                    break
            else:
                target_meta = meta
                break

    return target_chunk, target_meta


def _get_debug_raw_snapshot(agent1_json: dict | None) -> dict:
    debug = _get_debug_section(agent1_json)
    snap = debug.get("snapshot")
    if isinstance(snap, dict):
        raw = snap.get("raw")
        if isinstance(raw, dict):
            return raw
    legacy = debug.get("latest_raw_snapshot")
    return legacy if isinstance(legacy, dict) else {}


def _render_main_views(
    question_text: str,
    agent1_payload: dict | None,
    agent2_payload: dict | None,
    *,
    rag_info_override: Dict[str, Any] | None = None,
) -> None:
    flags_snapshot = st.session_state.get("_data_flags", {}).copy()
    structured_payload = structured_adapter(agent1_payload)
    weather_payload = weather_adapter(question_text, enabled=flags_snapshot.get("use_weather", False))
    external_payload = external_adapter(question_text, enabled=flags_snapshot.get("use_external", False))
    if rag_info_override is not None:
        rag_info = rag_info_override
    else:
        rag_info = _compute_rag_info(question_text, flags_snapshot)

    retrieval_payload = rag_info.get("payload") if isinstance(rag_info, dict) else None
    if rag_info.get("error"):
        st.warning(f"RetrievalTool 오류: {rag_info['error']}")
    if retrieval_payload is not None:
        st.session_state['_latest_retrieval'] = retrieval_payload
    else:
        st.session_state['_latest_retrieval'] = None
    st.session_state['_latest_rag_info'] = rag_info
    st.session_state['_latest_rag_prompt_context'] = _prepare_rag_prompt_context(rag_info)

    fail_soft_payload = compose_fail_soft_answer(
        question_text,
        structured_payload,
        weather_payload,
        external_payload,
        rag_info,
        flags_snapshot,
    )
    st.session_state['_latest_failsoft'] = fail_soft_payload

    try:
        overview_cached = st.session_state.get('_latest_overview', (None, None))
        if isinstance(agent2_payload, dict):
            if retrieval_payload:
                agent2_payload.setdefault("evidence", retrieval_payload.get("evidence", []))
                agent2_payload.setdefault("retrieval_chunks", retrieval_payload.get("chunks", []))
            agent2_payload.setdefault("used_data", fail_soft_payload.get("used_data"))
            if APP_MODE == "debug":
                agent2_payload.setdefault("caveats", fail_soft_payload.get("caveats"))

        if APP_MODE == "debug" and show_debug:
            render_fail_soft_answer(fail_soft_payload, rag_info=rag_info)
        render_summary_view(
            agent1_payload,
            agent2_payload or {},
            overview_df=overview_cached[0],
            table_dict=overview_cached[1],
            retrieval_payload=retrieval_payload,
        )
    except Exception:
        st.error("요약 뷰를 렌더링하는 중 오류가 발생했습니다.")
        st.code(traceback.format_exc())

def render_debug_view(agent1_json: dict | None, show_raw: bool = DEBUG_SHOW_RAW) -> None:
    debug = _get_debug_section(agent1_json)
    if not debug:
        st.info("디버그 정보가 없습니다.")
        return

    question_type = st.session_state.get("_latest_question_type")
    rag_info_state = st.session_state.get("_latest_rag_info")
    rag_prompt_state = st.session_state.get("_latest_rag_prompt_context")
    prompt_trace = st.session_state.get("_latest_prompt_trace")
    response_trace = st.session_state.get("_latest_response_trace")
    report_markdown = _build_debug_report_markdown(
        agent1_json,
        question_type=question_type,
        rag_info=rag_info_state,
        rag_prompt_context=rag_prompt_state,
        prompt_trace=prompt_trace,
        response_trace=response_trace,
    )
    if report_markdown:
        st.markdown(report_markdown)

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
    age_details = get_age_bucket_details(agent1_json or {})
    if age_details:
        st.caption("연령 분포 결정")
        st.table(pd.DataFrame(age_details))

    render_info = debug.get("render", {}) or {}
    table_dict = render_info.get("table_dict")
    if isinstance(table_dict, dict) and table_dict:
        st.markdown("#### 렌더 테이블")
        st.table(pd.DataFrame([table_dict]))

    rag_info_state = st.session_state.get("_latest_rag_info")
    rag_prompt_state = st.session_state.get("_latest_rag_prompt_context") or {}
    st.markdown("#### RAG 상태")
    if isinstance(rag_info_state, dict):
        summary = {
            "requested": rag_info_state.get("requested"),
            "enabled": rag_info_state.get("enabled"),
            "selected_doc_ids": rag_info_state.get("selected_doc_ids"),
            "threshold": rag_info_state.get("threshold"),
            "max_score": rag_info_state.get("max_score"),
            "mode": rag_info_state.get("mode"),
            "error": rag_info_state.get("error"),
        }
        st.write(summary)
        payload = rag_info_state.get("payload") or {}
        chunk_rows = []
        for chunk in (payload.get("chunks") or [])[:5]:
            chunk_rows.append(
                {
                    "doc_id": chunk.get("doc_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "score": chunk.get("score"),
                }
            )
        if chunk_rows:
            st.table(pd.DataFrame(chunk_rows))
        else:
            st.write("근거 없음 또는 임계값 미달")
        prompt_note = rag_prompt_state.get("prompt_note")
        if prompt_note:
            st.caption(_shorten_snippet(prompt_note, limit=200))
    else:
        st.write("RAG 정보가 없습니다.")

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


def _render_sources_footer(used_data: Dict[str, Any]) -> None:
    if not used_data:
        return

    st.caption("Sources Used")
    cols = st.columns(4)
    structured_text = "✅ Structured" if used_data.get("structured") else "⚪️ Structured"
    weather_text = "✅ Weather" if used_data.get("weather") else "⚪️ Weather"
    external_text = "✅ External" if used_data.get("external") else "⚪️ External"
    rag_info = used_data.get("rag") or {}
    rag_enabled = bool(rag_info.get("enabled"))
    rag_hits = rag_info.get("hits") or 0
    rag_label = "✅ RAG" if rag_enabled and rag_hits else ("⚪️ RAG" if rag_enabled else "🚫 RAG")
    rag_details = []
    if rag_hits:
        rag_details.append(f"hits={rag_hits}")
    max_score = rag_info.get("max_score")
    if isinstance(max_score, (int, float)):
        rag_details.append(f"max={max_score:.2f}")
    threshold = rag_info.get("threshold")
    if isinstance(threshold, (int, float)):
        rag_details.append(f"θ={threshold:.2f}")
    mode = rag_info.get("mode")
    if mode:
        rag_details.append(str(mode))
    selected_docs = rag_info.get("selected_docs") or []
    if selected_docs:
        rag_details.append(f"docs={len(selected_docs)}")
    elif rag_info.get("requested") and not rag_enabled:
        rag_details.append("docs=0")
    rag_text = rag_label + (" (" + ", ".join(rag_details) + ")" if rag_details else "")

    cols[0].markdown(structured_text)
    cols[1].markdown(weather_text)
    cols[2].markdown(external_text)
    cols[3].markdown(rag_text)


def render_fail_soft_answer(
    payload: Optional[Dict[str, Any]],
    *,
    rag_info: Optional[Dict[str, Any]] = None,
) -> None:
    st.subheader("Fail-soft 응답")
    if not payload:
        st.info("응답이 생성되지 않았습니다.")
        return

    segments = payload.get("segments") or []
    if not segments:
        st.info("응답이 생성되지 않았습니다.")
    rag_error = (rag_info or {}).get("error") if rag_info else None

    for segment in segments:
        cols = st.columns([12, 1])
        with cols[0]:
            st.markdown(f"- {segment.get('text')}")
        with cols[1]:
            evidence_chunk = segment.get("evidence")
            evidence_meta = segment.get("evidence_meta")
            source = segment.get("source")
            if evidence_chunk:
                _render_evidence_badge(evidence_chunk, evidence_meta)
            elif source == "rag":
                rag_enabled = bool((rag_info or {}).get("enabled"))
                if rag_enabled:
                    _render_evidence_badge(None, None, tooltip="관련 근거 없음")
                elif rag_error:
                    _render_evidence_badge(None, None, tooltip=f"RAG 오류: {rag_error}")
                else:
                    _render_evidence_badge(None, None, disabled=True, tooltip="RAG 비활성화")

    caveats = payload.get("caveats") or []
    if caveats:
        unique_caveats = []
        for item in caveats:
            if item and item not in unique_caveats:
                unique_caveats.append(item)
        if unique_caveats:
            st.caption("주의 사항")
            for item in unique_caveats:
                st.markdown(f"- {item}")

    if rag_error and all("RAG 오류" not in item for item in caveats):
        st.error(f"RAG 오류: {rag_error}")

    _render_sources_footer(payload.get("used_data") or {})

def _render_evidence_badge(
    chunk: dict | None,
    evidence_meta: dict | None,
    *,
    disabled: bool = False,
    tooltip: str | None = None,
) -> None:
    if disabled:
        label = "📎 (OFF)"
        if tooltip:
            label += f" — {tooltip}"
        st.caption(label)
        return

    if not chunk:
        if tooltip:
            st.caption(f"📎 {tooltip}")
        else:
            st.write("")
        return

    popover_fn = getattr(st, "popover", None)
    score = chunk.get("score")
    score_text = f"{float(score):.3f}" if score is not None else "—"
    doc_id = chunk.get("doc_id") or "—"
    chunk_id = chunk.get("chunk_id") or "—"
    title = evidence_meta.get("title") if isinstance(evidence_meta, dict) else None
    uri = evidence_meta.get("uri") if isinstance(evidence_meta, dict) else None

    def _short_label(value: str) -> str:
        text = str(value or "")
        if len(text) <= 8:
            return text
        return text[:5] + "…"

    display_token = chunk.get("chunk_id") or chunk.get("doc_id") or "근거"
    badge_label = f"📎 {_short_label(display_token)}·{score_text}"

    body_lines = [f"**문서 제목:** {title or doc_id}"]
    body_lines.append(f"**문서 ID:** {doc_id}")
    body_lines.append(f"**Chunk ID:** {chunk_id}")
    body_lines.append(f"**유사도:** {score_text}")
    text = chunk.get("text") or "—"
    body_lines.append("\n**내용 발췌**\n")
    body_lines.append(text)
    if uri:
        body_lines.append(f"\n[원본 열기]({uri})")

    if callable(popover_fn):
        with popover_fn(badge_label):
            for line in body_lines:
                st.markdown(line)
    else:  # pragma: no cover - fallback for older Streamlit
        with st.expander(badge_label, expanded=False):
            for line in body_lines:
                st.markdown(line)


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


def _collect_major_customers(agent1_json: dict) -> str:
    buckets = merge_age_buckets(agent1_json or {})
    if not buckets:
        return "—"
    segments: list[str] = []
    for bucket in buckets[:3]:
        label = bucket.get("label") or bucket.get("key") or "—"
        value = bucket.get("value")
        if isinstance(value, (int, float)):
            segments.append(f"{label} {value:.1f}%")
        else:
            segments.append(f"{label} —")
    return ", ".join(segments) if segments else "—"


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


def _build_diagnosis(agent1_json: dict) -> List[str]:
    try:
        return three_line_diagnosis(agent1_json or {})
    except Exception:
        return ["요약 생성 오류", "데이터 확인 필요", "—"]


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


def render_summary_view(
    agent1_json: dict,
    agent2_json: dict,
    overview_df: pd.DataFrame | None = None,
    table_dict: dict | None = None,
    retrieval_payload: dict | None = None,
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
    is_public_mode = APP_MODE == "public"
    try:
        print("📊 overview_table:", json.dumps(overview_df.to_dict(orient="records"), ensure_ascii=False))
    except Exception:
        pass

    if is_public_mode:
        st.subheader("현황 요약")
        if table_dict:
            summary_df = pd.DataFrame(list(table_dict.items()), columns=["항목", "값"]).head(3)
            st.table(summary_df)
        else:
            st.info("요약 정보가 없습니다.")
    else:
        st.subheader("현황 표")
        st.table(overview_df)

    st.subheader("한 줄 진단")
    diagnosis_lines = _build_diagnosis(agent1_json)
    if not diagnosis_lines:
        diagnosis_lines = ["요약 정보 없음", "—", "—"]
    if retrieval_payload and retrieval_payload.get("chunks"):
        best_chunk = retrieval_payload["chunks"][0]
        evidence_meta = None
        for item in retrieval_payload.get("evidence", []):
            if item.get("doc_id") == best_chunk.get("doc_id") and item.get("chunk_id") == best_chunk.get("chunk_id"):
                evidence_meta = item
                break
        cols = st.columns([12, 1])
        with cols[0]:
            for line in diagnosis_lines:
                st.markdown(f"- {line}")
        with cols[1]:
            _render_evidence_badge(best_chunk, evidence_meta)
    else:
        for line in diagnosis_lines:
            st.markdown(f"- {line}")

    if not is_public_mode:
        period_text, goal_lines = _build_goal_lines(agent1_json)
        st.subheader("목표")
        st.markdown(f"- 기간 가정: {period_text}")
        for line in goal_lines:
            st.markdown(f"- {line}")

    agent2_payload = agent2_json or {}
    answers = agent2_payload.get("answers") or agent2_payload.get("recommendations") or []
    if not isinstance(answers, list):
        answers = []

    st.subheader("아이디어 제안")
    if not answers:
        st.info("아이디어 제안이 제공되지 않았습니다.")
    for idx, answer in enumerate(answers[:4], start=1):
        with st.container():
            st.markdown(f"**{idx}. {answer.get('idea_title', '—')}**")
            st.markdown(f"- 대상: {answer.get('audience', '—')}")
            st.markdown(f"- 채널: {_format_list(answer.get('channels'))}")
            st.markdown(f"- 실행: {_format_list(answer.get('execution'))}")
            st.markdown(f"- 카피 샘플: {_format_list(answer.get('copy_samples'))}")
            st.markdown(f"- 측정: {_format_list(answer.get('measurement'))}")

            evidence_items = answer.get("evidence") or []
            if evidence_items:
                st.markdown("**근거**")
                for entry in evidence_items:
                    line = _format_evidence_line(entry)
                    cols = st.columns([12, 1])
                    with cols[0]:
                        st.markdown(line)
                    with cols[1]:
                        source = str(entry.get("source") or "").upper()
                        if source == "RAG":
                            chunk, meta = _match_evidence_chunk(entry, retrieval_payload)
                            if chunk:
                                _render_evidence_badge(chunk, meta)
                            else:
                                _render_evidence_badge(
                                    None,
                                    None,
                                    disabled=True,
                                    tooltip="RAG 근거 매칭 실패",
                                )
                        elif source in {"STRUCTURED", "WEATHER", "EXTERNAL"}:
                            _render_evidence_badge(
                                None,
                                None,
                                disabled=True,
                                tooltip=f"{source} 근거",
                            )
                        else:
                            tooltip = "근거 없음" if source in {"NONE", ""} else source
                            _render_evidence_badge(None, None, disabled=True, tooltip=tooltip)
            else:
                st.markdown("**근거**")
                cols = st.columns([12, 1])
                with cols[0]:
                    st.markdown("- 근거 없음")
                with cols[1]:
                    _render_evidence_badge(None, None, disabled=True, tooltip="근거 없음")

    if not is_public_mode:
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

        if retrieval_payload and retrieval_payload.get("chunks"):
            st.subheader("임베디드 근거")
            for idx, chunk in enumerate(retrieval_payload.get("chunks", [])[:5], start=1):
                meta = None
                for item in retrieval_payload.get("evidence", []):
                    if item.get("doc_id") == chunk.get("doc_id") and item.get("chunk_id") == chunk.get("chunk_id"):
                        meta = item
                        break
                cols = st.columns([12, 1])
                with cols[0]:
                    preview = str(chunk.get("text") or "—")
                    preview = preview.strip()
                    if len(preview) > 160:
                        preview = preview[:160].rstrip() + "…"
                    st.markdown(f"{idx}. {preview}")
                with cols[1]:
                    _render_evidence_badge(chunk, meta)

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

data_flags = st.session_state.get("_data_flags", {})
st.sidebar.header("Data Sources")
mode_options = ["public", "debug"]
mode_labels = {"public": "Public", "debug": "Debug"}
mode_index = mode_options.index(APP_MODE) if APP_MODE in mode_options else 0
selected_mode = st.sidebar.selectbox(
    "App Mode",
    mode_options,
    index=mode_index,
    format_func=lambda val: mode_labels.get(val, val),
)
if selected_mode != APP_MODE:
    st.session_state["_app_mode"] = selected_mode
    if selected_mode != "debug":
        st.session_state["show_debug_checkbox"] = False
    st.experimental_rerun()

data_flags["use_weather"] = st.sidebar.toggle(
    "Use Weather Data",
    value=bool(data_flags.get("use_weather", False)),
)
data_flags["use_external"] = st.sidebar.toggle(
    "Use External APIs",
    value=bool(data_flags.get("use_external", False)),
)

rag_root_exists = RAG_ROOT_PATH.exists()
if rag_root_exists:
    st.sidebar.caption(f"RAG Root: {RAG_ROOT_PATH}")
else:
    st.sidebar.info(f"RAG_ROOT 경로({RAG_ROOT_PATH})가 없어 RAG를 사용할 수 없습니다.")

if RAG_CATALOG_ERROR:
    st.sidebar.warning(RAG_CATALOG_ERROR)

rag_toggle_disabled = (
    RETRIEVAL_TOOL is None
    or RETRIEVAL_INIT_ERROR is not None
    or not rag_root_exists
    or RAG_CATALOG_ERROR is not None
)
if rag_toggle_disabled:
    data_flags["use_rag"] = False
    st.sidebar.toggle(
        "Use RAG (Embedded Docs)",
        value=False,
        disabled=True,
        help="RetrievalTool이 준비되지 않아 비활성화되었습니다.",
    )
    data_flags["rag_selected_ids"] = []
else:
    data_flags["use_rag"] = st.sidebar.toggle(
        "Use RAG (Embedded Docs)",
        value=bool(data_flags.get("use_rag", True)),
    )

rag_threshold_default = float(data_flags.get("rag_threshold", 0.35))
data_flags["rag_threshold"] = float(
    st.sidebar.slider(
        "RAG Threshold",
        min_value=0.2,
        max_value=0.6,
        value=rag_threshold_default,
        step=0.05,
        disabled=rag_toggle_disabled,
    )
)
data_flags["rag_top_k"] = int(
    st.sidebar.slider(
        "RAG top_k",
        min_value=3,
        max_value=10,
        value=int(data_flags.get("rag_top_k", 5)),
        step=1,
        disabled=rag_toggle_disabled,
    )
)
rag_modes = ["auto", "always", "off"]
rag_mode_value = data_flags.get("rag_mode", "auto")
if rag_mode_value not in rag_modes:
    rag_mode_value = "auto"
data_flags["rag_mode"] = st.sidebar.selectbox(
    "RAG Mode",
    options=rag_modes,
    index=rag_modes.index(rag_mode_value),
    disabled=rag_toggle_disabled,
)

if not rag_toggle_disabled and data_flags.get("use_rag"):
    current_filter = data_flags.get("rag_filter", "")
    current_filter = st.sidebar.text_input(
        "RAG Search Filter (optional)",
        value=current_filter,
        placeholder="문서명, 태그, ID 검색",
    )
    data_flags["rag_filter"] = current_filter

    def _matches_filter(record: Dict[str, Any], query: str) -> bool:
        if not query:
            return True
        haystack_parts = [
            str(record.get("title") or ""),
            str(record.get("document_id") or ""),
        ]
        tags = record.get("tags") or []
        haystack_parts.extend(str(tag) for tag in tags)
        haystack = " ".join(haystack_parts).lower()
        return query.lower() in haystack

    filtered_catalog = [
        item for item in RAG_CATALOG if _matches_filter(item, current_filter.strip())
    ]

    if not filtered_catalog:
        st.sidebar.info("필터와 일치하는 문서가 없습니다. 필터를 비워주세요.")

    label_map = {
        f"{item['title']} ({item['document_id']})": item["document_id"]
        for item in filtered_catalog
    }
    options = list(label_map.keys())

    previous_selection = data_flags.get("rag_selected_ids") or []
    default_labels = [label for label, doc_id in label_map.items() if doc_id in previous_selection]
    if not default_labels and not previous_selection and len(options) <= 20:
        default_labels = options

    selected_labels = st.sidebar.multiselect(
        "Select RAG Documents",
        options,
        default=default_labels,
    )
    selected_ids = [label_map[label] for label in selected_labels]
    data_flags["rag_selected_ids"] = selected_ids

    if data_flags.get("use_rag") and not selected_ids:
        st.sidebar.info("선택된 RAG 문서가 없어 이번 실행에서는 RAG를 사용하지 않습니다.")
elif not rag_toggle_disabled:
    # Toggle가 꺼져 있으면 기존 선택을 유지하되 필터 입력만 초기화하지 않습니다.
    data_flags.setdefault("rag_selected_ids", data_flags.get("rag_selected_ids", []))

st.session_state["_data_flags"] = data_flags

# ===== 탭 구성 =====
analysis_tab, sources_tab = st.tabs(["📈 분석", "📚 Embedded Sources"])

with analysis_tab:
    default_q = "성동구 {고향***} 기준으로, 재방문율 4주 플랜 작성해줘."
    question = st.text_input("질문을 입력하세요", value=default_q)
    st.caption("상호는 반드시 {}로 감싸 주세요. 예) 성동구 {동대******}")

    run_analysis = st.button("분석 실행", type="primary")
    if run_analysis:
        from bigcon_2agent_mvp_v3 import (
            agent1_pipeline,
            build_agent2_prompt,
            call_gemini_agent2,
            AGENT2_PROMPT_TRACE,
            AGENT2_RESPONSE_TRACE,
            infer_question_type,
        )

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
                st.session_state['_latest_question'] = question
                st.success("Agent-1 JSON 생성 완료")
            except Exception:
                st.error("Agent-1 실행 오류")
                st.code(traceback.format_exc())
                st.stop()

        flags_snapshot = st.session_state.get("_data_flags", {}).copy()
        rag_info_for_prompt = _compute_rag_info(question, flags_snapshot)
        rag_prompt_context = _prepare_rag_prompt_context(rag_info_for_prompt)
        st.session_state['_latest_rag_info'] = rag_info_for_prompt
        question_type = infer_question_type(question)
        st.session_state['_latest_question_type'] = question_type
        st.session_state['_latest_rag_prompt_context'] = rag_prompt_context

        with st.spinner("Agent-2: 카드 생성 중..."):
            try:
                os.environ["GEMINI_API_KEY"] = API_KEY
                prompt_text = build_agent2_prompt(
                    a1,
                    question_text=question,
                    question_type=question_type,
                    rag_context=rag_prompt_context,
                )
                if isinstance(AGENT2_PROMPT_TRACE, dict):
                    st.session_state["_latest_prompt_trace"] = dict(AGENT2_PROMPT_TRACE)
                else:
                    st.session_state["_latest_prompt_trace"] = {}
                result = call_gemini_agent2(
                    prompt_text,
                    question_type=question_type,
                    agent1_json=a1,
                )
                st.success("Agent-2 카드 생성 완료")
                st.session_state['_latest_agent2'] = result
                if isinstance(AGENT2_RESPONSE_TRACE, dict):
                    st.session_state['_latest_response_trace'] = dict(AGENT2_RESPONSE_TRACE)
                else:
                    st.session_state['_latest_response_trace'] = {}
            except Exception:
                st.error("Agent-2 실행 오류")
                st.code(traceback.format_exc())
                st.stop()

        _render_main_views(question, a1, result, rag_info_override=rag_info_for_prompt)

    elif st.session_state.get('_latest_agent1') and st.session_state.get('_latest_agent2'):
        latest_agent1 = st.session_state.get('_latest_agent1')
        latest_agent2 = st.session_state.get('_latest_agent2')
        question_snapshot = st.session_state.get('_latest_question', question)
        _render_main_views(question_snapshot, latest_agent1, latest_agent2)

    latest_agent2 = st.session_state.get('_latest_agent2')
    if isinstance(latest_agent2, dict):
        with st.expander("🧾 Agent-2 출력(JSON) 보기", expanded=False):
            st.json(latest_agent2)
    latest_retrieval = st.session_state.get('_latest_retrieval')
    if latest_retrieval:
        with st.expander("📎 Retrieval Evidence (JSON)", expanded=False):
            st.json(latest_retrieval)
    latest_agent1 = st.session_state.get('_latest_agent1')
    if isinstance(latest_agent1, dict):
        with st.expander("🔎 Agent-1 출력(JSON) 보기", expanded=False):
            st.json(latest_agent1)

    rag_flags = st.session_state.get("_data_flags", {})
    if RETRIEVAL_TOOL is not None and RETRIEVAL_INIT_ERROR is None:
        col_lo, col_hi = st.columns(2)
        if col_lo.button("임계값 낮추기 (-0.05)"):
            new_threshold = max(0.2, float(rag_flags.get("rag_threshold", 0.35)) - 0.05)
            st.session_state["_data_flags"]["rag_threshold"] = round(new_threshold, 2)
            st.experimental_rerun()
        if col_hi.button("임계값 높이기 (+0.05)"):
            new_threshold = min(0.6, float(rag_flags.get("rag_threshold", 0.35)) + 0.05)
            st.session_state["_data_flags"]["rag_threshold"] = round(new_threshold, 2)
            st.experimental_rerun()

    if show_debug:
        latest_agent1 = st.session_state.get('_latest_agent1')
        with st.expander("🔍 디버그 상세", expanded=True):
            render_debug_view(latest_agent1, show_raw=DEBUG_SHOW_RAW)

    if not st.session_state.get("_intro_shown"):
        st.info("✅ 업로드 성공! 이제 질문 입력 후 [분석 실행]을 눌러 카드 결과를 확인해보세요.")
        st.session_state["_intro_shown"] = True

with sources_tab:
    st.subheader("임베디드 소스 카탈로그")
    st.caption(f"RAG Root: {RAG_ROOT_PATH}")
    if RETRIEVAL_INIT_ERROR:
        st.error(f"RetrievalTool 초기화 실패: {RETRIEVAL_INIT_ERROR}")
    elif RETRIEVAL_TOOL is None:
        st.info("RetrievalTool이 비활성화되어 있습니다.")
    elif not RAG_ROOT_PATH.exists():
        st.info("data/rag 경로가 존재하지 않습니다. corpus/와 indices/를 구성한 뒤 다시 시도하세요.")
    elif RAG_CATALOG_ERROR:
        st.warning(RAG_CATALOG_ERROR)
    else:
        catalog_df = pd.DataFrame(RAG_CATALOG)
        if catalog_df.empty:
            st.info("등록된 임베디드 문서가 없습니다. data/rag/indices 폴더를 확인하세요.")
        else:
            display_columns = [
                "title",
                "document_id",
                "num_chunks",
                "embedding_model",
                "created_at",
                "origin_path",
            ]
            optional_cols = [col for col in ["tags", "year"] if col in catalog_df.columns]
            st.dataframe(
                catalog_df[display_columns + optional_cols],
                use_container_width=True,
            )
            doc_ids = catalog_df["document_id"].tolist()
            selected_doc = st.selectbox("미리보기 문서", doc_ids, index=0 if doc_ids else None)
            if selected_doc:
                preview_chunks = RETRIEVAL_TOOL.preview_chunks(selected_doc)
                manifest_row = catalog_df[catalog_df["document_id"] == selected_doc].iloc[0]
                st.markdown(f"**문서 제목:** {manifest_row['title']}")
                origin_path = manifest_row.get("origin_path")
                if origin_path:
                    st.markdown(f"[원본 열기]({origin_path})")
                tags = manifest_row.get("tags") or []
                if tags:
                    st.caption("태그: " + ", ".join(str(tag) for tag in tags))
                if not preview_chunks:
                    st.info("프리뷰 가능한 청크가 없습니다.")
                else:
                    for chunk in preview_chunks:
                        with st.expander(f"Chunk {chunk.get('chunk_id')}"):
                            st.write(chunk.get("text") or "—")
