"""Fail-soft composition helpers for the Streamlit consulting app."""

from __future__ import annotations

from typing import Any, Dict, Optional

SANITIZED_METRIC_MAP: Dict[str, str] = {
    "revisit_pct": "MCT_UE_CLN_REU_RAT",
    "new_pct": "MCT_UE_CLN_NEW_RAT",
    "youth_pct": "M12_MAL_1020_RAT",
    "flow_pct": "RC_M1_SHC_FLP_UE_CLN_RAT",
    "resident_pct": "RC_M1_SHC_RSD_UE_CLN_RAT",
    "work_pct": "RC_M1_SHC_WP_UE_CLN_RAT",
}

FAILSOFT_GENERIC_NOTE = "데이터 소스가 비활성화/부족하여 일반적 권고만 제공합니다."


def structured_adapter(agent1_json: Optional[dict]) -> Dict[str, Any]:
    debug = (agent1_json or {}).get("debug")
    snapshot = debug.get("snapshot") if isinstance(debug, dict) else {}
    sanitized = snapshot.get("sanitized") if isinstance(snapshot, dict) else {}
    raw_snapshot = snapshot.get("raw") if isinstance(snapshot, dict) else {}
    panel_info = debug.get("panel") if isinstance(debug, dict) else {}
    latest_period = None
    if isinstance(panel_info, dict):
        latest_period = panel_info.get("latest_ta_ym") or panel_info.get("latest_period")
    context = (agent1_json or {}).get("context") or {}
    kpis = (agent1_json or {}).get("kpis") or {}
    available = bool(sanitized or raw_snapshot or kpis)
    return {
        "available": available,
        "sanitized": sanitized if isinstance(sanitized, dict) else {},
        "raw": raw_snapshot if isinstance(raw_snapshot, dict) else {},
        "panel": panel_info if isinstance(panel_info, dict) else {},
        "latest_period": latest_period,
        "context": context,
        "kpis": kpis,
    }


def weather_adapter(question: str, *, enabled: bool) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    return {
        "enabled": True,
        "available": False,
        "note": "날씨 데이터 소스가 아직 연동되지 않았습니다.",
        "question": question,
    }


def external_adapter(question: str, *, enabled: bool) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    return {
        "enabled": True,
        "available": False,
        "note": "외부 API 연동이 비활성 상태입니다.",
        "question": question,
    }


def rag_adapter(
    question: str,
    retrieval_tool: Any,
    *,
    enabled: bool,
    top_k: int,
    threshold: float,
    mode: str,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": bool(enabled) and mode != "off" and retrieval_tool is not None,
        "mode": mode,
        "threshold": threshold,
        "top_k": top_k,
        "error": None,
        "payload": None,
        "available": False,
        "max_score": None,
        "include_evidence": False,
    }

    if not info["enabled"]:
        if info["requested"] and retrieval_tool is None:
            info["error"] = "RetrievalTool 미구성"
        return info

    try:
        payload = retrieval_tool.retrieve(
            question,
            top_k=int(top_k),
            threshold=threshold,
            mode=mode if mode in {"auto", "always"} else "auto",
        )
    except Exception as exc:  # pragma: no cover - UI safeguard
        info["error"] = str(exc)
        return info

    info["payload"] = payload
    info["available"] = bool(payload.get("chunks"))
    info["max_score"] = payload.get("max_score")
    info["include_evidence"] = bool(payload.get("include_evidence"))
    return info


def _format_metric_lines(structured: Dict[str, Any]) -> list[str]:
    sanitized = structured.get("sanitized") or {}
    period = structured.get("latest_period") or structured.get("panel", {}).get("latest_ta_ym") or "—"
    lines: list[str] = []
    for key in ["revisit_pct", "new_pct", "flow_pct", "resident_pct", "work_pct", "youth_pct"]:
        if key not in SANITIZED_METRIC_MAP:
            continue
        value = sanitized.get(key)
        if value is None:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        lines.append(f"{num:.1f}% {SANITIZED_METRIC_MAP[key]} ({period})")
    return lines[:4]


def compose_fail_soft_answer(
    query_text: str,
    structured: Dict[str, Any],
    weather: Optional[Dict[str, Any]],
    external: Optional[Dict[str, Any]],
    rag_info: Dict[str, Any],
    flags: Dict[str, Any],
) -> Dict[str, Any]:
    segments: list[Dict[str, Any]] = []
    caveats: list[str] = []
    evidence: list[Dict[str, Any]] = []

    structured_available = bool(structured.get("available"))
    if structured_available:
        metric_lines = _format_metric_lines(structured)
        if metric_lines:
            segments.append(
                {
                    "text": "핵심 KPI: " + ", ".join(metric_lines),
                    "source": "structured",
                    "evidence": None,
                    "evidence_meta": None,
                }
            )
        else:
            latest_period = structured.get("latest_period") or "—"
            segments.append(
                {
                    "text": f"구조화 KPI 확보 (최신월 {latest_period})",
                    "source": "structured",
                    "evidence": None,
                    "evidence_meta": None,
                }
            )
    else:
        caveats.append("구조화 KPI 미확정: Agent-1 결과를 확인하세요.")

    if weather is None:
        if not flags.get("use_weather"):
            caveats.append("날씨 데이터 OFF: 기상 관련 제안 제외")
    else:
        if weather.get("available"):
            note = weather.get("note") or "날씨 보조 지표 활용"
            segments.append(
                {
                    "text": f"날씨 참고: {note}",
                    "source": "weather",
                    "evidence": None,
                    "evidence_meta": None,
                }
            )
        else:
            caveats.append(weather.get("note") or "날씨 데이터 확보 불가")

    if external is None:
        if not flags.get("use_external"):
            caveats.append("외부 API OFF: 외부 참고 지표 제외")
    else:
        if external.get("available"):
            note = external.get("note") or "외부 데이터 참고"
            segments.append(
                {
                    "text": f"외부 참고: {note}",
                    "source": "external",
                    "evidence": None,
                    "evidence_meta": None,
                }
            )
        else:
            caveats.append(external.get("note") or "외부 API 응답 없음")

    rag_payload = rag_info.get("payload") or {}
    rag_chunks = rag_payload.get("chunks") or []
    rag_evidence_list = rag_payload.get("evidence") or []
    rag_enabled = bool(rag_info.get("enabled"))
    rag_requested = bool(rag_info.get("requested"))
    rag_threshold = flags.get("rag_threshold")
    if rag_chunks:
        for chunk in rag_chunks[:2]:
            preview = str(chunk.get("text") or "").strip()
            if len(preview) > 160:
                preview = preview[:160].rstrip() + "…"
            evidence_meta = None
            for item in rag_evidence_list:
                if item.get("doc_id") == chunk.get("doc_id") and item.get("chunk_id") == chunk.get("chunk_id"):
                    evidence_meta = item
                    break
            segments.append(
                {
                    "text": f"문서 근거: {preview}",
                    "source": "rag",
                    "evidence": chunk,
                    "evidence_meta": evidence_meta,
                }
            )
        evidence.extend(rag_evidence_list)
    else:
        if rag_info.get("error"):
            caveats.append(f"RAG 오류: {rag_info['error']}")
        elif rag_enabled and rag_threshold is not None:
            caveats.append(f"RAG 임계값 {rag_threshold:.2f}: 관련 근거 없음")
        elif rag_requested and not rag_enabled:
            caveats.append("RAG 데이터 OFF: 문서 근거 제외")

    if not rag_requested:
        msg = "RAG 데이터 OFF: 문서 근거 제외"
        if msg not in caveats:
            caveats.append(msg)

    used_data = {
        "structured": structured_available,
        "weather": bool(weather and weather.get("available")),
        "external": bool(external and external.get("available")),
        "rag": {
            "enabled": rag_enabled,
            "hits": len(rag_chunks),
            "max_score": rag_info.get("max_score"),
            "threshold": rag_threshold,
            "mode": rag_info.get("mode"),
        },
    }

    if not structured_available and not used_data["weather"] and not used_data["external"] and not rag_chunks:
        generic_text = f"{FAILSOFT_GENERIC_NOTE}"
        if query_text:
            generic_text = f"{query_text} 관련 {FAILSOFT_GENERIC_NOTE}"
        segments = [
            {
                "text": generic_text,
                "source": "generic",
                "evidence": None,
                "evidence_meta": None,
            }
        ]
        if FAILSOFT_GENERIC_NOTE not in caveats:
            caveats.append(FAILSOFT_GENERIC_NOTE)

    answer_text = "\n".join(segment["text"] for segment in segments)

    return {
        "answer": answer_text,
        "segments": segments,
        "used_data": used_data,
        "evidence": evidence,
        "caveats": caveats,
        "rag": {
            "enabled": rag_enabled,
            "requested": rag_requested,
            "payload": rag_payload,
        },
    }
