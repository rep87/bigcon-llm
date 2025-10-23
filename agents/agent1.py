"""Rule-based diagnostics for selecting metrics for Agent-2."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

from data.loader import profile_columns

APP_VERSION = "2024.09-data-provenance"
TELEMETRY_DIR = Path(__file__).resolve().parent.parent / "telemetry"

MIN_COVERAGE_BY_QUESTION = {
    "Q1": 0.6,
    "Q2": 0.5,
    "Q3": 0.5,
}

QUESTION_TAGS = {
    "Q1": {"age", "gender", "workplace", "residence", "floating", "channel", "revisit", "acquisition"},
    "Q2": {"revisit", "retention", "acquisition", "workplace", "residence"},
    "Q3": {"sales", "volume", "ticket", "delivery", "revisit", "acquisition", "quality"},
}

KEYWORD_TAGS = {
    "재방문": "revisit",
    "신규": "acquisition",
    "직장": "workplace",
    "주거": "residence",
    "유동": "floating",
    "외지": "floating",
    "배달": "delivery",
    "매출": "sales",
    "객단가": "ticket",
    "이용": "volume",
    "승인": "quality",
}

REASON_TEMPLATES = {
    "revisit": "질문에 재방문 조건이 있어 재방문율을 확인합니다.",
    "acquisition": "신규 고객 확보 여부를 판단하기 위한 지표입니다.",
    "workplace": "직장 생활권 고객 비중이 채널 선택에 영향을 줍니다.",
    "residence": "주거 생활권 고객 비중으로 상권 적합성을 판단합니다.",
    "floating": "외지·유동 고객 비중을 확인해 외부 유입 채널을 조정합니다.",
    "age": "연령대별 비중을 확인해 메시지 타깃을 정합니다.",
    "gender": "성별 비중을 확인해 카피 톤을 조정합니다.",
    "sales": "매출 규모를 파악해 제안 강도를 조절합니다.",
    "volume": "이용량 추세를 확인해 성장 기회를 점검합니다.",
    "ticket": "객단가를 확인해 프로모션 수준을 설정합니다.",
    "delivery": "배달 비중이 온라인 vs. 오프라인 채널 선택에 영향을 줍니다.",
    "quality": "승인율을 통해 결제 경험 품질을 확인합니다.",
}


@dataclass
class SelectionContext:
    question_types: Set[str]
    keyword_tags: Set[str]
    must_have_tags: Set[str]


def _infer_question_types(question: str) -> SelectionContext:
    question = question or ""
    question_types: Set[str] = set()
    if "재방문" in question or "채널" in question or "카페" in question:
        question_types.add("Q1")
    if "재방문" in question and ("30" in question or "삼십" in question):
        question_types.add("Q2")
    if "요식" in question or "문제" in question or "원인" in question or "매출" in question or "배달" in question:
        question_types.add("Q3")
    if not question_types:
        question_types.add("Q1")

    keyword_tags: Set[str] = set()
    for keyword, tag in KEYWORD_TAGS.items():
        if keyword in question:
            keyword_tags.add(tag)

    must_have_tags: Set[str] = set()
    if "재방문" in question:
        must_have_tags.add("revisit")
    if "직장" in question:
        must_have_tags.add("workplace")
    if "배달" in question:
        must_have_tags.add("delivery")

    return SelectionContext(question_types, keyword_tags, must_have_tags)


def _score_profile(entry: Dict[str, Any], context: SelectionContext) -> float:
    meta = entry.get("meta") or {}
    tags = set(meta.get("role_tags") or [])
    question_roles = set(meta.get("question_roles") or [])
    score = 0.0
    overlap = context.question_types & question_roles
    if overlap:
        score += 2.0 * len(overlap)
    coverage = float(entry.get("recent_coverage") or 0.0)
    score += coverage
    tag_targets: Set[str] = set()
    for q in context.question_types:
        tag_targets.update(QUESTION_TAGS.get(q, set()))
    score += 1.5 * len(tags & context.keyword_tags)
    score += 1.0 * len(tags & tag_targets)
    if entry.get("sufficiency") == "High":
        score += 0.5
    return score


def _passes_threshold(entry: Dict[str, Any], context: SelectionContext) -> bool:
    coverage = float(entry.get("recent_coverage") or 0.0)
    meta = entry.get("meta") or {}
    question_roles = set(meta.get("question_roles") or [])
    for q in context.question_types:
        if q in question_roles:
            required = MIN_COVERAGE_BY_QUESTION.get(q, 0.5)
            if coverage <= 0:
                return False
            if coverage < required:
                return False
    return True


def _pick_reason_tag(entry: Dict[str, Any], context: SelectionContext) -> str | None:
    meta = entry.get("meta") or {}
    tags = list(meta.get("role_tags") or [])
    for tag in tags:
        if tag in context.keyword_tags:
            return tag
    for tag in tags:
        for q in context.question_types:
            if tag in QUESTION_TAGS.get(q, set()):
                return tag
    return tags[0] if tags else None


def _build_reason(label: str, tag: str | None, question: str) -> str:
    if not tag:
        return f"{label} 지표가 질문 맥락과 직접 연결됩니다."
    if tag == "revisit" and "30" in question:
        return "질문이 ‘재방문율 30% 이하’ 조건이므로 핵심 지표입니다."
    template = REASON_TEMPLATES.get(tag)
    if template:
        return template
    return f"{label} 지표가 질문에 언급된 핵심 키워드와 연결됩니다."


def _ensure_telemetry_dir() -> None:
    TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)


def _write_telemetry(payload: Dict[str, Any], merchant_id: str) -> None:
    _ensure_telemetry_dir()
    timestamp = payload.get("timestamp")
    if not timestamp:
        from datetime import datetime

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        payload["timestamp"] = timestamp
    filename = TELEMETRY_DIR / f"run_{merchant_id}_{timestamp}.json"
    with filename.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def diagnose_and_select(question: str, merchant_id: str) -> Dict[str, Any]:
    context = _infer_question_types(question)
    profiles = profile_columns(merchant_id)

    scored: List[Dict[str, Any]] = []
    for entry in profiles:
        entry_meta = entry.get("meta") or {}
        if entry_meta.get("feature_id") is None:
            continue
        entry["unit"] = entry_meta.get("unit")
        entry["feature_id"] = entry_meta.get("feature_id")
        entry_score = _score_profile(entry, context)
        entry["_score"] = entry_score
        entry["_passes"] = _passes_threshold(entry, context)
        scored.append(entry)

    scored.sort(key=lambda x: x.get("_score", 0.0), reverse=True)

    selected: List[Dict[str, Any]] = []
    selected_tags: Set[str] = set()
    for entry in scored:
        if not entry.get("_passes"):
            continue
        if len(selected) >= 8:
            break
        selected.append(entry)
        meta = entry.get("meta") or {}
        selected_tags.update(meta.get("role_tags") or [])

    for must_tag in context.must_have_tags:
        if must_tag in selected_tags:
            continue
        candidates = [
            item
            for item in scored
            if must_tag in (item.get("meta", {}).get("role_tags") or [])
            and item not in selected
        ]
        if candidates:
            selected.append(candidates[0])
            selected_tags.update(candidates[0].get("meta", {}).get("role_tags") or [])

    selection_reasons: List[Dict[str, Any]] = []
    selected_features: List[Dict[str, Any]] = []
    sufficiency_low = 0

    for entry in profiles:
        entry["selected"] = entry in selected
        entry.pop("_score", None)
        entry.pop("_passes", None)
        if entry.get("selected"):
            meta = entry.get("meta") or {}
            tag = _pick_reason_tag(entry, context)
            reason = _build_reason(meta.get("label", entry.get("column", "")), tag, question)
            entry["reason"] = reason
            selection_reasons.append({"column": entry.get("column"), "reason": reason})
            selected_features.append(
                {
                    "id": meta.get("feature_id"),
                    "column": entry.get("column"),
                    "label": meta.get("label", entry.get("column")),
                    "source": entry.get("source"),
                    "period": meta.get("default_period", "recent_3m"),
                    "value": entry.get("value_recent"),
                    "latest_period": entry.get("recency"),
                    "time_span": entry.get("time_span"),
                }
            )
            if entry.get("sufficiency") == "Low":
                sufficiency_low += 1
        else:
            entry["reason"] = None

    sufficiency_summary = "데이터 충분"
    if selected_features and sufficiency_low / max(len(selected_features), 1) >= 0.4:
        sufficiency_summary = "데이터 부족"

    telemetry_payload = {
        "app_version": APP_VERSION,
        "question": question,
        "merchant_id": merchant_id,
        "profile_table": [
            {
                key: value
                for key, value in entry.items()
                if key not in {"meta"}
            }
            for entry in profiles
        ],
        "selected_features": selected_features,
        "selection_reasons": selection_reasons,
        "sufficiency_summary": sufficiency_summary,
    }

    from datetime import datetime

    telemetry_payload["timestamp"] = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    telemetry_payload["filters"] = {"question_types": sorted(context.question_types)}
    telemetry_payload["time_windows"] = {"recent_window": 3}

    _write_telemetry(telemetry_payload, merchant_id)

    return {
        "profile_table": [
            {key: value for key, value in entry.items() if key != "meta"}
            for entry in profiles
        ],
        "selected_features": selected_features,
        "selection_reasons": selection_reasons,
        "sufficiency_summary": sufficiency_summary,
    }
