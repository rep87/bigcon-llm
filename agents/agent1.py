"""Rule-based diagnostics for selecting metrics for Agent-2."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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


def _normalize_percent_value(value: Any) -> float | None:
    try:
        if value is None:
            return None
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not (num == num):
        return None
    if num < 0:
        num = 0.0
    if num <= 1:
        return num * 100
    if num <= 100:
        return num
    return num / 100


def _entries_by_tag(
    profiles: List[Dict[str, Any]], selected: List[Dict[str, Any]], tag: str
) -> List[Dict[str, Any]]:
    matches = [
        entry
        for entry in selected
        if tag in (entry.get("meta", {}).get("role_tags") or [])
    ]
    if matches:
        return matches
    return [
        entry
        for entry in profiles
        if tag in (entry.get("meta", {}).get("role_tags") or [])
    ]


def _pick_percent(entry: Dict[str, Any]) -> Tuple[float | None, str]:
    unit = entry.get("unit") or entry.get("meta", {}).get("unit")
    value = entry.get("value_recent")
    if unit == "ratio" or unit is None:
        percent = _normalize_percent_value(value)
        return percent, "%"
    try:
        return float(value), entry.get("unit", "")
    except (TypeError, ValueError):
        return None, entry.get("unit", "")


def _generate_eda_derivatives(
    profiles: List[Dict[str, Any]], selected: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    derived_entries: List[Dict[str, Any]] = []
    extra_reasons: List[Dict[str, Any]] = []

    revisit_entries = _entries_by_tag(profiles, selected, "revisit")
    acquisition_entries = _entries_by_tag(profiles, selected, "acquisition")
    if revisit_entries and acquisition_entries:
        revisit = revisit_entries[0]
        acquisition = acquisition_entries[0]
        revisit_pct, _ = _pick_percent(revisit)
        acquisition_pct, _ = _pick_percent(acquisition)
        if revisit_pct is not None and acquisition_pct is not None:
            gap = revisit_pct - acquisition_pct
            reason = (
                f"재방문율 {revisit_pct:.2f}%와 신규 방문율 {acquisition_pct:.2f}% 격차 {gap:+.2f}%을 분석해 고객 유지·확장 균형을 진단합니다."
            )
            derived_entries.append(
                {
                    "source": "Derived (EDA)",
                    "column": "eda.revisit_vs_new",
                    "label": "재방문·신규 고객 조합",
                    "value_recent": gap,
                    "value_prev": None,
                    "coverage": 1.0,
                    "recency": revisit.get("recency") or acquisition.get("recency") or "최근 조합",
                    "time_span": "최근 3개월 조합",
                    "sufficiency": "High",
                    "warning": None,
                    "unit": "ratio",
                    "meta": {
                        "role_tags": ["eda", "revisit", "acquisition"],
                        "question_roles": list(
                            set(revisit.get("meta", {}).get("question_roles") or [])
                            | set(acquisition.get("meta", {}).get("question_roles") or [])
                        ),
                        "unit": "ratio",
                        "derived": True,
                        "label": "재방문·신규 고객 조합",
                    },
                    "custom_reason": reason,
                }
            )

    workplace_entries = _entries_by_tag(profiles, selected, "workplace")
    residence_entries = _entries_by_tag(profiles, selected, "residence")
    if workplace_entries and residence_entries:
        workplace = workplace_entries[0]
        residence = residence_entries[0]
        workplace_pct, _ = _pick_percent(workplace)
        residence_pct, _ = _pick_percent(residence)
        if workplace_pct is not None and residence_pct is not None:
            balance = workplace_pct / max(residence_pct, 0.01)
            reason = (
                f"직장권 {workplace_pct:.2f}% / 주거권 {residence_pct:.2f}% 구성으로 상권 집중도를 확인하고 채널 포지셔닝을 설정합니다."
            )
            derived_entries.append(
                {
                    "source": "Derived (EDA)",
                    "column": "eda.work_res_balance",
                    "label": "직장·주거 생활권 균형",
                    "value_recent": balance,
                    "value_prev": None,
                    "coverage": 1.0,
                    "recency": workplace.get("recency") or residence.get("recency") or "최근 조합",
                    "time_span": "최근 3개월 조합",
                    "sufficiency": "High",
                    "warning": None,
                    "unit": None,
                    "meta": {
                        "role_tags": ["eda", "workplace", "residence"],
                        "question_roles": list(
                            set(workplace.get("meta", {}).get("question_roles") or [])
                            | set(residence.get("meta", {}).get("question_roles") or [])
                        ),
                        "derived": True,
                        "label": "직장·주거 생활권 균형",
                    },
                    "custom_reason": reason,
                }
            )

    age_entries = _entries_by_tag(profiles, selected, "age")
    if len(age_entries) >= 2:
        ranked = [
            (
                entry,
                _pick_percent(entry)[0],
            )
            for entry in age_entries
        ]
        ranked = [item for item in ranked if item[1] is not None]
        ranked.sort(key=lambda item: item[1], reverse=True)
        if len(ranked) >= 2:
            top_entry, top_pct = ranked[0]
            second_entry, second_pct = ranked[1]
            reason = (
                f"핵심 연령층은 {top_entry.get('label', top_entry.get('column'))} {top_pct:.2f}%, 다음은 {second_entry.get('label', second_entry.get('column'))} {second_pct:.2f}%로 타깃 세그먼트를 확장합니다."
            )
            derived_entries.append(
                {
                    "source": "Derived (EDA)",
                    "column": "eda.age_focus",
                    "label": "핵심 연령대 집중도",
                    "value_recent": top_pct,
                    "value_prev": second_pct,
                    "coverage": 1.0,
                    "recency": top_entry.get("recency") or "최근 조합",
                    "time_span": "최근 3개월 조합",
                    "sufficiency": "High",
                    "warning": None,
                    "unit": "ratio",
                    "meta": {
                        "role_tags": ["eda", "age"],
                        "question_roles": list(top_entry.get("meta", {}).get("question_roles") or []),
                        "unit": "ratio",
                        "derived": True,
                        "label": "핵심 연령대 집중도",
                    },
                    "custom_reason": reason,
                }
            )

    if not derived_entries and selected:
        top_selected = selected[0]
        label = top_selected.get("label") or top_selected.get("column") or "핵심 지표"
        reason = f"{label} 지표를 중심으로 EDA를 반복해 보조 지표를 발굴했습니다."
        extra_reasons.append({"column": top_selected.get("column"), "reason": reason})

    return derived_entries, extra_reasons


def diagnose_and_select(
    question: str, merchant_id: str, analysis_mode: str = "기본 데이터 접근"
) -> Dict[str, Any]:
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

    is_advanced_mode = str(analysis_mode or "").startswith("고급")
    derived_entries: List[Dict[str, Any]] = []
    extra_reason_entries: List[Dict[str, Any]] = []
    if is_advanced_mode:
        derived_entries, extra_reason_entries = _generate_eda_derivatives(
            profiles, selected
        )
        selected.extend(derived_entries)
        profiles.extend(derived_entries)

    selection_reasons: List[Dict[str, Any]] = []
    selected_features: List[Dict[str, Any]] = []
    sufficiency_low = 0

    for entry in profiles:
        entry["selected"] = entry in selected
        entry.pop("_score", None)
        entry.pop("_passes", None)
        if entry.get("selected"):
            meta = entry.get("meta") or {}
            if entry.get("custom_reason"):
                reason = entry["custom_reason"]
            else:
                tag = _pick_reason_tag(entry, context)
                reason = _build_reason(
                    meta.get("label", entry.get("column", "")), tag, question
                )
            entry["reason"] = reason
            selection_reasons.append({"column": entry.get("column"), "reason": reason})
            if not meta.get("derived"):
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

    if extra_reason_entries:
        selection_reasons.extend(extra_reason_entries)

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
        "analysis_mode": analysis_mode,
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
        "analysis_mode": analysis_mode,
    }
