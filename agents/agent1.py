"""Rule-based diagnostics for selecting metrics for Agent-2."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from data.loader import profile_columns

APP_VERSION = "2024.09-data-provenance"
TELEMETRY_DIR = Path(__file__).resolve().parent.parent / "telemetry"

AnalysisMode = Literal["basic", "advanced"]

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


DERIVED_RECIPES = [
    {
        "id": "youth_share",
        "label": "청년(10~20) 비중",
        "formula": "M12_MAL_1020_RAT + M12_FME_1020_RAT",
        "source": "set3",
        "unit": "%",
    },
    {
        "id": "worker_share",
        "label": "직장권 비중",
        "formula": "RC_M1_SHC_WP_UE_CLN_RAT",
        "source": "set3",
        "unit": "%",
    },
    {
        "id": "loyalty_bias",
        "label": "유지편향(재방문-신규)",
        "formula": "MCT_UE_CLN_REU_RAT - MCT_UE_CLN_NEW_RAT",
        "source": "set3",
        "unit": "pp",
    },
    {
        "id": "revisit_delta",
        "label": "재방문율 증감(최근3M-직전3M)",
        "formula": "DELTA(MCT_UE_CLN_REU_RAT)",
        "source": "set3",
        "unit": "pp",
    },
    {
        "id": "ticket_trend",
        "label": "객단가 추이(최근3M-직전3M)",
        "formula": "DELTA(RC_M1_AV_NP_AT)",
        "source": "set2",
        "unit": "원",
    },
    {
        "id": "traffic_trend",
        "label": "이용건 추이(최근3M-직전3M)",
        "formula": "DELTA(RC_M1_TO_UE_CT)",
        "source": "set2",
        "unit": "건",
    },
    {
        "id": "unique_rate",
        "label": "고객/이용건 비율",
        "formula": "RC_M1_UE_CUS_CN / RC_M1_TO_UE_CT",
        "source": "set2",
        "unit": "",
    },
    {
        "id": "delivery_dependency",
        "label": "배달 의존도",
        "formula": "DLV_SAA_RAT",
        "source": "set2",
        "unit": "%",
    },
]


@dataclass
class SelectionContext:
    question_types: Set[str]
    keyword_tags: Set[str]
    must_have_tags: Set[str]


SUFFICIENCY_RANK = {"Low": 0, "Medium": 1, "High": 2}


def normalize_percent(value: Any) -> Optional[float]:
    """Normalize ratio-like values to 0~100 scale with guard rails."""

    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not (num == num):
        return None
    if num < 0:
        num = 0.0
    if 0 <= num <= 1:
        return round(num * 100, 2)
    if num > 100:
        return round(num / 100, 2)
    return round(num, 2)


def _is_percent_column(column: str | None, meta: Dict[str, Any]) -> bool:
    unit = str(meta.get("unit") or "").lower()
    if unit in {"ratio", "percent", "%", "pct"}:
        return True
    if not column:
        return False
    suffixes = ("_rat", "_rate", "_ratio", "_shr", "_share", "_pct")
    column_lower = column.lower()
    return column_lower.endswith(suffixes)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not (num == num):
        return None
    return float(num)


def _normalize_metric_value(column: str, meta: Dict[str, Any], value: Any) -> Optional[float]:
    if value is None:
        return None
    if _is_percent_column(column, meta):
        return normalize_percent(value)
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not (num == num):
        return None
    return round(num, 2)


def _strip_internal_fields(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        cleaned.append({key: value for key, value in row.items() if not key.startswith("_")})
    return cleaned


def _worst_sufficiency(values: List[Optional[str]]) -> str:
    rank = min((SUFFICIENCY_RANK.get(val or "Low", 0) for val in values), default=0)
    for key, score in SUFFICIENCY_RANK.items():
        if score == rank:
            return key
    return "Low"


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


def build_profile_table_basic(
    raw_profiles: List[Dict[str, Any]], question: str
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in raw_profiles:
        meta = entry.get("meta") or {}
        column = entry.get("column") or meta.get("column")
        unit_value = meta.get("unit") or None
        if not unit_value and _is_percent_column(column, meta):
            unit_value = "ratio"

        row: Dict[str, Any] = {
            "source": entry.get("source"),
            "column": column,
            "label": meta.get("label") or entry.get("label") or column,
            "value_recent": _normalize_metric_value(column, meta, entry.get("value_recent")),
            "value_prev": _normalize_metric_value(column, meta, entry.get("value_prev")),
            "coverage": _safe_float(entry.get("coverage")),
            "recency": entry.get("recency"),
            "time_span": entry.get("time_span"),
            "sufficiency": entry.get("sufficiency") or "Low",
            "selected": False,
            "unit": unit_value,
            "warning": entry.get("warning"),
            "reason": None,
            "_meta": meta,
            "_recent_coverage": _safe_float(entry.get("recent_coverage")),
        }
        rows.append(row)
    return rows


def _extract_recipe_columns(formula: str) -> List[str]:
    formula = formula.strip()
    if formula.upper().startswith("DELTA(") and formula.endswith(")"):
        inner = formula[6:-1].strip()
        return [inner]
    tokens = []
    current = []
    for char in formula:
        if char.isalnum() or char == "_":
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    # filter out numeric literals
    columns = [token for token in tokens if not token.replace(".", "", 1).isdigit()]
    return columns


def _evaluate_recipe_expression(expr: str, values: Dict[str, Optional[float]]) -> Optional[float]:
    import ast

    def _eval(node: ast.AST) -> Optional[float]:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if abs(right) < 1e-9:
                    return None
                return left / right
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = _eval(node.operand)
            if operand is None:
                return None
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.Constant):
            try:
                return float(node.value)
            except (TypeError, ValueError):
                return None
        if isinstance(node, ast.Name):
            return values.get(node.id)
        return None

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    return _eval(tree)


def _compute_recipe_value(
    recipe: Dict[str, Any], base_rows: Dict[str, Dict[str, Any]]
) -> Optional[float]:
    formula = recipe.get("formula", "").strip()
    if not formula:
        return None
    if formula.upper().startswith("DELTA(") and formula.endswith(")"):
        column = formula[6:-1].strip()
        row = base_rows.get(column)
        if not row:
            return None
        recent = row.get("value_recent")
        prev = row.get("value_prev")
        if recent is None or prev is None:
            return None
        return round(recent - prev, 2)

    column_names = _extract_recipe_columns(formula)
    value_map: Dict[str, Optional[float]] = {}
    for name in column_names:
        row = base_rows.get(name)
        value_map[name] = None if not row else _safe_float(row.get("value_recent"))
    if any(value is None for value in value_map.values()):
        return None
    evaluated = _evaluate_recipe_expression(formula, value_map)
    if evaluated is None:
        return None
    return round(evaluated, 2)


def build_derived_features(profile_basic: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base_map = {row.get("column"): row for row in profile_basic if row.get("column")}
    derived_rows: List[Dict[str, Any]] = []
    for recipe in DERIVED_RECIPES:
        required_columns = _extract_recipe_columns(recipe.get("formula", ""))
        base_rows = [base_map.get(col) for col in required_columns if base_map.get(col)]
        if required_columns and len(base_rows) < len(required_columns):
            continue
        value = _compute_recipe_value(recipe, base_map)
        if value is None:
            continue
        coverage_candidates = [row.get("coverage") for row in base_rows if row]
        suff_candidates = [row.get("sufficiency") for row in base_rows if row]
        recency_candidates = [row.get("recency") for row in base_rows if row]
        time_span_candidates = [row.get("time_span") for row in base_rows if row]
        coverage = min((val for val in coverage_candidates if val is not None), default=None)
        sufficiency = _worst_sufficiency(suff_candidates or ["Medium"])
        recency = next((val for val in recency_candidates if val), None)
        time_span = next((val for val in time_span_candidates if val), None)
        unit_raw = recipe.get("unit") or ""
        unit_map = {"%": "ratio", "원": "currency", "건": "count", "pp": "pp"}
        normalized_unit = unit_map.get(unit_raw, unit_raw or None)
        derived_rows.append(
            {
                "source": recipe.get("source", "derived"),
                "column": f"derived.{recipe['id']}",
                "label": recipe.get("label", recipe["id"]),
                "value_recent": value,
                "value_prev": None,
                "coverage": coverage,
                "recency": recency,
                "time_span": time_span,
                "sufficiency": sufficiency,
                "selected": True,
                "unit": normalized_unit,
                "warning": None,
                "reason": None,
                "_recipe": recipe,
            }
        )
    return derived_rows


def build_profile_table_advanced(derived_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return derived_rows


def _format_value_for_reason(value: Any, unit: Optional[str]) -> str:
    if value is None:
        return "값 없음"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if unit == "ratio":
        percent = normalize_percent(num)
        if percent is None:
            return "값 없음"
        return f"{percent:.2f}%"
    if unit == "currency":
        return f"{num:,.0f}원"
    if unit == "count":
        return f"{num:,.0f}건"
    if unit == "pp":
        sign = "+" if num > 0 else ""
        return f"{sign}{num:.2f}pp"
    return f"{num:.2f}"


def enrich_reasons_with_eda(
    reasons: List[Dict[str, Any]],
    profile_basic: List[Dict[str, Any]],
    profile_advanced: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    enriched = list(reasons)
    seen = {item.get("column") for item in reasons}
    for row in profile_advanced:
        column = row.get("column")
        if column in seen:
            continue
        recipe = row.get("_recipe") or {}
        recipe_id = recipe.get("id")
        label = row.get("label", column or "파생 지표")
        unit = row.get("unit")
        value = row.get("value_recent")
        reason_text: str
        if recipe_id == "revisit_delta":
            if value is None:
                reason_text = "재방문율 변화량을 확인했으나 수치가 부족합니다."
            elif value > 0:
                reason_text = (
                    f"최근 3개월 대비 {abs(value):.2f}pp 상승하여 성장 모멘텀을 활용합니다."
                )
            elif value < 0:
                reason_text = (
                    f"최근 3개월 대비 {abs(value):.2f}pp 하락이 확인되어 재방문 인센티브가 필요합니다."
                )
            else:
                reason_text = "최근 3개월과 직전 3개월의 재방문율 변화가 없어 추세를 유지합니다."
        elif recipe_id == "ticket_trend":
            if value is None:
                reason_text = "객단가 추이를 확인했으나 수치가 부족합니다."
            elif value > 0:
                reason_text = (
                    f"최근 3개월 객단가가 {abs(value):,.0f}원 상승해 고가 전략을 검토합니다."
                )
            elif value < 0:
                reason_text = (
                    f"최근 3개월 객단가가 {abs(value):,.0f}원 하락해 할인 프로모션을 점검합니다."
                )
            else:
                reason_text = "객단가 변동이 없어 기본 전략을 유지합니다."
        elif recipe_id == "traffic_trend":
            if value is None:
                reason_text = "이용건 추이를 확인했으나 수치가 부족합니다."
            elif value > 0:
                reason_text = (
                    f"최근 이용건이 {abs(value):,.0f}건 증가해 성장 채널을 확대합니다."
                )
            elif value < 0:
                reason_text = (
                    f"최근 이용건이 {abs(value):,.0f}건 감소해 방문 유입 캠페인이 필요합니다."
                )
            else:
                reason_text = "이용건 변동이 없어 안정적 운영을 유지합니다."
        elif recipe_id == "loyalty_bias":
            value_text = _format_value_for_reason(value, "pp")
            reason_text = (
                f"재방문율과 신규율의 격차({value_text})를 확인해 유지/확장 균형을 조정합니다."
            )
        elif recipe_id == "youth_share":
            value_text = _format_value_for_reason(value, "ratio")
            reason_text = f"10~20대 비중 {value_text}를 확인해 젊은 타깃 채널을 선정합니다."
        elif recipe_id == "worker_share":
            value_text = _format_value_for_reason(value, "ratio")
            reason_text = f"직장권 비중 {value_text}로 출퇴근 시간대 집중 캠페인을 제안합니다."
        elif recipe_id == "unique_rate":
            value_text = _format_value_for_reason(value, None)
            reason_text = f"고객·이용건 비율 {value_text}를 참고해 충성 고객 비중을 판단합니다."
        elif recipe_id == "delivery_dependency":
            value_text = _format_value_for_reason(value, "ratio")
            reason_text = f"배달 의존도 {value_text}를 확인해 온/오프 채널 믹스를 조정합니다."
        else:
            value_text = _format_value_for_reason(value, unit)
            reason_text = f"{label} — 파생 분석 값 {value_text}를 근거로 활용합니다."
        enriched.append({"column": column, "label": label, "reason": reason_text})
        seen.add(column)
    return enriched


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


def select_features(
    profile_basic: List[Dict[str, Any]],
    context: SelectionContext,
    question: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    row_map = {
        row.get("column"): row
        for row in profile_basic
        if row.get("column")
    }

    scored: List[Dict[str, Any]] = []
    for column, row in row_map.items():
        meta = row.get("_meta") or {}
        feature_id = meta.get("feature_id")
        if not feature_id:
            continue
        entry = {
            "column": column,
            "source": row.get("source"),
            "value_recent": row.get("value_recent"),
            "value_prev": row.get("value_prev"),
            "coverage": row.get("coverage"),
            "recent_coverage": row.get("_recent_coverage")
            if row.get("_recent_coverage") is not None
            else row.get("coverage"),
            "sufficiency": row.get("sufficiency"),
            "recency": row.get("recency"),
            "time_span": row.get("time_span"),
            "meta": meta,
        }
        entry["_score"] = _score_profile(entry, context)
        entry["_passes"] = _passes_threshold(entry, context)
        scored.append(entry)

    scored.sort(key=lambda x: x.get("_score", 0.0), reverse=True)

    selected_entries: List[Dict[str, Any]] = []
    selected_tags: Set[str] = set()
    for entry in scored:
        if not entry.get("_passes"):
            continue
        if len(selected_entries) >= 8:
            break
        selected_entries.append(entry)
        meta = entry.get("meta") or {}
        selected_tags.update(meta.get("role_tags") or [])

    for must_tag in context.must_have_tags:
        if must_tag in selected_tags:
            continue
        candidate = next(
            (
                item
                for item in scored
                if must_tag in (item.get("meta", {}).get("role_tags") or [])
                and item not in selected_entries
            ),
            None,
        )
        if candidate:
            selected_entries.append(candidate)
            selected_tags.update(candidate.get("meta", {}).get("role_tags") or [])

    selection_reasons: List[Dict[str, Any]] = []
    selected_features: List[Dict[str, Any]] = []

    for row in profile_basic:
        row["selected"] = False
        row["reason"] = None

    for entry in selected_entries:
        column = entry.get("column")
        row = row_map.get(column)
        meta = entry.get("meta") or {}
        label = (row or {}).get("label") or meta.get("label") or column or "지표"
        tag = _pick_reason_tag(entry, context)
        reason = _build_reason(label, tag, question)
        if row:
            row["selected"] = True
            row["reason"] = reason
        selection_reasons.append({"column": column, "label": label, "reason": reason})
        if not meta.get("derived"):
            selected_features.append(
                {
                    "id": meta.get("feature_id"),
                    "column": column,
                    "label": label,
                    "source": (row or {}).get("source"),
                    "period": meta.get("default_period", "recent_3m"),
                    "value": (row or {}).get("value_recent"),
                    "latest_period": (row or {}).get("recency"),
                    "time_span": (row or {}).get("time_span"),
                }
            )

    base_entries = [entry for entry in selected_entries if not entry.get("meta", {}).get("derived")]
    if base_entries:
        high_mid = sum(
            1
            for entry in base_entries
            if (row_map.get(entry.get("column")) or {}).get("sufficiency")
            in {"High", "Medium"}
        )
        ratio = high_mid / max(len(base_entries), 1)
    else:
        ratio = 1.0
    sufficiency_summary = "데이터 충분" if ratio >= 0.6 else "데이터 부족"

    return selected_features, selection_reasons, sufficiency_summary


def diagnose_and_select(
    question: str,
    merchant_id: str,
    analysis_mode: AnalysisMode = "basic",
    **kwargs,
) -> Dict[str, Any]:
    context = _infer_question_types(question)
    raw_profiles = profile_columns(merchant_id)
    profile_basic = build_profile_table_basic(raw_profiles, question)

    selected_features, selection_reasons, sufficiency_summary = select_features(
        profile_basic, context, question
    )

    derived_rows: List[Dict[str, Any]] = []
    if analysis_mode == "advanced":
        derived_rows = build_derived_features(profile_basic)
        if derived_rows:
            selection_reasons = enrich_reasons_with_eda(
                selection_reasons, profile_basic, derived_rows
            )

    profile_advanced = build_profile_table_advanced(derived_rows) if derived_rows else None

    from datetime import datetime

    telemetry_payload = {
        "app_version": APP_VERSION,
        "question": question,
        "merchant_id": merchant_id,
        "analysis_mode": analysis_mode,
        "profile_basic": _strip_internal_fields(profile_basic),
        "profile_advanced": _strip_internal_fields(derived_rows)
        if derived_rows
        else None,
        "selected_features": selected_features,
        "selection_reasons": selection_reasons,
        "sufficiency_summary": sufficiency_summary,
        "timestamp": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "filters": {"question_types": sorted(context.question_types)},
        "time_windows": {"recent_window": 3},
    }

    _write_telemetry(telemetry_payload, merchant_id)

    return {
        "profile_basic": _strip_internal_fields(profile_basic),
        "profile_advanced": _strip_internal_fields(profile_advanced)
        if profile_advanced
        else None,
        "selected_features": selected_features,
        "selection_reasons": selection_reasons,
        "sufficiency_summary": sufficiency_summary,
    }
