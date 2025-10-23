"""Formatting helpers for Agent-2 marketing plan disclosure."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from data.loader import get_catalog


def _format_value(value: Any, unit: str | None) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if unit == "ratio":
        return f"{numeric * 100:.1f}%"
    if unit == "currency":
        return f"{numeric:,.0f}원"
    if unit == "count":
        return f"{numeric:,.0f}건"
    return f"{numeric:,.2f}"


def _resolve_label(feature: Dict[str, Any], catalog: Dict[str, Dict[str, Any]]) -> str:
    column = feature.get("column")
    meta = catalog.get(column or "") if column else None
    if meta:
        return meta.get("label") or feature.get("label") or column or "지표"
    return feature.get("label") or column or "지표"


def generate_plan(
    question: str,
    merchant_id: str,
    selected_features: Iterable[Dict[str, Any]],
    sufficiency_summary: str,
) -> Dict[str, Any]:
    catalog = get_catalog()
    lines: List[str] = []

    for feature in selected_features or []:
        column = feature.get("column")
        meta = catalog.get(column or "") if column else None
        label = _resolve_label(feature, catalog)
        unit = meta.get("unit") if meta else None
        period_label = feature.get("period") or meta.get("default_period") if meta else None
        value_text = _format_value(feature.get("value"), unit)
        source = feature.get("source") or (meta.get("source") if meta else None)
        latest_period = feature.get("latest_period") or feature.get("recency")
        extras: List[str] = []
        if period_label:
            extras.append(f"{period_label}={value_text}")
        else:
            extras.append(f"값={value_text}")
        if source:
            extras.append(str(source))
        if latest_period:
            extras.append(str(latest_period))
        line = f"{label} ({', '.join(extras)})"
        lines.append(line)

    tone_message = "선택된 지표를 기반으로 제안을 구성했습니다."
    if sufficiency_summary == "데이터 부족":
        tone_message = "데이터 커버리지가 낮아 보수적으로 제안을 제시합니다. 추가 데이터 확보가 필요합니다."

    return {
        "used_metrics_lines": lines,
        "tone_message": tone_message,
        "sufficiency_summary": sufficiency_summary,
    }
