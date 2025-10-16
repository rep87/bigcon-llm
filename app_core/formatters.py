"""Deterministic formatters and diagnostics for Agent-1 derived metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


AGE_LABEL_MAP: Dict[str, str] = {
    "1020": "10‒20대",
    "2029": "20대",
    "2030": "20‒30대",
    "3039": "30대",
    "30": "30대",
    "3040": "30‒40대",
    "4049": "40대",
    "40": "40대",
    "4050": "40‒50대",
    "5059": "50대",
    "50": "50대",
    "60": "60대",
    "60+": "60대+",
    "60_plus": "60대+",
    "70": "70대",
}

GENDER_ALIASES: Dict[str, set[str]] = {
    "F": {"F", "f", "FME", "female", "여", "여성"},
    "M": {"M", "m", "MAL", "male", "남", "남성"},
}


@dataclass
class AgeMergeRecord:
    """Intermediate record describing how an age bucket was derived."""

    key: str
    label: str
    combined_value: Optional[float]
    combined_hint: str
    female_value: Optional[float]
    female_hint: str
    male_value: Optional[float]
    male_hint: str
    final_value: Optional[float]
    source: str
    included: bool
    notes: List[str]


def to_float_pct(value: Any) -> Tuple[Optional[float], str]:
    """Convert value to a 0-100 percentage with origin hint.

    Returns (numeric_value, origin_hint) where origin_hint is one of
    {"p100", "p1", "str", "none", "bad"}.
    """

    if value is None:
        return None, "none"

    hint = "p100"
    numeric: Optional[float]

    if isinstance(value, bool):
        return None, "bad"

    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None, "none"
        hint = "str"
        text = text.replace("%", "").replace(",", "")
        try:
            numeric = float(text)
        except (TypeError, ValueError):
            return None, "bad"

    if numeric is None or math.isnan(numeric):  # pragma: no cover - defensive
        return None, "bad"

    if 0.0 <= numeric <= 1.0:
        numeric *= 100.0
        hint = "p1"

    if numeric < 0.0 or numeric > 100.0:
        return None, "bad"

    return round(numeric, 1), hint if hint in {"p100", "p1", "str"} else "p100"


def _snapshot_sections(agent1: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    debug = (agent1 or {}).get("debug") or {}
    snapshot = debug.get("snapshot") or {}
    sanitized = snapshot.get("sanitized") or {}
    raw = snapshot.get("raw") or {}
    kpis = (agent1 or {}).get("kpis") or {}
    return sanitized if isinstance(sanitized, dict) else {}, raw if isinstance(raw, dict) else {}, kpis


def _collect_allowlist(agent1: Dict[str, Any], sanitized: Dict[str, Any], kpis: Dict[str, Any]) -> List[str]:
    allowlist: List[str] = []
    candidates: Iterable[Any] = []
    if isinstance(kpis.get("age_allowlist"), (list, tuple, set)):
        candidates = kpis.get("age_allowlist")
        allowlist.extend(str(code) for code in candidates if code is not None)
    if isinstance(sanitized.get("age_allowlist"), (list, tuple, set)):
        candidates = sanitized.get("age_allowlist")
        allowlist.extend(str(code) for code in candidates if code is not None)
    return list(dict.fromkeys(allowlist))


def _iter_age_entries(value: Any) -> Iterable[Tuple[str, Optional[str], Any]]:
    if isinstance(value, dict):
        for key, val in value.items():
            yield str(key), None, val
    elif isinstance(value, Sequence):
        for item in value:
            if isinstance(item, dict):
                code = item.get("code") or item.get("key") or item.get("id") or item.get("label")
                if code is None:
                    continue
                label = item.get("label") or item.get("name")
                val = item.get("value")
                if val is None:
                    # allow alternate keys for numeric value
                    for alt in ("percent", "ratio", "pct", "share"):
                        if alt in item:
                            val = item[alt]
                            break
                yield str(code), label, val


def _collect_combined_distribution(
    sanitized: Dict[str, Any], kpis: Dict[str, Any]
) -> Dict[str, Tuple[Optional[float], str]]:
    combined: Dict[str, Tuple[Optional[float], str]] = {}
    labels: Dict[str, str] = {}
    for container in (
        sanitized.get("age_distribution"),
        kpis.get("age_distribution"),
    ):
        for code, label, raw in _iter_age_entries(container):
            numeric, _ = to_float_pct(raw)
            if numeric is None:
                continue
            combined[code] = (numeric, label or AGE_LABEL_MAP.get(code, code))
            if label:
                labels[code] = label
    return combined


def _collect_gender_distribution(
    sanitized: Dict[str, Any], raw: Dict[str, Any], kpis: Dict[str, Any]
) -> Dict[str, Dict[str, Tuple[Optional[float], str]]]:
    by_gender: Dict[str, Dict[str, Tuple[Optional[float], str]]] = {}

    for container in (
        sanitized.get("age_by_gender"),
        sanitized.get("age_gender"),
        kpis.get("age_by_gender"),
        kpis.get("age_gender"),
    ):
        if not isinstance(container, dict):
            continue
        for code, mapping in container.items():
            if not isinstance(mapping, dict):
                continue
            record: Dict[str, Tuple[Optional[float], str]] = by_gender.setdefault(str(code), {})
            for gender_alias, aliases in GENDER_ALIASES.items():
                for alias in aliases:
                    if alias in mapping:
                        value, hint = to_float_pct(mapping[alias])
                        if value is not None:
                            record[gender_alias] = (value, hint)
                        break

    pattern = re.compile(r"M12_(MAL|FME)_([0-9+]+)_RAT", re.IGNORECASE)
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        base_key = key[:-4] if key.endswith("_raw") else key
        match = pattern.fullmatch(base_key)
        if not match:
            continue
        gender_token, bucket = match.groups()
        gender_norm: Optional[str] = None
        for gender, aliases in GENDER_ALIASES.items():
            if gender_token in aliases:
                gender_norm = gender
                break
        if gender_norm is None:
            continue
        numeric, hint = to_float_pct(value)
        if numeric is None:
            continue
        by_gender.setdefault(str(bucket), {})[gender_norm] = (numeric, hint)

    return by_gender


def _guard_percent(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if math.isnan(value):  # pragma: no cover - defensive
        return None
    if value < 0.0 or value > 100.0:
        return None
    return round(value, 1)


def _build_age_records(agent1: Dict[str, Any]) -> Tuple[List[AgeMergeRecord], List[AgeMergeRecord]]:
    sanitized, raw, kpis = _snapshot_sections(agent1)
    allowlist = _collect_allowlist(agent1, sanitized, kpis)
    combined = _collect_combined_distribution(sanitized, kpis)
    by_gender = _collect_gender_distribution(sanitized, raw, kpis)

    candidate_keys = sorted(set(combined.keys()) | set(by_gender.keys()))
    if allowlist:
        allow = set(allowlist)
        candidate_keys = [code for code in candidate_keys if code in allow]

    all_records: List[AgeMergeRecord] = []
    included: List[AgeMergeRecord] = []

    for code in candidate_keys:
        combined_entry = combined.get(code)
        combined_value = combined_entry[0] if combined_entry else None
        combined_label = combined_entry[1] if combined_entry else None
        combined_hint = "none"
        if combined_entry:
            _, combined_hint = to_float_pct(combined_value)

        gender_map = by_gender.get(code) or {}
        female_value, female_hint = (gender_map.get("F") or (None, "none"))
        male_value, male_hint = (gender_map.get("M") or (None, "none"))

        fm_sum: Optional[float] = None
        notes: List[str] = []
        if female_value is not None and male_value is not None:
            total = female_value + male_value
            fm_sum = _guard_percent(total)
            if fm_sum is None:
                notes.append("F+M 합이 0–100 범위를 벗어남")

        final_value: Optional[float] = None
        source = "skipped"

        if combined_value is not None:
            final_value = _guard_percent(combined_value)
            if final_value is not None:
                source = "combined"
        if final_value is None and fm_sum is not None:
            final_value = fm_sum
            if final_value is not None:
                source = "F+M"

        if (
            final_value is not None
            and combined_value is not None
            and fm_sum is not None
            and abs(combined_value - fm_sum) > 0.5
        ):
            notes.append(f"combined {combined_value:.1f} vs F+M {fm_sum:.1f}")

        label = AGE_LABEL_MAP.get(code, combined_label or code)

        record = AgeMergeRecord(
            key=code,
            label=label,
            combined_value=_guard_percent(combined_value),
            combined_hint=combined_hint,
            female_value=_guard_percent(female_value),
            female_hint=female_hint,
            male_value=_guard_percent(male_value),
            male_hint=male_hint,
            final_value=_guard_percent(final_value),
            source=source,
            included=final_value is not None,
            notes=notes,
        )

        all_records.append(record)
        if record.included and record.final_value is not None:
            included.append(record)

    included.sort(
        key=lambda item: (
            -item.final_value if isinstance(item.final_value, (int, float)) else 1e9,
            item.key,
        )
    )

    return included, all_records


def merge_age_buckets(agent1: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return deterministic age buckets for rendering."""

    included, _ = _build_age_records(agent1)
    return [
        {
            "key": record.key,
            "label": record.label,
            "value": record.final_value,
            "source": record.source,
            "combined": record.combined_value,
            "female": record.female_value,
            "male": record.male_value,
            "notes": list(record.notes),
        }
        for record in included
        if record.final_value is not None
    ]


def get_age_buckets(agent1: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Return a sorted list of (label, percent) tuples for valid age buckets."""

    return [(item["label"], item["value"]) for item in merge_age_buckets(agent1)]


def get_age_bucket_details(agent1: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detailed debug information for age merge decisions."""

    _, all_records = _build_age_records(agent1)
    details: List[Dict[str, Any]] = []
    for record in all_records:
        details.append(
            {
                "key": record.key,
                "label": record.label,
                "final_value": record.final_value,
                "source": record.source,
                "combined_value": record.combined_value,
                "female_value": record.female_value,
                "male_value": record.male_value,
                "included": record.included,
                "notes": "; ".join(record.notes) if record.notes else "",
            }
        )
    return details


def three_line_diagnosis(agent1: Dict[str, Any]) -> List[str]:
    """Build exactly three summary lines for the overview diagnosis."""

    sanitized, _, kpis = _snapshot_sections(agent1)
    lines: List[str] = []

    mix_detail = sanitized.get("customer_mix_detail") or kpis.get("customer_mix_detail") or {}
    mix_labels = ["유동", "직장", "거주"]
    mix_parts: List[str] = []
    if isinstance(mix_detail, dict):
        for label in mix_labels:
            value, _ = to_float_pct(mix_detail.get(label))
            if value is None:
                mix_parts.append(f"{label} —")
            else:
                mix_parts.append(f"{label} {value:.1f}%")
    if not mix_parts:
        mix_parts = ["유형 데이터 부족"]
    lines.append(" · ".join(mix_parts))

    age_buckets = merge_age_buckets(agent1)
    if age_buckets:
        top_bucket = age_buckets[0]
        lines.append(f"최다 연령 {top_bucket['label']} {top_bucket['value']:.1f}%")
    else:
        lines.append("연령 데이터 부족")

    new_pct, _ = to_float_pct(sanitized.get("new_pct") or kpis.get("new_rate_avg"))
    revisit_pct, _ = to_float_pct(sanitized.get("revisit_pct") or kpis.get("revisit_rate_avg"))
    if new_pct is None and revisit_pct is None:
        lines.append("신규·재방문 데이터 부족")
    else:
        parts = []
        parts.append(f"신규 {new_pct:.1f}%" if new_pct is not None else "신규 —")
        parts.append(f"재방문 {revisit_pct:.1f}%" if revisit_pct is not None else "재방문 —")
        lines.append(" · ".join(parts))

    while len(lines) < 3:
        lines.append("데이터 부족")

    return lines[:3]


__all__ = [
    "to_float_pct",
    "merge_age_buckets",
    "get_age_buckets",
    "get_age_bucket_details",
    "three_line_diagnosis",
]

