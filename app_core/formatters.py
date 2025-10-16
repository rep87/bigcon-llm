"""Utility formatters for deterministic KPI rendering."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _to_float(value: Any) -> Optional[float]:
    """Best-effort conversion to float while stripping percent symbols."""

    if value is None:
        return None
    if isinstance(value, bool):
        # Treat booleans as invalid for percentages
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("%", "").replace(",", "")
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _pct_guard(number: Optional[float]) -> Optional[float]:
    """Normalize a numeric value into the 0-100 range with one decimal place."""

    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        number *= 100.0
    if math.isnan(number):  # pragma: no cover - defensive
        return None
    if number < 0.0 or number > 100.0:
        return None
    return round(number, 1)


def _close(a: float, b: float, tol: float = 0.5) -> bool:
    return abs(a - b) <= tol


_AGE_LABEL_MAP: Dict[str, str] = {
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


_GENDER_KEYS = {
    "F": {"F", "f", "FME", "female", "여", "여성"},
    "M": {"M", "m", "MAL", "male", "남", "남성"},
}


@dataclass
class AgeBucket:
    code: str
    label: str
    percent: Optional[float]
    source: str
    female: Optional[float]
    male: Optional[float]
    combined: Optional[float]
    fm_sum: Optional[float]
    notes: str = ""
    included: bool = False


def _label_for_code(code: str, fallback: Optional[str] = None) -> str:
    if not code:
        return fallback or "—"
    key = str(code)
    if key in _AGE_LABEL_MAP:
        return _AGE_LABEL_MAP[key]
    if fallback:
        return fallback
    return key


def _collect_allowlist(agent1: Dict[str, Any]) -> List[str]:
    allowlist: List[str] = []
    kpis = (agent1 or {}).get("kpis") or {}
    candidate = kpis.get("age_allowlist")
    if isinstance(candidate, (list, tuple, set)):
        allowlist.extend([str(code) for code in candidate if code is not None])
    debug_snapshot = ((agent1 or {}).get("debug") or {}).get("snapshot") or {}
    sanitized = debug_snapshot.get("sanitized")
    if isinstance(sanitized, dict):
        candidate = sanitized.get("age_allowlist")
        if isinstance(candidate, (list, tuple, set)):
            allowlist.extend([str(code) for code in candidate if code is not None])
    return list(dict.fromkeys(allowlist))


def _combined_distribution(agent1: Dict[str, Any]) -> Dict[str, Tuple[Optional[float], Optional[str]]]:
    combined: Dict[str, Tuple[Optional[float], Optional[str]]] = {}
    sources: Sequence[Any] = []
    kpis = (agent1 or {}).get("kpis") or {}
    dist = kpis.get("age_distribution")
    if isinstance(dist, list):
        sources = dist
    else:
        snapshot = ((agent1 or {}).get("debug") or {}).get("snapshot") or {}
        sanitized = snapshot.get("sanitized")
        if isinstance(sanitized, dict):
            dist = sanitized.get("age_distribution")
            if isinstance(dist, list):
                sources = dist
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        code = entry.get("code")
        label = entry.get("label")
        if code is None and label:
            code = label
        if code is None:
            continue
        code_str = str(code)
        value = _pct_guard(_to_float(entry.get("value")))
        combined[code_str] = (value, label)
    return combined


def _gender_distribution(agent1: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    by_gender: Dict[str, Dict[str, Optional[float]]] = {}
    snapshot = ((agent1 or {}).get("debug") or {}).get("snapshot") or {}
    raw = snapshot.get("raw")
    if not isinstance(raw, dict):
        return by_gender
    pattern = re.compile(r"M12_(MAL|FME)_([0-9+]+)_RAT", re.IGNORECASE)
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        cleaned_key = key[:-4] if key.endswith("_raw") else key
        match = pattern.fullmatch(cleaned_key)
        if not match:
            continue
        gender_code, bucket_code = match.groups()
        gender_norm = None
        for gkey, aliases in _GENDER_KEYS.items():
            if gender_code in aliases:
                gender_norm = gkey
                break
        if gender_norm is None:
            continue
        numeric = _pct_guard(_to_float(value))
        if numeric is None:
            continue
        bucket = str(bucket_code)
        by_gender.setdefault(bucket, {})[gender_norm] = numeric
    return by_gender


def _build_age_records(agent1: Dict[str, Any]) -> List[AgeBucket]:
    combined = _combined_distribution(agent1)
    by_gender = _gender_distribution(agent1)
    allowlist = _collect_allowlist(agent1)
    combined_codes = set(combined.keys())
    gender_codes = set(by_gender.keys())
    candidate_keys = sorted(combined_codes | gender_codes)
    if allowlist:
        allow = {str(code) for code in allowlist}
        candidate_keys = [code for code in candidate_keys if code in allow]

    records: List[AgeBucket] = []

    for code in candidate_keys:
        combined_value, combined_label = combined.get(code, (None, None))
        genders = by_gender.get(code, {})
        female = genders.get("F")
        male = genders.get("M")
        fm_sum = None
        fm_note = ""
        if female is not None and male is not None:
            fm_candidate = _pct_guard(female + male)
            if fm_candidate is None:
                fm_note = "F+M 합이 범위를 벗어남"
            else:
                fm_sum = fm_candidate
        elif genders:
            fm_note = "단일 성별 데이터만 존재"

        label = _label_for_code(code, combined_label)
        chosen_value: Optional[float] = None
        source = "skipped"

        if combined_value is not None:
            chosen_value = combined_value
            source = "combined"
        elif fm_sum is not None:
            chosen_value = fm_sum
            source = "F+M"

        if chosen_value is None and fm_sum is not None:
            # Combined missing but F+M valid
            chosen_value = fm_sum
            source = "F+M"

        note_parts: List[str] = []
        if fm_note:
            note_parts.append(fm_note)
        if (
            combined_value is not None
            and fm_sum is not None
            and not _close(combined_value, fm_sum)
        ):
            note_parts.append(
                f"combined {combined_value:.1f} vs F+M {fm_sum:.1f}"
            )

        included = chosen_value is not None
        if not included and not note_parts and not genders and combined_value is None:
            note_parts.append("데이터 없음")

        records.append(
            AgeBucket(
                code=code,
                label=label,
                percent=chosen_value,
                source=source,
                female=female,
                male=male,
                combined=combined_value,
                fm_sum=fm_sum,
                notes="; ".join(note_parts),
                included=included,
            )
        )

    records.sort(
        key=lambda item: (
            0 if item.included else 1,
            -(item.percent if isinstance(item.percent, (int, float)) else -1),
            item.code,
        )
    )

    return records


def get_age_buckets(agent1: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Return a sorted list of (label, percent) tuples for valid age buckets."""

    records = _build_age_records(agent1)
    buckets: List[Tuple[str, float]] = []
    for record in records:
        if not record.included or record.percent is None:
            continue
        buckets.append((record.label, record.percent))
    buckets.sort(key=lambda item: item[1], reverse=True)
    return buckets


def get_age_bucket_details(agent1: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Provide detailed debug information for how age buckets were derived."""

    records = _build_age_records(agent1)
    details: List[Dict[str, Any]] = []
    for record in records:
        details.append(
            {
                "code": record.code,
                "label": record.label,
                "percent": record.percent,
                "source": record.source,
                "female": record.female,
                "male": record.male,
                "combined": record.combined,
                "sum_fm": record.fm_sum,
                "included": record.included,
                "notes": record.notes,
            }
        )
    return details

