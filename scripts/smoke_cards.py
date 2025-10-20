"""Smoke checks for Agent-2 JSON parsing and fallback generation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bigcon_2agent_mvp_v3 import (  # noqa: E402
    llm_json_safe_parse,
    _structured_fallback_cards,  # type: ignore[attr-defined]
    _resolve_schema_for_question,  # type: ignore[attr-defined]
)


def _print_header(title: str) -> None:
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)


def malformed_json_repair() -> None:
    sample = """
    ```json
    {'answers': [{'idea_title': '테스트 아이디어', 'audience': '핵심 고객', 'channels': ['SNS'], 'execution': ['step 1', 'step 2'], 'copy_samples': ['샘플 카피'], 'measurement': ['metric'], 'evidence': [{'source': 'STRUCTURED', 'key': 'age_distribution.1020', 'value': '68.6%', 'period': 'recent', 'snippet': '연령 데이터'}]},],}
    ```
    """
    _, validator, _ = _resolve_schema_for_question("Q1_CAFE_CHANNELS")
    parsed, logs = llm_json_safe_parse(sample, validator)
    print("parse_success:", bool(parsed))
    print("passes:", [entry["pass"] + ("✔" if entry["success"] else "✖") for entry in logs])
    if parsed:
        print("answers_count:", len(parsed.get("answers", [])))
    else:
        print("last_error:", logs[-1] if logs else None)


def fallback_generation() -> None:
    agent1_stub = {
        "debug": {
            "snapshot": {
                "sanitized": {
                    "age_distribution": [
                        {"code": "1020", "label": "10‒20대", "value": 68.6},
                        {"code": "3039", "label": "30대", "value": 18.6},
                    ],
                    "customer_mix_detail": {"유동": 94.0, "직장": 4.5, "거주": 1.5},
                    "new_pct": 12.4,
                    "revisit_pct": 39.5,
                    "latest_ta_ym": "202404",
                }
            }
        }
    }
    fallback = _structured_fallback_cards(agent1_stub, "Q1_CAFE_CHANNELS")
    print("fallback_answers:", len(fallback.get("answers", [])))
    print("first_answer_keys:", list(fallback.get("answers", [{}])[0].keys()))


def main() -> None:
    _print_header("Malformed JSON repair test")
    malformed_json_repair()
    _print_header("Structured fallback generation")
    fallback_generation()


if __name__ == "__main__":
    main()
