"""Lightweight smoke checks for deterministic formatting and prompt wiring."""

from __future__ import annotations

import json

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ruff: noqa: E402
from app_core.formatters import merge_age_buckets, three_line_diagnosis
import bigcon_2agent_mvp_v3 as agent2


def _print(title: str, payload) -> None:
    print(f"\n== {title} ==")
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload)


def check_age_merge() -> None:
    agent1 = {
        "debug": {
            "snapshot": {
                "sanitized": {
                    "age_distribution": [
                        {"code": "1020", "value": 68.6},
                        {"code": "3039", "value": 18.6},
                        {"code": "5059", "value": 7.7},
                        {"code": "4049", "value": 3.9},
                        {"code": "60+", "value": 1.2},
                    ],
                    "age_by_gender": {
                        "3039": {"F": 15.77, "M": 2.83},
                        "5059": {"F": 7.7},
                    },
                    "customer_mix_detail": {"유동": 0.94, "직장": 0.04, "거주": 0.02},
                    "new_pct": 0.31,
                    "revisit_pct": 0.69,
                }
            }
        }
    }

    buckets = merge_age_buckets(agent1)
    _print("Age buckets", buckets)

    lines = three_line_diagnosis(agent1)
    _print("Three-line diagnosis", lines)


def check_prompt_with_rag_block() -> None:
    agent1_stub = {"debug": {"snapshot": {"sanitized": {}}}}
    rag_context = {
        "enabled": True,
        "requested": True,
        "selection_missing": False,
        "selected_doc_ids": ["doc-demo"],
        "threshold": 0.35,
        "mode": "auto",
        "max_score": 0.28,
        "hits": 2,
        "chunks": [
            {"doc_id": "doc-demo", "chunk_id": "c0", "score": 0.28, "text": "demo"},
            {"doc_id": "doc-demo", "chunk_id": "c1", "score": 0.23, "text": "demo2"},
        ],
        "top_scores": [0.28, 0.23],
    }

    _ = agent2.build_agent2_prompt(
        agent1_stub,
        question_text="카페 채널 아이디어 알려줘",
        question_type="Q1_CAFE_CHANNELS",
        rag_context=rag_context,
    )
    trace = dict(agent2.AGENT2_PROMPT_TRACE)
    _print("Agent-2 prompt trace", trace)


def main() -> None:
    check_age_merge()
    check_prompt_with_rag_block()


if __name__ == "__main__":
    main()

