"""Lightweight smoke checks for deterministic formatting and prompt wiring."""

from __future__ import annotations

import compileall
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ruff: noqa: E402
import pandas as pd

import app_core.formatters as formatters
import app_core.panel_extract as panel_extract
import app_core.summary_blocks as summary_blocks
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

    buckets = formatters.merge_age_buckets(agent1)
    _print("Age buckets", buckets)

    lines = formatters.three_line_diagnosis(agent1)
    _print("Three-line diagnosis", lines)


def check_summary_blocks() -> None:
    raw = {
        "ENCODED_MCT": ["m1", "m1", "m1", "m1", "m1"],
        "TA_YM": ["202205", "202302", "202303", "202304", "202305"],
        "M12_MAL_1020_RAT": [10, 10, 10, 10, 10],
        "M12_MAL_30_RAT": [15, 15, 15, 15, 15],
        "M12_MAL_40_RAT": [10, 10, 10, 10, 10],
        "M12_MAL_50_RAT": [5, 5, 5, 5, 5],
        "M12_MAL_60_RAT": [10, 10, 10, 10, 10],
        "M12_FME_1020_RAT": [15, 15, 15, 15, 15],
        "M12_FME_30_RAT": [10, 10, 10, 10, 10],
        "M12_FME_40_RAT": [10, 10, 10, 10, 10],
        "M12_FME_50_RAT": [10, 10, 10, 10, 10],
        "M12_FME_60_RAT": [5, 5, 5, 5, 5],
        "MCT_UE_CLN_REU_RAT": [82, 86, 85, 84, 83],
        "MCT_UE_CLN_NEW_RAT": [18, 14, 15, 16, 17],
        "RC_M1_SHC_RSD_UE_CLN_RAT": [30, 30, 30, 30, 30],
        "RC_M1_SHC_WP_UE_CLN_RAT": [40, 40, 40, 40, 40],
        "RC_M1_SHC_FLP_UE_CLN_RAT": [30, 30, 30, 30, 30],
        "EXTRA": [1, 2, 3, 4, 5],
    }
    df = pd.DataFrame(raw)

    subset = panel_extract.subset_needed(df)
    assert list(subset.columns) == panel_extract.NEEDED

    summary = summary_blocks.pick_latest_baseline_trend_yoy(subset, "m1")
    required_keys = {"latest_ym", "age_top3", "flow", "kpi", "trend"}
    assert required_keys.issubset(summary)
    assert len(summary.get("age_top3") or []) <= 3
    age_codes = [item.get("code") for item in summary.get("age_all") or []]
    assert len(age_codes) == len(set(age_codes))
    assert summary.get("flow", {}).keys() >= {"유동", "직장", "거주"}
    assert summary.get("yoy") is not None

    _print("Panel summary", summary)


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
    check_summary_blocks()
    check_prompt_with_rag_block()
    compileall.compile_dir(str(PROJECT_ROOT), quiet=1)


if __name__ == "__main__":
    main()

