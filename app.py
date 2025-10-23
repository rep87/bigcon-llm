from __future__ import annotations

import os as _os

_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_os.environ.setdefault("TRANSFORMERS_NO_ACCELERATE", "1")
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
_os.environ.setdefault("OMP_NUM_THREADS", "2")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import sys, platform

print(
    "[BOOT]",
    {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    },
)

# --- minimal safe import block (panel_extract) ---
import sys, os, traceback

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if not sys.path or sys.path[0] != APP_ROOT:
    if APP_ROOT in sys.path:
        sys.path.remove(APP_ROOT)
    sys.path.insert(0, APP_ROOT)

try:  # pragma: no cover - defensive bootstrap
    from app_core.import_utils import ensure_repo_on_sys_path, import_or_raise
except Exception as import_utils_error:  # pragma: no cover - fallback path
    print(
        f"[import-utils-fallback] using inline helpers due to: {import_utils_error!r}"
    )

    def ensure_repo_on_sys_path(anchor_file: str) -> str:
        repo_root = os.path.dirname(os.path.abspath(anchor_file))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        return repo_root

    def import_or_raise(mod_name: str):
        try:
            return __import__(mod_name, fromlist=["*"])
        except Exception:
            print("[import-utils-fallback] cwd=", os.getcwd())
            print("[import-utils-fallback] sys.path[:5]=", sys.path[:5])
            raise

ensure_repo_on_sys_path(__file__)

try:
    pe = import_or_raise("app_core.panel_extract")
    NEEDED = getattr(pe, "NEEDED", None) \
          or getattr(pe, "REQUIRED_COLS", None) \
          or getattr(pe, "NEEDED_COLS", None)
    subset_needed = getattr(pe, "subset_needed", None) \
                 or getattr(pe, "select_needed", None) \
                 or getattr(pe, "get_required_subset", None)
    if NEEDED is None or subset_needed is None:
        avail = [n for n in dir(pe) if not n.startswith("_")]
        raise ImportError(f"[panel_extract] required symbols missing; avail={sorted(avail)} file={getattr(pe,'__file__','<?>')}")
except Exception as e:
    print("[import-fallback] panel_extract failed:", repr(e))
    traceback.print_exc()
    # keep the app alive with a minimal inline implementation
    NEEDED = [
        "ENCODED_MCT","TA_YM",
        "M12_MAL_1020_RAT","M12_MAL_30_RAT","M12_MAL_40_RAT","M12_MAL_50_RAT","M12_MAL_60_RAT",
        "M12_FME_1020_RAT","M12_FME_30_RAT","M12_FME_40_RAT","M12_FME_50_RAT","M12_FME_60_RAT",
        "MCT_UE_CLN_REU_RAT","MCT_UE_CLN_NEW_RAT",
        "RC_M1_SHC_RSD_UE_CLN_RAT","RC_M1_SHC_WP_UE_CLN_RAT","RC_M1_SHC_FLP_UE_CLN_RAT",
    ]
    def subset_needed(df):
        missing = [c for c in NEEDED if c not in df.columns]
        if missing:
            raise KeyError(f"[panel_fallback] Missing required columns: {sorted(missing)}")
        return df[NEEDED].copy()

    def _fallback_pct(value):
        try:
            if value is None:
                return None
            val = float(value)
        except Exception:
            return None
        if val < 0 or val > 100:
            return None
        return float(val)

    def _fallback_latest_valid_row(df, mct_id, eps: float = 0.5):
        import pandas as pd  # local import to avoid hard dependency at module import time

        sub = df[df["ENCODED_MCT"].astype(str) == str(mct_id)].copy()
        if sub.empty:
            raise ValueError(f"[panel_fallback] No rows for ENCODED_MCT={mct_id}")

        sub["__TA_YM_INT__"] = pd.to_numeric(sub["TA_YM"], errors="coerce").astype("Int64")
        sub = sub.dropna(subset=["__TA_YM_INT__"])
        if sub.empty:
            raise ValueError("[panel_fallback] TA_YM not parseable")

        sub = sub.sort_values("__TA_YM_INT__", ascending=False)
        row = sub.iloc[0].copy()
        row.attrs["guard_fallback"] = True
        return row, int(row["__TA_YM_INT__"])

    # register a synthetic module so downstream imports that depend on panel_extract keep working
    from types import ModuleType

    fallback_module = ModuleType("app_core.panel_extract")
    fallback_module.__dict__.update(
        {
            "__file__": "<panel_extract_fallback>",
            "__package__": "app_core",
            "SAFE_COLS": {
                "id": ["ENCODED_MCT"],
                "period": ["TA_YM"],
                "age_gender": [
                    "M12_MAL_1020_RAT",
                    "M12_MAL_30_RAT",
                    "M12_MAL_40_RAT",
                    "M12_MAL_50_RAT",
                    "M12_MAL_60_RAT",
                    "M12_FME_1020_RAT",
                    "M12_FME_30_RAT",
                    "M12_FME_40_RAT",
                    "M12_FME_50_RAT",
                    "M12_FME_60_RAT",
                ],
                "kpi": ["MCT_UE_CLN_REU_RAT", "MCT_UE_CLN_NEW_RAT"],
                "flow": [
                    "RC_M1_SHC_RSD_UE_CLN_RAT",
                    "RC_M1_SHC_WP_UE_CLN_RAT",
                    "RC_M1_SHC_FLP_UE_CLN_RAT",
                ],
            },
            "NEEDED": NEEDED,
            "NEEDED_COLS": NEEDED,
            "REQUIRED_COLS": NEEDED,
            "subset_needed": subset_needed,
            "select_needed": subset_needed,
            "get_required_subset": subset_needed,
            "_pct": _fallback_pct,
            "latest_valid_row": _fallback_latest_valid_row,
        }
    )
    sys.modules.pop("app_core.panel_extract", None)
    sys.modules["app_core.panel_extract"] = fallback_module
# --- end minimal safe import block ---

import html
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.agent1 import diagnose_and_select
from agents.agent2 import generate_plan

import app_core.config as app_config
import app_core.failsoft as failsoft
import app_core.formatters as formatters
import app_core.summary_blocks as summary_blocks
import pandas as pd
import streamlit as st


def _render_evidence_badge(
    metric: dict | str | None = None,
    meta: dict | None = None,
    *,
    metric_id: str | None = None,
    label: str | None = None,
    value: Any | None = None,
    period: str | None = None,
    comment: str | None = None,
    disabled: bool = False,
    tooltip: str | None = None,
) -> None:
    """Render a compact evidence badge with graceful fallbacks."""

    metric_dict = metric if isinstance(metric, dict) else {}
    meta_dict = meta if isinstance(meta, dict) else {}

    metric_key = metric_id or meta_dict.get("metric_id") or metric_dict.get("id")
    base_label = (
        label
        or meta_dict.get("label")
        or meta_dict.get("title")
        or metric_dict.get("label")
        or metric_dict.get("title")
        or metric_key
        or "근거"
    )

    if value is not None:
        base_value = value
    elif meta_dict.get("value") is not None:
        base_value = meta_dict.get("value")
    else:
        base_value = metric_dict.get("value")

    base_period = (
        period
        or meta_dict.get("period")
        or meta_dict.get("latest_period")
        or meta_dict.get("year")
        or metric_dict.get("period")
        or "—"
    )

    base_comment = (
        comment
        or meta_dict.get("comment")
        or metric_dict.get("comment")
        or tooltip
        or "근거 확인"
    )

    percent_value = _normalize_ratio_to_percent(base_value)
    if percent_value is not None:
        value_text = f"{percent_value:.2f}%"
    elif base_value is None:
        value_text = "—"
    else:
        value_text = str(base_value)

    border_color = "#cccccc" if disabled else "#2f8bfd"
    bg_color = "#f6f6f6" if disabled else "#f0f6ff"
    tooltip_attr = f" title=\"{html.escape(str(tooltip))}\"" if tooltip else ""
    badge_html = (
        f"<div style='border:1px solid {border_color};"
        f"border-radius:6px;padding:6px;margin-bottom:4px;"
        f"background-color:{bg_color};font-size:0.85rem;'"
        f"{tooltip_attr}>"
        f"<b>{html.escape(str(base_label))}</b>: {html.escape(value_text)}"
        f" ({html.escape(str(base_period))}) — {html.escape(str(base_comment))}"
        "</div>"
    )
    st.markdown(badge_html, unsafe_allow_html=True)


def _mem_mb() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        return -1


def safe_step(label: str, fn, *args, **kwargs):
    try:
        with st.spinner(f"{label} 실행 중..."):
            return fn(*args, **kwargs)
    except Exception as e:
        st.error(f"{label} 중 오류가 발생했습니다.")
        st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        st.stop()


from rag.preview import rag_preview_search
from rag.retrieval_tool import build_query_encoder, QueryEncoder


RAG_THRESHOLD_DEFAULT = 0.35


@st.cache_resource(show_spinner=False)
def get_cached_encoder() -> QueryEncoder:
    model = build_query_encoder(device="cpu")
    return QueryEncoder(model)


def _ensure_retriever_encoder(enabled: bool) -> QueryEncoder | None:
    if not enabled:
        return None
    if RETRIEVAL_TOOL is None or RETRIEVAL_INIT_ERROR is not None:
        return None
    encoder = get_cached_encoder()
    try:
        if hasattr(RETRIEVAL_TOOL, "set_query_encoder_factory"):
            RETRIEVAL_TOOL.set_query_encoder_factory(get_cached_encoder)  # type: ignore[arg-type]
        if hasattr(RETRIEVAL_TOOL, "set_query_encoder"):
            RETRIEVAL_TOOL.set_query_encoder(encoder)  # type: ignore[attr-defined]
    except Exception:
        pass
    return encoder


def _pick_best_chunk(
    chunks: list[dict] | None, metas: list[dict] | None
) -> Tuple[Any, Any]:
    if not chunks:
        return None, None
    target_chunk = chunks[0]
    target_meta = metas[0] if metas else None
    return target_chunk, target_meta

# Optional import-time sanity check to catch indentation leaks during development.
if os.getenv("STREAMLIT_COMPILE_GUARD", "0") == "1":  # pragma: no cover - dev aid only
    import py_compile

    py_compile.compile(__file__, doraise=True)

try:
    from rag import RetrievalTool
except Exception:  # pragma: no cover - optional dependency path
    RetrievalTool = None


def _get_debug_raw_snapshot(agent1_json: dict | None) -> dict:
    debug = _get_debug_section(agent1_json)
    snap = debug.get("snapshot")
    if isinstance(snap, dict):
        raw = snap.get("raw")
        if isinstance(raw, dict):
            return raw
    legacy = debug.get("latest_raw_snapshot")
    return legacy if isinstance(legacy, dict) else {}


def _render_main_views(
    question_text: str,
    agent1_payload: dict | None,
    agent2_payload: dict | None,
    *,
    rag_info_override: Dict[str, Any] | None = None,
) -> None:
    flags_snapshot = st.session_state.get("_data_flags", {}).copy()
    structured_payload = failsoft.structured_adapter(agent1_payload)
    weather_payload = failsoft.weather_adapter(question_text, enabled=flags_snapshot.get("use_weather", False))
    external_payload = failsoft.external_adapter(question_text, enabled=flags_snapshot.get("use_external", False))
    if rag_info_override is not None:
        rag_info = rag_info_override
    else:
        rag_info = _compute_rag_info(question_text, flags_snapshot)

    retrieval_payload = rag_info.get("payload") if isinstance(rag_info, dict) else None
    if rag_info.get("error"):
        st.warning(f"RetrievalTool 오류: {rag_info['error']}")
    if retrieval_payload is not None:
        st.session_state['_latest_retrieval'] = retrieval_payload
    else:
        st.session_state['_latest_retrieval'] = None
    st.session_state['_latest_rag_info'] = rag_info
    st.session_state['_latest_rag_prompt_context'] = _prepare_rag_prompt_context(rag_info)

    fail_soft_payload = failsoft.compose_fail_soft_answer(
        question_text,
        structured_payload,
        weather_payload,
        external_payload,
        rag_info,
        flags_snapshot,
    )
    st.session_state['_latest_failsoft'] = fail_soft_payload

    try:
        overview_cached = st.session_state.get('_latest_overview', (None, None, None))
        if isinstance(agent2_payload, dict):
            if retrieval_payload:
                agent2_payload.setdefault("evidence", retrieval_payload.get("evidence", []))
                agent2_payload.setdefault("retrieval_chunks", retrieval_payload.get("chunks", []))
            agent2_payload.setdefault("used_data", fail_soft_payload.get("used_data"))

        cards = []
        if isinstance(agent2_payload, dict):
            raw_cards = (
                agent2_payload.get("answers")
                or agent2_payload.get("recommendations")
                or []
            )
            if isinstance(raw_cards, list):
                cards = raw_cards

        if cards and isinstance(cards[0], dict) and retrieval_payload:
            include_evidence = retrieval_payload.get("include_evidence")
            if include_evidence:
                top_chunks = (
                    retrieval_payload.get("chunks", [])[: retrieval_payload.get("used_k", 0)]
                )
                rag_evi: list[dict[str, Any]] = []
                for chunk in top_chunks:
                    rag_evi.append(
                        {
                            "source": "RAG",
                            "doc_id": chunk.get("doc_id"),
                            "chunk_id": chunk.get("chunk_id"),
                            "score": round(float(chunk.get("score", 0.0)), 3),
                            "preview": (chunk.get("text") or "")[:180],
                        }
                    )
                if rag_evi:
                    existing = cards[0].get("evidence")
                    if isinstance(existing, list):
                        cards[0]["evidence"] = existing + rag_evi
                    else:
                        cards[0]["evidence"] = rag_evi

        render_summary_view(
            agent1_payload,
            agent2_payload or {},
            overview_df=overview_cached[0],
            table_dict=overview_cached[1],
            panel_summary=overview_cached[2],
            retrieval_payload=retrieval_payload,
            question_text=question_text,
        )
    except Exception:
        st.error("요약 뷰를 렌더링하는 중 오류가 발생했습니다.")
        st.code(traceback.format_exc())

    preview_query = (
        ((agent1_payload or {}).get("context") or {})
        .get("parsed", {})
        .get("original_question")
        or question_text
    )
    selected_doc_ids = st.session_state.get("rag_selected_docs")
    if selected_doc_ids is None:
        selected_doc_ids = (
            st.session_state.get("_data_flags", {}).get("rag_selected_ids")
        )
    rag_mode_value = str(st.session_state.get("_data_flags", {}).get("rag_mode", "auto"))
    render_rag_preview(preview_query, selected_doc_ids, rag_mode_value)


def run_retrieval_preview(
    query_text: str,
    selected_docs: list[str] | None,
    *,
    rag_mode: str,
) -> dict:
    cleaned_docs = [str(doc_id) for doc_id in (selected_docs or []) if str(doc_id)]
    rag_requested = rag_mode != "off"
    query_text = (query_text or "").strip()
    base_payload = {
        "query": query_text,
        "model_name": "unknown",
        "k": 8,
        "threshold": RAG_THRESHOLD_DEFAULT,
        "selected_docs": cleaned_docs if cleaned_docs else None,
        "items": [],
        "reason": "rag_disabled" if not rag_requested else "no_docs",
        "error": None,
    }
    if not rag_requested:
        return base_payload
    if RETRIEVAL_TOOL is None or RETRIEVAL_INIT_ERROR is not None:
        payload = dict(base_payload)
        payload.update(
            {
                "reason": "unavailable",
                "error": RETRIEVAL_INIT_ERROR or "RetrievalTool unavailable",
            }
        )
        return payload
    if not cleaned_docs:
        return base_payload

    _ensure_retriever_encoder(True)
    start_ts = time.time()
    print(
        f"[STEP] RAG preview start mem={_mem_mb()}MB docs={len(cleaned_docs)} query_len={len(query_text)}"
    )
    payload = rag_preview_search(
        query_text,
        k=8,
        threshold=RAG_THRESHOLD_DEFAULT,
        selected_docs=cleaned_docs,
        retriever=RETRIEVAL_TOOL,
    )
    hits = payload.get("items", []) if isinstance(payload, dict) else []
    print(
        f"[STEP] RAG preview done mem={_mem_mb()}MB dt={time.time()-start_ts:.2f}s hits={len(hits)}"
    )
    if isinstance(payload, dict):
        payload.setdefault("selected_docs", cleaned_docs)
        payload.setdefault("query", query_text)
        payload.setdefault("threshold", RAG_THRESHOLD_DEFAULT)
        payload.setdefault("k", 8)
        payload.setdefault("reason", "ok")
        payload.setdefault("error", None)
        return payload
    return base_payload


def render_rag_preview(
    query_text: str, selected_doc_ids: list[str] | None, rag_mode: str
) -> None:
    payload = run_retrieval_preview(
        query_text,
        selected_doc_ids,
        rag_mode=rag_mode,
    )
    hits = payload.get("items", []) or []
    model_name = payload.get("model_name", "unknown")
    reason = payload.get("reason", "ok")
    error_text = payload.get("error")

    with st.expander("🔍 RAG 유사도 미리보기", expanded=False):
        selected_display = payload.get("selected_docs") or "ALL"
        st.caption(
            f"Query encoder: **{model_name}** · threshold: {payload.get('threshold')} · selected: {selected_display}"
        )
        if reason != "ok":
            reason_text = f"Reason: `{reason}`"
            if error_text:
                reason_text += f" · error: `{error_text}`"
            st.caption(reason_text)

        if not hits:
            st.write("No retrieved chunks.")
            return

        rows = []
        for idx, hit in enumerate(hits, start=1):
            score = getattr(hit, "score", None)
            doc_id = getattr(hit, "doc_id", None)
            chunk_id = getattr(hit, "chunk_id", None)
            text = getattr(hit, "text", None)
            if isinstance(hit, dict):
                score = hit.get("score")
                doc_id = hit.get("doc_id")
                chunk_id = hit.get("chunk_id")
                text = hit.get("text")
            preview = (text or "")[:200]
            rows.append(
                {
                    "rank": idx,
                    "score": round(float(score or 0.0), 3),
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "preview": preview,
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

def _mask_name(raw: str) -> str:
    if not raw:
        return "—"
    name = str(raw)
    if "*" in name:
        return name
    trimmed = name.strip()
    if len(trimmed) <= 1:
        return trimmed or "—"
    if len(trimmed) == 2:
        return trimmed[0] + "*"
    return trimmed[:2] + ("*" * max(4, len(trimmed) - 2))


def _extract_merchant_name(agent1_json: dict) -> str:
    context = (agent1_json or {}).get("context", {})
    keys = [
        "merchant_masked_name",
        "masked_name",
        "merchant_name_masked",
        "merchant_name",
        "store_name",
    ]
    for key in keys:
        val = context.get(key)
        if val:
            return _mask_name(val)
    return "—"


def _normalize_ratio_to_percent(value: Any) -> float | None:
    """Normalize a ratio or percent-like value into 0-100 scale."""

    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not (num == num):  # NaN guard
        return None
    if num < 0:
        num = 0.0
    if num <= 1:
        return num * 100
    if num <= 100:
        return num
    return num / 100


def _format_percent(value) -> str:
    percent = _normalize_ratio_to_percent(value)
    if percent is None:
        return "—"
    return f"{percent:.2f}%"


def _format_pp_delta(value: Any) -> str:
    try:
        if value is None:
            return "—"
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    if not (num == num):  # NaN guard
        return "—"
    return f"{num:+.1f}"


def _format_ratio_percent(value: Any) -> str:
    percent = _normalize_ratio_to_percent(value)
    if percent is None:
        return "—"
    return f"{percent:.2f}%"


def _format_metric_value(value: Any, unit: str | None) -> str:
    if value is None:
        return "—"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if unit == "ratio":
        percent = _normalize_ratio_to_percent(num)
        if percent is None:
            return "—"
        return f"{percent:.2f}%"
    if unit == "currency":
        return f"{num:,.0f}원"
    if unit == "count":
        return f"{num:,.0f}"
    if unit == "pp":
        sign = "+" if num > 0 else ""
        return f"{sign}{num:.2f}pp"
    return f"{num:,.2f}"


def _resolve_panel_path() -> Path:
    override = app_config.get_setting("PANEL_CSV_PATH", None)
    if override:
        return Path(str(override)).expanduser()

    data_root = app_config.get_setting("DATA_DIR", None)
    base = Path(str(data_root)).expanduser() if data_root else Path("data")
    return (base / "shinhan" / "big_data_set3_f.csv").expanduser()


@lru_cache(maxsize=1)
def _load_panel_csv_cached(path_str: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path_str, usecols=NEEDED)
    except ValueError:
        frame = pd.read_csv(path_str)
    frame = subset_needed(frame)
    frame["ENCODED_MCT"] = frame["ENCODED_MCT"].astype(str)
    return frame


def _get_panel_dataframe() -> pd.DataFrame | None:
    path = _resolve_panel_path()
    if not path.exists():
        return None
    try:
        return _load_panel_csv_cached(path.as_posix())
    except Exception:
        return None


def _collect_major_customers(agent1_json: dict) -> str:
    buckets = formatters.merge_age_buckets(agent1_json or {})
    if not buckets:
        return "—"
    segments: list[str] = []
    visible_limit = 3
    for bucket in buckets[:visible_limit]:
        label = bucket.get("label") or bucket.get("key") or "—"
        value = bucket.get("value")
        if isinstance(value, (int, float)):
            segments.append(f"{label} {value:.1f}%")
        else:
            segments.append(f"{label} —")
    hidden_count = max(0, len(buckets) - visible_limit)
    if hidden_count:
        segments.append(f"+ {hidden_count}개 더")
    return ", ".join(segments) if segments else "—"


def _format_customer_mix(detail: dict | None) -> str:
    if not detail or not isinstance(detail, dict):
        return "—"
    ordered_labels = ["유동", "거주", "직장"]
    parts: list[str] = []
    for label in ordered_labels:
        value = detail.get(label)
        if value is not None:
            percent = _format_percent(value)
            if percent != "—":
                parts.append(f"{label} {percent}")
    for label, value in detail.items():
        if label in ordered_labels or value is None:
            continue
        percent = _format_percent(value)
        if percent != "—":
            parts.append(f"{label} {percent}")
    return ", ".join(parts[:3]) if parts else "—"


def _extract_merchant_id(agent1_json: dict | None) -> str | None:
    if not agent1_json:
        return None
    context = (agent1_json.get("context") or {}) if isinstance(agent1_json, dict) else {}
    merchant = context.get("merchant") or {}
    merchant_id = merchant.get("encoded_mct") or merchant.get("ENCODED_MCT")
    if merchant_id is None:
        debug_section = _get_debug_section(agent1_json)
        resolve_block = debug_section.get("resolve") if isinstance(debug_section, dict) else None
        if isinstance(resolve_block, dict):
            merchant_id = resolve_block.get("resolved_merchant_id")
    if merchant_id is None:
        return None
    return str(merchant_id)


def _collect_overview_row(
    agent1_json: dict,
) -> tuple[pd.DataFrame, dict, dict | None]:
    context = (agent1_json or {}).get("context", {}) or {}
    parsed = context.get("parsed", {}) or {}
    merchant = context.get("merchant", {}) or {}

    industry_candidate = (
        merchant.get("category")
        or parsed.get("merchant_industry_label")
        or parsed.get("industry")
    )
    industry_labels = {
        "cafe": "카페",
        "restaurant": "음식점",
        "retail": "소매",
    }
    industry = industry_labels.get(industry_candidate, industry_candidate or "—")

    addr = (
        merchant.get("address")
        or context.get("address_masked")
        or context.get("address")
        or context.get("addr_base")
    )
    if isinstance(addr, (list, tuple)):
        addr = " / ".join(str(v) for v in addr if v)
    address = addr if addr else "—"

    merchant_id = _extract_merchant_id(agent1_json)

    panel_summary: dict | None = None
    panel_df = _get_panel_dataframe()
    if merchant_id and isinstance(panel_df, pd.DataFrame):
        try:
            subset = panel_df[panel_df["ENCODED_MCT"] == merchant_id]
            if not subset.empty:
                panel_summary = summary_blocks.pick_latest_baseline_trend_yoy(subset, merchant_id)
        except Exception:
            panel_summary = None

    def _fmt_age_text(summary: dict | None) -> str:
        if not summary:
            return "—"
        age_top3 = summary.get("age_top3") or []
        parts: list[str] = []
        for bucket in age_top3:
            label = bucket.get("label") or bucket.get("code") or "—"
            pct = _format_percent(bucket.get("value"))
            parts.append(f"{label} {pct}")
        if not parts:
            return "—"
        hidden = max(0, len(summary.get("age_all") or []) - len(age_top3))
        if hidden:
            parts.append(f"+ {hidden}개 더")
        return ", ".join(parts)

    def _fmt_flow_text(summary: dict | None) -> str:
        if not summary:
            return "—"
        flow = summary.get("flow") or {}
        order = ["유동", "직장", "거주"]
        parts = []
        valid = False
        for key in order:
            pct = _format_percent(flow.get(key))
            if pct != "—":
                valid = True
            parts.append(f"{key} {pct}")
        return ", ".join(parts) if valid else "—"

    def _fmt_kpi_text(summary: dict | None) -> str:
        if not summary:
            return "—"
        kpi = summary.get("kpi") or {}
        new_pct = _format_percent(kpi.get("new_rate"))
        revisit_pct = _format_percent(kpi.get("revisit_rate"))
        if new_pct == "—" and revisit_pct == "—":
            return "—"
        return f"신규 {new_pct} / 재방문 {revisit_pct}"

    def _fmt_trend_text(summary: dict | None) -> str:
        if not summary:
            return "—"
        trend = summary.get("trend") or {}
        if "revisit_pp_vs_2m" in trend or "new_pp_vs_2m" in trend:
            revisit = _format_pp_delta(trend.get("revisit_pp_vs_2m"))
            new = _format_pp_delta(trend.get("new_pp_vs_2m"))
            if revisit == "—" and new == "—":
                return "—"
            return f"최근 3개월: 재방문 {revisit}pp / 신규 {new}pp"
        if "revisit_pp_vs_1m" in trend or "new_pp_vs_1m" in trend:
            revisit = _format_pp_delta(trend.get("revisit_pp_vs_1m"))
            new = _format_pp_delta(trend.get("new_pp_vs_1m"))
            if revisit == "—" and new == "—":
                return "—"
            return f"최근 2개월: 재방문 {revisit}pp / 신규 {new}pp"
        return "—"

    def _fmt_yoy_text(summary: dict | None) -> str:
        if not summary:
            return "—"
        yoy = summary.get("yoy")
        if not yoy:
            return "—"
        revisit = _format_pp_delta(yoy.get("revisit_pp_yoy"))
        new = _format_pp_delta(yoy.get("new_pp_yoy"))
        if revisit == "—" and new == "—":
            return "전년동월: —"
        return f"전년동월: 재방문 {revisit}pp / 신규 {new}pp"

    def _fmt_latest_month(summary: dict | None) -> str:
        if not summary:
            return "—"
        latest = summary.get("latest_ym")
        if not latest:
            return "—"
        ym = str(latest)
        if len(ym) < 6:
            return "—"
        year = ym[:4]
        month = ym[4:6]
        if not month.isdigit():
            return "—"
        return f"{year}.{month}"

    table_payload = {
        "업종": industry,
        "주소": address,
        "대표월": _fmt_latest_month(panel_summary),
        "주요 고객층": _fmt_age_text(panel_summary),
        "유입유형": _fmt_flow_text(panel_summary),
        "KPI": _fmt_kpi_text(panel_summary),
        "추세": _fmt_trend_text(panel_summary),
        "YoY": _fmt_yoy_text(panel_summary),
    }

    return pd.DataFrame([table_payload]), table_payload, panel_summary
def _build_goal_lines(agent1_json: dict) -> tuple[str, list[str]]:
    period = (agent1_json or {}).get("period", {})
    months = period.get("months")
    weeks_requested = period.get("weeks_requested")
    if weeks_requested:
        try:
            weeks_val = int(weeks_requested)
        except (TypeError, ValueError):
            weeks_val = None
        if weeks_val and months:
            period_text = f"향후 {weeks_val}주 (약 {months}개월)"
        elif weeks_val:
            period_text = f"향후 {weeks_val}주"
        else:
            period_text = "기간 정보 —"
    elif months:
        period_text = f"최근 {months}개월"
    else:
        period_text = "기간 정보 —"

    debug_snapshot = _get_debug_snapshot(agent1_json)
    kpis = (agent1_json or {}).get("kpis", {})
    mapping = [
        ("revisit_rate_avg", "재방문율"),
        ("new_rate_avg", "신규 고객 비중"),
        ("youth_share_avg", "청년 고객 비중"),
    ]
    lines = []
    for key, label in mapping:
        sanitized_key = {
            "revisit_rate_avg": "revisit_pct",
            "new_rate_avg": "new_pct",
            "youth_share_avg": "youth_pct",
        }.get(key)
        value = debug_snapshot.get(sanitized_key, kpis.get(key))
        if value is not None:
            lines.append(f"{label}: 현황 {_format_percent(value)} → 목표 구간 —")
    if not lines:
        lines.append("KPI 목표 구간 —")
    return period_text, lines[:3]


def _format_list(values) -> str:
    if not values:
        return "—"
    if isinstance(values, (list, tuple)):
        items = [str(v) for v in values if v]
        return " · ".join(items) if items else "—"
    return str(values)


def _render_status_summary(
    table_dict: dict | None,
    panel_summary: dict | None,
) -> None:
    if not table_dict:
        st.info("요약 정보가 없습니다.")
        return

    rows = [
        ("업종", table_dict.get("업종", "—")),
        ("주소", table_dict.get("주소", "—")),
        ("대표월", table_dict.get("대표월", "—")),
        ("주요 고객층", table_dict.get("주요 고객층", "—")),
        ("유입유형", table_dict.get("유입유형", "—")),
        ("KPI", table_dict.get("KPI", "—")),
        ("추세", table_dict.get("추세", "—")),
        ("YoY", table_dict.get("YoY", "—")),
    ]
    summary_df = pd.DataFrame(rows, columns=["항목", "값"])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if panel_summary is None:
        st.caption("패널 요약 정보가 없습니다.")
        return

    if len(panel_summary.get("age_all") or []) > 3:
        details = []
        for bucket in panel_summary.get("age_all", []):
            label = bucket.get("label") or bucket.get("code") or "—"
            pct = _format_percent(bucket.get("value"))
            details.append(f"{label} {pct}")
        with st.expander("전체 연령 비중 보기", expanded=False):
            st.markdown(", ".join(details) if details else "—")

    if panel_summary.get("guard_fallback"):
        st.caption("최근 월 데이터로 대체된 요약입니다.")


MODE_LABELS = {
    "basic": "기본 데이터 접근",
    "advanced": "고급 데이터 접근 (EDA/조합 포함)",
}


def render_profile_table(
    rows: List[Dict[str, Any]], title: str, accent: bool = False
) -> None:
    st.subheader(title)
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return

    working = df.copy()
    working["최근값"] = working.apply(
        lambda r: _format_metric_value(r.get("value_recent"), r.get("unit", None)),
        axis=1,
    )
    working["직전값"] = working.apply(
        lambda r: _format_metric_value(r.get("value_prev"), r.get("unit", None)),
        axis=1,
    )
    working["커버리지"] = working.get("coverage", pd.Series([None] * len(working))).apply(
        _format_ratio_percent
    )
    working["선택"] = working.get("selected", pd.Series([False] * len(working))).apply(
        lambda value: "선택" if bool(value) else ""
    )

    display_df = pd.DataFrame(
        {
            "Source": working.get("source", pd.Series(["—"] * len(working))),
            "Column": working.get("column", pd.Series(["—"] * len(working))),
            "Label": working.get("label", pd.Series(["—"] * len(working))),
            "최근값": working["최근값"],
            "직전값": working["직전값"],
            "커버리지": working["커버리지"],
            "최신월": working.get("recency", pd.Series(["—"] * len(working))),
            "가용기간": working.get("time_span", pd.Series(["—"] * len(working))),
            "충분도": working.get("sufficiency", pd.Series(["—"] * len(working))),
            "선택": working["선택"],
        }
    )

    def _style_row(row: pd.Series) -> List[str]:
        highlight = row.get("선택") == "선택"
        border_color = "#ff4b4b" if highlight else ("#2f8bfd" if accent else "")
        style = f"border-top:{border_color}; border-bottom:{border_color};" if border_color else ""
        return [style] * len(row)

    st.dataframe(
        display_df.style.apply(_style_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )


def render_selection_reasons(reasons: List[Dict[str, Any]]) -> None:
    st.markdown("### 🎯 선택 사유 (한 줄 설명)")
    if not reasons:
        st.markdown("🧩 선택된 근거가 없습니다.")
        return
    for item in reasons:
        label = item.get("label") or item.get("column") or "지표"
        reason = item.get("reason") or "사유 없음"
        st.markdown(f"🧩 **{label}** — {reason}")


def _render_data_profile_section(
    question_text: str | None,
    merchant_title: str,
    merchant_id: str | None,
    analysis_mode: str,
) -> dict | None:
    st.caption(
        "신한카드 패널 데이터에서 해당 가맹점의 모든 주요 지표를 수집해 표로 정리했습니다."
    )
    if not merchant_id:
        st.info("가맹점 ID를 확인할 수 없어 데이터 프로파일을 표시할 수 없습니다.")
        return None

    cache = st.session_state.setdefault("_data_profile_cache", {})
    cache_key = f"{merchant_id}::{question_text or ''}::{analysis_mode}"
    if cache_key in cache:
        profile_payload = cache[cache_key]
    else:
        try:
            profile_payload = diagnose_and_select(
                question_text or "",
                merchant_id,
                analysis_mode=analysis_mode,
            )
        except Exception as exc:  # pragma: no cover - defensive for UI
            st.warning(f"데이터 프로파일 생성에 실패했습니다: {exc}")
            return None
        cache[cache_key] = profile_payload

    profile_basic = profile_payload.get("profile_basic") or []
    profile_advanced = profile_payload.get("profile_advanced") or []
    selected_features = profile_payload.get("selected_features") or []
    selection_reasons = profile_payload.get("selection_reasons") or []
    sufficiency_summary = profile_payload.get("sufficiency_summary") or "데이터 충분"

    st.markdown(f"**{merchant_title}** 가맹점의 최신 데이터 스냅샷입니다.")

    if not profile_basic:
        st.info("표시할 프로파일 데이터가 없습니다.")
    else:
        render_profile_table(profile_basic, "신한카드 데이터 요약 (원본 컬럼)")

    if analysis_mode == "advanced" and profile_advanced:
        render_profile_table(profile_advanced, "파생(조합) 컬럼 요약", accent=True)

    with st.expander("선택 기준 보기(룰 카드)", expanded=False):
        st.markdown(
            "- Q1: 연령·성별/생활권/재방문·신규 지표, 최근 3개월 커버리지 60% 이상\n"
            "- Q2: 재방문·신규·생활권 지표, 커버리지 50% 이상\n"
            "- Q3: 매출·이용·배달 지표, 커버리지 50% 이상\n"
            "- 질문 키워드와 role_tags 매칭으로 선택 사유를 자동 생성합니다."
        )

    if sufficiency_summary == "데이터 부족":
        st.warning("선택된 지표의 커버리지가 부족해 보수적인 해석이 필요합니다.")
    else:
        st.success("선택된 지표의 커버리지가 충분합니다.")

    mode_label = MODE_LABELS.get(analysis_mode, analysis_mode)
    st.caption(
        f"현재 모드: {mode_label} — 선택된 지표 {len(selected_features)}개, 선택 사유 {len(selection_reasons)}개"
    )

    return profile_payload


def render_summary_view(
    agent1_json: dict,
    agent2_json: dict,
    overview_df: pd.DataFrame | None = None,
    table_dict: dict | None = None,
    panel_summary: dict | None = None,
    retrieval_payload: dict | None = None,
    question_text: str | None = None,
) -> None:
    merchant_title = _extract_merchant_name(agent1_json)
    merchant_id = _extract_merchant_id(agent1_json)
    st.header(f"📊 {merchant_title} 가맹점 방문 고객 현황 분석")

    mode_options = [
        ("기본 데이터 접근", "basic"),
        ("고급 데이터 접근 (EDA/조합 포함)", "advanced"),
    ]
    default_mode = st.session_state.get("analysis_mode", "basic")
    codes = [code for _, code in mode_options]
    try:
        default_index = codes.index(default_mode)
    except ValueError:
        default_index = 0
    selected_label = st.radio(
        "데이터 접근 수준 선택",
        [label for label, _ in mode_options],
        index=default_index,
        horizontal=True,
    )
    analysis_mode = next(code for label, code in mode_options if label == selected_label)
    st.session_state["analysis_mode"] = analysis_mode

    st.markdown(
        "> 💡 **이 화면은 실제 신한카드 데이터를 기반으로 분석한 과정을 시각적으로 보여줍니다.**  "
        "<br>- ‘기본 데이터 접근’은 단순 지표 수준,  "
        "<br>- ‘고급 데이터 접근’은 조합/EDA를 포함한 확장 분석입니다.  "
        "<br>각 단계별로 어떤 데이터를 근거로 판단했는지를 명확히 드러내,  "
        "<br>사용자가 ‘데이터를 보고 판단했다’는 신뢰를 전달합니다.",
        unsafe_allow_html=True,
    )

    if question_text:
        st.markdown(f"📍 **질문:** {question_text}")
    else:
        st.markdown("📍 **질문:** —")
    st.divider()

    context = (agent1_json or {}).get("context", {})
    if context and not context.get("merchant"):
        st.warning("질문과 정확히 일치하는 가맹점을 찾지 못해 표본 전체 요약을 보여드립니다.")

    debug_info = _get_debug_section(agent1_json)
    if overview_df is None or table_dict is None:
        render_info = debug_info.get("render") if isinstance(debug_info, dict) else None
        if isinstance(render_info, dict) and isinstance(render_info.get("table_dict"), dict):
            table_dict = render_info.get("table_dict")
            panel_summary = render_info.get("panel_summary")
            if overview_df is None:
                overview_df = pd.DataFrame([table_dict])
        else:
            overview_df, table_dict, panel_summary = _collect_overview_row(agent1_json)
            if isinstance(debug_info, dict):
                cache_block = debug_info.setdefault("render", {})
                cache_block["table_dict"] = table_dict
                cache_block["panel_summary"] = panel_summary
    if overview_df is None:
        if table_dict:
            overview_df = pd.DataFrame([table_dict])
        else:
            overview_df = pd.DataFrame()
    is_public_mode = True
    try:
        print(
            "📊 overview_table:",
            json.dumps(overview_df.to_dict(orient="records"), ensure_ascii=False),
        )
    except Exception:
        pass

    if is_public_mode:
        st.subheader("현황 요약")
        _render_status_summary(table_dict, panel_summary)
    else:
        st.subheader("현황 표")
        st.table(overview_df)

    st.markdown("### 📊 신한카드 데이터 요약 (전 컬럼)")
    profile_payload = _render_data_profile_section(
        question_text, merchant_title, merchant_id, analysis_mode
    )
    st.divider()

    selection_reasons = []
    if profile_payload:
        selection_reasons = profile_payload.get("selection_reasons") or []

    render_selection_reasons(selection_reasons)
    st.divider()

    if not is_public_mode:
        period_text, goal_lines = _build_goal_lines(agent1_json)
        st.subheader("목표")
        st.markdown(f"- 기간 가정: {period_text}")
        for line in goal_lines:
            st.markdown(f"- {line}")

    agent2_payload = agent2_json or {}
    answers = agent2_payload.get("answers") or agent2_payload.get("recommendations") or []
    if not isinstance(answers, list):
        answers = []

    data_flags = st.session_state.get("_data_flags", {})
    if not isinstance(data_flags, dict):
        data_flags = {}
    use_structured = bool(data_flags.get("use_structured", True))
    use_weather = bool(data_flags.get("use_weather", False))
    use_external = bool(data_flags.get("use_external", False))

    used_metrics_block = None
    if profile_payload is not None:
        used_metrics_block = generate_plan(
            question_text or "",
            merchant_id or "",
            profile_payload.get("selected_features") or [],
            profile_payload.get("sufficiency_summary", "데이터 충분"),
        )

    st.markdown("### 📑 Agent-2의 마케팅 제안")
    if used_metrics_block:
        st.markdown("📑 **사용 지표**")
        lines = used_metrics_block.get("used_metrics_lines") or []
        if lines:
            for line in lines:
                st.markdown(f"- {line}")
        else:
            st.markdown("- 활용 가능한 지표가 선택되지 않았습니다.")
        tone_message = used_metrics_block.get("tone_message")
        if tone_message:
            st.caption(tone_message)

    if not answers:
        st.info("아이디어 제안이 제공되지 않았습니다.")
    for idx, answer in enumerate(answers[:4], start=1):
        with st.container():
            st.markdown(f"**{idx}. {answer.get('idea_title', '—')}**")
            st.markdown(f"- 대상: {answer.get('audience', '—')}")
            st.markdown(f"- 채널: {_format_list(answer.get('channels'))}")
            st.markdown(f"- 실행: {_format_list(answer.get('execution'))}")
            st.markdown(f"- 카피 샘플: {_format_list(answer.get('copy_samples'))}")
            st.markdown(f"- 측정: {_format_list(answer.get('measurement'))}")

            evidence_items = answer.get("evidence") or []
            if evidence_items:
                st.markdown("**근거**")
                for entry in evidence_items:
                    line = _format_evidence_line(entry)
                    cols = st.columns([12, 1])
                    with cols[0]:
                        st.markdown(line)
                    with cols[1]:
                        source = str(entry.get("source") or "").upper()
                        if source == "RAG":
                            chunk, meta = _match_evidence_chunk(entry, retrieval_payload)
                            if chunk:
                                _render_evidence_badge(chunk, meta)
                            else:
                                _render_evidence_badge(
                                    None,
                                    None,
                                    disabled=True,
                                    tooltip="RAG 근거 매칭 실패",
                                )
                        elif source == "STRUCTURED":
                            _render_evidence_badge(
                                None,
                                None,
                                disabled=not use_structured,
                                tooltip="STRUCTURED 근거",
                            )
                        elif source == "WEATHER":
                            _render_evidence_badge(
                                None,
                                None,
                                disabled=not use_weather,
                                tooltip="WEATHER 근거",
                            )
                        elif source == "EXTERNAL":
                            _render_evidence_badge(
                                None,
                                None,
                                disabled=not use_external,
                                tooltip="EXTERNAL 근거",
                            )
                        else:
                            tooltip = "근거 없음" if source in {"NONE", ""} else source
                            _render_evidence_badge(None, None, disabled=True, tooltip=tooltip)
            else:
                st.markdown("**근거**")
                cols = st.columns([12, 1])
                with cols[0]:
                    st.markdown("- 근거 없음")
                with cols[1]:
                    _render_evidence_badge(None, None, disabled=True, tooltip="근거 없음")

    if not is_public_mode:
        limits = (agent1_json or {}).get("limits", [])
        st.subheader("한계/데이터 보강")
        st.markdown("**현재 한계**")
        if limits:
            for item in limits[:5]:
                st.markdown(f"- {item}")
        else:
            st.markdown("- 한계 정보가 제공되지 않았습니다.")

        improvement_suggestions = []
        for item in limits:
            text = str(item)
            if "날씨" in text:
                improvement_suggestions.append("날씨 데이터 연계를 통해 우천 가설을 검증합니다.")
            elif "표본" in text or "데이터" in text:
                improvement_suggestions.append("누락 구간을 점검해 표본을 보강합니다.")
        if not improvement_suggestions:
            improvement_suggestions.append("다음 스프린트에서 결합 데이터 소스를 재점검합니다.")

        st.markdown("**다음 스프린트 보강 계획**")
        for suggestion in improvement_suggestions[:3]:
            st.markdown(f"- {suggestion}")

        if retrieval_payload and retrieval_payload.get("chunks"):
            st.subheader("임베디드 근거")
            for idx, chunk in enumerate(retrieval_payload.get("chunks", [])[:5], start=1):
                meta = None
                for item in retrieval_payload.get("evidence", []):
                    if (
                        item.get("doc_id") == chunk.get("doc_id")
                        and item.get("chunk_id") == chunk.get("chunk_id")
                    ):
                        meta = item
                        break
                cols = st.columns([12, 1])
                with cols[0]:
                    preview = str(chunk.get("text") or "—").strip()
                    if len(preview) > 160:
                        preview = preview[:160].rstrip() + "…"
                    st.markdown(f"{idx}. {preview}")
                with cols[1]:
                    _render_evidence_badge(chunk, meta)

    st.divider()
    st.markdown(f"📂 접근 수준: **{mode}**")
DEFAULT_RAG_ROOT = "data/rag"
RAG_ROOT = str(app_config.get_setting("RAG_ROOT", DEFAULT_RAG_ROOT))
RAG_EMBED_VERSION = str(app_config.get_setting("RAG_EMBED_VERSION", "embed_v1"))
DEFAULT_QUESTION = "{페로**********} 카페의 주요 방문 고객 특성에 따른 마케팅 채널 추천 및 홍보안을 작성"
RAG_ROOT_PATH = Path(RAG_ROOT).expanduser()
RETRIEVAL_INIT_ERROR: str | None = None
RETRIEVAL_TOOL: object | None = None
RAG_CATALOG: list[Dict[str, Any]] = []
RAG_CATALOG_ERROR: str | None = None

if RetrievalTool is not None:
    try:
        RETRIEVAL_TOOL = RetrievalTool(root=RAG_ROOT, embed_version=RAG_EMBED_VERSION)
    except Exception as exc:  # pragma: no cover - defensive guard for UI
        RETRIEVAL_INIT_ERROR = str(exc)
else:  # pragma: no cover - module missing
    RETRIEVAL_INIT_ERROR = "rag.RetrievalTool 모듈을 불러오지 못했습니다."

if RETRIEVAL_TOOL is not None and RETRIEVAL_INIT_ERROR is None:
    if not RAG_ROOT_PATH.exists():
        RAG_CATALOG_ERROR = f"RAG_ROOT 경로({RAG_ROOT_PATH})가 존재하지 않습니다."
    else:
        try:
            catalog_entries = RETRIEVAL_TOOL.load_catalog()
            for entry in catalog_entries:
                origin_path = entry.origin_path
                origin_uri = origin_path
                if origin_path:
                    path_obj = Path(origin_path)
                    if not path_obj.is_absolute():
                        origin_uri = (RAG_ROOT_PATH / path_obj).as_posix()
                    else:
                        origin_uri = path_obj.as_posix()
                RAG_CATALOG.append(
                    {
                        "document_id": entry.doc_id,
                        "title": entry.title,
                        "num_chunks": entry.num_chunks,
                        "embedding_model": entry.embedding_model,
                        "created_at": entry.created_at,
                        "origin_path": origin_uri,
                        "tags": list(entry.tags or []),
                        "year": entry.year,
                    }
                )
        except Exception as exc:  # pragma: no cover - defensive guard
            RAG_CATALOG_ERROR = str(exc)


def _get_debug_section(agent1_json: dict | None) -> dict:
    debug = (agent1_json or {}).get("debug")
    return debug if isinstance(debug, dict) else {}


def _ensure_session_defaults() -> None:
    if "user_query_text" not in st.session_state:
        st.session_state["user_query_text"] = DEFAULT_QUESTION
    if "seeded_default_question" not in st.session_state:
        st.session_state["seeded_default_question"] = True
    if "_data_flags" not in st.session_state:
        st.session_state["_data_flags"] = {
            "use_weather": False,
            "use_external": False,
            "rag_top_k": 5,
            "rag_mode": "auto",
            "rag_filter": "",
            "rag_selected_ids": [],
        }


def _get_debug_snapshot(agent1_json: dict | None) -> dict:
    debug = _get_debug_section(agent1_json)
    snap = debug.get("snapshot")
    if isinstance(snap, dict):
        sanitized = snap.get("sanitized")
        if isinstance(sanitized, dict):
            return sanitized
    legacy = debug.get("sanitized_snapshot")
    return legacy if isinstance(legacy, dict) else {}


def _compute_rag_info(question_text: str, flags_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    selected_docs = [
        str(doc_id)
        for doc_id in (flags_snapshot.get("rag_selected_ids") or [])
        if str(doc_id)
    ]
    rag_mode = str(flags_snapshot.get("rag_mode", "auto"))
    rag_requested = rag_mode != "off"
    rag_top_k = int(flags_snapshot.get("rag_top_k", 5))
    rag_enabled = rag_requested and bool(selected_docs)
    _ensure_retriever_encoder(rag_enabled)
    return failsoft.rag_adapter(
        question_text,
        RETRIEVAL_TOOL,
        enabled=rag_enabled,
        top_k=rag_top_k,
        threshold=RAG_THRESHOLD_DEFAULT,
        mode=rag_mode,
        requested=rag_requested,
        doc_ids=selected_docs,
    )


def _prepare_rag_prompt_context(rag_info: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(rag_info, dict):
        return None
    payload = rag_info.get("payload") or {}
    chunks = payload.get("chunks") or []
    evidence = payload.get("evidence") or []
    context = {
        "enabled": bool(rag_info.get("enabled")),
        "requested": bool(rag_info.get("requested")),
        "selection_missing": bool(rag_info.get("selection_missing")),
        "selected_doc_ids": list(rag_info.get("selected_doc_ids") or []),
        "threshold": rag_info.get("threshold"),
        "mode": rag_info.get("mode"),
        "max_score": rag_info.get("max_score"),
        "hits": len(chunks),
        "chunks": [dict(chunk) for chunk in chunks],
        "evidence": [dict(item) for item in evidence],
        "error": rag_info.get("error"),
        "catalog_size": rag_info.get("catalog_size"),
        "top_scores": list(payload.get("top_scores") or []),
        "encoder_info": payload.get("encoder_info"),
        "doc_specs": payload.get("doc_specs"),
        "retrieval_warnings": payload.get("warnings"),
    }
    return context


def _shorten_snippet(text: Any, limit: int = 140) -> str:
    snippet = str(text or "").strip()
    if len(snippet) > limit:
        snippet = snippet[:limit].rstrip() + "…"
    return snippet


def _format_evidence_line(entry: Dict[str, Any]) -> str:
    source = str(entry.get("source") or "NONE").upper()
    key = entry.get("key") or "—"
    value = entry.get("value")
    if value is None:
        value_text = "—"
    else:
        value_text = str(value)
    period = entry.get("period")
    snippet = entry.get("snippet")
    parts = [f"[{source}] {key}: {value_text}"]
    if period:
        parts.append(f"({period})")
    line = " ".join(parts)
    if snippet:
        line += f" — {_shorten_snippet(snippet)}"
    return f"- {line}"


def _match_evidence_chunk(
    entry: Dict[str, Any],
    retrieval_payload: Dict[str, Any] | None,
) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    if not isinstance(entry, dict) or not retrieval_payload:
        return None, None
    doc_id = entry.get("doc_id")
    if not doc_id:
        return None, None
    chunk_id = entry.get("chunk_id")
    chunks = retrieval_payload.get("chunks") or []
    evidence_list = retrieval_payload.get("evidence") or []
    chunk_id_str = str(chunk_id) if chunk_id is not None else None
    matching_chunks = [
        chunk for chunk in chunks if str(chunk.get("doc_id")) == str(doc_id)
    ]
    if chunk_id_str is not None:
        exact_chunks = [
            chunk
            for chunk in matching_chunks
            if str(chunk.get("chunk_id")) == chunk_id_str
        ]
        if exact_chunks:
            matching_chunks = exact_chunks

    matching_metas = [
        meta for meta in evidence_list if str(meta.get("doc_id")) == str(doc_id)
    ]
    if chunk_id_str is not None:
        exact_metas = [
            meta
            for meta in matching_metas
            if str(meta.get("chunk_id")) == chunk_id_str
        ]
        if exact_metas:
            matching_metas = exact_metas

    return _pick_best_chunk(matching_chunks, matching_metas)


def _get_debug_raw_snapshot(agent1_json: dict | None) -> dict:
    debug = _get_debug_section(agent1_json)
    snap = debug.get("snapshot")
    if isinstance(snap, dict):
        raw = snap.get("raw")
        if isinstance(raw, dict):
            return raw
    legacy = debug.get("latest_raw_snapshot")
    return legacy if isinstance(legacy, dict) else {}


def main() -> None:
    _ensure_session_defaults()

    # ===== 경로 & 키 =====
    DATA_DIR = Path("data")
    SHINHAN_DIR = DATA_DIR / "shinhan"
    EXTERNAL_DIR = DATA_DIR / "external"

    API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    if not API_KEY:
        st.warning("GEMINI_API_KEY가 설정되지 않았습니다. (앱 설정의 App secrets에 등록하세요)")

    st.sidebar.header("⚙️ 설정")
    st.sidebar.slider(
        "LLM 출력 토큰 한도",
        min_value=256,
        max_value=8192,
        step=256,
        value=int(st.session_state.get("llm_out_max", 2048)),
        key="llm_out_max",
    )

    # ===== 사이드바: 데이터 상태 =====
    st.sidebar.header("데이터 상태")
    st.sidebar.write(f"📁 SHINHAN_DIR 존재: {SHINHAN_DIR.exists()}")
    st.sidebar.write(f"📁 EXTERNAL_DIR 존재: {EXTERNAL_DIR.exists()}")

    data_flags = st.session_state.get("_data_flags", {}).copy()
    st.sidebar.header("Data Sources")

    data_flags["use_weather"] = st.sidebar.toggle(
        "Use Weather Data",
        value=bool(data_flags.get("use_weather", False)),
    )
    data_flags["use_external"] = st.sidebar.toggle(
        "Use External APIs",
        value=bool(data_flags.get("use_external", False)),
    )

    rag_root_exists = RAG_ROOT_PATH.exists()
    if rag_root_exists:
        st.sidebar.caption(f"RAG Root: {RAG_ROOT_PATH}")
    else:
        st.sidebar.info(f"RAG_ROOT 경로({RAG_ROOT_PATH})가 없어 RAG를 사용할 수 없습니다.")

    if RAG_CATALOG_ERROR:
        st.sidebar.warning(RAG_CATALOG_ERROR)

    rag_disabled = (
        RETRIEVAL_TOOL is None
        or RETRIEVAL_INIT_ERROR is not None
        or not rag_root_exists
        or RAG_CATALOG_ERROR is not None
    )
    data_flags["rag_top_k"] = int(
        st.sidebar.slider(
            "RAG top_k",
            min_value=3,
            max_value=10,
            value=int(data_flags.get("rag_top_k", 5)),
            step=1,
            disabled=rag_disabled,
        )
    )
    rag_modes = ["off", "auto", "always"]
    rag_mode_value = data_flags.get("rag_mode", "auto")
    if rag_mode_value not in rag_modes:
        rag_mode_value = "auto"
    if rag_disabled:
        rag_mode_value = "off"
    data_flags["rag_mode"] = st.sidebar.selectbox(
        "RAG Mode",
        options=rag_modes,
        index=rag_modes.index(rag_mode_value),
        disabled=rag_disabled,
    )

    if rag_disabled:
        data_flags["rag_selected_ids"] = []
        st.session_state["rag_selected_docs"] = []
    elif data_flags.get("rag_mode") != "off":
        current_filter = data_flags.get("rag_filter", "")
        current_filter = st.sidebar.text_input(
            "RAG Search Filter (optional)",
            value=current_filter,
            placeholder="문서명, 태그, ID 검색",
        )
        data_flags["rag_filter"] = current_filter

        def _matches_filter(record: Dict[str, Any], query: str) -> bool:
            if not query:
                return True
            haystack_parts = [
                str(record.get("title") or ""),
                str(record.get("document_id") or ""),
            ]
            tags = record.get("tags") or []
            haystack_parts.extend(str(tag) for tag in tags)
            haystack = " ".join(haystack_parts).lower()
            return query.lower() in haystack

        filtered_catalog = [
            item for item in RAG_CATALOG if _matches_filter(item, current_filter.strip())
        ]

        if not filtered_catalog:
            st.sidebar.info("필터와 일치하는 문서가 없습니다. 필터를 비워주세요.")

        label_map = {
            f"{item['title']} ({item['document_id']})": item["document_id"]
            for item in filtered_catalog
        }
        options = list(label_map.keys())

        previous_selection = data_flags.get("rag_selected_ids") or []
        default_labels = [label for label, doc_id in label_map.items() if doc_id in previous_selection]
        if not default_labels and not previous_selection and len(options) <= 20:
            default_labels = options

        selected_labels = st.sidebar.multiselect(
            "Select RAG Documents",
            options,
            default=default_labels,
        )
        selected_ids = [label_map[label] for label in selected_labels]
        data_flags["rag_selected_ids"] = selected_ids
        st.session_state["rag_selected_docs"] = selected_ids

        if data_flags.get("rag_mode") != "off" and not selected_ids:
            st.sidebar.info("선택된 RAG 문서가 없어 이번 실행에서는 RAG를 사용하지 않습니다.")
    elif not rag_disabled:
        # Toggle가 꺼져 있으면 기존 선택을 유지하되 필터 입력만 초기화하지 않습니다.
        data_flags.setdefault("rag_selected_ids", data_flags.get("rag_selected_ids", []))
        st.session_state.setdefault(
            "rag_selected_docs", data_flags.get("rag_selected_ids", [])
        )

    st.session_state["_data_flags"] = data_flags

    # ===== 탭 구성 =====
    analysis_tab, sources_tab = st.tabs(["📈 분석", "📚 Embedded Sources"])

    with analysis_tab:
        if "user_query_text" not in st.session_state:
            st.session_state["user_query_text"] = DEFAULT_QUESTION

        question = st.text_area(
            "질문을 입력하세요",
            key="user_query_text",
            placeholder=DEFAULT_QUESTION,
            height=120,
        )
        st.caption("상호는 반드시 {}로 감싸 주세요. 예) 성동구 {동대******}")

        run_analysis = st.button("분석 실행", type="primary")
        if run_analysis:
            from bigcon_2agent_mvp_v3 import (
                agent1_pipeline,
                build_agent2_prompt,
                call_gemini_agent2,
                AGENT2_PROMPT_TRACE,
                AGENT2_RESPONSE_TRACE,
                infer_question_type,
            )

            def _run_agent1_step():
                step_t0 = time.time()
                print(f"[STEP] Agent-1 start mem={_mem_mb()}MB")
                agent1_payload = agent1_pipeline(question, SHINHAN_DIR, EXTERNAL_DIR)
                try:
                    overview_df, table_dict, panel_summary = _collect_overview_row(agent1_payload)
                except Exception:
                    overview_df, table_dict, panel_summary = pd.DataFrame(), {}, None
                if isinstance(agent1_payload, dict):
                    dbg = _get_debug_section(agent1_payload)
                    render_cache = dbg.setdefault("render", {})
                    render_cache["table_dict"] = table_dict
                    render_cache["panel_summary"] = panel_summary
                    agent1_payload["debug"] = dbg
                st.session_state["_latest_overview"] = (
                    overview_df,
                    table_dict,
                    panel_summary,
                )
                st.session_state["_latest_agent1"] = agent1_payload
                st.session_state["_latest_question"] = question
                st.success("Agent-1 JSON 생성 완료")
                print(
                    f"[STEP] Agent-1 done  mem={_mem_mb()}MB dt={time.time()-step_t0:.2f}s"
                )
                return agent1_payload

            a1 = safe_step("Agent-1", _run_agent1_step)

            def _prepare_rag_assets():
                flags_snapshot = st.session_state.get("_data_flags", {}).copy()
                rag_info_payload = _compute_rag_info(question, flags_snapshot)
                rag_prompt_context = _prepare_rag_prompt_context(rag_info_payload)
                question_type_value = infer_question_type(question)
                st.session_state["_latest_rag_info"] = rag_info_payload
                st.session_state["_latest_question_type"] = question_type_value
                st.session_state["_latest_rag_prompt_context"] = rag_prompt_context
                return flags_snapshot, rag_info_payload, rag_prompt_context, question_type_value

            (
                flags_snapshot,
                rag_info_for_prompt,
                rag_prompt_context,
                question_type,
            ) = safe_step("RAG 구성", _prepare_rag_assets)

            selected_for_preview = flags_snapshot.get("rag_selected_ids")
            rag_mode_for_preview = str(flags_snapshot.get("rag_mode", "auto"))
            rag_preview_payload = safe_step(
                "RAG 미리보기",
                run_retrieval_preview,
                question,
                selected_for_preview,
                rag_mode=rag_mode_for_preview,
            )
            st.session_state["_latest_rag_preview"] = rag_preview_payload

            def _run_agent2_step(
                agent1_json: dict | None,
                question_text: str,
                question_type_value: str | None,
                rag_context: Dict[str, Any] | None,
            ):
                os.environ["GEMINI_API_KEY"] = API_KEY
                prompt_text = build_agent2_prompt(
                    agent1_json,
                    question_text=question_text,
                    question_type=question_type_value,
                    rag_context=rag_context,
                )
                if isinstance(AGENT2_PROMPT_TRACE, dict):
                    st.session_state["_latest_prompt_trace"] = dict(AGENT2_PROMPT_TRACE)
                else:
                    st.session_state["_latest_prompt_trace"] = {}
                agent2_payload = call_gemini_agent2(
                    prompt_text,
                    question_type=question_type_value,
                    agent1_json=agent1_json,
                )
                st.success("Agent-2 카드 생성 완료")
                st.session_state["_latest_agent2"] = agent2_payload
                if isinstance(AGENT2_RESPONSE_TRACE, dict):
                    st.session_state["_latest_response_trace"] = dict(AGENT2_RESPONSE_TRACE)
                else:
                    st.session_state["_latest_response_trace"] = {}
                return agent2_payload

            result = safe_step(
                "Agent-2",
                _run_agent2_step,
                a1,
                question,
                question_type,
                rag_prompt_context,
            )

            _render_main_views(
                question,
                a1,
                result,
                rag_info_override=rag_info_for_prompt,
            )

        elif st.session_state.get('_latest_agent1') and st.session_state.get('_latest_agent2'):
            latest_agent1 = st.session_state.get('_latest_agent1')
            latest_agent2 = st.session_state.get('_latest_agent2')
            question_snapshot = st.session_state.get('_latest_question', question)
            _render_main_views(question_snapshot, latest_agent1, latest_agent2)

        latest_agent2 = st.session_state.get('_latest_agent2')
        if isinstance(latest_agent2, dict):
            with st.expander("🧾 Agent-2 출력(JSON) 보기", expanded=False):
                st.json(latest_agent2)
        latest_retrieval = st.session_state.get('_latest_retrieval')
        if latest_retrieval:
            with st.expander("📎 Retrieval Evidence (JSON)", expanded=False):
                st.json(latest_retrieval)
        latest_agent1 = st.session_state.get('_latest_agent1')
        if isinstance(latest_agent1, dict):
            with st.expander("🔎 Agent-1 출력(JSON) 보기", expanded=False):
                st.json(latest_agent1)

        if not st.session_state.get("_intro_shown"):
            st.info("✅ 업로드 성공! 이제 질문 입력 후 [분석 실행]을 눌러 카드 결과를 확인해보세요.")
            st.session_state["_intro_shown"] = True

    with sources_tab:
        st.subheader("임베디드 소스 카탈로그")
        st.caption(f"RAG Root: {RAG_ROOT_PATH}")
        if RETRIEVAL_INIT_ERROR:
            st.error(f"RetrievalTool 초기화 실패: {RETRIEVAL_INIT_ERROR}")
        elif RETRIEVAL_TOOL is None:
            st.info("RetrievalTool이 비활성화되어 있습니다.")
        elif not RAG_ROOT_PATH.exists():
            st.info("data/rag 경로가 존재하지 않습니다. corpus/와 indices/를 구성한 뒤 다시 시도하세요.")
        elif RAG_CATALOG_ERROR:
            st.warning(RAG_CATALOG_ERROR)
        else:
            catalog_df = pd.DataFrame(RAG_CATALOG)
            if catalog_df.empty:
                st.info("등록된 임베디드 문서가 없습니다. data/rag/indices 폴더를 확인하세요.")
            else:
                display_columns = [
                    "title",
                    "document_id",
                    "num_chunks",
                    "embedding_model",
                    "created_at",
                    "origin_path",
                ]
                optional_cols = [col for col in ["tags", "year"] if col in catalog_df.columns]
                st.dataframe(
                    catalog_df[display_columns + optional_cols],
                    use_container_width=True,
                )
                doc_ids = catalog_df["document_id"].tolist()
                selected_doc = st.selectbox("미리보기 문서", doc_ids, index=0 if doc_ids else None)
                if selected_doc:
                    preview_chunks = RETRIEVAL_TOOL.preview_chunks(selected_doc)
                    manifest_row = catalog_df[catalog_df["document_id"] == selected_doc].iloc[0]
                    st.markdown(f"**문서 제목:** {manifest_row['title']}")
                    origin_path = manifest_row.get("origin_path")
                    if origin_path:
                        st.markdown(f"[원본 열기]({origin_path})")
                    tags = manifest_row.get("tags") or []
                    if tags:
                        st.caption("태그: " + ", ".join(str(tag) for tag in tags))
                    if not preview_chunks:
                        st.info("프리뷰 가능한 청크가 없습니다.")
                    else:
                        for chunk in preview_chunks:
                            with st.expander(f"Chunk {chunk.get('chunk_id')}"):
                                st.write(chunk.get("text") or "—")


if __name__ == "__main__":
    main()
