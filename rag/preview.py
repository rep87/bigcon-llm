from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class RagHit:
    doc_id: str
    chunk_id: str
    score: float
    text: str
    origin_path: Optional[str] = None


def _extract_model_name(payload: dict, retriever) -> str:
    encoder_info = payload.get("encoder_info") if isinstance(payload, dict) else None
    if isinstance(encoder_info, dict):
        active = encoder_info.get("active") or {}
        model = active.get("model") or active.get("model_name")
        if model:
            return str(model)
        configured = encoder_info.get("configured") or {}
        model = configured.get("model") or configured.get("model_name")
        if model:
            return str(model)
    settings = getattr(retriever, "encoder_settings", None)
    if settings is not None:
        model = getattr(settings, "model_name", None)
        if model:
            return str(model)
    model_attr = getattr(retriever, "model_name", None)
    if callable(model_attr):
        try:
            return str(model_attr())
        except Exception:  # pragma: no cover - defensive guard
            pass
    return "unknown"


def _normalise_selected(selected_docs: Optional[Iterable[str]]) -> Optional[list[str]]:
    if selected_docs is None:
        return None
    cleaned: list[str] = []
    for item in selected_docs:
        if not item:
            continue
        doc_id = str(item)
        if doc_id not in cleaned:
            cleaned.append(doc_id)
    return cleaned


def rag_preview_search(
    query: str,
    *,
    k: int = 8,
    threshold: float = 0.10,
    selected_docs: Optional[list[str]] = None,
    retriever=None,
) -> dict:
    """
    Returns a read-only preview payload:
    {
      "query": query,
      "model_name": <runtime query encoder name>,
      "k": k,
      "threshold": threshold,
      "selected_docs": selected_docs,     # None or list
      "items": [RagHit...],               # sorted by score desc
      "reason": "ok|empty_index|no_hit|below_threshold|error",
      "error": str|None
    }
    """

    try:
        from rag import RetrievalTool  # local import to avoid circular at module import time
    except Exception:  # pragma: no cover - optional dependency path
        RetrievalTool = None  # type: ignore

    try:
        if retriever is None:
            if RetrievalTool is None:
                raise RuntimeError("RetrievalTool unavailable")
            retriever = RetrievalTool()

        query_text = (query or "").strip()
        selected_list = _normalise_selected(selected_docs)

        try:
            catalog_entries = retriever.load_catalog()
        except AttributeError:
            catalog_entries = []
        catalog_ids = [getattr(entry, "doc_id", None) for entry in catalog_entries]
        catalog_ids = [doc_id for doc_id in catalog_ids if doc_id]

        payload: dict[str, object] = {}
        model_name = "unknown"
        items: list[RagHit] = []
        reason = "ok"
        error_text: Optional[str] = None

        if not catalog_ids:
            model_name = getattr(retriever, "model_name", lambda: "unknown")()
            return {
                "query": query_text,
                "model_name": model_name,
                "k": k,
                "threshold": threshold,
                "selected_docs": selected_list,
                "items": [],
                "reason": "empty_index",
                "error": None,
            }

        if not query_text:
            model_name = getattr(retriever, "model_name", lambda: "unknown")()
            return {
                "query": query_text,
                "model_name": model_name,
                "k": k,
                "threshold": threshold,
                "selected_docs": selected_list or catalog_ids,
                "items": [],
                "reason": "no_hit",
                "error": None,
            }

        if selected_list is not None and not selected_list:
            return {
                "query": query_text,
                "model_name": getattr(retriever, "model_name", lambda: "unknown")(),
                "k": k,
                "threshold": threshold,
                "selected_docs": selected_list,
                "items": [],
                "reason": "no_hit",
                "error": None,
            }

        search_doc_ids: Optional[list[str]]
        if selected_list:
            search_doc_ids = list(selected_list)
        else:
            search_doc_ids = list(catalog_ids)

        top_k = max(int(k), 1)
        effective_k = max(top_k, 8)

        payload = retriever.retrieve(
            query_text,
            top_k=effective_k,
            threshold=threshold,
            mode="auto",
            doc_ids=search_doc_ids,
        )
        model_name = _extract_model_name(payload, retriever)
        error_text = payload.get("error") if isinstance(payload, dict) else None

        hits_data = []
        if isinstance(payload, dict):
            hits_data = payload.get("chunks") or []

        if not hits_data:
            try:
                preview_payload = retriever.retrieve(
                    query_text,
                    top_k=effective_k,
                    threshold=0.0,
                    mode="always",
                    doc_ids=search_doc_ids,
                )
                if isinstance(preview_payload, dict):
                    alt_hits = preview_payload.get("chunks") or []
                    if alt_hits:
                        hits_data = alt_hits
            except Exception:
                pass

        for entry in hits_data:
            if not isinstance(entry, dict):
                continue
            hit = RagHit(
                doc_id=str(entry.get("doc_id") or ""),
                chunk_id=str(entry.get("chunk_id") or ""),
                score=float(entry.get("score") or 0.0),
                text=(entry.get("text") or "")[:500],
                origin_path=entry.get("origin_path"),
            )
            items.append(hit)

        items.sort(key=lambda h: -h.score)

        max_score = None
        if isinstance(payload, dict):
            max_score = payload.get("max_score")
            if isinstance(max_score, (int, float)):
                max_score = float(max_score)
            else:
                max_score = None

        if error_text:
            reason = "error"
        elif not items:
            reason = "no_hit"
        elif max_score is not None and max_score < float(threshold):
            reason = "below_threshold"
        else:
            reason = "ok"

        return {
            "query": query_text,
            "model_name": model_name,
            "k": effective_k,
            "threshold": threshold,
            "selected_docs": search_doc_ids,
            "items": items,
            "reason": reason,
            "error": error_text,
        }
    except Exception as exc:  # pragma: no cover - defensive guard
        return {
            "query": query,
            "model_name": "unknown",
            "k": k,
            "threshold": threshold,
            "selected_docs": selected_docs,
            "items": [],
            "reason": "error",
            "error": repr(exc),
        }
