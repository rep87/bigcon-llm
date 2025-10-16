"""Minimal file-based retrieval tool for embedded corpora.

The retrieval tool scans embedding indices stored under ``indices/<version>``
inside a configurable RAG data root. Each indexed document has a folder
containing ``manifest.json``, ``chunks.parquet`` (with chunk metadata), and
``vectors.npy`` (with L2-normalised embedding vectors). Originals remain under
``corpus/`` beneath the same root so evidence can link back to the source
files.

This module keeps the implementation intentionally lightweight so the main
Streamlit app can import and query it without additional services.
"""

from __future__ import annotations

import os
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


_TOKEN_REGEX = re.compile(r"\w+", re.UNICODE)


@dataclass
class _DocumentEntry:
    doc_id: str
    title: str
    origin_path: str
    embedding_model: str | None
    created_at: str | None
    num_chunks: int
    manifest_path: Path
    tags: tuple[str, ...] | None = None
    year: int | None = None

    def as_dict(self) -> dict:
        return {
            "document_id": self.doc_id,
            "title": self.title,
            "num_chunks": self.num_chunks,
            "embedding_model": self.embedding_model,
            "created_at": self.created_at,
            "origin_path": self.origin_path,
            "tags": list(self.tags or ()),
            "year": self.year,
        }


class RetrievalTool:
    """A small helper around local embedding indices."""

    def __init__(self, root: str | Path | None = None, embed_version: str = "embed_v1") -> None:
        resolved_root = root or os.getenv("RAG_ROOT") or "data/rag"
        root_path = Path(resolved_root).expanduser().resolve()
        self.root_path = root_path
        self.embed_version = embed_version
        self.indices_dir = self.root_path / "indices" / embed_version
        self.corpus_dir = self.root_path / "corpus"
        self._catalog: list[_DocumentEntry] | None = None
        self._index_cache: dict[str, dict[str, Any]] = {}
        self._hybrid_alpha = 0.3

    # ------------------------------------------------------------------
    # catalog helpers
    # ------------------------------------------------------------------
    def load_catalog(self) -> list[_DocumentEntry]:
        """Load manifest metadata for all indexed documents."""
        if self._catalog is not None:
            return self._catalog

        entries: list[_DocumentEntry] = []
        if not self.indices_dir.exists():
            self._catalog = []
            return self._catalog

        for manifest_path in sorted(self.indices_dir.glob("*/manifest.json")):
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            folder = manifest_path.parent
            doc_id = str(data.get("document_id") or folder.name)
            title = str(data.get("title") or doc_id)
            origin_path = str(data.get("origin_path") or data.get("source_path") or "")
            origin_path = origin_path.replace("\\", "/")
            embedding_model = data.get("embedding_model")
            created_at = data.get("created_at")
            num_chunks = int(data.get("num_chunks") or self._count_chunks(folder))
            tags = self._normalise_tags(data.get("tags"))
            year = self._normalise_year(data.get("year"), created_at)
            entries.append(
                _DocumentEntry(
                    doc_id=doc_id,
                    title=title,
                    origin_path=origin_path,
                    embedding_model=embedding_model,
                    created_at=created_at,
                    num_chunks=num_chunks,
                    manifest_path=manifest_path,
                    tags=tags,
                    year=year,
                )
            )
        self._catalog = entries
        return entries

    def get_doc_list(self) -> pd.DataFrame:
        """Return the catalog as a dataframe for UI rendering."""
        records = [entry.as_dict() for entry in self.load_catalog()]
        if not records:
            return pd.DataFrame(
                columns=[
                    "document_id",
                    "title",
                    "num_chunks",
                    "embedding_model",
                    "created_at",
                    "origin_path",
                    "tags",
                    "year",
                ]
            )
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        *,
        threshold: float | None = None,
        mode: str = "auto",
        doc_ids: Optional[Sequence[str]] = None,
    ) -> dict:
        """Retrieve the most relevant chunks for ``query``.

        Parameters
        ----------
        query:
            User query text.
        top_k:
            Maximum number of chunks to surface.
        filters:
            Optional doc-level filters. Supports ``doc_id`` or ``doc_ids``.
        threshold:
            Minimum cosine score required for evidence to be considered when
            ``mode`` is ``"auto"``. When ``None`` the check is skipped.
        mode:
            ``"auto"`` (default) hides evidence when the best score is below the
            threshold, ``"always"`` surfaces evidence regardless, and any other
            value behaves like ``"auto"``.
        """
        query = (query or "").strip()
        if not query:
            return {"chunks": [], "evidence": [], "used_k": 0, "selected_doc_ids": []}

        doc_entries = self.load_catalog()
        if not doc_entries:
            return {"chunks": [], "evidence": [], "used_k": 0, "selected_doc_ids": []}

        allowed_docs: Optional[set[str]] = None
        if doc_ids is not None:
            cleaned = [str(item) for item in doc_ids if str(item)]
            if not cleaned:
                return {
                    "chunks": [],
                    "evidence": [],
                    "used_k": 0,
                    "selected_doc_ids": [],
                }
            allowed_docs = set(cleaned)
        if filters:
            ids = filters.get("doc_ids") or filters.get("doc_id")
            if isinstance(ids, str):
                allowed_docs = {ids}
            elif isinstance(ids, Iterable):
                allowed_docs = {str(item) for item in ids}

        vectors_list: list[np.ndarray] = []
        meta_rows: list[dict[str, Any]] = []
        dim = None

        tokenised_query = self._tokenise(f"query: {query}")

        for entry in doc_entries:
            if allowed_docs and entry.doc_id not in allowed_docs:
                continue
            doc_index = self._load_doc_index(entry)
            if doc_index is None:
                continue
            vectors = doc_index["vectors"]
            chunks_df: pd.DataFrame = doc_index["chunks"]
            if vectors.size == 0 or chunks_df.empty:
                continue
            if dim is None:
                dim = vectors.shape[1]
            elif vectors.shape[1] != dim:
                # Skip documents with incompatible dimensions.
                continue
            vectors_list.append(vectors)
            token_sets = doc_index["token_sets"]
            token_lists = doc_index.get("token_lists") or []
            token_counts = doc_index.get("token_counts") or []
            for i, row in chunks_df.iterrows():
                tokens_list = token_lists[i] if i < len(token_lists) else []
                counts = token_counts[i] if i < len(token_counts) else Counter(tokens_list)
                token_set = token_sets[i] if i < len(token_sets) else set(tokens_list)
                meta_rows.append(
                    {
                        "doc_id": entry.doc_id,
                        "chunk_id": row.get("chunk_id", i),
                        "text": row.get("text", ""),
                        "start": row.get("start"),
                        "end": row.get("end"),
                        "title": entry.title,
                        "origin_path": entry.origin_path,
                        "token_set": token_set,
                        "token_list": tokens_list,
                        "token_counts": counts,
                        "length": len(tokens_list),
                    }
                )

        if not vectors_list or dim is None:
            return {"chunks": [], "evidence": [], "used_k": 0, "selected_doc_ids": sorted(allowed_docs or [])}

        all_vectors = np.vstack(vectors_list)
        query_vec = self._embed_query(tokenised_query, dim)
        if query_vec is None:
            return {
                "chunks": [],
                "evidence": [],
                "used_k": 0,
                "selected_doc_ids": sorted(allowed_docs or []),
            }

        scores = all_vectors @ query_vec

        bm25_scores = np.zeros(len(meta_rows), dtype=np.float32)
        if meta_rows and tokenised_query:
            total_docs = len(meta_rows)
            doc_freq: Counter[str] = Counter()
            lengths = []
            for meta in meta_rows:
                token_list = meta.get("token_list") or []
                lengths.append(len(token_list))
                doc_freq.update(set(token_list))
            avg_len = float(sum(lengths) / max(len(lengths), 1)) or 1.0
            k1 = 1.5
            b = 0.75
            for token in tokenised_query:
                df = doc_freq.get(token, 0)
                if df == 0:
                    continue
                idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
                for idx, meta in enumerate(meta_rows):
                    counts: Counter[str] = meta.get("token_counts") or Counter()
                    tf = counts.get(token, 0)
                    if tf <= 0:
                        continue
                    length = meta.get("length") or 0
                    denom = tf + k1 * (1 - b + b * (length / avg_len))
                    bm25_scores[idx] += idf * (tf * (k1 + 1)) / denom

        if bm25_scores.size and bm25_scores.max() > 0:
            bm25_norm = bm25_scores / bm25_scores.max()
            scores = (1 - self._hybrid_alpha) * scores + self._hybrid_alpha * bm25_norm

        query_tokens = set(tokenised_query)
        if query_tokens:
            for idx, meta in enumerate(meta_rows):
                chunk_tokens = meta.get("token_set") or set()
                if query_tokens.issubset(chunk_tokens):
                    scores[idx] += 0.03

        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        chunks: list[dict[str, Any]] = []
        for idx in top_indices:
            meta = meta_rows[idx]
            score = float(scores[idx])
            chunk_text = str(meta["text"])
            chunks.append(
                {
                    "doc_id": meta["doc_id"],
                    "chunk_id": meta["chunk_id"],
                    "text": chunk_text,
                    "score": score,
                    "start": meta.get("start"),
                    "end": meta.get("end"),
                }
            )

        max_score_value = None
        if chunks:
            max_score_value = float(max(chunk["score"] for chunk in chunks))
        threshold = threshold if threshold is not None else None
        raw_used_k = len(chunks)
        include_evidence = True
        if mode != "always":
            if threshold is not None:
                if not chunks:
                    include_evidence = False
                elif max_score_value is None or not (max_score_value >= threshold):
                    include_evidence = False

        evidence: list[dict[str, Any]] = []
        if include_evidence:
            evidence_map: dict[str, dict[str, Any]] = {}
            for idx, chunk in enumerate(chunks):
                doc_id = chunk["doc_id"]
                meta = meta_rows[top_indices[idx]]
                ev = evidence_map.get(doc_id)
                if (ev is None) or (chunk["score"] > ev["score"]):
                    uri = meta.get("origin_path") or ""
                    evidence_map[doc_id] = {
                        "doc_id": doc_id,
                        "title": meta.get("title") or doc_id,
                        "uri": uri,
                        "score": chunk["score"],
                        "chunk_id": chunk["chunk_id"],
                    }

            evidence = sorted(evidence_map.values(), key=lambda item: item["score"], reverse=True)
        else:
            chunks = []

        return {
            "chunks": chunks,
            "evidence": evidence,
            "used_k": len(chunks),
            "raw_used_k": raw_used_k,
            "include_evidence": include_evidence,
            "max_score": max_score_value,
            "threshold": threshold,
            "mode": mode,
            "selected_doc_ids": sorted(allowed_docs or []),
            "catalog_size": len(doc_entries),
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def preview_chunks(self, doc_id: str, limit: int = 2) -> list[dict[str, Any]]:
        """Return the first ``limit`` chunks for preview purposes."""
        entry = next((item for item in self.load_catalog() if item.doc_id == doc_id), None)
        if entry is None:
            return []
        doc_index = self._load_doc_index(entry)
        if doc_index is None:
            return []
        chunks_df: pd.DataFrame = doc_index["chunks"]
        preview = []
        for _, row in chunks_df.head(limit).iterrows():
            preview.append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "text": row.get("text", ""),
                    "start": row.get("start"),
                    "end": row.get("end"),
                }
            )
        return preview

    def _count_chunks(self, folder: Path) -> int:
        chunks_path = folder / "chunks.parquet"
        if not chunks_path.exists():
            return 0
        try:
            df = pd.read_parquet(chunks_path, columns=["chunk_id"])
        except Exception:
            return 0
        return int(len(df))

    def _load_doc_index(self, entry: _DocumentEntry) -> Optional[dict[str, Any]]:
        if entry.doc_id in self._index_cache:
            return self._index_cache[entry.doc_id]

        folder = entry.manifest_path.parent
        vectors_path = folder / "vectors.npy"
        chunks_path = folder / "chunks.parquet"
        if not vectors_path.exists() or not chunks_path.exists():
            return None
        try:
            vectors = np.load(vectors_path)
            chunks_df = pd.read_parquet(chunks_path)
        except Exception:
            return None
        if len(chunks_df) != len(vectors):
            # Shape mismatch, skip the document.
            return None
        token_lists = [self._tokenise(f"passage: {text}") for text in chunks_df.get("text", [])]
        token_sets = [set(tokens) for tokens in token_lists]
        token_counts = [Counter(tokens) for tokens in token_lists]
        payload = {
            "vectors": vectors,
            "chunks": chunks_df,
            "token_sets": token_sets,
            "token_lists": token_lists,
            "token_counts": token_counts,
        }
        self._index_cache[entry.doc_id] = payload
        return payload

    def _tokenise(self, text: str) -> list[str]:
        return [match.group(0).lower() for match in _TOKEN_REGEX.finditer(text or "")]

    def _embed_query(self, tokens: list[str], dim: int) -> Optional[np.ndarray]:
        if dim <= 0:
            return None
        vec = np.zeros(dim, dtype=np.float32)
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _normalise_tags(self, value: Any) -> tuple[str, ...] | None:
        if not value:
            return None
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, Iterable):
            items = [str(item).strip() for item in value if str(item).strip()]
        else:
            items = []
        return tuple(dict.fromkeys(items)) if items else None

    def _normalise_year(self, value: Any, created_at: Any) -> int | None:
        try:
            if isinstance(value, (int, float)):
                year = int(value)
                return year if 1900 <= year <= 2100 else None
            if isinstance(value, str) and value.isdigit() and len(value) == 4:
                year = int(value)
                return year if 1900 <= year <= 2100 else None
        except Exception:  # pragma: no cover - defensive guard
            pass
        if isinstance(created_at, str) and len(created_at) >= 4 and created_at[:4].isdigit():
            try:
                year = int(created_at[:4])
                return year if 1900 <= year <= 2100 else None
            except Exception:  # pragma: no cover - defensive guard
                return None
        return None

