"""Minimal file-based retrieval tool for embedded corpora.

The retrieval tool scans embedding indices stored under ``indices/<version>``
inside the repository root. Each indexed document has a folder containing
``manifest.json``, ``chunks.parquet`` (with chunk metadata), and ``vectors.npy``
(with L2-normalised embedding vectors).

This module keeps the implementation intentionally lightweight so the main
Streamlit app can import and query it without additional services.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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


class RetrievalTool:
    """A small helper around local embedding indices."""

    def __init__(self, root: str | Path = "repo", embed_version: str = "embed_v1") -> None:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            # Fallback to current working directory when the provided root is missing.
            root_path = Path.cwd()
        self.root_path = root_path.resolve()
        self.embed_version = embed_version
        self.indices_dir = self.root_path / "indices" / embed_version
        self.corpus_dir = self.root_path / "corpus"
        self._catalog: list[_DocumentEntry] | None = None
        self._index_cache: dict[str, dict[str, Any]] = {}

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
            embedding_model = data.get("embedding_model")
            created_at = data.get("created_at")
            num_chunks = int(data.get("num_chunks") or self._count_chunks(folder))
            entries.append(
                _DocumentEntry(
                    doc_id=doc_id,
                    title=title,
                    origin_path=origin_path,
                    embedding_model=embedding_model,
                    created_at=created_at,
                    num_chunks=num_chunks,
                    manifest_path=manifest_path,
                )
            )
        self._catalog = entries
        return entries

    def get_doc_list(self) -> pd.DataFrame:
        """Return the catalog as a dataframe for UI rendering."""
        records = [
            {
                "document_id": entry.doc_id,
                "title": entry.title,
                "num_chunks": entry.num_chunks,
                "embedding_model": entry.embedding_model,
                "created_at": entry.created_at,
                "origin_path": entry.origin_path,
            }
            for entry in self.load_catalog()
        ]
        if not records:
            return pd.DataFrame(
                columns=[
                    "document_id",
                    "title",
                    "num_chunks",
                    "embedding_model",
                    "created_at",
                    "origin_path",
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
    ) -> dict:
        """Retrieve the most relevant chunks for ``query``."""
        query = (query or "").strip()
        if not query:
            return {"chunks": [], "evidence": [], "used_k": 0}

        doc_entries = self.load_catalog()
        if not doc_entries:
            return {"chunks": [], "evidence": [], "used_k": 0}

        allowed_docs: Optional[set[str]] = None
        if filters:
            ids = filters.get("doc_ids") or filters.get("doc_id")
            if isinstance(ids, str):
                allowed_docs = {ids}
            elif isinstance(ids, Iterable):
                allowed_docs = {str(item) for item in ids}

        vectors_list: list[np.ndarray] = []
        meta_rows: list[dict[str, Any]] = []
        dim = None

        tokenised_query = self._tokenise(query)

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
            text_tokens = doc_index["token_sets"]
            for i, row in chunks_df.iterrows():
                meta_rows.append(
                    {
                        "doc_id": entry.doc_id,
                        "chunk_id": row.get("chunk_id", i),
                        "text": row.get("text", ""),
                        "start": row.get("start"),
                        "end": row.get("end"),
                        "title": entry.title,
                        "origin_path": entry.origin_path,
                        "token_set": text_tokens[i],
                    }
                )

        if not vectors_list or dim is None:
            return {"chunks": [], "evidence": [], "used_k": 0}

        all_vectors = np.vstack(vectors_list)
        query_vec = self._embed_query(tokenised_query, dim)
        if query_vec is None:
            return {"chunks": [], "evidence": [], "used_k": 0}

        scores = all_vectors @ query_vec
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
        return {"chunks": chunks, "evidence": evidence, "used_k": len(chunks)}

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
        token_sets = [self._tokenise(str(text)) for text in chunks_df.get("text", [])]
        token_sets = [set(tokens) for tokens in token_sets]
        payload = {"vectors": vectors, "chunks": chunks_df, "token_sets": token_sets}
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

