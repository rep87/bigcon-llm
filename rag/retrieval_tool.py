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

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app_core.config import get_flag, get_setting
from app_core.embeddings import EmbedSettings, QueryEncoder


_TOKEN_REGEX = re.compile(r"\w+", re.UNICODE)


@dataclass
class _DocumentEntry:
    doc_id: str
    title: str
    origin_path: str
    embedding_model: str | None
    tokenizer: str | None
    vector_dim: int | None
    normalize: bool | None
    prefix_query: str | None
    prefix_passage: str | None
    model_revision: str | None
    created_at: str | None
    num_chunks: int
    manifest_path: Path
    tags: tuple[str, ...] | None = None
    year: int | None = None
    raw_manifest: Dict[str, Any] | None = None

    def as_dict(self) -> dict:
        return {
            "document_id": self.doc_id,
            "title": self.title,
            "num_chunks": self.num_chunks,
            "embedding_model": self.embedding_model,
            "tokenizer": self.tokenizer,
            "vector_dim": self.vector_dim,
            "normalize": self.normalize,
            "prefix_query": self.prefix_query,
            "prefix_passage": self.prefix_passage,
            "model_revision": self.model_revision,
            "created_at": self.created_at,
            "origin_path": self.origin_path,
            "tags": list(self.tags or ()),
            "year": self.year,
        }


class RetrievalTool:
    """A small helper around local embedding indices."""

    def __init__(self, root: str | Path | None = None, embed_version: str = "embed_v1") -> None:
        resolved_root = root or get_setting("RAG_ROOT", "data/rag") or "data/rag"
        root_path = Path(resolved_root).expanduser().resolve()
        self.root_path = root_path
        self.embed_version = embed_version
        self.indices_dir = self.root_path / "indices" / embed_version
        self.corpus_dir = self.root_path / "corpus"
        self._catalog: list[_DocumentEntry] | None = None
        self._index_cache: dict[str, dict[str, Any]] = {}
        self._hybrid_alpha = 0.3
        backend_value = str(get_setting("EMBED_BACKEND", "e5") or "e5").lower()
        if backend_value not in {"e5", "hbw"}:
            backend_value = "e5"
        model_name_value = str(
            get_setting("EMBED_MODEL_NAME", "intfloat/multilingual-e5-base")
            or "intfloat/multilingual-e5-base"
        )
        prefix_query = str(get_setting("EMBED_PREFIX_QUERY", "query: ") or "query: ")
        prefix_passage = str(get_setting("EMBED_PREFIX_PASSAGE", "passage: ") or "passage: ")
        normalize_flag = get_flag("EMBED_NORMALIZE", True)
        self.encoder_settings = EmbedSettings(
            backend=backend_value,  # type: ignore[arg-type]
            model_name=model_name_value,
            prefix_query=prefix_query,
            prefix_passage=prefix_passage,
            normalize=bool(normalize_flag),
        )
        self.autoswitch = get_flag("RAG_AUTOSWITCH", True)
        self._encoder_cache: dict[
            Tuple[str, str, str, str, bool], QueryEncoder
        ] = {}

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
            tokenizer = data.get("tokenizer") or data.get("tokenizer_name")
            vector_dim = data.get("vector_dim") or data.get("embedding_dim")
            try:
                vector_dim = int(vector_dim) if vector_dim is not None else None
            except Exception:
                vector_dim = None
            normalize = self._coerce_bool(
                data.get("normalize", data.get("normalise"))
            )
            prefix_query, prefix_passage = self._extract_prefixes(data)
            model_revision = data.get("model_revision") or data.get("revision")
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
                    tokenizer=str(tokenizer) if tokenizer else None,
                    vector_dim=vector_dim,
                    normalize=normalize,
                    prefix_query=prefix_query,
                    prefix_passage=prefix_passage,
                    model_revision=str(model_revision) if model_revision else None,
                    created_at=created_at,
                    num_chunks=num_chunks,
                    manifest_path=manifest_path,
                    tags=tags,
                    year=year,
                    raw_manifest=data,
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
        """Retrieve the most relevant chunks for ``query``."""

        filters = filters or {}
        query_text = (query or "").strip()
        doc_entries = self.load_catalog()
        effective_mode = mode if mode in {"auto", "always"} else "auto"

        allowed_docs: Optional[set[str]] = None
        cleaned_selection: list[str] = []
        if doc_ids is not None:
            cleaned_selection = [str(item) for item in doc_ids if str(item)]
            if not cleaned_selection:
                payload = self._base_payload(len(doc_entries), [], threshold, effective_mode)
                payload["encoder_info"] = self._build_encoder_info(
                    [],
                    None,
                    self.encoder_settings,
                    warnings=[],
                    mode="strict",
                    skipped=[],
                    consistent=True,
                )
                return payload
            allowed_docs = set(cleaned_selection)

        if filters:
            ids = filters.get("doc_ids") or filters.get("doc_id")
            if isinstance(ids, str):
                allowed_docs = {ids}
            elif isinstance(ids, Iterable):
                allowed_docs = {str(item) for item in ids if str(item)}

        selected_default = sorted(allowed_docs) if allowed_docs else []
        payload = self._base_payload(len(doc_entries), selected_default, threshold, effective_mode)

        if not query_text or not doc_entries:
            payload["encoder_info"] = self._build_encoder_info(
                [],
                None,
                self.encoder_settings,
                warnings=[],
                mode="strict",
                skipped=[],
                consistent=bool(doc_entries),
            )
            return payload

        vectors_list: list[np.ndarray] = []
        meta_rows: list[dict[str, Any]] = []
        doc_specs: list[dict[str, Any]] = []
        warnings: list[str] = []
        skipped_docs: list[str] = []
        used_doc_ids: list[str] = []
        dim: int | None = None

        tokenised_query = self._tokenise(query_text)

        for entry in doc_entries:
            if allowed_docs and entry.doc_id not in allowed_docs:
                continue
            doc_index = self._load_doc_index(entry)
            if doc_index is None:
                skipped_docs.append(entry.doc_id)
                continue
            vectors = doc_index["vectors"]
            chunks_df: pd.DataFrame = doc_index["chunks"]
            if vectors.size == 0 or chunks_df.empty:
                skipped_docs.append(entry.doc_id)
                continue
            doc_dim = int(vectors.shape[1])
            if dim is None:
                dim = doc_dim
            elif doc_dim != dim:
                warnings.append(f"{entry.doc_id}: vector_dim {doc_dim} != {dim}")
                skipped_docs.append(entry.doc_id)
                continue

            spec = self._build_spec(entry, doc_dim)
            doc_specs.append(spec)
            vectors_list.append(vectors)
            used_doc_ids.append(entry.doc_id)

            token_lists = doc_index.get("token_lists") or []
            token_sets = doc_index.get("token_sets") or []
            token_counts = doc_index.get("token_counts") or []

            for idx, row in chunks_df.iterrows():
                tokens_list = token_lists[idx] if idx < len(token_lists) else []
                counts = token_counts[idx] if idx < len(token_counts) else Counter(tokens_list)
                token_set = token_sets[idx] if idx < len(token_sets) else set(tokens_list)
                meta_rows.append(
                    {
                        "doc_id": entry.doc_id,
                        "chunk_id": row.get("chunk_id", idx),
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
            if allowed_docs:
                warnings.append("선택된 문서에서 사용할 수 있는 임베딩이 없습니다.")
            payload["warnings"] = warnings
            payload["doc_specs"] = doc_specs
            payload["selected_doc_ids"] = sorted(set(used_doc_ids) or selected_default)
            payload["encoder_info"] = self._build_encoder_info(
                doc_specs,
                None,
                self.encoder_settings,
                warnings=warnings,
                mode="strict",
                skipped=skipped_docs,
                consistent=len(doc_specs) <= 1,
            )
            return payload

        decision = self._decide_active_settings(doc_specs)
        active_settings: EmbedSettings = decision["settings"]
        warnings.extend(decision["warnings"])

        try:
            encoder = self._get_encoder(active_settings)
        except Exception as exc:  # pragma: no cover - defensive guard
            error_msg = f"임베딩 초기화 실패: {exc}"
            warnings.append(error_msg)
            payload["error"] = error_msg
            payload["warnings"] = warnings
            payload["doc_specs"] = doc_specs
            payload["selected_doc_ids"] = sorted(set(used_doc_ids) or selected_default)
            payload["encoder_info"] = self._build_encoder_info(
                doc_specs,
                None,
                active_settings,
                warnings=warnings,
                mode=decision["mode"],
                skipped=skipped_docs,
                consistent=decision["consistent"],
            )
            return payload

        try:
            if active_settings.backend == "hbw":
                query_vec = encoder.encode_query(query_text, target_dim=dim)
            else:
                query_vec = encoder.encode_query(query_text)
        except Exception as exc:  # pragma: no cover - defensive guard
            error_msg = f"쿼리 임베딩 실패: {exc}"
            warnings.append(error_msg)
            payload["error"] = error_msg
            payload["warnings"] = warnings
            payload["doc_specs"] = doc_specs
            payload["selected_doc_ids"] = sorted(set(used_doc_ids) or selected_default)
            payload["encoder_info"] = self._build_encoder_info(
                doc_specs,
                encoder,
                active_settings,
                warnings=warnings,
                mode=decision["mode"],
                skipped=skipped_docs,
                consistent=decision["consistent"],
            )
            return payload

        if query_vec.shape[0] != dim:
            error_msg = f"쿼리 임베딩 차원 {query_vec.shape[0]} != {dim}"
            warnings.append(error_msg)
            payload["error"] = error_msg
            payload["warnings"] = warnings
            payload["doc_specs"] = doc_specs
            payload["selected_doc_ids"] = sorted(set(used_doc_ids) or selected_default)
            payload["encoder_info"] = self._build_encoder_info(
                doc_specs,
                encoder,
                active_settings,
                warnings=warnings,
                mode=decision["mode"],
                skipped=skipped_docs,
                consistent=decision["consistent"],
            )
            return payload

        all_vectors = np.vstack(vectors_list)
        scores = all_vectors @ query_vec

        bm25_scores = np.zeros(len(meta_rows), dtype=np.float32)
        if meta_rows and tokenised_query:
            total_chunks = len(meta_rows)
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
                idf = math.log((total_chunks - df + 0.5) / (df + 0.5) + 1.0)
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
                if query_tokens and query_tokens.issubset(chunk_tokens):
                    scores[idx] += 0.03

        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        ordered_scores = [float(scores[idx]) for idx in top_indices.tolist()]
        ranked_chunks: list[dict[str, Any]] = []
        for idx in top_indices:
            meta = meta_rows[idx]
            ranked_chunks.append(
                {
                    "doc_id": meta["doc_id"],
                    "chunk_id": meta["chunk_id"],
                    "text": str(meta.get("text", "")),
                    "score": float(scores[idx]),
                    "start": meta.get("start"),
                    "end": meta.get("end"),
                }
            )

        max_score_value = float(max((chunk["score"] for chunk in ranked_chunks), default=0.0)) if ranked_chunks else None
        raw_used_k = len(ranked_chunks)
        include_evidence = True
        if effective_mode != "always":
            threshold_value = threshold if threshold is not None else None
            if threshold_value is not None:
                if not ranked_chunks:
                    include_evidence = False
                elif max_score_value is None or max_score_value < threshold_value:
                    include_evidence = False
        evidence: list[dict[str, Any]] = []
        chunks_for_return: list[dict[str, Any]]
        if include_evidence:
            evidence_map: dict[str, dict[str, Any]] = {}
            for idx, chunk in enumerate(ranked_chunks):
                doc_id = chunk["doc_id"]
                meta = meta_rows[top_indices[idx]]
                existing = evidence_map.get(doc_id)
                if existing is None or chunk["score"] > existing["score"]:
                    evidence_map[doc_id] = {
                        "doc_id": doc_id,
                        "title": meta.get("title") or doc_id,
                        "uri": meta.get("origin_path") or "",
                        "score": chunk["score"],
                        "chunk_id": chunk["chunk_id"],
                    }
            evidence = sorted(evidence_map.values(), key=lambda item: item["score"], reverse=True)
            chunks_for_return = ranked_chunks
        else:
            chunks_for_return = []

        payload.update(
            {
                "chunks": chunks_for_return,
                "evidence": evidence,
                "used_k": len(chunks_for_return),
                "raw_used_k": raw_used_k,
                "include_evidence": include_evidence,
                "max_score": max_score_value,
                "top_scores": ordered_scores,
                "selected_doc_ids": sorted(set(used_doc_ids) or selected_default),
                "doc_specs": doc_specs,
                "warnings": list(dict.fromkeys(warnings)),
                "encoder_info": self._build_encoder_info(
                    doc_specs,
                    encoder,
                    active_settings,
                    warnings=warnings,
                    mode=decision["mode"],
                    skipped=skipped_docs,
                    consistent=decision["consistent"],
                ),
            }
        )
        return payload

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
            return None
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            return None
        normalise_vectors = self.encoder_settings.normalize or entry.normalize is not False
        if normalise_vectors:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vectors = (vectors / norms).astype(np.float32)
            normalized_flag = True
        else:
            normalized_flag = bool(entry.normalize)

        text_series = chunks_df.get("text")
        if text_series is None:
            text_values = [""] * len(chunks_df)
        else:
            text_values = [str(value) for value in text_series.tolist()]
        token_lists = [self._tokenise(value) for value in text_values]
        token_sets = [set(tokens) for tokens in token_lists]
        token_counts = [Counter(tokens) for tokens in token_lists]
        payload = {
            "vectors": vectors,
            "chunks": chunks_df,
            "token_sets": token_sets,
            "token_lists": token_lists,
            "token_counts": token_counts,
            "normalized": normalized_flag,
        }
        self._index_cache[entry.doc_id] = payload
        if entry.vector_dim is None:
            entry.vector_dim = int(vectors.shape[1])
        if entry.normalize is None:
            entry.normalize = normalized_flag
        return payload

    def _tokenise(self, text: str) -> list[str]:
        return [match.group(0).lower() for match in _TOKEN_REGEX.finditer(text or "")]

    def _base_payload(
        self,
        catalog_size: int,
        selected_doc_ids: Sequence[str],
        threshold: float | None,
        mode: str,
    ) -> dict:
        return {
            "chunks": [],
            "evidence": [],
            "used_k": 0,
            "raw_used_k": 0,
            "include_evidence": False,
            "max_score": None,
            "threshold": threshold if threshold is not None else None,
            "mode": mode,
            "selected_doc_ids": list(selected_doc_ids),
            "catalog_size": catalog_size,
            "top_scores": [],
            "error": None,
            "encoder_info": None,
            "doc_specs": [],
            "warnings": [],
        }

    def _build_spec(self, entry: _DocumentEntry, vector_dim: int) -> dict:
        manifest_model = None
        if entry.raw_manifest and isinstance(entry.raw_manifest, dict):
            manifest_model = entry.raw_manifest.get("embedding_model")
        return {
            "doc_id": entry.doc_id,
            "title": entry.title,
            "origin_path": entry.origin_path,
            "embedding_model": entry.embedding_model or manifest_model,
            "tokenizer": entry.tokenizer,
            "vector_dim": vector_dim,
            "normalize": entry.normalize,
            "prefix_query": entry.prefix_query,
            "prefix_passage": entry.prefix_passage,
            "model_revision": entry.model_revision,
            "manifest_path": str(entry.manifest_path),
        }

    def _get_encoder(self, settings: EmbedSettings) -> QueryEncoder:
        key = (
            settings.backend,
            settings.model_name,
            settings.prefix_query,
            settings.prefix_passage,
            bool(settings.normalize),
        )
        encoder = self._encoder_cache.get(key)
        if encoder is None:
            encoder = QueryEncoder(settings)
            self._encoder_cache[key] = encoder
        return encoder

    def _decide_active_settings(self, doc_specs: Sequence[dict]) -> dict:
        if not doc_specs:
            return {
                "settings": self.encoder_settings,
                "mode": "strict",
                "consistent": True,
                "warnings": [],
            }

        warnings: list[str] = []
        signatures = {self._spec_signature(spec) for spec in doc_specs}
        consistent = len(signatures) <= 1
        target_spec = doc_specs[0]
        active_settings = self.encoder_settings
        mode = "strict"

        if consistent:
            target_spec = doc_specs[0]
            backend = "hbw" if self._spec_is_legacy(target_spec) else "e5"
            model_name = target_spec.get("embedding_model") or (
                "legacy_hbw" if backend == "hbw" else self.encoder_settings.model_name
            )
            normalize = target_spec.get("normalize")
            prefix_query = target_spec.get("prefix_query") or self.encoder_settings.prefix_query
            prefix_passage = target_spec.get("prefix_passage") or self.encoder_settings.prefix_passage
            target_settings = EmbedSettings(
                backend=backend,  # type: ignore[arg-type]
                model_name=str(model_name),
                prefix_query=str(prefix_query),
                prefix_passage=str(prefix_passage),
                normalize=self.encoder_settings.normalize if normalize is None else bool(normalize),
            )
            if not self._settings_equal(target_settings, self.encoder_settings):
                if self.autoswitch:
                    active_settings = target_settings
                    mode = "autoswitch"
                else:
                    warnings.append("임베딩 설정이 인덱스와 달라 autoswitch를 적용하지 않았습니다.")
            else:
                active_settings = target_settings
        else:
            warnings.append("선택된 문서의 임베딩 설정이 서로 달라 단일 백엔드로 강제합니다.")

        return {
            "settings": active_settings,
            "mode": mode,
            "consistent": consistent,
            "warnings": warnings,
        }

    def _settings_equal(self, lhs: EmbedSettings, rhs: EmbedSettings) -> bool:
        return (
            lhs.backend == rhs.backend
            and lhs.model_name == rhs.model_name
            and lhs.prefix_query == rhs.prefix_query
            and lhs.prefix_passage == rhs.prefix_passage
            and bool(lhs.normalize) == bool(rhs.normalize)
        )

    def _spec_signature(self, spec: dict) -> Tuple[str, str, int, bool, str, str]:
        return (
            str(spec.get("embedding_model") or "legacy_hbw").lower(),
            str(spec.get("tokenizer") or "").lower(),
            int(spec.get("vector_dim") or 0),
            bool(spec.get("normalize") if spec.get("normalize") is not None else True),
            str(spec.get("prefix_query") or ""),
            str(spec.get("prefix_passage") or ""),
        )

    def _spec_is_legacy(self, spec: dict) -> bool:
        model_name = str(spec.get("embedding_model") or "").lower()
        tokenizer = str(spec.get("tokenizer") or "").lower()
        if not model_name and not tokenizer:
            return True
        legacy_tokens = {"legacy", "hbw", "hash", "bow"}
        return any(token in model_name for token in legacy_tokens) or any(
            token in tokenizer for token in legacy_tokens
        )

    def _settings_to_dict(self, settings: EmbedSettings) -> dict:
        return {
            "backend": settings.backend,
            "model": settings.model_name,
            "prefix_query": settings.prefix_query,
            "prefix_passage": settings.prefix_passage,
            "normalize": bool(settings.normalize),
        }

    def _build_encoder_info(
        self,
        doc_specs: Sequence[dict],
        encoder: QueryEncoder | None,
        active_settings: EmbedSettings,
        *,
        warnings: Sequence[str],
        mode: str,
        skipped: Sequence[str],
        consistent: bool,
    ) -> dict:
        encoder_state = encoder.info() if encoder is not None else None
        return {
            "configured": self._settings_to_dict(self.encoder_settings),
            "active": self._settings_to_dict(active_settings),
            "encoder_state": encoder_state,
            "mode": mode,
            "consistent": bool(consistent),
            "doc_specs": list(doc_specs),
            "warnings": list(dict.fromkeys(str(item) for item in warnings if item)),
            "skipped_doc_ids": list(dict.fromkeys(str(item) for item in skipped if item)),
        }

    def _coerce_bool(self, value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        return None

    def _extract_prefixes(self, manifest: Dict[str, Any]) -> tuple[str | None, str | None]:
        if not isinstance(manifest, dict):
            return (None, None)
        block = (
            manifest.get("instruction_prefix")
            or manifest.get("prefix")
            or manifest.get("prefixes")
        )
        query_prefix = None
        passage_prefix = None
        if isinstance(block, dict):
            query_prefix = block.get("query") or block.get("query_prefix")
            passage_prefix = block.get("passage") or block.get("doc")
        if query_prefix is None:
            query_prefix = manifest.get("query_prefix")
        if passage_prefix is None:
            passage_prefix = manifest.get("passage_prefix")
        return (
            str(query_prefix) if isinstance(query_prefix, str) else None,
            str(passage_prefix) if isinstance(passage_prefix, str) else None,
        )

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

