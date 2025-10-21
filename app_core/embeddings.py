"""Embedding backends for the Streamlit retrieval workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


Backend = Literal["e5", "hbw"]


@dataclass
class EmbedSettings:
    """Configuration shared between query and passage encoders."""

    backend: Backend
    model_name: str
    prefix_query: str = "query: "
    prefix_passage: str = "passage: "
    normalize: bool = True


class QueryEncoder:
    """Lazily initialised embedding encoder for queries and passages."""

    def __init__(self, settings: EmbedSettings):
        self.s = settings
        self._model = None
        self._tokenizer = None
        self._tokenizer_name: str | None = None
        self.dim: int | None = None

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        if self.s.backend == "e5":
            try:
                from sentence_transformers import SentenceTransformer
                from transformers import AutoTokenizer
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "sentence-transformers/transformers 설치가 필요합니다."
                ) from exc

            model = SentenceTransformer(self.s.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.s.model_name, use_fast=True)
            self._model = model
            self._tokenizer = tokenizer
            self._tokenizer_name = getattr(tokenizer, "name_or_path", self.s.model_name)
            self.dim = int(model.get_sentence_embedding_dimension())
        elif self.s.backend == "hbw":
            from rag.legacy_hbw import HBWEncoder  # lazy import for legacy path

            encoder = HBWEncoder()
            self._model = encoder
            self._tokenizer_name = "legacy_hbw"
            self.dim = encoder.dim
        else:  # pragma: no cover - guard for unexpected config
            raise ValueError(f"Unknown embedding backend: {self.s.backend}")

    def _post(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32)
        if not vec.size:
            return vec
        if self.s.normalize:
            norms = np.linalg.norm(vec, axis=-1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vec = vec / norms
        return vec.astype("float32", copy=False)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def encode_query(self, text: str, *, target_dim: int | None = None) -> np.ndarray:
        self._ensure_model()
        prepared = f"{self.s.prefix_query}{text or ''}"

        if self.s.backend == "e5":
            model = self._model
            assert model is not None  # for type checkers
            vec = model.encode(
                prepared,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            vec = np.asarray(vec, dtype=np.float32)
            if vec.ndim > 1:
                vec = vec[0]
            vec = self._post(vec)
            if target_dim is not None and vec.shape[0] != target_dim:
                raise ValueError(
                    f"Query/vector dimension mismatch: {vec.shape[0]} vs {target_dim}"
                )
            return vec

        encoder = self._model
        assert encoder is not None  # pragma: no cover - guarded above
        dim = int(target_dim or self.dim or 1536)
        result = encoder.encode_query(
            prepared,
            dim=dim,
            normalize=self.s.normalize,
        )
        self.dim = dim
        return np.asarray(result, dtype=np.float32)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_model()

        if self.s.backend == "e5":
            model = self._model
            assert model is not None
            prefixed = [f"{self.s.prefix_passage}{text}" for text in texts]
            vecs = model.encode(
                prefixed,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            return self._post(np.asarray(vecs, dtype=np.float32))

        encoder = self._model
        assert encoder is not None  # pragma: no cover
        dim = self.dim or 1536
        prefixed = [f"{self.s.prefix_passage}{text}" for text in texts]
        vecs = encoder.encode_passages(prefixed, dim=dim, normalize=self.s.normalize)
        self.dim = dim
        return np.asarray(vecs, dtype=np.float32)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def info(self) -> dict:
        return {
            "backend": self.s.backend,
            "model": self.s.model_name,
            "tokenizer": self._tokenizer_name or self.s.model_name,
            "prefix_query": self.s.prefix_query,
            "prefix_passage": self.s.prefix_passage,
            "normalize": self.s.normalize,
            "dim": self.dim,
        }


__all__ = ["Backend", "EmbedSettings", "QueryEncoder"]
