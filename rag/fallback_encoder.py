"""Fallback encoder for CPU-only retrieval when E5 is unavailable."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

_TOKEN = re.compile(r"\w+", re.UNICODE)


class BowFallbackEncoder:
    """Simple hashed bag-of-words encoder with L2 normalisation."""

    def __init__(self, dim: int = 2048) -> None:
        self.dim = int(dim) if dim else 2048

    def _encode_counts(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN.findall(text or ""):
            hashed = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16)
            index = hashed % self.dim
            vec[index] += 1.0
        return vec

    def encode(self, text: str, *, normalize: bool = True) -> np.ndarray:
        vec = self._encode_counts(text.lower())
        if normalize:
            norm = float(np.linalg.norm(vec)) or 1.0
            vec = vec / norm
        return vec.astype(np.float32)

    def encode_query(
        self,
        text: str,
        *,
        prefix: str | None = None,
        normalize: bool = True,
        target_dim: int | None = None,
    ) -> np.ndarray:
        prepared = f"{prefix or ''}{text or ''}".strip()
        dim = int(target_dim) if target_dim else self.dim
        if dim != self.dim:
            encoder = BowFallbackEncoder(dim=dim)
            return encoder.encode(prepared, normalize=normalize)
        return self.encode(prepared, normalize=normalize)

    def model_name(self) -> str:
        return f"bow-fallback-{self.dim}"

    def backend(self) -> str:
        return "bow"

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.backend(),
            "model": self.model_name(),
            "dim": self.dim,
        }


__all__ = ["BowFallbackEncoder"]
