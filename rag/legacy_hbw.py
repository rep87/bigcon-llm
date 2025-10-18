"""Legacy hashed bag-of-words encoder used by earlier indices."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np


_TOKEN_REGEX = re.compile(r"\w+", re.UNICODE)


class HBWEncoder:
    """Simple SHA-1 hashing encoder for backward compatibility."""

    def __init__(self, dim: int = 1536):
        self.dim = int(dim)

    def _tokenise(self, text: str) -> list[str]:
        return [match.group(0).lower() for match in _TOKEN_REGEX.finditer(text or "")]

    def _embed(self, tokens: Iterable[str], *, dim: int, normalize: bool) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            vec[idx] += 1.0
        if normalize:
            norm = np.linalg.norm(vec)
            if norm:
                vec = vec / norm
        return vec.astype("float32", copy=False)

    def encode_query(self, text: str, *, dim: int | None = None, normalize: bool = True) -> np.ndarray:
        target_dim = int(dim or self.dim or 1536)
        self.dim = target_dim
        tokens = self._tokenise(text)
        return self._embed(tokens, dim=target_dim, normalize=normalize)

    def encode_passages(
        self,
        texts: Iterable[str],
        *,
        dim: int | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        target_dim = int(dim or self.dim or 1536)
        self.dim = target_dim
        rows = [self._embed(self._tokenise(text), dim=target_dim, normalize=normalize) for text in texts]
        if not rows:
            return np.zeros((0, target_dim), dtype=np.float32)
        return np.vstack(rows)


__all__ = ["HBWEncoder"]
