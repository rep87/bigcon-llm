"""Minimal CPU-only reproduction script for query encoding."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TRANSFORMERS_NO_ACCELERATE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

print("torch:", torch.__version__, "cuda?", torch.cuda.is_available())

from sentence_transformers import SentenceTransformer


model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")

param_device = "unknown"
try:
    for param in model._first_module().parameters():  # type: ignore[attr-defined]
        param_device = str(param.device)
        break
except Exception:
    pass

print("param.device:", param_device)

vector = model.encode(
    "ping",
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
)

print("vec shape:", vector.shape, "dtype:", vector.dtype)
