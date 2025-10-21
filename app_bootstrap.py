"""Bootstrap entrypoint ensuring CPU-only initialization before app import."""

import os
import sys


def _apply_env_guards() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TRANSFORMERS_NO_ACCELERATE", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")


def _set_default_device() -> None:
    try:
        import torch  # type: ignore

        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    except Exception:
        pass


def main() -> None:
    _apply_env_guards()
    _set_default_device()

    if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import app  # noqa: WPS433  # pylint: disable=import-outside-toplevel,unused-import


if __name__ == "__main__":
    main()
