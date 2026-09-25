#!/usr/bin/env python3
"""
gpu_utils.py — Centralized GPU detection & helpers

Provides a single source of truth for CUDA availability across the pipeline.
All GPU-capable modules import from here to avoid redundant detection.

Auto-detects:
  1. PyTorch CUDA (for blocking / feature vectorization)
  2. XGBoost CUDA (for model training / inference)

Falls back gracefully to CPU if nothing is available.
"""
import logging
import os

# Fix OpenMP duplicate library error on Windows (PyTorch + XGBoost both link libiomp5md.dll)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger(__name__)

# ── PyTorch CUDA detection ────────────────────────────────────────────────────
HAS_TORCH_CUDA = False
TORCH_DEVICE = "cpu"
_torch = None

try:
    import torch as _torch
    if _torch.cuda.is_available():
        HAS_TORCH_CUDA = True
        TORCH_DEVICE = "cuda"
        _gpu_name = _torch.cuda.get_device_name(0)
        _gpu_mem = _torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logger.info(f"GPU detected: {_gpu_name} ({_gpu_mem:.1f} GB) — PyTorch CUDA enabled")
    else:
        logger.info("PyTorch installed but CUDA not available — using CPU")
except ImportError:
    logger.info("PyTorch not installed — GPU features disabled (blocking, features)")

# ── XGBoost CUDA detection ────────────────────────────────────────────────────
HAS_XGBOOST_CUDA = False

try:
    import xgboost as _xgb
    # Quick smoke test: train a tiny model on GPU
    import numpy as _np
    _X = _np.random.randn(10, 2).astype(_np.float32)
    _y = _np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    try:
        _clf = _xgb.XGBClassifier(
            n_estimators=2, max_depth=1, device="cuda",
            tree_method="hist", verbosity=0,
        )
        _clf.fit(_X, _y)
        HAS_XGBOOST_CUDA = True
        logger.info("XGBoost CUDA verified — model training will use GPU")
    except Exception:
        logger.info("XGBoost installed but CUDA training failed — using CPU")
    del _X, _y
except ImportError:
    pass

# ── Public helpers ────────────────────────────────────────────────────────────

def get_torch():
    """Return the torch module (or None if unavailable)."""
    return _torch


def get_device():
    """Return 'cuda' or 'cpu' for PyTorch operations."""
    return TORCH_DEVICE


def xgboost_device():
    """Return 'cuda' or 'cpu' for XGBoost tree_method='hist'."""
    return "cuda" if HAS_XGBOOST_CUDA else "cpu"


def gpu_summary() -> str:
    """Return a human-readable summary of GPU capabilities."""
    lines = [
        f"PyTorch CUDA:  {'YES' if HAS_TORCH_CUDA else 'NO'}  (device={TORCH_DEVICE})",
        f"XGBoost CUDA:  {'YES' if HAS_XGBOOST_CUDA else 'NO'}  (device={xgboost_device()})",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(gpu_summary())
