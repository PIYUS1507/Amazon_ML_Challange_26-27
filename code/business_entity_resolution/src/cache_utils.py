#!/usr/bin/env python3
"""
cache_utils.py — Fine-grained step and sub-step caching for ML Pipeline.

Provides transparent disk caching for:
  - Pandas DataFrames (.parquet via pyarrow or .pkl)
  - Numpy arrays / matrices (.npz or .npy)
  - Dictionaries of candidate pairs, ID lists, metadata (.pkl with protocol 5)
  - Models and arbitrary artifacts

Features:
  - Sub-step granular caching: preprocess, blocking, features, training data, predictions
  - Instant re-runs: skips expensive computations if outputs are already cached
  - Memory-safe: triggers garbage collection after saving/loading to prevent OOM
  - Safe parameter tagging: avoids mixing dev-sample and full-run cache
  - Toggleable via --no-cache / --force flags
"""
import os
import gc
import sys
import time
import pickle
import logging
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StepCache:
    """
    Manages cached artifacts for pipeline steps and sub-steps.
    """
    def __init__(self, cache_dir: str = "cache", enabled: bool = True, tag: str = ""):
        self.cache_dir = os.path.abspath(cache_dir)
        self.enabled = enabled
        self.tag = f"_{tag}" if tag and not tag.startswith("_") else tag
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _resolve_filename(self, key: str, ext: str = None) -> Tuple[str, str]:
        """Returns (full_filepath, chosen_extension)."""
        base_name = f"{key}{self.tag}"
        if ext is not None:
            return os.path.join(self.cache_dir, f"{base_name}{ext}"), ext
        # Check if already exists with known extensions
        for candidate_ext in [".parquet", ".npz", ".pkl"]:
            p = os.path.join(self.cache_dir, f"{base_name}{candidate_ext}")
            if os.path.exists(p):
                return p, candidate_ext
        # Default extension if saving new
        return os.path.join(self.cache_dir, f"{base_name}.pkl"), ".pkl"

    def exists(self, key: str, ext: str = None) -> bool:
        """Check if cached file exists."""
        if not self.enabled:
            return False
        path, _ = self._resolve_filename(key, ext)
        return os.path.exists(path)

    def save(self, key: str, data: Any, ext: str = None) -> str:
        """
        Save data to cache disk with optimal format and compression.
        """
        if not self.enabled:
            return ""

        t0 = time.time()
        base_name = f"{key}{self.tag}"

        if ext is None:
            if isinstance(data, pd.DataFrame):
                ext = ".parquet"
            elif isinstance(data, (np.ndarray, dict)) and all(isinstance(v, np.ndarray) for v in (data.values() if isinstance(data, dict) else [])):
                ext = ".npz"
            else:
                ext = ".pkl"

        filepath = os.path.join(self.cache_dir, f"{base_name}{ext}")

        if isinstance(data, pd.DataFrame):
            try:
                data.to_parquet(filepath, index=False, engine="pyarrow")
            except Exception:
                filepath = os.path.join(self.cache_dir, f"{base_name}.pkl")
                ext = ".pkl"
                with open(filepath, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif ext == ".npz" and isinstance(data, dict):
            np.savez_compressed(filepath, **data)
        elif ext == ".npz" and isinstance(data, np.ndarray):
            np.savez_compressed(filepath, arr=data)
        elif ext == ".npy" and isinstance(data, np.ndarray):
            np.save(filepath, data)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        elapsed = time.time() - t0
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info(f"💾 [CACHE SAVED] '{key}' -> {os.path.basename(filepath)} ({size_mb:.1f} MB) in {elapsed:.2f}s")
        gc.collect()
        return filepath

    def load(self, key: str, ext: str = None) -> Any:
        """
        Load data from cache disk.
        """
        if not self.enabled:
            raise FileNotFoundError(f"Cache disabled, cannot load {key}")

        filepath, detected_ext = self._resolve_filename(key, ext)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cache miss for '{key}': {filepath} does not exist")

        t0 = time.time()
        size_mb = os.path.getsize(filepath) / (1024 * 1024)

        if detected_ext == ".parquet":
            data = pd.read_parquet(filepath, engine="pyarrow")
        elif detected_ext == ".npz":
            with np.load(filepath, allow_pickle=True) as loaded:
                if len(loaded.files) == 1 and "arr" in loaded.files:
                    data = loaded["arr"]
                else:
                    data = {k: loaded[k] for k in loaded.files}
        elif detected_ext == ".npy":
            data = np.load(filepath, allow_pickle=True)
        else:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

        elapsed = time.time() - t0
        logger.info(f"⚡ [CACHE HIT] Loaded '{key}' from {os.path.basename(filepath)} ({size_mb:.1f} MB) in {elapsed:.2f}s")
        return data

    def run_or_load(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        step_desc: str = "",
        ext: str = None,
        force: bool = False,
    ) -> Any:
        """
        If cached and not force, loads from disk.
        Otherwise executes compute_fn(), saves to cache, and returns result.
        """
        desc = step_desc or key
        if self.enabled and not force and self.exists(key, ext):
            try:
                return self.load(key, ext)
            except Exception as e:
                logger.warning(f"Failed to load cache for '{key}' ({e}) — recomputing...")

        logger.info(f"⏳ [CACHE MISS] Running sub-step: {desc}...")
        t0 = time.time()
        result = compute_fn()
        elapsed = time.time() - t0
        logger.info(f"✓ Sub-step completed: {desc} in {elapsed:.1f}s")

        if self.enabled:
            try:
                self.save(key, result, ext)
            except Exception as e:
                logger.warning(f"Failed to save cache for '{key}': {e}")

        return result

    def clear(self):
        """Remove all files in cache dir."""
        if os.path.exists(self.cache_dir):
            for f in os.listdir(self.cache_dir):
                fp = os.path.join(self.cache_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
            logger.info(f"Cache cleared: {self.cache_dir}")
