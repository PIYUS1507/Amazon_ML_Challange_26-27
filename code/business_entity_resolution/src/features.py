#!/usr/bin/env python3
"""
Step 3: Feature Engineering for Pairwise Matching (Optimized & Multi-Threaded)

Computes 20 pairwise similarity features between candidate pairs using:
  - Multi-threaded shared-memory parallelism (OpenMP-style multi-core execution)
  - rapidfuzz AVX2 C++ engine for Levenshtein, token sort, token set, and LCS
  - Instant hash dict lookups (bypassing slow pandas .at[] index lookups)
  - Real-time progress reporting with throughput (pairs/sec) and ETA
  - Memory-efficient batch allocation

Features:
  Name (10): Levenshtein ratio, token sort ratio, token set ratio,
             Jaccard tokens, Jaccard bigrams, prefix-3, prefix-5,
             LCS ratio, length ratio, exact match
  Address (7): Levenshtein ratio, token sort ratio, Jaccard tokens,
               Jaccard bigrams, numeric overlap, prefix-5, LCS ratio
  Meta (3): country match, cross name1∩addr2 tokens, cross name2∩addr1 tokens
"""
import os
import re
import gc
import time
import logging
import concurrent.futures
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Number of worker threads for parallel feature extraction
N_WORKERS = min(os.cpu_count() or 4, 16)

# ── High-speed string similarity backend ──────────────────────────────────────
try:
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz.distance import LCSseq, Prefix

    HAS_RAPIDFUZZ = True

    def _lev_ratio_batch(queries: list, targets: list) -> np.ndarray:
        return np.array([rfuzz.ratio(a, b) for a, b in zip(queries, targets)], dtype=np.float32) / 100.0

    def _tok_sort_batch(queries: list, targets: list) -> np.ndarray:
        return np.array([rfuzz.token_sort_ratio(a, b) for a, b in zip(queries, targets)], dtype=np.float32) / 100.0

    def _tok_set_batch(queries: list, targets: list) -> np.ndarray:
        return np.array([rfuzz.token_set_ratio(a, b) for a, b in zip(queries, targets)], dtype=np.float32) / 100.0

    def _lcs_ratio_batch(queries: list, targets: list) -> np.ndarray:
        return np.array([LCSseq.normalized_similarity(a, b) for a, b in zip(queries, targets)], dtype=np.float32)

    def _prefix_match_batch(queries: list, targets: list, n: int) -> np.ndarray:
        return np.array([Prefix.normalized_similarity(a[:n], b[:n]) for a, b in zip(queries, targets)], dtype=np.float32)

except ImportError:
    import difflib
    HAS_RAPIDFUZZ = False

    def _lev_ratio_batch(queries: list, targets: list) -> np.ndarray:
        return np.array([difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(queries, targets)], dtype=np.float32)

    def _tok_sort_batch(queries: list, targets: list) -> np.ndarray:
        def _ts(a, b):
            a2, b2 = " ".join(sorted(a.split())), " ".join(sorted(b.split()))
            return difflib.SequenceMatcher(None, a2, b2).ratio()
        return np.array([_ts(a, b) for a, b in zip(queries, targets)], dtype=np.float32)

    def _tok_set_batch(queries: list, targets: list) -> np.ndarray:
        def _tset(a, b):
            ta, tb = set(a.split()), set(b.split())
            union = ta | tb
            return len(ta & tb) / len(union) if union else 0.0
        return np.array([_tset(a, b) for a, b in zip(queries, targets)], dtype=np.float32)

    def _lcs_ratio_batch(queries: list, targets: list) -> np.ndarray:
        out = np.zeros(len(queries), dtype=np.float32)
        for i, (a, b) in enumerate(zip(queries, targets)):
            if not a and not b:
                out[i] = 1.0
            elif a and b:
                m = difflib.SequenceMatcher(None, a, b)
                lcs = m.find_longest_match(0, len(a), 0, len(b)).size
                out[i] = 2.0 * lcs / (len(a) + len(b))
        return out

    def _prefix_match_batch(queries: list, targets: list, n: int) -> np.ndarray:
        out = np.zeros(len(queries), dtype=np.float32)
        for i, (a, b) in enumerate(zip(queries, targets)):
            pa, pb = a[:n], b[:n]
            if not pa and not pb:
                out[i] = 1.0
            elif pa and pb:
                out[i] = sum(c1 == c2 for c1, c2 in zip(pa, pb)) / max(len(pa), len(pb))
        return out


# ── Vectorized token & numeric helpers ────────────────────────────────────────

def _jaccard_tokens_batch(a_list: list, b_list: list) -> np.ndarray:
    out = np.zeros(len(a_list), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        ta, tb = set(a.split()), set(b.split())
        union = len(ta | tb)
        out[i] = len(ta & tb) / union if union > 0 else (1.0 if not ta and not tb else 0.0)
    return out


def _jaccard_bigrams_batch(a_list: list, b_list: list) -> np.ndarray:
    out = np.zeros(len(a_list), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        ba = {a[j:j+2] for j in range(len(a)-1)} if len(a) > 1 else set()
        bb = {b[j:j+2] for j in range(len(b)-1)} if len(b) > 1 else set()
        union = len(ba | bb)
        out[i] = len(ba & bb) / union if union > 0 else (1.0 if not ba and not bb else 0.0)
    return out


_DIGIT_RE = re.compile(r'\d+')

def _numeric_overlap_batch(a_list: list, b_list: list) -> np.ndarray:
    out = np.zeros(len(a_list), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        na = set(_DIGIT_RE.findall(a))
        nb = set(_DIGIT_RE.findall(b))
        union = len(na | nb)
        out[i] = len(na & nb) / union if union > 0 else 1.0
    return out


def _length_ratio_batch(a_list: list, b_list: list) -> np.ndarray:
    la = np.array([len(a) for a in a_list], dtype=np.float32)
    lb = np.array([len(b) for b in b_list], dtype=np.float32)
    mx = np.maximum(la, lb)
    mn = np.minimum(la, lb)
    both_zero = mx == 0
    return np.where(both_zero, 1.0, mn / np.where(both_zero, 1.0, mx))


def _cross_tok_batch(a_list: list, b_list: list) -> np.ndarray:
    out = np.zeros(len(a_list), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        ta = set(a.split())
        if not ta:
            out[i] = 0.0
        else:
            tb = set(b.split())
            out[i] = len(ta & tb) / len(ta)
    return out


# ── Feature names ─────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "name_lev_ratio", "name_token_sort", "name_token_set",
    "name_jaccard_tok", "name_jaccard_bigram",
    "name_prefix3", "name_prefix5", "name_lcs", "name_len_ratio", "name_exact",
    "addr_lev_ratio", "addr_token_sort", "addr_jaccard_tok", "addr_jaccard_bigram",
    "addr_numeric_overlap", "addr_prefix5", "addr_lcs",
    "country_match",
    "cross_n1_a2", "cross_n2_a1",
]


# ── Worker function for sub-batch parallelization ─────────────────────────────

def _compute_features_sub_batch(
    sub_n1: list, sub_n2: list,
    sub_a1: list, sub_a2: list,
    sub_c1: list, sub_c2: list,
) -> np.ndarray:
    """Compute all 20 features for a slice of pairs in parallel."""
    m = len(sub_n1)
    sub_X = np.zeros((m, len(FEATURE_NAMES)), dtype=np.float32)

    # Name features (0-9)
    sub_X[:, 0] = _lev_ratio_batch(sub_n1, sub_n2)
    sub_X[:, 1] = _tok_sort_batch(sub_n1, sub_n2)
    sub_X[:, 2] = _tok_set_batch(sub_n1, sub_n2)
    sub_X[:, 3] = _jaccard_tokens_batch(sub_n1, sub_n2)
    sub_X[:, 4] = _jaccard_bigrams_batch(sub_n1, sub_n2)
    sub_X[:, 5] = _prefix_match_batch(sub_n1, sub_n2, 3)
    sub_X[:, 6] = _prefix_match_batch(sub_n1, sub_n2, 5)
    sub_X[:, 7] = _lcs_ratio_batch(sub_n1, sub_n2)
    sub_X[:, 8] = _length_ratio_batch(sub_n1, sub_n2)
    sub_X[:, 9] = np.array([1.0 if a == b else 0.0 for a, b in zip(sub_n1, sub_n2)], dtype=np.float32)

    # Address features (10-16)
    sub_X[:, 10] = _lev_ratio_batch(sub_a1, sub_a2)
    sub_X[:, 11] = _tok_sort_batch(sub_a1, sub_a2)
    sub_X[:, 12] = _jaccard_tokens_batch(sub_a1, sub_a2)
    sub_X[:, 13] = _jaccard_bigrams_batch(sub_a1, sub_a2)
    sub_X[:, 14] = _numeric_overlap_batch(sub_a1, sub_a2)
    sub_X[:, 15] = _prefix_match_batch(sub_a1, sub_a2, 5)
    sub_X[:, 16] = _lcs_ratio_batch(sub_a1, sub_a2)

    # Meta features (17-19)
    sub_X[:, 17] = np.array([1.0 if a == b else 0.0 for a, b in zip(sub_c1, sub_c2)], dtype=np.float32)
    sub_X[:, 18] = _cross_tok_batch(sub_n1, sub_a2)
    sub_X[:, 19] = _cross_tok_batch(sub_n2, sub_a1)

    return sub_X


# ── Main feature computation ──────────────────────────────────────────────────

def build_feature_matrix(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    candidate_pairs: Dict[str, set],
    batch_size: int = 100_000,
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Build feature matrix for all candidate pairs in vectorized multi-threaded batches.
    Utilizes multi-core OpenMP-style parallel processing across all CPU threads.
    """
    logger.info("Preparing fast entity lookup dictionaries...")
    t_prep = time.time()
    s1_names = dict(zip(s1_df["entity_id"].to_numpy(), s1_df["name_clean"].fillna("").astype(str).to_numpy()))
    s1_addrs = dict(zip(s1_df["entity_id"].to_numpy(), s1_df["addr_clean"].fillna("").astype(str).to_numpy()))
    s1_countries = dict(zip(s1_df["entity_id"].to_numpy(), s1_df["country_clean"].fillna("").astype(str).to_numpy()))

    s23_names = dict(zip(s23_df["entity_id"].to_numpy(), s23_df["name_clean"].fillna("").astype(str).to_numpy()))
    s23_addrs = dict(zip(s23_df["entity_id"].to_numpy(), s23_df["addr_clean"].fillna("").astype(str).to_numpy()))
    s23_countries = dict(zip(s23_df["entity_id"].to_numpy(), s23_df["country_clean"].fillna("").astype(str).to_numpy()))
    logger.info(f"  Lookup maps built in {time.time()-t_prep:.1f}s")

    # Flatten all pairs
    logger.info("Flattening candidate pairs...")
    pair_ids: List[Tuple[str, str]] = []
    s23_set = set(s23_names.keys())

    for s1_id, cand_ids in candidate_pairs.items():
        if s1_id not in s1_names:
            continue
        for cand_id in cand_ids:
            if cand_id in s23_set:
                pair_ids.append((s1_id, cand_id))

    n = len(pair_ids)
    logger.info(f"Total candidate pairs to featurize: {n:,} (using {N_WORKERS} parallel threads)")

    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), []

    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    t_start = time.time()

    # Worker chunk size within each batch
    sub_chunk_size = max(10_000, batch_size // N_WORKERS)

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch = pair_ids[batch_start:batch_end]

            n1 = [s1_names.get(s1, "") for s1, _ in batch]
            n2 = [s23_names.get(s23, "") for _, s23 in batch]
            a1 = [s1_addrs.get(s1, "") for s1, _ in batch]
            a2 = [s23_addrs.get(s23, "") for _, s23 in batch]
            c1 = [s1_countries.get(s1, "") for s1, _ in batch]
            c2 = [s23_countries.get(s23, "") for _, s23 in batch]

            batch_len = len(batch)
            tasks = []

            for sub_start in range(0, batch_len, sub_chunk_size):
                sub_end = min(sub_start + sub_chunk_size, batch_len)
                tasks.append((
                    batch_start + sub_start,
                    batch_start + sub_end,
                    executor.submit(
                        _compute_features_sub_batch,
                        n1[sub_start:sub_end], n2[sub_start:sub_end],
                        a1[sub_start:sub_end], a2[sub_start:sub_end],
                        c1[sub_start:sub_end], c2[sub_start:sub_end],
                    )
                ))

            for g_start, g_end, future in tasks:
                X[g_start:g_end, :] = future.result()

            done = batch_end
            if done % 200_000 == 0 or done == n:
                elapsed = time.time() - t_start
                rate = done / max(elapsed, 0.001)
                remaining = (n - done) / max(rate, 0.001) / 60.0
                pct = 100.0 * done / n
                logger.info(f"  Features: {done:,}/{n:,} pairs ({pct:.1f}%) — {rate:,.0f} pairs/sec — ETA {remaining:.1f} min")

    del s1_names, s1_addrs, s1_countries, s23_names, s23_addrs, s23_countries, s23_set
    gc.collect()
    logger.info(f"Feature matrix complete: {X.shape} in {time.time()-t_start:.1f}s")
    return X, pair_ids


if __name__ == "__main__":
    print(f"Features ({len(FEATURE_NAMES)}):", FEATURE_NAMES)
    print(f"rapidfuzz available: {HAS_RAPIDFUZZ}")
    print(f"Workers: {N_WORKERS}")
