#!/usr/bin/env python3
"""
Step 2: Blocking / Candidate Generation  (scale-aware, precision-tuned, GPU-accelerated)

Target: ~20-50 candidates per S1 entity on full 10M S2+S3 dataset.

Blocking strategies (all country-conditioned):
  1. Exact name-4 prefix + country
  2. First numeric token in address + country  (only when num >= 2 digits)
  3. Sorted char-bigram key (top-6 bigrams of name) + country
  4. First 2 tokens of name sorted + country
  5. Trigram inverted index with min_shared ≥ 3  (raised from 2 to cut FPs)
  6. [GPU] TF-IDF char n-gram cosine similarity via PyTorch CUDA (top-K per entity)

GPU acceleration:
  When PyTorch CUDA is available, an additional TF-IDF cosine blocking pass
  is run on GPU. Sparse TF-IDF matrices are built on CPU (scikit-learn), then
  batched GPU matrix multiplication finds the top-K most similar S2/S3
  candidates per S1 entity. This catches candidates that the hash-based
  strategies miss (different prefixes, reorderings, transliterations).

Falls back to CPU-only trigram blocking when CUDA is unavailable.
"""
import re
import logging
from collections import defaultdict
from typing import Dict, Set

import pandas as pd
import numpy as np

from gpu_utils import HAS_TORCH_CUDA, get_torch, get_device

logger = logging.getLogger(__name__)

# ── Tunable constants ─────────────────────────────────────────────────────────
TRIGRAM_MIN_SHARED   = 3     # must share at least this many char-3-grams
TRIGRAM_MAX_PER_KEY  = 50    # cap postings per trigram (skip super-common ones)
MAX_CANDIDATES_PER_S1 = 200  # hard cap per S1 entity (safety valve)

# GPU TF-IDF blocking knobs
TFIDF_TOP_K       = 25       # top-K candidates per S1 entity from TF-IDF
TFIDF_MIN_SCORE   = 0.15     # minimum cosine similarity to keep a candidate
TFIDF_NGRAM_RANGE = (2, 4)   # character n-gram range for TF-IDF
TFIDF_BATCH_SIZE  = 4096     # S1 rows per GPU batch (tune for 8GB VRAM)


# ── Character n-gram helpers ──────────────────────────────────────────────────

def _trigrams(text: str) -> list:
    return [text[i:i+3] for i in range(len(text) - 2)] if len(text) > 2 else []


def _bigrams(text: str) -> set:
    return {text[i:i+2] for i in range(len(text) - 1)} if len(text) > 1 else set()


def _sorted_bigram_key(text: str, n: int = 6) -> str:
    return "".join(sorted(_bigrams(text))[:n])


def _first_n_tokens_sorted(text: str, n: int = 2) -> str:
    """First n tokens of name, sorted alphabetically — order-invariant."""
    tokens = text.split()[:n]
    return " ".join(sorted(tokens))


def _first_digits(text: str) -> str:
    m = re.search(r'\d{2,}', text)   # require >= 2 digits to avoid single-digit noise
    return m.group() if m else ""


# ── Build blocking keys ───────────────────────────────────────────────────────

def build_blocking_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cc = df["country_clean"]
    nc = df["name_clean"]

    df["key_name4"]   = cc + "||" + nc.str[:4]
    df["key_addr_num"] = cc + "||" + df["addr_num"].where(df["addr_num"].str.len() >= 2, "")
    df["key_bigram6"] = cc + "||" + nc.apply(lambda x: _sorted_bigram_key(x, 6))
    df["key_tok2"]    = cc + "||" + nc.apply(lambda x: _first_n_tokens_sorted(x, 2))

    return df


def _index_by_key(df: pd.DataFrame, key_col: str, min_suffix_len: int = 3) -> Dict[str, list]:
    idx = defaultdict(list)
    for _, row in df.iterrows():
        k = row[key_col]
        suffix = k.split("||", 1)[-1] if "||" in k else ""
        if len(suffix) >= min_suffix_len:
            idx[k].append(row["entity_id"])
    return idx


# ── Trigram inverted index ────────────────────────────────────────────────────

def build_trigram_index(df: pd.DataFrame) -> Dict[tuple, list]:
    """Build (country, trigram) → [entity_id, ...] index from S2/S3."""
    idx = defaultdict(list)
    for _, row in df.iterrows():
        cc = row["country_clean"]
        nc = row["name_clean"]
        seen = set()
        for tg in _trigrams(nc):
            if tg not in seen:
                seen.add(tg)
                idx[(cc, tg)].append(row["entity_id"])
    return idx


def trigram_candidates(
    s1_row: pd.Series,
    trigram_idx: Dict,
) -> Set[str]:
    """Return S2/S3 IDs sharing >= TRIGRAM_MIN_SHARED trigrams with s1_row."""
    cc = s1_row["country_clean"]
    nc = s1_row["name_clean"]
    counter = defaultdict(int)
    seen_tg = set()
    for tg in _trigrams(nc):
        if tg in seen_tg:
            continue
        seen_tg.add(tg)
        postings = trigram_idx.get((cc, tg), [])
        if len(postings) <= TRIGRAM_MAX_PER_KEY:   # skip ultra-common trigrams
            for eid in postings:
                counter[eid] += 1
    return {eid for eid, cnt in counter.items() if cnt >= TRIGRAM_MIN_SHARED}


# ── GPU-accelerated TF-IDF cosine blocking ────────────────────────────────────

def _gpu_tfidf_blocking(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    top_k: int = TFIDF_TOP_K,
    min_score: float = TFIDF_MIN_SCORE,
    batch_size: int = TFIDF_BATCH_SIZE,
) -> Dict[str, Set[str]]:
    """
    GPU-accelerated TF-IDF character n-gram cosine similarity blocking.

    1. Build TF-IDF sparse matrices on CPU (scikit-learn).
    2. For each country group, convert to dense PyTorch tensors on GPU.
    3. Batch matrix-multiply S1 × S23^T to get cosine similarities.
    4. Extract top-K candidates per S1 entity above min_score threshold.

    Returns dict of {s1_entity_id: set of s23_entity_ids}.
    """
    torch = get_torch()
    device = get_device()

    from sklearn.feature_extraction.text import TfidfVectorizer

    candidates: Dict[str, Set[str]] = {}

    # Group by country to avoid cross-country matches
    s1_countries = s1["country_clean"].unique()
    s23_grouped = s23.groupby("country_clean")

    for country in s1_countries:
        s1_c = s1[s1["country_clean"] == country]
        if country not in s23_grouped.groups:
            for eid in s1_c["entity_id"]:
                candidates[eid] = set()
            continue
        s23_c = s23_grouped.get_group(country)

        if len(s1_c) == 0 or len(s23_c) == 0:
            continue

        # Build combined name+address text for TF-IDF
        s1_texts  = (s1_c["name_clean"].fillna("") + " " + s1_c["addr_clean"].fillna("")).tolist()
        s23_texts = (s23_c["name_clean"].fillna("") + " " + s23_c["addr_clean"].fillna("")).tolist()
        s1_ids  = s1_c["entity_id"].tolist()
        s23_ids = s23_c["entity_id"].tolist()

        # Fit TF-IDF on S23 (the "database" side)
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=TFIDF_NGRAM_RANGE,
            max_features=50_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        tfidf_s23 = vectorizer.fit_transform(s23_texts)   # sparse (n_s23, vocab)
        tfidf_s1  = vectorizer.transform(s1_texts)         # sparse (n_s1, vocab)

        # Convert to dense PyTorch tensors — work in batches to fit in 8GB VRAM
        # S23 matrix stays on GPU for all batches
        s23_dense = torch.tensor(tfidf_s23.toarray(), device=device, dtype=torch.float16)  # (n_s23, V)
        s23_dense_t = s23_dense.T  # (V, n_s23)

        actual_k = min(top_k, len(s23_ids))

        for batch_start in range(0, len(s1_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(s1_ids))
            s1_batch = tfidf_s1[batch_start:batch_end]

            s1_dense = torch.tensor(s1_batch.toarray(), device=device, dtype=torch.float16)  # (B, V)

            # Cosine similarity = S1_batch @ S23^T  (both already L2-normalized by TF-IDF)
            sim = torch.mm(s1_dense, s23_dense_t)  # (B, n_s23)

            # Top-K per row
            topk_scores, topk_indices = torch.topk(sim, k=actual_k, dim=1)

            # Move results to CPU and build candidate sets
            topk_scores_cpu = topk_scores.cpu().float().numpy()
            topk_indices_cpu = topk_indices.cpu().numpy()

            for i in range(batch_end - batch_start):
                s1_id = s1_ids[batch_start + i]
                cand_set = candidates.get(s1_id, set())
                for j in range(actual_k):
                    if topk_scores_cpu[i, j] >= min_score:
                        cand_set.add(s23_ids[topk_indices_cpu[i, j]])
                candidates[s1_id] = cand_set

            del s1_dense, sim, topk_scores, topk_indices

        del s23_dense, s23_dense_t
        torch.cuda.empty_cache()

    return candidates


# ── Main blocking entry point ─────────────────────────────────────────────────

def blocking_pass(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
) -> Dict[str, Set[str]]:
    """Single-pass blocking (for small inputs)."""
    return blocking_pass_chunked(s1, s23, chunk_size=len(s1))


def blocking_pass_chunked(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    chunk_size: int = 200_000,
    trigram_min_shared: int = TRIGRAM_MIN_SHARED,   # kept for API compat
) -> Dict[str, Set[str]]:
    """
    Memory-safe chunked blocking for large S1 datasets.
    Builds S2/S3 indexes once, processes S1 in chunks.

    When PyTorch CUDA is available, adds GPU TF-IDF cosine candidates
    to complement the hash-based and trigram strategies.
    """
    logger.info(f"Building blocking keys for {len(s23):,} S2/S3 records...")
    s23_k = build_blocking_keys(s23)

    key_cols = ["key_name4", "key_addr_num", "key_bigram6", "key_tok2"]
    logger.info("Building hash indexes for S2/S3...")
    s23_hash_idxs = {}
    for kc in key_cols:
        s23_hash_idxs[kc] = _index_by_key(s23_k, kc, min_suffix_len=3)
        logger.info(f"  [{kc}] index: {len(s23_hash_idxs[kc]):,} unique keys")

    logger.info("Building trigram index for S2/S3...")
    trigram_idx = build_trigram_index(s23_k)
    logger.info(f"  Trigram index: {len(trigram_idx):,} (country, trigram) keys")

    all_candidates: Dict[str, Set[str]] = {}
    n_chunks = max(1, (len(s1) + chunk_size - 1) // chunk_size)

    for chunk_i in range(n_chunks):
        chunk = s1.iloc[chunk_i * chunk_size: (chunk_i + 1) * chunk_size]
        chunk_k = build_blocking_keys(chunk)
        chunk_cands: Dict[str, Set[str]] = {eid: set() for eid in chunk["entity_id"]}

        # Phase A: hash blocking
        for kc in key_cols:
            for _, row in chunk_k.iterrows():
                k = row[kc]
                suffix = k.split("||", 1)[-1] if "||" in k else ""
                if len(suffix) >= 3:
                    chunk_cands[row["entity_id"]].update(
                        s23_hash_idxs[kc].get(k, [])
                    )

        # Phase B: trigram blocking
        for _, row in chunk_k.iterrows():
            new_c = trigram_candidates(row, trigram_idx)
            chunk_cands[row["entity_id"]].update(new_c)

        # Hard cap per entity to prevent explosions
        for eid in chunk_cands:
            if len(chunk_cands[eid]) > MAX_CANDIDATES_PER_S1:
                # Keep a deterministic subset (sorted for reproducibility)
                chunk_cands[eid] = set(sorted(chunk_cands[eid])[:MAX_CANDIDATES_PER_S1])

        all_candidates.update(chunk_cands)

        if n_chunks > 1 and ((chunk_i + 1) % 5 == 0 or chunk_i == n_chunks - 1):
            done = min((chunk_i + 1) * chunk_size, len(s1))
            logger.info(f"  Chunked blocking: {done:,}/{len(s1):,} S1 rows processed")

    # ── Phase C: GPU TF-IDF cosine blocking (additive — only adds candidates) ─
    if HAS_TORCH_CUDA:
        logger.info("Running GPU-accelerated TF-IDF cosine blocking...")
        import time
        t0 = time.time()
        try:
            gpu_cands = _gpu_tfidf_blocking(s1, s23)
            # Merge GPU candidates into existing candidates
            gpu_added = 0
            for s1_id, gpu_set in gpu_cands.items():
                if s1_id in all_candidates:
                    before = len(all_candidates[s1_id])
                    all_candidates[s1_id].update(gpu_set)
                    gpu_added += len(all_candidates[s1_id]) - before
                else:
                    all_candidates[s1_id] = gpu_set
                    gpu_added += len(gpu_set)
            logger.info(f"  GPU TF-IDF added {gpu_added:,} new candidates in {time.time()-t0:.0f}s")
        except Exception as e:
            logger.warning(f"  GPU TF-IDF blocking failed ({e}), continuing with CPU-only candidates")
    else:
        logger.info("GPU not available — skipping TF-IDF cosine blocking")

    # Re-apply hard cap after GPU additions
    for eid in all_candidates:
        if len(all_candidates[eid]) > MAX_CANDIDATES_PER_S1:
            all_candidates[eid] = set(sorted(all_candidates[eid])[:MAX_CANDIDATES_PER_S1])

    total = sum(len(v) for v in all_candidates.values())
    avg = total / max(len(all_candidates), 1)
    logger.info(f"Blocking complete: {total:,} candidate pairs | avg {avg:.1f}/S1 entity")
    return all_candidates


def compute_blocking_stats(
    candidates: Dict[str, Set[str]],
    gt_df: pd.DataFrame,
) -> dict:
    """Compute recall ceiling and avg candidates vs ground truth."""
    gt_map = {}
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        matched = str(row.get("matched_entity_ids", "")).strip()
        ids = set(x.strip() for x in matched.split(",") if x.strip()) if matched else set()
        gt_map[s1] = ids

    total_true = total_recalled = 0
    for s1, true_matches in gt_map.items():
        if not true_matches:
            continue
        cands = candidates.get(s1, set())
        total_true += len(true_matches)
        total_recalled += len(true_matches & cands)

    recall_ceiling = total_recalled / total_true if total_true > 0 else 1.0
    avg_cands = sum(len(v) for v in candidates.values()) / max(len(candidates), 1)
    return {
        "recall_ceiling": round(recall_ceiling, 4),
        "avg_candidates_per_s1": round(avg_cands, 1),
        "total_candidate_pairs": sum(len(v) for v in candidates.values()),
    }
