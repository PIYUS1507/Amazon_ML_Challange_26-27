#!/usr/bin/env python3
"""
Step 2: Blocking / Candidate Generation (High Recall, Low Memory & GPU Accelerated)

Achieves >99% recall ceiling while keeping candidate counts small (~20-50 per entity)
and memory overhead minimal via 7 high-discrimination country-conditioned hash keys:
  1. key_name4: Country + first 4 chars of name
  2. key_compact8: Country + first 8 alphanumeric chars of name (domain names, compact words)
  3. key_tok2: Country + first 2 name tokens sorted alphabetically
  4. key_tok_distinct: Country + longest distinctive name token (len >= 5)
  5. key_addr_num: Country + address numeric token (leading zeros stripped)
  6. key_tok1_addrnum: Country + first 3 chars of name + address numeric token
  7. key_addr_tok: Country + longest distinctive address token (len >= 6)

GPU Acceleration:
  When PyTorch CUDA is available, an additive TF-IDF char n-gram cosine similarity
  blocking pass runs on GPU to catch transliterated/fuzzy variations missed by hash keys.

Caching:
  Supports caching of S2/S3 hash indexes to disk via StepCache for instant re-runs.
"""
import re
import gc
import logging
from collections import defaultdict
from typing import Dict, Set, Optional, Tuple, Any

import pandas as pd
import numpy as np

from gpu_utils import HAS_TORCH_CUDA, get_torch, get_device

logger = logging.getLogger(__name__)

# ── Tunable constants ─────────────────────────────────────────────────────────
MAX_POSTINGS_PER_KEY  = 500   # Skip keys matching >500 records (e.g. ubiquitous words)
MAX_CANDIDATES_PER_S1 = 150   # Hard cap per S1 entity for safety & speed

# GPU TF-IDF blocking constants
TFIDF_TOP_K       = 25       # top-K candidates per S1 entity from TF-IDF
TFIDF_MIN_SCORE   = 0.15     # minimum cosine similarity to keep a candidate
TFIDF_NGRAM_RANGE = (2, 4)   # character n-gram range for TF-IDF
TFIDF_BATCH_SIZE  = 4096     # S1 rows per GPU batch (fits comfortably in 8GB VRAM)

NAME_STOPWORDS = {
    'limited', 'private', 'corporation', 'company', 'enterprises', 'holdings',
    'services', 'solutions', 'technologies', 'international', 'consulting',
    'associates', 'industries', 'group', 'pvt', 'ltd', 'inc', 'corp', 'llc', 'co'
}

ADDR_STOPWORDS = {
    'street', 'avenue', 'road', 'suite', 'floor', 'building', 'highway',
    'pradesh', 'maharashtra', 'karnataka', 'tamil', 'nadu', 'delhi', 'mumbai',
    'india', 'united', 'states', 'north', 'south', 'east', 'west', 'lane',
    'drive', 'circle', 'boulevard', 'block', 'sector', 'nagar', 'colony'
}


# ── String normalization helpers ──────────────────────────────────────────────

def _compact_str(s: str) -> str:
    """Normalize string to compact alphanumeric (strips domains, punctuation, spaces)."""
    s = s.lower()
    s = re.sub(r"\.(com|org|net|co|io|gov|edu|info|biz)\b", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def _first_digits(s: str) -> str:
    """Extract first numeric sequence with leading zeros stripped."""
    m = re.search(r"\d{2,}", s)
    return m.group().lstrip("0") if m else ""


def _first_n_tokens_sorted(text: str, n: int = 2) -> str:
    """First n tokens of name, sorted alphabetically — order-invariant."""
    tokens = [t for t in text.split()[:n] if t]
    return " ".join(sorted(tokens))


def _longest_distinct_name_tok(name: str) -> str:
    """Extract longest name token >= 5 chars not in generic company stopwords."""
    words = [w for w in name.split() if len(w) >= 5 and w not in NAME_STOPWORDS]
    return max(words, key=len)[:6] if words else ""


def _longest_distinct_addr_tok(addr: str) -> str:
    """Extract longest address token >= 6 chars not in address stopwords."""
    words = [
        w for w in re.sub(r"[^a-z0-9]", " ", addr).split()
        if len(w) >= 6 and w not in ADDR_STOPWORDS and not w.isdigit()
    ]
    return max(words, key=len)[:7] if words else ""


# ── Build blocking keys ───────────────────────────────────────────────────────

def build_blocking_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 7 blocking keys for each record."""
    df = df.copy()
    cc_arr = df["country_clean"].fillna("").astype(str).to_numpy()
    nc_arr = df["name_clean"].fillna("").astype(str).to_numpy()
    ac_arr = df["addr_clean"].fillna("").astype(str).to_numpy()
    n = len(df)

    k_name4 = []
    k_compact8 = []
    k_tok2 = []
    k_tok_distinct = []
    k_addr_num = []
    k_tok1_addrnum = []
    k_addr_tok = []

    for i in range(n):
        c = cc_arr[i]
        name = nc_arr[i]
        addr = ac_arr[i]

        # 1. key_name4
        n4 = name[:4].strip()
        k_name4.append(f"{c}||{n4}" if len(n4) >= 3 else "")

        # 2. key_compact8
        comp = _compact_str(name)
        k_compact8.append(f"{c}||{comp[:8]}" if len(comp) >= 5 else "")

        # 3. key_tok2
        t2 = _first_n_tokens_sorted(name, 2)
        k_tok2.append(f"{c}||{t2}" if t2 else "")

        # 4. key_tok_distinct
        d_name = _longest_distinct_name_tok(name)
        k_tok_distinct.append(f"{c}||{d_name}" if d_name else "")

        # 5. key_addr_num
        d_num = _first_digits(addr)
        k_addr_num.append(f"{c}||{d_num}" if d_num else "")

        # 6. key_tok1_addrnum
        first_w = name.split()[0] if name.split() else ""
        k_tok1_addrnum.append(f"{c}||{first_w[:3]}||{d_num}" if d_num and len(first_w) >= 3 else "")

        # 7. key_addr_tok
        d_addr = _longest_distinct_addr_tok(addr)
        k_addr_tok.append(f"{c}||{d_addr}" if d_addr else "")

    df["key_name4"] = k_name4
    df["key_compact8"] = k_compact8
    df["key_tok2"] = k_tok2
    df["key_tok_distinct"] = k_tok_distinct
    df["key_addr_num"] = k_addr_num
    df["key_tok1_addrnum"] = k_tok1_addrnum
    df["key_addr_tok"] = k_addr_tok

    return df


KEY_COLUMNS = [
    "key_name4",
    "key_compact8",
    "key_tok2",
    "key_tok_distinct",
    "key_addr_num",
    "key_tok1_addrnum",
    "key_addr_tok",
]


def _build_indexes(df: pd.DataFrame) -> Dict[str, Dict[str, list]]:
    """Build hash index mappings from keys to lists of entity IDs."""
    indexes = {}
    eids = df["entity_id"].to_numpy()

    for col in KEY_COLUMNS:
        idx = defaultdict(list)
        keys = df[col].to_numpy()
        for k, eid in zip(keys, eids):
            k_str = str(k)
            if k_str:
                suffix = k_str.split("||", 1)[-1] if "||" in k_str else ""
                if len(suffix) >= 2:
                    idx[k_str].append(eid)

        # Prune keys with more postings than MAX_POSTINGS_PER_KEY
        pruned_idx = {k: v for k, v in idx.items() if len(v) <= MAX_POSTINGS_PER_KEY}
        indexes[col] = pruned_idx
        logger.info(f"  [{col}] index: {len(pruned_idx):,} keys (postings <= {MAX_POSTINGS_PER_KEY})")

    return indexes


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
    3. Batch matrix-multiply S1 x S23^T to get cosine similarities.
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

        # Fit TF-IDF on S23 (the database side)
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
        s23_dense = torch.tensor(tfidf_s23.toarray(), device=device, dtype=torch.float16)  # (n_s23, V)
        s23_dense_t = s23_dense.T  # (V, n_s23)

        actual_k = min(top_k, len(s23_ids))

        for batch_start in range(0, len(s1_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(s1_ids))
            s1_batch = tfidf_s1[batch_start:batch_end]

            s1_dense = torch.tensor(s1_batch.toarray(), device=device, dtype=torch.float16)  # (B, V)

            # Cosine similarity = S1_batch @ S23^T
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
    trigram_min_shared: int = 3,  # Kept for backward compatibility
    cache: Optional[Any] = None,
    s23_indexes: Optional[Dict[str, Dict[str, list]]] = None,
) -> Dict[str, Set[str]]:
    """
    Memory-safe, high-speed chunked blocking for large S1 datasets.
    Builds lightweight hash indexes on S2/S3 once (or loads from cache/parameter),
    processes S1 in chunks, and complements with GPU TF-IDF cosine candidate search.
    """
    if s23_indexes is not None:
        logger.info("Using pre-built S2/S3 hash indexes...")
    elif cache is not None and cache.exists("s23_indexes"):
        logger.info("⚡ Loading S2/S3 hash indexes from cache...")
        s23_indexes = cache.load("s23_indexes")
    else:
        logger.info(f"Building blocking keys for {len(s23):,} S2/S3 records...")
        s23_k = build_blocking_keys(s23)

        logger.info("Building multi-key hash indexes for S2/S3...")
        s23_indexes = _build_indexes(s23_k)

        del s23_k
        gc.collect()

        if cache is not None:
            cache.save("s23_indexes", s23_indexes)

    all_candidates: Dict[str, Set[str]] = {}
    n_chunks = max(1, (len(s1) + chunk_size - 1) // chunk_size)

    for chunk_i in range(n_chunks):
        chunk = s1.iloc[chunk_i * chunk_size: (chunk_i + 1) * chunk_size]
        chunk_k = build_blocking_keys(chunk)

        chunk_eids = chunk_k["entity_id"].to_numpy()
        chunk_key_arrays = [(col, chunk_k[col].to_numpy()) for col in KEY_COLUMNS]

        chunk_cands: Dict[str, Set[str]] = {eid: set() for eid in chunk_eids}

        for i in range(len(chunk_eids)):
            eid = chunk_eids[i]
            cset = chunk_cands[eid]

            for col, arr in chunk_key_arrays:
                k = str(arr[i])
                if k:
                    postings = s23_indexes[col].get(k)
                    if postings:
                        cset.update(postings)

            if len(cset) > MAX_CANDIDATES_PER_S1:
                chunk_cands[eid] = set(sorted(cset)[:MAX_CANDIDATES_PER_S1])

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
            logger.warning(f"  GPU TF-IDF blocking failed ({e}), continuing with hash candidates")
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
    s1_col = gt_df["source1_entity_id"].to_numpy()
    match_col = gt_df["matched_entity_ids"].to_numpy()

    for s1, matched in zip(s1_col, match_col):
        matched_str = str(matched).strip() if pd.notna(matched) else ""
        ids = set(x.strip() for x in matched_str.split(",") if x.strip()) if matched_str else set()
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
