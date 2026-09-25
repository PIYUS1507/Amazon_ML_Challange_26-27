#!/usr/bin/env python3
"""
Step 2: Blocking / Candidate Generation  (scale-aware, precision-tuned)

Target: ~20-50 candidates per S1 entity on full 10M S2+S3 dataset.

Blocking strategies (all country-conditioned):
  1. Exact name-4 prefix + country
  2. First numeric token in address + country  (only when num >= 2 digits)
  3. Sorted char-bigram key (top-6 bigrams of name) + country
  4. First 2 tokens of name sorted + country
  5. Trigram inverted index with min_shared ≥ 3  (raised from 2 to cut FPs)

Removed:
  - name-prefix-3  (too broad → explodes bucket sizes)
  - consonant-only key  (also too broad)
  - min_shared=2 trigrams  (generates ~500 candidates/entity on full data)
"""
import re
import logging
from collections import defaultdict
from typing import Dict, Set

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Tunable constants ─────────────────────────────────────────────────────────
TRIGRAM_MIN_SHARED   = 3     # must share at least this many char-3-grams
TRIGRAM_MAX_PER_KEY  = 50    # cap postings per trigram (skip super-common ones)
MAX_CANDIDATES_PER_S1 = 200  # hard cap per S1 entity (safety valve)


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
