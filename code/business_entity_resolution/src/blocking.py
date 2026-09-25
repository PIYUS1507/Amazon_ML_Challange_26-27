#!/usr/bin/env python3
"""
Step 2: Blocking / Candidate Generation  (scale-aware version)
Generates candidate pairs using multiple blocking keys to maximize recall
while reducing the search space (Reduction Ratio).

Scale: S1=2.2M, S2+S3=10.3M → ~22 trillion naive pairs → we reduce to ~O(30) per S1

Blocking strategies (all country-conditioned):
  1. Name prefix-4 hash
  2. Address first numeric token (building/street number)
  3. Sorted character bigram key (top-5 bigrams)
  4. First token of name
  5. Name prefix-3 + country (broader catch)
  6. Trigram inverted index (sparse token-overlap, char 3-grams)
     → replaces dense TF-IDF to avoid OOM at 10M scale
"""
import re
import logging
from collections import defaultdict
from typing import Dict, Set

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ── Utility helpers ───────────────────────────────────────────────────────────

def _bigrams(text: str) -> set:
    return {text[i:i+2] for i in range(len(text) - 1)} if len(text) > 1 else set()


def _trigrams(text: str) -> set:
    return {text[i:i+3] for i in range(len(text) - 2)} if len(text) > 2 else set()


def _sorted_bigram_key(text: str, n: int = 5) -> str:
    bgs = sorted(_bigrams(text))
    return "".join(bgs[:n])


def _first_digits(text: str) -> str:
    m = re.search(r'\d+', text)
    return m.group() if m else ""


# ── Build blocking keys ───────────────────────────────────────────────────────

def build_blocking_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cc = df["country_clean"]
    nc = df["name_clean"]

    df["key_name4"]    = cc + "||" + nc.str[:4]
    df["key_addr_num"] = cc + "||" + df["addr_num"]
    df["key_bigram5"]  = cc + "||" + nc.apply(lambda x: _sorted_bigram_key(x, 5))
    df["key_tok1"]     = cc + "||" + df["name_prefix"]
    df["key_name3"]    = cc + "||" + nc.str[:3]
    # Soundex-like: vowel-stripped prefix (catches more transliterations)
    df["key_consonant"] = cc + "||" + nc.str[:6].str.replace(r'[aeiou\s]', '', regex=True).str[:4]

    return df


def _index_by_key(df: pd.DataFrame, key_col: str) -> Dict[str, list]:
    idx = defaultdict(list)
    for _, row in df.iterrows():
        k = row[key_col]
        suffix = k.split("||", 1)[-1] if "||" in k else ""
        if suffix and len(suffix) >= 2:  # skip empty/single-char keys
            idx[k].append(row["entity_id"])
    return idx


# ── Trigram inverted index (scalable alternative to dense TF-IDF) ─────────────

def build_trigram_index(df: pd.DataFrame) -> Dict[str, list]:
    """
    Build inverted index: (country, trigram) → list of entity_ids.
    Uses name trigrams only (most discriminative).
    """
    idx = defaultdict(list)
    for _, row in df.iterrows():
        cc = row["country_clean"]
        nc = row["name_clean"]
        for tg in _trigrams(nc):
            if len(tg) == 3 and not tg.isspace():
                idx[(cc, tg)].append(row["entity_id"])
    return idx


def trigram_candidates(
    s1_row: pd.Series,
    trigram_idx: Dict,
    min_shared: int = 2,
    max_per_trigram: int = 100,
) -> Set[str]:
    """
    For a single S1 row, find S2/S3 candidates by trigram overlap.
    Uses a counter: candidate must share ≥ min_shared trigrams with S1.
    Caps per-trigram list to avoid runaway common trigrams (e.g. "the").
    """
    cc = s1_row["country_clean"]
    nc = s1_row["name_clean"]
    counter = defaultdict(int)
    for tg in _trigrams(nc):
        if len(tg) == 3 and not tg.isspace():
            for eid in trigram_idx.get((cc, tg), [])[:max_per_trigram]:
                counter[eid] += 1
    return {eid for eid, cnt in counter.items() if cnt >= min_shared}


# ── Main blocking entry point ─────────────────────────────────────────────────

def blocking_pass(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    trigram_min_shared: int = 2,
) -> Dict[str, Set[str]]:
    """
    For each S1 entity, return a set of candidate S2/S3 entity_ids.

    Two phases:
      Phase A — Hash blocking (5 key types, O(N) per type)
      Phase B — Trigram inverted index (scalable approximate name similarity)
    """
    logger.info(f"Building blocking keys for {len(s1):,} S1 and {len(s23):,} S2/S3 records...")
    s1_k  = build_blocking_keys(s1)
    s23_k = build_blocking_keys(s23)

    candidates: Dict[str, Set[str]] = {eid: set() for eid in s1["entity_id"]}

    # ── Phase A: Hash-based blocking ─────────────────────────────────────────
    key_cols = ["key_name4", "key_addr_num", "key_bigram5", "key_tok1",
                "key_name3", "key_consonant"]

    for key_col in key_cols:
        s23_idx = _index_by_key(s23_k, key_col)
        hits = 0
        for _, row in s1_k.iterrows():
            k = row[key_col]
            suffix = k.split("||", 1)[-1] if "||" in k else ""
            if suffix and len(suffix) >= 2:
                new = s23_idx.get(k, [])
                candidates[row["entity_id"]].update(new)
                hits += len(new)
        logger.info(f"  [{key_col}] added {hits:,} candidate links")

    after_hash = sum(len(v) for v in candidates.values())
    logger.info(f"After hash blocking: {after_hash:,} total candidate pairs")

    # ── Phase B: Trigram inverted index ──────────────────────────────────────
    logger.info("Building trigram inverted index for S2/S3...")
    trigram_idx = build_trigram_index(s23_k)
    logger.info(f"  Trigram index size: {len(trigram_idx):,} (country, trigram) keys")

    logger.info("Running trigram candidate lookup for S1 entities...")
    tg_added = 0
    for _, row in s1_k.iterrows():
        new_cands = trigram_candidates(row, trigram_idx, min_shared=trigram_min_shared)
        before = len(candidates[row["entity_id"]])
        candidates[row["entity_id"]].update(new_cands)
        tg_added += len(candidates[row["entity_id"]]) - before

    total_cands = sum(len(v) for v in candidates.values())
    logger.info(f"Trigram blocking added {tg_added:,} new links")
    logger.info(f"Final candidate pool: {total_cands:,} pairs across {len(candidates):,} S1 entities")
    logger.info(f"Average candidates per S1 entity: {total_cands / max(len(candidates),1):.1f}")

    return candidates


def blocking_pass_chunked(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    chunk_size: int = 200_000,
    trigram_min_shared: int = 2,
) -> Dict[str, Set[str]]:
    """
    Memory-safe chunked version of blocking_pass for very large S1.
    Processes S1 in chunks, keeping the S2/S3 index in memory once.
    """
    logger.info(f"Chunked blocking: {len(s1):,} S1 rows in chunks of {chunk_size:,}")

    # Build S2/S3 indexes once
    s23_k = build_blocking_keys(s23)
    key_cols = ["key_name4", "key_addr_num", "key_bigram5", "key_tok1",
                "key_name3", "key_consonant"]

    logger.info("Building hash indexes for S2/S3...")
    s23_hash_idxs = {kc: _index_by_key(s23_k, kc) for kc in key_cols}

    logger.info("Building trigram index for S2/S3...")
    trigram_idx = build_trigram_index(s23_k)
    logger.info(f"  Trigram index: {len(trigram_idx):,} keys")

    all_candidates: Dict[str, Set[str]] = {}

    n_chunks = (len(s1) + chunk_size - 1) // chunk_size
    for chunk_i in range(n_chunks):
        chunk = s1.iloc[chunk_i * chunk_size : (chunk_i + 1) * chunk_size]
        chunk_k = build_blocking_keys(chunk)

        chunk_cands: Dict[str, Set[str]] = {eid: set() for eid in chunk["entity_id"]}

        # Hash blocking
        for kc in key_cols:
            for _, row in chunk_k.iterrows():
                k = row[kc]
                suffix = k.split("||", 1)[-1] if "||" in k else ""
                if suffix and len(suffix) >= 2:
                    chunk_cands[row["entity_id"]].update(s23_hash_idxs[kc].get(k, []))

        # Trigram blocking
        for _, row in chunk_k.iterrows():
            new_c = trigram_candidates(row, trigram_idx, min_shared=trigram_min_shared)
            chunk_cands[row["entity_id"]].update(new_c)

        all_candidates.update(chunk_cands)

        if (chunk_i + 1) % 5 == 0 or chunk_i == n_chunks - 1:
            done = (chunk_i + 1) * chunk_size
            logger.info(f"  Chunked blocking: {min(done, len(s1)):,}/{len(s1):,} S1 rows processed")

    total = sum(len(v) for v in all_candidates.values())
    logger.info(f"Chunked blocking complete: {total:,} total candidate pairs")
    return all_candidates


def compute_blocking_stats(
    candidates: Dict[str, Set[str]],
    gt_df: pd.DataFrame,
) -> dict:
    """Compute recall ceiling and avg candidates against ground truth."""
    gt_map = {}
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        matched = str(row.get("matched_entity_ids", "")).strip()
        ids = set(x.strip() for x in matched.split(",") if x.strip()) if matched else set()
        gt_map[s1] = ids

    total_true = 0
    total_recalled = 0
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
