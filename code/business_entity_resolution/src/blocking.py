#!/usr/bin/env python3
"""
Step 2: Blocking / Candidate Generation
Generates candidate pairs using multiple blocking keys to maximize recall
while reducing the search space (Reduction Ratio).

Blocking strategies:
  1. Country + Name prefix (first 3 chars)
  2. Country + Address numeric token (building number)
  3. Country + TF-IDF cosine similarity top-k (approximate)
  4. Country + Sorted Bigram set overlap
"""
import re
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def _bigrams(text: str) -> Set[str]:
    """Character bigrams of a string."""
    return {text[i:i+2] for i in range(len(text) - 1)} if len(text) > 1 else set()


def _sorted_bigram_key(text: str, n: int = 4) -> str:
    """Take top-n sorted bigrams as a blocking key."""
    bgs = sorted(_bigrams(text))
    return "".join(bgs[:n])


def build_blocking_keys(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Compute blocking keys for S2/S3 records.
    Returns same df with added key columns.
    """
    df = df.copy()
    cc = df["country_clean"]
    nc = df["name_clean"]
    ac = df["addr_clean"]

    # Key 1: country + first 4 chars of name
    df["key_name_prefix"] = cc + "||" + nc.str[:4]

    # Key 2: country + first numeric token in address
    df["key_addr_num"] = cc + "||" + df["addr_num"]

    # Key 3: country + sorted bigram key of name (first 5 chars)
    df["key_bigram"] = cc + "||" + nc.apply(_sorted_bigram_key)

    # Key 4: country + first token of name
    df["key_name_tok"] = cc + "||" + df["name_prefix"]

    return df


def _index_by_key(df: pd.DataFrame, key_col: str) -> Dict[str, List[str]]:
    """Build inverted index: key -> list of entity_ids."""
    idx = defaultdict(list)
    for _, row in df.iterrows():
        k = row[key_col]
        if k and k.split("||")[-1]:  # skip empty keys
            idx[k].append(row["entity_id"])
    return idx


def blocking_pass(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    tfidf_top_k: int = 20,
) -> Dict[str, Set[str]]:
    """
    For each S1 entity, return a set of candidate S2/S3 entity IDs.
    Uses multiple blocking passes and merges candidates.
    """
    # Build keys for both sides
    s1_keyed = build_blocking_keys(s1, "S1")
    s23_keyed = build_blocking_keys(s23, "S23")

    candidates: Dict[str, Set[str]] = {eid: set() for eid in s1["entity_id"]}
    key_cols = ["key_name_prefix", "key_addr_num", "key_bigram", "key_name_tok"]

    # ── Hash-based blocking ───────────────────────────────────────────────────
    for key_col in key_cols:
        s23_index = _index_by_key(s23_keyed, key_col)
        for _, row in s1_keyed.iterrows():
            k = row[key_col]
            if k and k.split("||")[-1]:
                for cand_id in s23_index.get(k, []):
                    candidates[row["entity_id"]].add(cand_id)

    logger.info(f"After hash blocking: {sum(len(v) for v in candidates.values())} total candidates")

    # ── TF-IDF blocking (per country) ─────────────────────────────────────────
    countries = s1["country_clean"].unique()
    for country in countries:
        s1_c = s1_keyed[s1_keyed["country_clean"] == country]
        s23_c = s23_keyed[s23_keyed["country_clean"] == country]

        if s1_c.empty or s23_c.empty:
            continue

        # Use name + address as combined text
        s1_texts = (s1_c["name_clean"] + " " + s1_c["addr_clean"]).tolist()
        s23_texts = (s23_c["name_clean"] + " " + s23_c["addr_clean"]).tolist()
        s1_ids = s1_c["entity_id"].tolist()
        s23_ids = s23_c["entity_id"].tolist()

        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1, max_features=50000)
            all_texts = s1_texts + s23_texts
            vec.fit(all_texts)
            s1_mat = vec.transform(s1_texts)
            s23_mat = vec.transform(s23_texts)

            # Compute cosine similarity in batches to avoid OOM
            batch_size = 500
            for batch_start in range(0, len(s1_ids), batch_size):
                batch_end = min(batch_start + batch_size, len(s1_ids))
                sim = cosine_similarity(s1_mat[batch_start:batch_end], s23_mat)
                # Top-k per row
                for i, s1_id in enumerate(s1_ids[batch_start:batch_end]):
                    top_k_idx = np.argpartition(sim[i], -min(tfidf_top_k, sim.shape[1]))[-min(tfidf_top_k, sim.shape[1]):]
                    for j in top_k_idx:
                        if sim[i, j] > 0.25:  # Minimum similarity threshold
                            candidates[s1_id].add(s23_ids[j])

        except Exception as e:
            logger.warning(f"TF-IDF blocking failed for country={country}: {e}")

    total_cands = sum(len(v) for v in candidates.values())
    logger.info(f"After TF-IDF blocking: {total_cands} total candidates")
    return candidates


def compute_blocking_stats(
    candidates: Dict[str, Set[str]],
    ground_truth: pd.DataFrame,
) -> Dict:
    """
    Compute recall ceiling and reduction ratio against ground truth.
    Only usable during training/validation.
    """
    total_true = 0
    total_recalled = 0

    gt_map = {}
    for _, row in ground_truth.iterrows():
        s1 = row["source1_entity_id"]
        matched = str(row["matched_entity_ids"]).strip()
        ids = set(x.strip() for x in matched.split(",") if x.strip()) if matched else set()
        gt_map[s1] = ids

    for s1, true_matches in gt_map.items():
        if not true_matches:
            continue
        cands = candidates.get(s1, set())
        recalled = len(true_matches & cands)
        total_true += len(true_matches)
        total_recalled += recalled

    recall_ceiling = total_recalled / total_true if total_true > 0 else 1.0
    avg_candidates = sum(len(v) for v in candidates.values()) / max(len(candidates), 1)

    return {
        "recall_ceiling": recall_ceiling,
        "avg_candidates_per_s1": avg_candidates,
        "total_pairs": sum(len(v) for v in candidates.values()),
    }
