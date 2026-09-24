#!/usr/bin/env python3
"""
Step 3: Feature Engineering for Pairwise Matching
Computes rich similarity features between a candidate (S1, S2/S3) pair.

Features:
  - String similarity: Jaro-Winkler, Levenshtein ratio, token sort ratio
  - Token-based: Jaccard on word tokens, Jaccard on char bigrams
  - TF-IDF cosine: name, address
  - Numeric overlap: address numbers (building, pin, etc.)
  - Country match (binary)
  - Name length ratio
  - Prefix match (first 3/5 chars)
  - Longest common subsequence ratio
"""
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ── Optional fast edit-distance library ──────────────────────────────────────
try:
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz.distance import Levenshtein as RLevenshtein

    def jaro_winkler(a: str, b: str) -> float:
        return rfuzz.token_sort_ratio(a, b) / 100.0

    def levenshtein_ratio(a: str, b: str) -> float:
        return rfuzz.ratio(a, b) / 100.0

    def token_sort_ratio(a: str, b: str) -> float:
        return rfuzz.token_sort_ratio(a, b) / 100.0

    def token_set_ratio(a: str, b: str) -> float:
        return rfuzz.token_set_ratio(a, b) / 100.0

    HAS_RAPIDFUZZ = True

except ImportError:
    import difflib

    def jaro_winkler(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    def levenshtein_ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    def token_sort_ratio(a: str, b: str) -> float:
        a_sorted = " ".join(sorted(a.split()))
        b_sorted = " ".join(sorted(b.split()))
        return difflib.SequenceMatcher(None, a_sorted, b_sorted).ratio()

    def token_set_ratio(a: str, b: str) -> float:
        ta, tb = set(a.split()), set(b.split())
        inter = ta & tb
        union = ta | tb
        return len(inter) / len(union) if union else 0.0

    HAS_RAPIDFUZZ = False


def _jaccard_tokens(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _jaccard_bigrams(a: str, b: str) -> float:
    def bigrams(s):
        return set(s[i:i+2] for i in range(len(s) - 1)) if len(s) > 1 else set()
    ba, bb = bigrams(a), bigrams(b)
    if not ba and not bb:
        return 1.0
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def _numeric_overlap(a: str, b: str) -> float:
    """Overlap of numeric tokens (building numbers, pin codes)."""
    na = set(re.findall(r"\d+", a))
    nb = set(re.findall(r"\d+", b))
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def _prefix_match(a: str, b: str, n: int) -> float:
    """Fraction of first n chars that match."""
    pa, pb = a[:n], b[:n]
    if not pa and not pb:
        return 1.0
    if not pa or not pb:
        return 0.0
    matches = sum(c1 == c2 for c1, c2 in zip(pa, pb))
    return matches / max(len(pa), len(pb))


def _lcs_ratio(a: str, b: str) -> float:
    """LCS length ratio."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Use SequenceMatcher for LCS approximation
    import difflib
    m = difflib.SequenceMatcher(None, a, b)
    return 2 * m.find_longest_match(0, len(a), 0, len(b)).size / (len(a) + len(b))


def _length_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(la, lb) if max(la, lb) > 0 else 0.0


def compute_pair_features(
    s1_row: pd.Series,
    s23_row: pd.Series,
) -> np.ndarray:
    """
    Compute a feature vector for a candidate pair.
    Returns a 1D numpy array of floats.
    """
    n1 = s1_row.get("name_clean", "")
    n2 = s23_row.get("name_clean", "")
    a1 = s1_row.get("addr_clean", "")
    a2 = s23_row.get("addr_clean", "")
    c1 = s1_row.get("country_clean", "")
    c2 = s23_row.get("country_clean", "")

    feats = [
        # Name features (10)
        levenshtein_ratio(n1, n2),
        token_sort_ratio(n1, n2),
        token_set_ratio(n1, n2),
        _jaccard_tokens(n1, n2),
        _jaccard_bigrams(n1, n2),
        _prefix_match(n1, n2, 3),
        _prefix_match(n1, n2, 5),
        _lcs_ratio(n1, n2),
        _length_ratio(n1, n2),
        1.0 if n1 == n2 else 0.0,

        # Address features (7)
        levenshtein_ratio(a1, a2),
        token_sort_ratio(a1, a2),
        _jaccard_tokens(a1, a2),
        _jaccard_bigrams(a1, a2),
        _numeric_overlap(a1, a2),
        _prefix_match(a1, a2, 5),
        _lcs_ratio(a1, a2),

        # Country (1)
        1.0 if c1 == c2 else 0.0,

        # Cross features (2)
        # Name of one vs. address of other (trade names sometimes appear in addresses)
        _jaccard_tokens(n1, a2),
        _jaccard_tokens(n2, a1),
    ]
    return np.array(feats, dtype=np.float32)


FEATURE_NAMES = [
    "name_lev_ratio", "name_token_sort", "name_token_set",
    "name_jaccard_tok", "name_jaccard_bigram",
    "name_prefix3", "name_prefix5", "name_lcs", "name_len_ratio", "name_exact",
    "addr_lev_ratio", "addr_token_sort", "addr_jaccard_tok", "addr_jaccard_bigram",
    "addr_numeric_overlap", "addr_prefix5", "addr_lcs",
    "country_match",
    "cross_n1_a2", "cross_n2_a1",
]


def build_feature_matrix(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    candidate_pairs: Dict,
) -> Tuple[np.ndarray, list]:
    """
    Build feature matrix for all candidate pairs.
    Returns (X, pair_ids) where pair_ids is list of (s1_id, s23_id).
    """
    s1_index = s1_df.set_index("entity_id")
    s23_index = s23_df.set_index("entity_id")

    rows = []
    pair_ids = []

    for s1_id, cand_ids in candidate_pairs.items():
        if s1_id not in s1_index.index:
            continue
        s1_row = s1_index.loc[s1_id]
        for cand_id in cand_ids:
            if cand_id not in s23_index.index:
                continue
            s23_row = s23_index.loc[cand_id]
            feats = compute_pair_features(s1_row, s23_row)
            rows.append(feats)
            pair_ids.append((s1_id, cand_id))

    if not rows:
        return np.array([]).reshape(0, len(FEATURE_NAMES)), []

    return np.vstack(rows), pair_ids


if __name__ == "__main__":
    print("Feature names:", FEATURE_NAMES)
    print(f"Using rapidfuzz: {HAS_RAPIDFUZZ}")
