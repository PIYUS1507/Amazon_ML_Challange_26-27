#!/usr/bin/env python3
"""
Step 5: Output Generation
Writes matching_results.tsv and candidate_pairs.tsv in the correct format.
"""
import os
from typing import Dict, Set
import pandas as pd


def write_matching_results(
    matches: Dict[str, Set[str]],
    all_s1_ids: list,
    out_path: str,
):
    """
    Write matching_results.tsv.
    Every S1 entity must appear exactly once. Empty set → empty matched_entity_ids.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for s1_id in all_s1_ids:
        matched = matches.get(s1_id, set())
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(sorted(matched)),
        })
    df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    df.to_csv(out_path, sep="\t", index=False)
    n_matched = sum(1 for r in rows if r["matched_entity_ids"])
    print(f"✓ Wrote {len(rows)} rows to {out_path} ({n_matched} with matches, {len(rows)-n_matched} singletons)")


def write_candidate_pairs(
    candidates: Dict[str, Set[str]],
    all_s1_ids: list,
    out_path: str,
):
    """
    Write candidate_pairs.tsv.
    Every S1 entity must appear exactly once.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for s1_id in all_s1_ids:
        cands = candidates.get(s1_id, set())
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(sorted(cands)),
        })
    df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    df.to_csv(out_path, sep="\t", index=False)
    print(f"✓ Wrote {len(rows)} rows to {out_path}")
