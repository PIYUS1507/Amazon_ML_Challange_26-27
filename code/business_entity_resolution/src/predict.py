#!/usr/bin/env python3
"""
predict.py — Inference pipeline for Business Entity Resolution

Usage:
    python code/business_entity_resolution/src/predict.py \
        --test-dir dataset/test \
        --model models/matching_model.pkl \
        --output-dir output \
        --tfidf-top-k 25

This script:
  1. Loads the trained model
  2. Preprocesses test data (all 3 sources)
  3. Runs blocking to generate candidates
  4. Engineers features for all candidate pairs
  5. Runs inference and applies learned threshold
  6. Writes matching_results.tsv and candidate_pairs.tsv
  7. Runs the validator automatically
"""
import argparse
import logging
import os
import subprocess
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from preprocess import preprocess_df
from blocking import blocking_pass
from features import build_feature_matrix
from model import MatchingModel, predict_matches
from output import write_matching_results, write_candidate_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--model", default="models/matching_model.pkl")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--tfidf-top-k", type=int, default=25,
                        help="TF-IDF top-k candidates (use slightly more at inference)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override learned threshold (optional)")
    parser.add_argument("--skip-validate", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading model from {args.model}...")
    model = MatchingModel.load(args.model)
    logger.info(f"  Backend: {model.backend}, threshold: {model.threshold:.4f}")

    # ── Load test data ────────────────────────────────────────────────────────
    logger.info("Loading test data...")
    s1 = pd.read_csv(os.path.join(args.test_dir, "test_source1.tsv"), sep="\t", dtype=str)
    s2 = pd.read_csv(os.path.join(args.test_dir, "test_source2.tsv"), sep="\t", dtype=str)
    s3 = pd.read_csv(os.path.join(args.test_dir, "test_source3.tsv"), sep="\t", dtype=str)
    logger.info(f"  S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}")

    all_s1_ids = s1["entity_id"].tolist()

    # ── Preprocess ────────────────────────────────────────────────────────────
    logger.info("Preprocessing...")
    s1_p = preprocess_df(s1)
    s23 = pd.concat([s2, s3], ignore_index=True)
    s23_p = preprocess_df(s23)

    # ── Blocking ──────────────────────────────────────────────────────────────
    logger.info("Running blocking...")
    t0 = time.time()
    candidates = blocking_pass(s1_p, s23_p, tfidf_top_k=args.tfidf_top_k)
    logger.info(f"  Blocking done in {time.time()-t0:.1f}s")

    # Ensure every S1 entity has a key (even if empty candidate set)
    for s1_id in all_s1_ids:
        if s1_id not in candidates:
            candidates[s1_id] = set()

    total_cands = sum(len(v) for v in candidates.values())
    n_with_cands = sum(1 for v in candidates.values() if v)
    logger.info(f"  Total candidates: {total_cands}, S1 entities with candidates: {n_with_cands}")

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("Building feature matrix...")
    t0 = time.time()
    X, pair_ids = build_feature_matrix(s1_p, s23_p, candidates)
    logger.info(f"  Feature matrix: {X.shape} in {time.time()-t0:.1f}s")

    # ── Inference ─────────────────────────────────────────────────────────────
    logger.info("Running inference...")
    threshold = args.threshold if args.threshold is not None else model.threshold
    matches = predict_matches(model, X, pair_ids, threshold=threshold)

    # Ensure every S1 entity appears in matches (singletons)
    for s1_id in all_s1_ids:
        if s1_id not in matches:
            matches[s1_id] = set()

    n_with_matches = sum(1 for v in matches.values() if v)
    total_matches = sum(len(v) for v in matches.values())
    logger.info(f"  {n_with_matches} S1 entities matched, {len(matches)-n_with_matches} singletons")
    logger.info(f"  Total matched pairs: {total_matches}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    matching_out = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_out = os.path.join(args.output_dir, "candidate_pairs.tsv")

    write_matching_results(matches, all_s1_ids, matching_out)
    write_candidate_pairs(candidates, all_s1_ids, candidate_out)

    # ── Validate ──────────────────────────────────────────────────────────────
    if not args.skip_validate:
        logger.info("Running submission validator...")
        validator = os.path.join("utils", "validate_submission.py")
        if os.path.exists(validator):
            result = subprocess.run(
                [sys.executable, validator,
                 "--matching", matching_out,
                 "--candidate", candidate_out,
                 "--test-dir", args.test_dir],
                capture_output=True, text=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
                logger.warning("Submission validation FAILED — fix issues before uploading!")
            else:
                logger.info("Submission validation PASSED ✓")
        else:
            logger.warning(f"Validator not found at {validator}, skipping.")

    logger.info("Done! Ready to submit output/matching_results.tsv to the leaderboard.")


if __name__ == "__main__":
    main()
