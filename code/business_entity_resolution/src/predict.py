#!/usr/bin/env python3
"""
predict.py — Inference pipeline for Business Entity Resolution (with fine-grained sub-step caching)

Dataset scale: S1=1.7M test | S2+S3=~10M → chunked blocking required

Usage:
    python code/business_entity_resolution/src/predict.py \
        --test-dir "6ab10eb3b23ba_student_resource/student_resource/dataset/test" \
        --model models/matching_model.pkl \
        --output-dir output
"""
import argparse
import gc
import logging
import os
import subprocess
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from gpu_utils import gpu_summary
from cache_utils import StepCache
from preprocess import preprocess_df
from blocking import blocking_pass_chunked
from features import build_feature_matrix
from model import MatchingModel, predict_matches
from output import write_matching_results, write_candidate_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Path to validator (relative to project root)
VALIDATOR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
    "utils", "validate_submission.py"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir",
        default=r"6ab10eb3b23ba_student_resource/student_resource/dataset/test")
    parser.add_argument("--model", default="models/matching_model.pkl")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--chunk-size", type=int, default=200_000,
        help="S1 chunk size for blocking")
    parser.add_argument("--trigram-min", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=None,
        help="Override learned threshold")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--cache-dir", default="cache",
        help="Directory to store intermediate step caches (default: cache)")
    parser.add_argument("--no-cache", action="store_true",
        help="Disable caching and force recomputing all steps")
    parser.add_argument("--force-step", nargs="*", default=[],
        help="Specific sub-steps to force recompute (e.g. preprocess, blocking, features, inference, output)")
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    # ── GPU status ────────────────────────────────────────────────────────────
    logger.info("GPU capabilities:\n" + gpu_summary())

    cache = StepCache(cache_dir=args.cache_dir, enabled=not args.no_cache, tag="test")

    def should_force(step_name: str) -> bool:
        return args.no_cache or (step_name in args.force_step)

    # ── Sub-step: Load model ──────────────────────────────────────────────────
    logger.info(f"Loading model from {args.model}...")
    model = MatchingModel.load(args.model)
    logger.info(f"  Backend: {model.backend} | threshold: {model.threshold:.4f}")

    # ── Sub-step: Preprocessing ───────────────────────────────────────────────
    need_prep = should_force("preprocess") or not (
        cache.exists("test_s1_p") and cache.exists("test_s23_p") and cache.exists("test_s1_ids")
    )
    if not need_prep:
        logger.info("⚡ Loading preprocessed test datasets from cache...")
        s1_p = cache.load("test_s1_p")
        s23_p = cache.load("test_s23_p")
        all_s1_ids = cache.load("test_s1_ids")
        logger.info(f"  Loaded from cache: S1: {len(s1_p):,} | S2+S3: {len(s23_p):,}")
    else:
        logger.info("Loading test data...")
        s1 = pd.read_csv(os.path.join(args.test_dir, "test_source1.tsv"), sep="\t", dtype=str)
        s2 = pd.read_csv(os.path.join(args.test_dir, "test_source2.tsv"), sep="\t", dtype=str)
        s3 = pd.read_csv(os.path.join(args.test_dir, "test_source3.tsv"), sep="\t", dtype=str)
        logger.info(f"  S1: {len(s1):,} | S2: {len(s2):,} | S3: {len(s3):,}")

        all_s1_ids = s1["entity_id"].tolist()

        logger.info("Preprocessing...")
        t0 = time.time()
        s1_p = preprocess_df(s1)
        s23 = pd.concat([s2, s3], ignore_index=True)
        del s1, s2, s3
        gc.collect()

        s23_p = preprocess_df(s23)
        del s23
        gc.collect()
        logger.info(f"  Done in {time.time()-t0:.0f}s | S2+S3: {len(s23_p):,}")

        cache.save("test_s1_p", s1_p)
        cache.save("test_s23_p", s23_p)
        cache.save("test_s1_ids", all_s1_ids)

    # ── Sub-step: Blocking ────────────────────────────────────────────────────
    need_blocking = should_force("blocking") or not cache.exists("test_candidates")
    if not need_blocking:
        candidates = cache.load("test_candidates")
    else:
        logger.info("Running chunked blocking...")
        t0 = time.time()
        candidates = blocking_pass_chunked(
            s1_p, s23_p,
            chunk_size=args.chunk_size,
            trigram_min_shared=args.trigram_min,
            cache=cache,
        )
        logger.info(f"  Blocking done in {time.time()-t0:.0f}s")

        # Ensure every S1 entity is represented
        for s1_id in all_s1_ids:
            if s1_id not in candidates:
                candidates[s1_id] = set()

        cache.save("test_candidates", candidates)

    total_cands = sum(len(v) for v in candidates.values())
    n_with_cands = sum(1 for v in candidates.values() if v)
    logger.info(f"  {total_cands:,} candidate pairs | {n_with_cands:,} S1 entities have candidates")

    # ── Sub-step: Feature engineering ─────────────────────────────────────────
    need_features = should_force("features") or not cache.exists("test_features")
    if not need_features:
        feat_data = cache.load("test_features")
        X, pair_ids = feat_data["X"], feat_data["pair_ids"]
        logger.info(f"  ⚡ Loaded test feature matrix from cache: {X.shape}")
    else:
        logger.info("Building feature matrix for all candidate pairs...")
        t0 = time.time()
        X, pair_ids = build_feature_matrix(s1_p, s23_p, candidates)
        logger.info(f"  Feature matrix: {X.shape} in {time.time()-t0:.0f}s")
        cache.save("test_features", {"X": X, "pair_ids": pair_ids})

    # ── Sub-step: Inference ───────────────────────────────────────────────────
    threshold = args.threshold if args.threshold is not None else model.threshold
    thresh_key = f"test_matches_th_{threshold:.3f}"
    need_infer = should_force("inference") or not cache.exists(thresh_key)

    if not need_infer:
        matches = cache.load(thresh_key)
        logger.info(f"  ⚡ Loaded test matches from cache (threshold={threshold:.4f})")
    else:
        logger.info("Running inference...")
        logger.info(f"  Using threshold: {threshold:.4f}")
        matches = predict_matches(model, X, pair_ids, threshold=threshold)

        # Singletons (no candidates → no matches)
        for s1_id in all_s1_ids:
            if s1_id not in matches:
                matches[s1_id] = set()

        cache.save(thresh_key, matches)

    n_with_matches = sum(1 for v in matches.values() if v)
    total_matches = sum(len(v) for v in matches.values())
    n_singletons = len(matches) - n_with_matches
    logger.info(f"  {n_with_matches:,} S1 entities matched | {n_singletons:,} singletons")
    logger.info(f"  Total matched pairs: {total_matches:,}")

    # ── Sub-step: Write outputs ───────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    matching_out  = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_out = os.path.join(args.output_dir, "candidate_pairs.tsv")

    write_matching_results(matches, all_s1_ids, matching_out)
    write_candidate_pairs(candidates, all_s1_ids, candidate_out)

    # ── Sub-step: Validate ────────────────────────────────────────────────────
    if not args.skip_validate:
        validator = os.path.normpath(VALIDATOR_PATH)
        if not os.path.exists(validator):
            validator = os.path.join("utils", "validate_submission.py")

        if os.path.exists(validator):
            logger.info("Running official submission validator...")
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
                logger.warning("Submission validation FAILED — fix before uploading!")
            else:
                logger.info("Submission validation PASSED ✓")
        else:
            logger.warning("Validator not found — run manually before submitting.")

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\nDone in {elapsed:.1f} min. Upload output/matching_results.tsv to the portal.")


if __name__ == "__main__":
    main()
