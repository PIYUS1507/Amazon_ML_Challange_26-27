#!/usr/bin/env python3
"""
train.py — Full training pipeline for Business Entity Resolution

Usage:
    python code/business_entity_resolution/src/train.py \
        --train-dir dataset/train \
        --model-out models/matching_model.pkl \
        --val-fraction 0.2

This script:
  1. Loads and preprocesses training data
  2. Runs blocking to generate candidate pairs
  3. Engineers features for all candidate pairs
  4. Trains an XGBoost/LightGBM/sklearn classifier
  5. Tunes threshold on validation split for F_0.5
  6. Saves model to disk
  7. Prints validation F_0.5 score
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

# Allow running from repo root
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import preprocess_df
from blocking import blocking_pass, compute_blocking_stats
from features import build_feature_matrix, FEATURE_NAMES
from model import MatchingModel, build_training_data, predict_matches, compute_f05_score
from output import write_matching_results, write_candidate_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--model-out", default="models/matching_model.pkl")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of S1 entities to hold out for validation")
    parser.add_argument("--tfidf-top-k", type=int, default=20,
                        help="TF-IDF top-k candidates per entity")
    parser.add_argument("--neg-ratio", type=int, default=5,
                        help="Negatives per positive in training data")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_ground_truth(path: str):
    gt = pd.read_csv(path, sep="\t", dtype=str)
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    gt_map = {}
    for _, row in gt.iterrows():
        s1 = row["source1_entity_id"]
        matched = str(row["matched_entity_ids"]).strip()
        ids = set(x.strip() for x in matched.split(",") if x.strip()) if matched else set()
        gt_map[s1] = ids
    return gt_map


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading training data...")
    s1 = pd.read_csv(os.path.join(args.train_dir, "train_source1.tsv"), sep="\t", dtype=str)
    s2 = pd.read_csv(os.path.join(args.train_dir, "train_source2.tsv"), sep="\t", dtype=str)
    s3 = pd.read_csv(os.path.join(args.train_dir, "train_source3.tsv"), sep="\t", dtype=str)
    gt_map = load_ground_truth(os.path.join(args.train_dir, "train_ground_truth.tsv"))

    logger.info(f"  S1: {len(s1)} records, S2: {len(s2)} records, S3: {len(s3)} records")
    logger.info(f"  Ground truth: {len(gt_map)} S1 entities")

    # ── Train / Val split ─────────────────────────────────────────────────────
    all_s1_ids = s1["entity_id"].tolist()
    n_val = max(1, int(len(all_s1_ids) * args.val_fraction))
    val_ids_set = set(rng.choice(all_s1_ids, size=n_val, replace=False).tolist())
    train_ids_set = set(all_s1_ids) - val_ids_set

    s1_train = s1[s1["entity_id"].isin(train_ids_set)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_ids_set)].reset_index(drop=True)

    logger.info(f"  Train S1: {len(s1_train)}, Val S1: {len(s1_val)}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    logger.info("Preprocessing...")
    s1_train_p = preprocess_df(s1_train)
    s1_val_p = preprocess_df(s1_val)
    s23 = pd.concat([s2, s3], ignore_index=True)
    s23_p = preprocess_df(s23)

    # ── Blocking ──────────────────────────────────────────────────────────────
    logger.info("Running blocking on training split...")
    t0 = time.time()
    train_candidates = blocking_pass(s1_train_p, s23_p, tfidf_top_k=args.tfidf_top_k)
    logger.info(f"  Train blocking done in {time.time()-t0:.1f}s")

    stats = compute_blocking_stats(train_candidates, pd.read_csv(
        os.path.join(args.train_dir, "train_ground_truth.tsv"), sep="\t", dtype=str
    ).query("source1_entity_id in @train_ids_set"))
    logger.info(f"  Train blocking stats: {stats}")

    logger.info("Running blocking on validation split...")
    val_candidates = blocking_pass(s1_val_p, s23_p, tfidf_top_k=args.tfidf_top_k)
    val_stats = compute_blocking_stats(val_candidates, pd.read_csv(
        os.path.join(args.train_dir, "train_ground_truth.tsv"), sep="\t", dtype=str
    ).query("source1_entity_id in @val_ids_set"))
    logger.info(f"  Val blocking stats: {val_stats}")

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("Building feature matrix for training pairs...")
    t0 = time.time()
    X_train, train_pair_ids = build_feature_matrix(s1_train_p, s23_p, train_candidates)
    logger.info(f"  Training feature matrix: {X_train.shape} in {time.time()-t0:.1f}s")

    logger.info("Building feature matrix for validation pairs...")
    X_val, val_pair_ids = build_feature_matrix(s1_val_p, s23_p, val_candidates)
    logger.info(f"  Val feature matrix: {X_val.shape}")

    # ── Build training labels ─────────────────────────────────────────────────
    X_tr, y_tr = build_training_data(X_train, train_pair_ids, gt_map, neg_ratio=args.neg_ratio, seed=args.seed)
    y_val_labels = np.array([
        1 if s23_id in gt_map.get(s1_id, set()) else 0
        for s1_id, s23_id in val_pair_ids
    ])
    logger.info(f"  Training: {y_tr.sum()} pos / {(y_tr==0).sum()} neg")
    logger.info(f"  Validation: {y_val_labels.sum()} pos / {(y_val_labels==0).sum()} neg")

    # ── Train model ───────────────────────────────────────────────────────────
    logger.info("Training model...")
    t0 = time.time()
    model = MatchingModel()
    model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val_labels)
    logger.info(f"  Training done in {time.time()-t0:.1f}s, backend={model.backend}")
    logger.info(f"  Decision threshold: {model.threshold:.4f}")

    # ── Evaluate on validation set ────────────────────────────────────────────
    logger.info("Evaluating on validation set...")
    val_matches = predict_matches(model, X_val, val_pair_ids)
    # Add singletons (no candidates)
    for s1_id in s1_val["entity_id"]:
        if s1_id not in val_matches:
            val_matches[s1_id] = set()

    val_f05 = compute_f05_score(val_matches, gt_map, s1_val["entity_id"].tolist())
    logger.info(f"\n{'='*50}")
    logger.info(f"  VALIDATION F_0.5 = {val_f05:.4f}")
    logger.info(f"{'='*50}\n")

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save(args.model_out)
    logger.info(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
