#!/usr/bin/env python3
"""
train.py — Full training pipeline for Business Entity Resolution

Dataset scale: S1=2.2M | S2=5.0M | S3=5.3M | Total pairs ~22T (before blocking)

Usage:
    python code/business_entity_resolution/src/train.py \
        --train-dir "6ab10eb3b23ba_student_resource/student_resource/dataset/train" \
        --model-out models/matching_model.pkl \
        --val-fraction 0.1 \
        --sample-s1 200000

Steps:
  1. Load & preprocess training data
  2. Optional: sample a subset of S1 for fast iteration
  3. Chunked blocking → candidate pairs
  4. Feature engineering on candidate pairs
  5. Train XGBoost/LightGBM classifier
  6. Tune threshold for F_0.5 on validation split
  7. Save model
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from preprocess import preprocess_df
from blocking import blocking_pass_chunked, compute_blocking_stats
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
    parser.add_argument("--train-dir",
        default=r"6ab10eb3b23ba_student_resource/student_resource/dataset/train")
    parser.add_argument("--model-out", default="models/matching_model.pkl")
    parser.add_argument("--val-fraction", type=float, default=0.1,
        help="Fraction of S1 for validation (default 10% = ~220K entities)")
    parser.add_argument("--sample-s1", type=int, default=0,
        help="If >0, randomly sample this many S1 entities for fast dev runs. 0=use all.")
    parser.add_argument("--sample-s23", type=int, default=0,
        help="If >0, sample this many S2+S3 rows for fast dev runs (e.g. 500000). 0=use all.")
    parser.add_argument("--chunk-size", type=int, default=200_000,
        help="S1 chunk size for blocking (tune based on RAM)")
    parser.add_argument("--trigram-min", type=int, default=2,
        help="Min shared trigrams to form a candidate pair")
    parser.add_argument("--neg-ratio", type=int, default=5,
        help="Hard negatives per positive in training data")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_ground_truth(path: str) -> dict:
    logger.info(f"Loading ground truth from {path}...")
    gt = pd.read_csv(path, sep="\t", dtype=str)
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    gt_map = {}
    for _, row in gt.iterrows():
        s1 = row["source1_entity_id"]
        matched = str(row["matched_entity_ids"]).strip()
        ids = set(x.strip() for x in matched.split(",") if x.strip()) if matched else set()
        gt_map[s1] = ids
    n_with = sum(1 for v in gt_map.values() if v)
    logger.info(f"  Ground truth: {len(gt_map):,} S1 | {n_with:,} with matches | {len(gt_map)-n_with:,} singletons")
    return gt_map


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading training source files...")
    s1 = pd.read_csv(os.path.join(args.train_dir, "train_source1.tsv"), sep="\t", dtype=str)
    s2 = pd.read_csv(os.path.join(args.train_dir, "train_source2.tsv"), sep="\t", dtype=str)
    s3 = pd.read_csv(os.path.join(args.train_dir, "train_source3.tsv"), sep="\t", dtype=str)
    gt_map = load_ground_truth(os.path.join(args.train_dir, "train_ground_truth.tsv"))

    logger.info(f"  S1: {len(s1):,} | S2: {len(s2):,} | S3: {len(s3):,}")

    # ── Optional sampling for fast dev runs ───────────────────────────────────
    if args.sample_s1 > 0 and args.sample_s1 < len(s1):
        logger.info(f"Sampling {args.sample_s1:,} S1 entities for dev run...")
        sample_ids = rng.choice(s1["entity_id"].values, size=args.sample_s1, replace=False)
        s1 = s1[s1["entity_id"].isin(sample_ids)].reset_index(drop=True)
        logger.info(f"  Sampled S1: {len(s1):,}")

    if args.sample_s23 > 0:
        total_s23 = len(s2) + len(s3)
        if args.sample_s23 < total_s23:
            logger.info(f"Sampling {args.sample_s23:,} S2/S3 rows for dev run (from {total_s23:,})...")
            n2 = int(args.sample_s23 * len(s2) / total_s23)
            n3 = args.sample_s23 - n2
            s2 = s2.sample(n=min(n2, len(s2)), random_state=args.seed).reset_index(drop=True)
            s3 = s3.sample(n=min(n3, len(s3)), random_state=args.seed).reset_index(drop=True)
            logger.info(f"  Sampled S2: {len(s2):,} | S3: {len(s3):,}")

    all_s1_ids = s1["entity_id"].tolist()

    # ── Train / Val split ─────────────────────────────────────────────────────
    n_val = max(1000, int(len(all_s1_ids) * args.val_fraction))
    val_ids = set(rng.choice(all_s1_ids, size=n_val, replace=False).tolist())
    train_ids = set(all_s1_ids) - val_ids

    s1_train = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val   = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)
    logger.info(f"  Train S1: {len(s1_train):,} | Val S1: {len(s1_val):,}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    logger.info("Preprocessing S1, S2, S3...")
    t0 = time.time()
    s1_train_p = preprocess_df(s1_train)
    s1_val_p   = preprocess_df(s1_val)
    s23 = pd.concat([s2, s3], ignore_index=True)
    s23_p = preprocess_df(s23)
    logger.info(f"  Preprocessing done in {time.time()-t0:.0f}s | S2+S3: {len(s23_p):,}")

    # ── Blocking ──────────────────────────────────────────────────────────────
    logger.info("Blocking: training split...")
    t0 = time.time()
    train_candidates = blocking_pass_chunked(
        s1_train_p, s23_p,
        chunk_size=args.chunk_size,
        trigram_min_shared=args.trigram_min,
    )
    logger.info(f"  Train blocking: {time.time()-t0:.0f}s")

    # Compute recall ceiling on a quick sample of validation GT
    gt_df_train = pd.DataFrame([
        {"source1_entity_id": k, "matched_entity_ids": ",".join(v)}
        for k, v in gt_map.items() if k in train_ids
    ])
    if not gt_df_train.empty:
        stats = compute_blocking_stats(train_candidates, gt_df_train)
        logger.info(f"  Train blocking stats: {stats}")

    logger.info("Blocking: validation split...")
    t0 = time.time()
    val_candidates = blocking_pass_chunked(
        s1_val_p, s23_p,
        chunk_size=min(args.chunk_size, 50_000),
        trigram_min_shared=args.trigram_min,
    )
    logger.info(f"  Val blocking: {time.time()-t0:.0f}s")

    gt_df_val = pd.DataFrame([
        {"source1_entity_id": k, "matched_entity_ids": ",".join(v)}
        for k, v in gt_map.items() if k in val_ids
    ])
    if not gt_df_val.empty:
        val_stats = compute_blocking_stats(val_candidates, gt_df_val)
        logger.info(f"  Val blocking stats: {val_stats}")

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("Building feature matrix for training pairs...")
    t0 = time.time()
    X_train, train_pair_ids = build_feature_matrix(s1_train_p, s23_p, train_candidates)
    logger.info(f"  Train features: {X_train.shape} in {time.time()-t0:.0f}s")

    logger.info("Building feature matrix for validation pairs...")
    X_val, val_pair_ids = build_feature_matrix(s1_val_p, s23_p, val_candidates)
    logger.info(f"  Val features: {X_val.shape}")

    # ── Build labels ──────────────────────────────────────────────────────────
    X_tr, y_tr = build_training_data(
        X_train, train_pair_ids, gt_map,
        neg_ratio=args.neg_ratio, seed=args.seed
    )
    y_val_labels = np.array([
        1 if s23_id in gt_map.get(s1_id, set()) else 0
        for s1_id, s23_id in val_pair_ids
    ])
    logger.info(f"  Train: {y_tr.sum():,} pos / {(y_tr==0).sum():,} neg")
    logger.info(f"  Val:   {y_val_labels.sum():,} pos / {(y_val_labels==0).sum():,} neg")

    # ── Train model ───────────────────────────────────────────────────────────
    logger.info("Training model...")
    t0 = time.time()
    model = MatchingModel()
    model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val_labels)
    logger.info(f"  Model trained in {time.time()-t0:.0f}s | backend={model.backend} | threshold={model.threshold:.4f}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info("Evaluating on validation set...")
    val_matches = predict_matches(model, X_val, val_pair_ids)
    for s1_id in s1_val["entity_id"]:
        if s1_id not in val_matches:
            val_matches[s1_id] = set()

    val_f05 = compute_f05_score(val_matches, gt_map, s1_val["entity_id"].tolist())

    logger.info(f"\n{'='*55}")
    logger.info(f"  VALIDATION F_0.5 = {val_f05:.4f}")
    logger.info(f"  Total wall time  = {(time.time()-t_start)/60:.1f} min")
    logger.info(f"{'='*55}\n")

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    model.save(args.model_out)
    logger.info(f"Model saved → {args.model_out}")


if __name__ == "__main__":
    main()
