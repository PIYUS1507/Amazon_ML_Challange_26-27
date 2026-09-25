#!/usr/bin/env python3
"""
run_pipeline.py — One-shot script to run the full pipeline end-to-end (with fine-grained caching).

Dataset scale:
  Train  — S1: 2.2M rows | S2: 5.0M rows | S3: 5.3M rows
  Test   — S1: 1.7M rows | S2: 4.9M rows | S3: 5.1M rows

Usage:
    # Full run (with caching — subsequent runs skip completed steps & sub-steps)
    python run_pipeline.py

    # Fast dev run (sample 200K S1, takes ~30–60 min)
    python run_pipeline.py --dev

    # Force recompute specific sub-steps (e.g. blocking or features)
    python run_pipeline.py --force-step blocking features

    # Re-run from scratch (ignoring cache)
    python run_pipeline.py --no-cache

    # Skip training (use existing models/matching_model.pkl)
    python run_pipeline.py --skip-train
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "code", "business_entity_resolution", "src")

# ── Dataset paths ─────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(ROOT, "6ab10eb3b23ba_student_resource", "student_resource", "dataset")
TRAIN_DIR    = os.path.join(DATASET_ROOT, "train")
TEST_DIR     = os.path.join(DATASET_ROOT, "test")
VALIDATOR    = os.path.join(ROOT, "utils", "validate_submission.py")


def run(cmd, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        sys.exit(result.returncode)
    elapsed = time.time() - t0
    print(f"✓ {description} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true",
        help="Fast dev run: sample 200K S1 entities only")
    parser.add_argument("--cache-dir", default="cache",
        help="Directory to store intermediate step caches (default: cache)")
    parser.add_argument("--no-cache", action="store_true",
        help="Disable caching and force recomputing everything from scratch")
    parser.add_argument("--force-step", nargs="*", default=[],
        help="Specific sub-steps to force recompute (e.g. preprocess, blocking, features, training_data, model, evaluate)")
    parser.add_argument("--skip-train", action="store_true",
        help="Skip Step 1 (training) and use existing models/matching_model.pkl")
    args = parser.parse_args()

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, args.cache_dir), exist_ok=True)

    cache_args = ["--cache-dir", os.path.join(ROOT, args.cache_dir)]
    if args.no_cache:
        cache_args.append("--no-cache")
    if args.force_step:
        cache_args += ["--force-step"] + args.force_step

    model_path = os.path.join(ROOT, "models", "matching_model.pkl")

    train_cmd = [
        sys.executable, os.path.join(SRC, "train.py"),
        "--train-dir", TRAIN_DIR,
        "--model-out", model_path,
        "--val-fraction", "0.1",
        "--chunk-size", "200000",
        "--trigram-min", "2",
        "--neg-ratio", "5",
    ] + cache_args

    if args.dev:
        train_cmd += ["--sample-s1", "200000", "--sample-s23", "500000"]
        print("\n⚡ DEV MODE: sampling 200K S1 + 500K S2/S3 for fast iteration")

    predict_cmd = [
        sys.executable, os.path.join(SRC, "predict.py"),
        "--test-dir", TEST_DIR,
        "--model", model_path,
        "--output-dir", os.path.join(ROOT, "output"),
        "--chunk-size", "200000",
        "--trigram-min", "2",
    ] + cache_args

    validate_cmd = [
        sys.executable, VALIDATOR,
        "--matching",  os.path.join(ROOT, "output", "matching_results.tsv"),
        "--candidate", os.path.join(ROOT, "output", "candidate_pairs.tsv"),
        "--test-dir",  TEST_DIR,
    ]

    if args.skip_train:
        print("\n⏩ Skipping Step 1/3 (training) — using existing model")
    else:
        run(train_cmd, "Step 1/3: Training model")

    run(predict_cmd,  "Step 2/3: Generating test predictions")
    run(validate_cmd, "Step 3/3: Validating submission format")

    print(f"\n{'='*60}")
    print("  ✅  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Upload → output/matching_results.tsv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
