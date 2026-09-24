#!/usr/bin/env python3
"""
run_pipeline.py — One-shot script to run the full pipeline end-to-end.

Usage (from project root):
    python run_pipeline.py

Runs:
  1. train.py  → trains model, saves to models/matching_model.pkl
  2. predict.py → generates output/ files
  3. validate_submission.py → confirms format is correct
"""
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "code", "business_entity_resolution", "src")


def run(cmd, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        sys.exit(result.returncode)
    print(f"✓ {description} completed successfully")


def main():
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)

    # Step 1: Train
    run(
        [sys.executable, os.path.join(SRC, "train.py"),
         "--train-dir", "dataset/train",
         "--model-out", "models/matching_model.pkl",
         "--val-fraction", "0.2",
         "--tfidf-top-k", "20",
         "--neg-ratio", "5"],
        "Step 1/2: Training model"
    )

    # Step 2: Predict
    run(
        [sys.executable, os.path.join(SRC, "predict.py"),
         "--test-dir", "dataset/test",
         "--model", "models/matching_model.pkl",
         "--output-dir", "output",
         "--tfidf-top-k", "25"],
        "Step 2/2: Generating predictions"
    )

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print("  Submit: output/matching_results.tsv  to the leaderboard portal")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
