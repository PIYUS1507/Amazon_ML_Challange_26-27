#!/usr/bin/env python3
"""
Step 4: ML Matching Model
Trains a gradient-boosted classifier (XGBoost/LightGBM) on pairwise features
and predicts match probability for candidate pairs.

Training strategy:
  - Positive examples: ground-truth matched pairs
  - Negative examples: sampled non-matching candidate pairs (hard negatives from blocking)
  - Class imbalance handled via scale_pos_weight and threshold tuning for F_0.5

Threshold tuning:
  - Uses per-entity greedy threshold selection to maximize F_0.5
"""
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve

logger = logging.getLogger(__name__)


def _f_beta(precision, recall, beta=0.5):
    """F-beta score (beta=0.5 is precision-heavy)."""
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def best_threshold_f05(y_true, y_scores):
    """Find threshold that maximizes global F_0.5."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f05 = [_f_beta(p, r, 0.5) for p, r in zip(precision, recall)]
    best_idx = int(np.argmax(f05))
    best_t = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_t, f05[best_idx]


# ── Try to import XGBoost, fall back to LightGBM, fall back to sklearn RF ────
try:
    import xgboost as xgb

    class MatchingModel:
        def __init__(self, n_estimators=500, max_depth=6, lr=0.05):
            self.n_estimators = n_estimators
            self.model = None
            self.threshold = 0.5
            self.backend = "xgboost"

        def fit(self, X, y, X_val=None, y_val=None):
            ratio = (y == 0).sum() / max((y == 1).sum(), 1)
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=ratio,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            )
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.model.fit(
                X, y,
                eval_set=eval_set,
                verbose=False,
            )
            if X_val is not None:
                scores = self.model.predict_proba(X_val)[:, 1]
                self.threshold, f05 = best_threshold_f05(y_val, scores)
                logger.info(f"Tuned threshold={self.threshold:.3f}, Val F_0.5={f05:.4f}")

        def predict_proba(self, X):
            return self.model.predict_proba(X)[:, 1]

        def save(self, path):
            with open(path, "wb") as f:
                pickle.dump({"model": self.model, "threshold": self.threshold, "backend": self.backend}, f)

        @classmethod
        def load(cls, path):
            obj = cls()
            with open(path, "rb") as f:
                d = pickle.load(f)
            obj.model = d["model"]
            obj.threshold = d["threshold"]
            return obj

    logger.info("Using XGBoost backend")

except ImportError:
    try:
        import lightgbm as lgb

        class MatchingModel:
            def __init__(self, n_estimators=500, max_depth=6, lr=0.05):
                self.n_estimators = n_estimators
                self.model = None
                self.threshold = 0.5
                self.backend = "lightgbm"

            def fit(self, X, y, X_val=None, y_val=None):
                ratio = (y == 0).sum() / max((y == 1).sum(), 1)
                self.model = lgb.LGBMClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    is_unbalance=True,
                    n_jobs=-1,
                    random_state=42,
                    verbose=-1,
                )
                callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
                eval_set = [(X_val, y_val)] if X_val is not None else None
                self.model.fit(X, y, eval_set=eval_set, callbacks=callbacks if eval_set else [])
                if X_val is not None:
                    scores = self.model.predict_proba(X_val)[:, 1]
                    self.threshold, f05 = best_threshold_f05(y_val, scores)
                    logger.info(f"Tuned threshold={self.threshold:.3f}, Val F_0.5={f05:.4f}")

            def predict_proba(self, X):
                return self.model.predict_proba(X)[:, 1]

            def save(self, path):
                with open(path, "wb") as f:
                    pickle.dump({"model": self.model, "threshold": self.threshold, "backend": self.backend}, f)

            @classmethod
            def load(cls, path):
                obj = cls()
                with open(path, "rb") as f:
                    d = pickle.load(f)
                obj.model = d["model"]
                obj.threshold = d["threshold"]
                return obj

        logger.info("Using LightGBM backend")

    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier

        class MatchingModel:
            def __init__(self, n_estimators=200, max_depth=4, lr=0.1):
                self.n_estimators = n_estimators
                self.model = None
                self.threshold = 0.5
                self.backend = "sklearn_gb"

            def fit(self, X, y, X_val=None, y_val=None):
                from sklearn.utils.class_weight import compute_sample_weight
                weights = compute_sample_weight("balanced", y)
                self.model = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42,
                )
                self.model.fit(X, y, sample_weight=weights)
                if X_val is not None:
                    scores = self.model.predict_proba(X_val)[:, 1]
                    self.threshold, f05 = best_threshold_f05(y_val, scores)
                    logger.info(f"Tuned threshold={self.threshold:.3f}, Val F_0.5={f05:.4f}")

            def predict_proba(self, X):
                return self.model.predict_proba(X)[:, 1]

            def save(self, path):
                with open(path, "wb") as f:
                    pickle.dump({"model": self.model, "threshold": self.threshold, "backend": self.backend}, f)

            @classmethod
            def load(cls, path):
                obj = cls()
                with open(path, "rb") as f:
                    d = pickle.load(f)
                obj.model = d["model"]
                obj.threshold = d["threshold"]
                return obj

        logger.info("Using sklearn GradientBoosting backend (slow — install xgboost or lightgbm)")


def build_training_data(
    X_all: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    gt_map: Dict[str, set],
    neg_ratio: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create (X_train, y_train) from all candidate pair features.
    Positives: pairs in ground truth.
    Negatives: sampled from candidate pairs NOT in ground truth (hard negatives).
    neg_ratio: how many negatives per positive to sample.
    """
    rng = np.random.default_rng(seed)

    pos_idx, neg_idx = [], []
    for i, (s1_id, s23_id) in enumerate(pair_ids):
        true_matches = gt_map.get(s1_id, set())
        if s23_id in true_matches:
            pos_idx.append(i)
        else:
            neg_idx.append(i)

    logger.info(f"Training data: {len(pos_idx)} positives, {len(neg_idx)} negatives (raw)")

    # Sample negatives
    n_neg = min(len(neg_idx), neg_ratio * len(pos_idx))
    sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False).tolist()

    all_idx = pos_idx + sampled_neg
    X = X_all[all_idx]
    y = np.array([1] * len(pos_idx) + [0] * len(sampled_neg))

    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def predict_matches(
    model: "MatchingModel",
    X_all: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    threshold: Optional[float] = None,
) -> Dict[str, set]:
    """
    Run inference on all candidate pairs and return matches dict.
    """
    if threshold is None:
        threshold = model.threshold

    if len(X_all) == 0:
        # No candidates — all singletons
        return {s1: set() for s1, _ in pair_ids}

    scores = model.predict_proba(X_all)
    matches: Dict[str, set] = {}

    for (s1_id, s23_id), score in zip(pair_ids, scores):
        if s1_id not in matches:
            matches[s1_id] = set()
        if score >= threshold:
            matches[s1_id].add(s23_id)

    return matches


def compute_f05_score(
    predictions: Dict[str, set],
    gt_map: Dict[str, set],
    all_s1_ids: List[str],
) -> float:
    """
    Compute macro-averaged F_0.5 over all S1 entities.
    Singletons (no ground truth matches) score 1.0 iff prediction is empty.
    """
    f05_scores = []
    for s1_id in all_s1_ids:
        true_set = gt_map.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        if not true_set and not pred_set:
            f05_scores.append(1.0)
        elif not true_set and pred_set:
            f05_scores.append(0.0)
        elif true_set and not pred_set:
            f05_scores.append(0.0)
        else:
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f05_scores.append(_f_beta(prec, rec, 0.5))

    return float(np.mean(f05_scores))
