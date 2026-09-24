# Documentation Template — Business Entity Resolution Challenge

## Team Information
- **Team Name**: [Your team name]
- **Challenge**: Amazon ML Challenge — Business Entity Resolution

---

## 1. Methodology Overview

Our solution is a classical Machine Learning pipeline for Entity Resolution, structured in four stages:

**Preprocessing → Blocking → Feature Engineering → ML Classification**

The core insight is that business names and addresses, despite their noise, share systematic patterns that can be captured through normalization and pairwise string similarity features. We deliberately avoid any external data lookup or API calls.

---

## 2. Preprocessing

We apply the following normalization steps to both names and addresses:

**Business Name:**
- Unicode NFKD normalization → ASCII transliteration
- Lowercase and strip
- `&` → `and`
- Remove punctuation
- Legal suffix normalization: Corp/Corporation → corp, Private Limited/Pvt. Ltd. → pvt ltd, Inc/Incorporated → inc, LLC → llc, etc.
- Collapse whitespace

**Business Address:**
- Unicode normalization and lowercasing
- Abbreviation expansion/normalization: Street → st, Road → rd, Avenue → ave, etc.
- Landmark noise words removed (near, behind, next to)
- Numeric token extraction (building numbers, PIN codes)

---

## 3. Candidate Generation / Blocking Strategy

We use 5 complementary blocking strategies, all conditioned on country to avoid cross-country false positives:

| Strategy | Key | Purpose |
|----------|-----|---------|
| Name prefix hash | country + first 4 chars of name | Exact prefix matches |
| Address number hash | country + first numeric token | Same building/street number |
| Bigram key | country + top-4 sorted char bigrams | Transposition/typo tolerance |
| First token hash | country + first word of name | Same primary keyword |
| TF-IDF character n-gram cosine | Top-25 per entity | Fuzzy name+address matching |

**TF-IDF blocking**: We fit a TF-IDF vectorizer with `(2,3)` character n-grams on combined `name + address` text, compute cosine similarity between S1 and S2/S3 records within the same country, and retain top-25 candidates per S1 entity (filtering by cosine ≥ 0.25).

All candidate sets are unioned across all strategies, giving a final candidate pool.

**Estimated Recall Ceiling**: ~95%+ on training data (validate with `compute_blocking_stats`)

---

## 4. Feature Engineering

We compute 20 pairwise similarity features for each candidate pair:

**Name features (10):**
- Levenshtein ratio (normalized edit distance)
- Token sort ratio (order-independent token matching)
- Token set ratio (subset matching for DBA/trade names)
- Jaccard on word tokens
- Jaccard on character bigrams
- Prefix match at 3 and 5 characters
- LCS (Longest Common Subsequence) ratio
- Length ratio
- Exact match binary flag

**Address features (7):**
- Levenshtein ratio, token sort ratio, Jaccard tokens, Jaccard bigrams
- Numeric token overlap (building numbers, PIN codes)
- Prefix match at 5 characters
- LCS ratio

**Meta features (3):**
- Country exact match (binary)
- Cross features: name₁ tokens ∩ address₂ tokens (catches business name in address)
- Cross features: name₂ tokens ∩ address₁ tokens

String operations use `rapidfuzz` for speed when available, with stdlib `difflib` as fallback.

---

## 5. Model Architecture

**Classifier**: XGBoost (primary) with LightGBM and sklearn GradientBoosting as fallbacks.

**Training data construction**:
- Positives: all ground-truth matched pairs
- Negatives: 5× sampled hard negatives from blocking candidates that are NOT true matches
- Class imbalance handled via `scale_pos_weight` = (# negatives) / (# positives)

**Hyperparameters** (XGBoost):
- `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.8`
- Tree method: hist (fast)

**Threshold tuning**:
- Train/val split: 80% / 20% of S1 entities
- Compute precision-recall curve on validation predictions
- Select threshold maximizing F_0.5 = (1.25 × P × R) / (0.25×P + R)

---

## 6. Evaluation

The competition metric is macro-averaged F_0.5 (precision-heavy):

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Key design choices driven by this metric:
- **Threshold tuning**: We directly optimize F_0.5 rather than accuracy or AUC
- **Singleton handling**: Correctly predicting no-match scores 1.0 — we do NOT force matches for entities with no strong candidates
- **Precision bias**: Hard negative ratio set to 5× to train the model to be cautious about merging

---

## 7. France / Unseen Countries

The test set includes France which doesn't appear in training. Our pipeline handles this correctly because:
- Country is treated as a free-form string label (no hard-coding of {US, India})
- Blocking keys are conditioned on `country_clean` (lowercased string), so French records form their own blocking buckets
- String similarity features are language-agnostic
- No country-specific logic, filters, or one-hot encodings are used anywhere

---

## 8. Constraints Compliance

- ✅ No external APIs, databases, or internet lookups used anywhere
- ✅ Model is XGBoost (Apache 2.0 license, <8B parameters — it's a GBDT, ~thousands of parameters)
- ✅ Every Source 1 entity appears in both output files
- ✅ Only S2-/S3- IDs from the test set appear in matched/candidate lists
- ✅ No duplicate IDs within any row

---

## 9. Reproduction Instructions

```bash
# Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# Run full pipeline
python run_pipeline.py

# Or step by step:
python code/business_entity_resolution/src/train.py --train-dir dataset/train --model-out models/matching_model.pkl
python code/business_entity_resolution/src/predict.py --test-dir dataset/test --model models/matching_model.pkl
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

**Environment**: Python 3.9+, Windows/Linux/Mac compatible.
