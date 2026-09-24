# Business Entity Resolution — Solution

## Quick Start

```bash
# 1. Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# 2. Run the full pipeline (train + predict + validate)
python run_pipeline.py
```

Output files will be in `output/`:
- `matching_results.tsv` → **upload this to the leaderboard**
- `candidate_pairs.tsv` → required for final submission package

---

## Running Steps Individually

### Train only
```bash
python code/business_entity_resolution/src/train.py \
    --train-dir dataset/train \
    --model-out models/matching_model.pkl \
    --val-fraction 0.2
```

### Predict only (requires trained model)
```bash
python code/business_entity_resolution/src/predict.py \
    --test-dir dataset/test \
    --model models/matching_model.pkl \
    --output-dir output
```

### Validate submission format
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## Project Structure

```
Amazon ML Challange/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── preprocess.py   # Text normalization
│   │   ├── blocking.py     # Candidate generation
│   │   ├── features.py     # Pairwise similarity features
│   │   ├── model.py        # XGBoost/LightGBM classifier
│   │   ├── train.py        # Training pipeline
│   │   └── predict.py      # Inference pipeline
│   ├── README.md
│   └── requirements.txt
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── models/
│   └── matching_model.pkl
├── utils/
│   └── validate_submission.py
└── run_pipeline.py          # One-shot end-to-end runner
```

---

## Pipeline Architecture

### 1. Preprocessing (`preprocess.py`)
- Unicode normalization (NFKD → ASCII)
- Business name normalization: legal suffixes (Corp/Corporation → corp, Pvt Ltd, Inc, LLC...), punctuation, `&` → `and`
- Address normalization: road/street abbreviations, landmark noise removal
- Token extraction: word tokens, numeric tokens, name prefix

### 2. Blocking / Candidate Generation (`blocking.py`)
Four complementary blocking strategies (run per-country to reduce cross-country false positives):
1. **Name prefix hash** — first 4 chars of normalized name
2. **Address number hash** — first numeric token (building/street number)
3. **Sorted bigram key** — top-4 sorted character bigrams
4. **First token hash** — first word of normalized name
5. **TF-IDF character n-gram cosine** — top-25 most similar records per entity (sklearn TfidfVectorizer with `(2,3)` char_wb ngrams)

### 3. Feature Engineering (`features.py`)
20 pairwise similarity features computed for each candidate pair:

| Category | Features |
|----------|----------|
| Name | Levenshtein ratio, token sort ratio, token set ratio, Jaccard tokens, Jaccard bigrams, prefix-3, prefix-5, LCS ratio, length ratio, exact match |
| Address | Levenshtein ratio, token sort ratio, Jaccard tokens, Jaccard bigrams, numeric overlap, prefix-5, LCS ratio |
| Cross | Country match, name₁∩address₂, name₂∩address₁ |

Uses `rapidfuzz` for fast string operations when available.

### 4. ML Matching Model (`model.py`)
- **Classifier**: XGBoost (primary) → LightGBM (fallback) → sklearn GradientBoosting (last resort)
- **Class imbalance**: `scale_pos_weight` set to negative/positive ratio
- **Hard negatives**: 5× negative sampling from candidates that blocking surfaced but don't match
- **Threshold tuning**: precision-recall curve → maximize F_0.5 on held-out validation set

### 5. Evaluation
- Macro-averaged F_0.5 per S1 entity (matches competition metric exactly)
- Singletons score 1.0 for correct empty predictions

---

## Tuning Tips

| Hyperparameter | Effect |
|---|---|
| `--tfidf-top-k` | Recall ceiling ↑ but speed ↓ (try 25-50) |
| `--neg-ratio` | Model precision ↑ with higher ratio |
| `--val-fraction` | More data for threshold tuning |
| `--threshold` (predict) | Override threshold manually to trade precision/recall |
