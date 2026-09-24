# 🏢 Amazon ML Challenge 2026-27 — Business Entity Resolution

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost%2FLightGBM-orange)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**Matching business records across 3 noisy, independent data sources using ML.**

</div>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Why Is This Hard?](#-why-is-this-hard)
- [Dataset Description](#-dataset-description)
- [Evaluation Metric](#-evaluation-metric)
- [Our Solution](#-our-solution)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Output Format](#-output-format)
- [Constraints & Fair Play](#-constraints--fair-play)

---

## 🎯 Problem Statement

In large-scale commercial platforms, business identity data arrives from **multiple independent sources** — each contributing partial, noisy fragments of information about the same real-world entities. These fragments **share no common identifiers**, and the challenge of determining which records refer to the same business is known as **Entity Resolution (ER)**.

### The Task

Given business records from **3 independent data sources** with noisy and inconsistent fields, determine which records across sources refer to the **same real-world business entity**.

```
Source 1 (reference, deduplicated)
    "McDonald's Corporation, 123 Main St, US"
         ↕  ← Does this match? ↕
Source 2 (noisy)
    "McDonalds Corp., 123 Main Street, United States"

Source 3 (noisy)
    "MC DONALDS, Near City Mall, Main Rd, US"
```

- **Source 1** is the deduplicated reference source
- **Sources 2 & 3** may have zero, one, or many records matching each Source 1 entity
- No shared primary keys — you must infer matches purely from **name + address + country** similarity

---

## 🔥 Why Is This Hard?

The data is intentionally messy. Expect all of these noise patterns:

### Name Variations
| Type | Example |
|------|---------|
| Legal suffix inconsistency | `Corp` vs `Corporation` vs `Corp.` |
| Private/Limited variants | `Pvt. Ltd.` vs `Private Limited` vs `Pvt Ltd` |
| Abbreviations | `Tech` vs `Technologies`, `Intl` vs `International` |
| Punctuation differences | `R&D Solutions` vs `R and D Solutions` |
| Word-order transposition | `Solutions Tech Global` vs `Global Tech Solutions` |
| Typos & transliterations | `Relianc` vs `Reliance`, Hindi transliteration variants |
| DBA / Trade names | Legal entity name vs. trading name |

### Address Variations
| Type | Example |
|------|---------|
| Abbreviations | `Road` vs `Rd`, `Street` vs `St`, `Avenue` vs `Ave` |
| Missing components | No PIN code, no state, no city |
| Landmark-based references | `Near SBI ATM, MG Road` vs `42 MG Road` |
| Component reordering | `Block 4, Sector 12` vs `Sector 12, Block 4` |
| Municipal numbering differences | `42/A` vs `42A` vs `Flat 42` |
| Transliteration variants | Indian city names in different romanizations |

### Scale
- Dataset is **~1 GB** across 3 sources
- Must match every Source 1 entity — including **singletons** (entities with no true match in S2/S3)
- Test set includes **France** — a country not seen during training

---

## 📂 Dataset Description

### File Structure

```
dataset/
├── train/
│   ├── train_source1.tsv       ← Reference source (deduplicated)
│   ├── train_source2.tsv       ← Noisy source 2
│   ├── train_source3.tsv       ← Noisy source 3
│   └── train_ground_truth.tsv  ← True match labels
└── test/
    ├── test_source1.tsv        ← Generate matches for EVERY row here
    ├── test_source2.tsv
    └── test_source3.tsv
```

> ⚠️ All files are **tab-separated** (`.tsv`). Always read with `sep="\t"`:
> ```python
> df = pd.read_csv("train_source1.tsv", sep="\t")
> ```

### Source File Columns

| Column | Description |
|--------|-------------|
| `entity_id` | Unique ID. Prefix tells you the source: `S1-`, `S2-`, `S3-` |
| `business_name` | Business name (noisy — see patterns above) |
| `business_address` | Address (noisy, partial, landmark-based) |
| `country` | Country label — treat as **open-set string** (US, India, France, ...) |

### Ground Truth File Columns

| Column | Description |
|--------|-------------|
| `source1_entity_id` | A Source 1 entity |
| `matched_entity_ids` | Comma-separated list of matching S2/S3 IDs (empty = no match) |

### Example Ground Truth

```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S3-00812
S1-00002	S3-00004
S1-00003	
```

---

## 📊 Evaluation Metric

Submissions are evaluated using **F_β Score with β = 0.5** — a **precision-heavy** metric.

### Formula

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Computed as **macro-average**: F_0.5 per Source 1 entity, then averaged across all entities.

### Why Precision-Heavy?

In real-world entity resolution, **merging two distinct businesses** (false positive) is far more damaging than **missing a link** (false negative). F_0.5 weights **precision 2× over recall**.

### Singleton Scoring

A Source 1 entity with no true matches:
- Scores **1.0** when you correctly predict an empty list ✅
- Scores **0.0** when you predict any match for it ❌

Correctly identifying singletons earns full credit.

### Worked Example

```
Ground truth for S1-00001: [S2-00047, S3-00812]
Your prediction:           [S2-00047, S2-00193, S3-00812]

True Positives  = 2  (S2-00047, S3-00812)
False Positives = 1  (S2-00193)
False Negatives = 0

Precision = 2/3 = 0.667
Recall    = 2/2 = 1.000

F_0.5 = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) = 0.714
```

---

## 🧠 Our Solution

A 5-stage classical ML pipeline — no LLM, no external APIs, fully reproducible.

```
Raw TSVs
   │
   ▼
┌─────────────────────────────────────────────────┐
│  1. PREPROCESSING                                │
│     • Unicode → ASCII normalization              │
│     • Legal suffix normalization                 │
│       (Corp/Corporation → corp, Pvt Ltd, etc.)   │
│     • Address abbreviation normalization         │
│     • Numeric token extraction                   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  2. BLOCKING (Candidate Generation)              │
│     5 complementary strategies per country:      │
│     • Name prefix hash (first 4 chars)           │
│     • Address number hash (building number)      │
│     • Character bigram key                       │
│     • First-token hash                           │
│     • TF-IDF char n-gram cosine (top-25)         │
│                                                  │
│     Goal: recall ceiling ~95%+                   │
│     while reducing pairs from O(N²) to O(N·k)   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  3. FEATURE ENGINEERING (20 features/pair)       │
│     Name: Levenshtein, token sort/set, Jaccard,  │
│           bigrams, LCS, prefix-3/5, length ratio │
│     Addr: Levenshtein, Jaccard, numeric overlap, │
│           prefix-5, LCS                          │
│     Meta: country match, cross name↔address      │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  4. ML CLASSIFICATION                            │
│     • XGBoost (primary) / LightGBM / sklearn     │
│     • Hard negative mining from blocking         │
│     • Class imbalance: scale_pos_weight          │
│     • Threshold tuned to maximize F_0.5          │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  5. OUTPUT + VALIDATION                          │
│     • matching_results.tsv  (leaderboard upload) │
│     • candidate_pairs.tsv   (blocking audit)     │
│     • Auto-run format validator                  │
└─────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Amazon ML Challange/
│
├── 📄 run_pipeline.py                          ← One-shot: train + predict
├── 📄 Documentation_template.md               ← Methodology write-up
├── 📄 .gitignore
│
├── 📂 code/business_entity_resolution/
│   ├── 📄 README.md                           ← Developer guide
│   ├── 📄 requirements.txt
│   └── 📂 src/
│       ├── preprocess.py                      ← Text normalization
│       ├── blocking.py                        ← Candidate generation
│       ├── features.py                        ← 20 similarity features
│       ├── model.py                           ← ML classifier + threshold
│       ├── train.py                           ← Training pipeline
│       └── predict.py                         ← Inference pipeline
│
├── 📂 utils/
│   └── validate_submission.py                 ← Format checker (stdlib only)
│
├── 📂 dataset/                                ← ⚠️ gitignored (1 GB data)
│   ├── train/
│   └── test/
│
├── 📂 models/                                 ← ⚠️ gitignored (saved model)
│   └── matching_model.pkl
│
└── 📂 output/                                 ← ⚠️ gitignored (submission files)
    ├── matching_results.tsv                   ← Upload to leaderboard
    └── candidate_pairs.tsv                    ← Include in zip submission
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Place Dataset

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

### 3. Run Full Pipeline

```bash
python run_pipeline.py
```

This runs training (with 20% validation split + F_0.5 threshold tuning), then inference on the test set, then auto-validates the output format.

### 4. Submit

Upload `output/matching_results.tsv` to the leaderboard portal.

### Manual Steps (Optional)

```bash
# Train only
python code/business_entity_resolution/src/train.py \
    --train-dir dataset/train \
    --model-out models/matching_model.pkl \
    --val-fraction 0.2

# Predict only
python code/business_entity_resolution/src/predict.py \
    --test-dir dataset/test \
    --model models/matching_model.pkl \
    --output-dir output

# Validate format
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 📤 Output Format

### `matching_results.tsv` — upload this to the leaderboard

```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

### `candidate_pairs.tsv` — blocking candidates (for submission package)

```tsv
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```

### Rules (auto-checked by validator):
- ✅ Every Source 1 entity must have **exactly one row**
- ✅ Empty `matched_entity_ids` for singletons (do not omit the row)
- ✅ Only `S2-` or `S3-` IDs from the **test set** in the lists
- ✅ No duplicate IDs within a single list
- ✅ No duplicate `source1_entity_id` rows
- ✅ Final matches must be a **subset** of candidates

---

## ⚖️ Constraints & Fair Play

| Constraint | Status |
|------------|--------|
| No external APIs or data lookups | ✅ Complied |
| No geocoding / business registry lookups | ✅ Complied |
| Model license: MIT or Apache 2.0 | ✅ XGBoost (Apache 2.0) |
| Model size: ≤ 8 billion parameters | ✅ GBDT (~thousands of trees) |
| Country as open-set string (no hard-coding) | ✅ France handled automatically |
| All Source 1 entities in submission | ✅ Enforced by code |

---

## 🏆 Leaderboard

| Phase | Basis |
|-------|-------|
| Public Leaderboard | Subset of test set (real-time feedback) |
| Private Leaderboard | Remaining test set (final rankings) |

Final decision is based on the **private leaderboard**.

---

<div align="center">
<em>Built for Amazon ML Challenge 2026-27</em>
</div>
