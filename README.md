# Entity Resolution: Scalable Multilingual Business Record Linkage

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![GPU-Accelerated](https://img.shields.io/badge/CUDA-cuML%20%7C%20cuPy-green.svg)](https://rapids.ai/)
[![Metric](https://img.shields.io/badge/Evaluation-Macro%20F0.5-brightgreen.svg)]()
[![Challenge](https://img.shields.io/badge/Amazon%20ML%20Challenge-2026-blueviolet.svg)](https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604)

An end-to-end, high-performance machine learning pipeline for **Large-Scale Multilingual Business Entity Resolution**, engineered for the **Amazon ML Challenge 2026**.

The system links fragmented, noisy business identity records across multiple independent data sources (Source 1, Source 2, and Source 3) without common keys or external database lookups. It features an ultra-scalable candidate generation (blocking) engine, multilingual address and legal entity normalization, comprehensive pairwise feature extraction, and an $F_{0.5}$-optimized gradient boosting classifier.

---

## 📌 Table of Contents

- [Problem Overview](#-problem-overview)
- [Dataset Access & Organization](#-dataset-access--organization)
- [System Architecture](#-system-architecture)
  - [Stage 1: Multilingual Preprocessing & Normalization](#stage-1-multilingual-preprocessing--normalization)
  - [Stage 2: Scalable Candidate Generation (Blocking)](#stage-2-scalable-candidate-generation-blocking)
  - [Stage 3: Deep Feature Engineering](#stage-3-deep-feature-engineering)
  - [Stage 4: Classifier & $F_{0.5}$ Threshold Optimization](#stage-4-classifier--f05-threshold-optimization)
  - [Stage 5: Verification & Diagnostics](#stage-5-verification--diagnostics)
- [Evaluation Metric ($F_{0.5}$)](#-evaluation-metric-f_05)
- [Repository Structure](#-repository-structure)
- [Quickstart & Reproduction Guide](#-quickstart--reproduction-guide)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Preprocessing](#2-preprocessing)
  - [3. Candidate Generation (Blocking)](#3-candidate-generation-blocking)
  - [4. Model Training & Inference](#4-model-training--inference)
  - [5. Submission Validation](#5-submission-validation)
- [Failure Analysis & Diagnostics](#-failure-analysis--diagnostics)
- [Compliance & Constraints](#-compliance--constraints)
- [Authors & Acknowledgments](#-authors--acknowledgments)

---

## 🏢 Problem Overview

In modern commercial platforms, business identity data arrives asynchronously from diverse, heterogeneous sources:
- **Source 1 ($S_1$)**: Deduplicated reference entity dataset.
- **Source 2 ($S_2$) & Source 3 ($S_3$)**: Secondary sources containing partial, corrupted, or duplicate fragments.

The goal is to map each entity in $S_1$ to its matching counterparts in $S_2$ and $S_3$. A single $S_1$ entity may match **zero records** (singleton), **one record**, or **multiple records**.

```
Source 1 (Reference)            Source 2 / Source 3 (Noisy)
┌───────────────────────┐       ┌─────────────────────────────────────┐
│ S1-00001:             │       │ S2-00047: "Orelee Barbershop"       │
│ "Orelee's Barbershop" │ ────► │ 1795 Westchester Dr, High Point, NC │
│ 1795 Westchester Dr,  │       ├─────────────────────────────────────┤
│ High Point, NC        │       │ S3-00812: "Orelee Barber Shop Inc"  │
│                       │ ────► │ 1795 Westchester Drive, NC          │
└───────────────────────┘       └─────────────────────────────────────┘
```

### Key Technical Challenges
1. **Extreme Noise & Abbreviations**: Inconsistent legal suffixes (`Pvt Ltd`, `LLC`, `Corp`, `SARL`), informal trade names (DBA), punctuation differences, and spelling typos.
2. **Geographical Variation**: Landmark-based Indian addresses ("Near SBI ATM"), US unit/suite formatting, and French accents.
3. **Zero-Shot Domain Generalization**: The training set covers records from **United States** (`US`) and **India** (`IN`). The test set introduces a third country, **France** (`FR`), which does not appear in training. Solutions must generalize cross-lingually without overfitting to specific country vocabularies.
4. **Computational Scale**: Pairwise matching of millions of records ($O(N_1 \times (N_2 + N_3))$) requires aggressive candidate blocking capable of processing on commodity GPUs/CPUs without memory exhaustion.
5. **Precision-Heavy Metric**: The competition is evaluated using **Macro $F_{0.5}$**, penalizing false merges twice as heavily as false dismissals.

---

## 💾 Dataset Access & Organization

The dataset is provided by Amazon through the Unstop platform:

> 🔗 **Dataset Portal**: [CRP Amazon ML Challenge 2026 on Unstop](https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604)

### Data Schema
All challenge files are tab-separated values (`.tsv`):

| Column | Type | Description |
| :--- | :--- | :--- |
| `entity_id` | String | Unique record identifier (`S1-xxxxx`, `S2-xxxxx`, `S3-xxxxx`) |
| `business_name` | String | Commercial/legal business name (noisy, abbreviations, typos) |
| `business_address` | String | Physical address, street, city, state, postal code |
| `country` | String | Country identifier (`US`, `India` in train; `US`, `India`, `France` in test) |

Ground truth matching file (`train_ground_truth.tsv`):
- `source1_entity_id`: Identifier of the Source 1 entity.
- `matched_entity_ids`: Comma-delimited list of true match IDs from $S_2$ and $S_3$ (empty for singletons).

### Directory Setup
To run the pipeline, download the competition data and extract it under `_student_resource/student_resource/dataset/`:

```
_student_resource/
└── student_resource/
    └── dataset/
        ├── train/
        │   ├── train_source1.tsv
        │   ├── train_source2.tsv
        │   ├── train_source3.tsv
        │   └── train_ground_truth.tsv
        └── test/
            ├── test_source1.tsv
            ├── test_source2.tsv
            └── test_source3.tsv
```

---

## 🏗 System Architecture

The pipeline follows a modular, 5-stage architecture designed for stability, high throughput, and maximum precision:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Raw Data Sources (S1, S2, S3)                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            Stage 1: Multilingual Preprocessing & Normalization              │
│    • Unicode NFKD (French Diacritics)  • Domain & URL Stripping             │
│    • Legal Suffix Standardization      • Address & Thoroughfare Unification │
│    • Regex Postal/PIN Code Extractor   • ISO-3166-1 Country Standardization │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│             Stage 2: Scalable Candidate Generation (Blocking)               │
│    • TF-IDF Cosine Similarity Matrix   • GPU cuML NearestNeighbors / SpGEMM │
│    • CPU Multi-threaded SpGEMM         • Exact (Postal + Street No.) Bucket │
│    • Address-Only Fallback Channel     • Streamed output: candidate_pairs   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Stage 3: Deep Feature Engineering                        │
│    • Levenshtein & RapidFuzz Metrics   • Jaro-Winkler & Token Sort/Set      │
│    • Character & Word Jaccard Overlap  • Street & Postal Match Indicators   │
│    • State/City Geography Match        • Country & Source Indicators        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Stage 4: Classifier & F0.5 Optimization                     │
│    • XGBoost Gradient Boosted Trees    • GPU Histogram Training             │
│    • Entity-Level Group Split          • Automated F0.5 Threshold Sweep     │
│    • Singleton Calibration Gate        • matching_results.tsv & Model Check │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Stage 5: Validation & Diagnostics                       │
│    • validate_submission.py Check      • eval_recall.py (Coverage Check)    │
│    • Root-Cause Failure Diagnostics    • audit_preprocessing.py Verification│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Stage 1: Multilingual Preprocessing & Normalization
**Module**: [`preprocess.py`](file:///home/rover/Amazon%20ML%20challenge%20-%20Copy/Team%20Noir_submission/code/business_entity_resolution/src/preprocess.py)

Raw business records suffer from extensive formatting discrepancies:
- **Unicode NFKD Decomposition**: Decomposes non-ASCII glyphs (e.g., `é` $\to$ `e`, `ç` $\to$ `c`), ensuring clean matching across European and French test sets.
- **Domain Stripping**: Removes web protocols and URL suffixes embedded in business names (e.g., `acme-holdings.com` $\to$ `acme holdings`).
- **Legal Entity Standardization**: Hierarchically maps company types to uniform abbreviations:
  - US/India: `Private Limited` / `Pvt. Ltd.` $\to$ `pvt ltd`, `Incorporated` / `Inc.` $\to$ `inc`, `LLC` $\to$ `llc`.
  - France: `Société à Responsabilité Limitée` / `SARL` $\to$ `sarl`, `SAS`, `EURL`.
- **Address Standardization**: Unifies roadways (`Road` $\to$ `rd`, `Street` $\to$ `st`, `Avenue` $\to$ `ave`), directionals (`North` $\to$ `n`), and units (`Suite`, `Unit`, `Apt`).
- **PIN/Postal Extraction**: Parses 5-digit US ZIPs and 6-digit Indian PIN codes via compiled regular expressions.
- **Audit Verification**: Validated with [`audit_preprocessing.py`](file:///home/rover/Amazon%20ML%20challenge%20-%20Copy/Team%20Noir_submission/code/business_entity_resolution/src/audit_preprocessing.py) and documented in [`audit_report.md`](file:///home/rover/Amazon%20ML%20challenge%20-%20Copy/Team%20Noir_submission/code/business_entity_resolution/audit_report.md), confirming 100% address normalization without data destruction.

---

### Stage 2: Scalable Candidate Generation (Blocking)
**Modules**: [`blocking2.py`](file:///home/rover/Amazon%20ML%20challenge%20-%20Copy/Team%20Noir_submission/code/business_entity_resolution/src/blocking2.py), [`blocking.py`](file:///home/rover/Amazon%20ML%20challenge%20-%20Copy/Team%20Noir_submission/code/business_entity_resolution/src/blocking.py), [`fast_blocking.py`](file:///home/rover/Amazon%20ML%20challenge%20-%20Copy/Team%20Noir_submission/code/business_entity_resolution/src/fast_blocking.py)

To reduce the $O(N_1 \times (N_2 + N_3))$ search space while maintaining $\ge 90\%$ recall:
1. **Multi-Channel Retrieval Architecture**:
   - **TF-IDF Sub-word N-gram Channel**: Fits sub-word character (3–5) and word n-grams across candidate pools. Computes cosine top-$K$ candidates via GPU-native sparse matrix multiplication (`cuML` brute-force cosine `NearestNeighbors` or `cuPy` SpGEMM) or multi-threaded CPU matrix operations (`sparse_dot_topn`).
   - **Exact (Postal + Street Number) Bucket**: Catches cases where business names are completely dissimilar (e.g. trade name vs. legal entity), but street number and postal code match exactly.
   - **Address-Only Fallback**: Targets records where business names are null or empty.
2. **Two-Phase Caching Pipeline**:
   - **Phase A (Offline)**: Builds global vectorizer across samples, partitions candidates by country, and caches sparse CSR matrices (`.npz`) and metadata (`.pkl`).
   - **Phase B (Streaming Inference)**: Loads cached indices and streams top candidates in batches directly into `candidate_pairs.tsv`.

---

### Stage 3: Deep Feature Engineering
**Module**: [`feature_engineering.py`](file:///home/rover/Amazon%20ML%20challenge%20-%20Copy/Team%20Noir_submission/code/business_entity_resolution/src/feature_engineering.py)

Computes 25+ discriminative similarity features per candidate pair:
- **String & Token Metrics**:
  - Levenshtein distance & similarity ratio (via RapidFuzz C++ engine).
  - Jaro-Winkler similarity (rewarding prefix matches).
  - Token Sort Ratio & Token Set Ratio (handling word transpositions).
  - Word & Character Jaccard overlap.
  - Length differences and length ratios.
- **Geographic & Address Signals**:
  - Address token Jaccard similarity.
  - Postal code match flag ($+1.0$ match, $0.0$ mismatch, $-1.0$ missing).
  - Street number exact match indicator.
  - State and city taxonomy match.
- **Relational & Origin Metadata**:
  - Same-country binary indicator (100% invariant to unseen countries like France).
  - TF-IDF blocking cosine similarity score.
  - Source flag (`is_s3`) to account for differing noise distributions between Source 2 and Source 3.

---

### Stage 4: Classifier & $F_{0.5}$ Threshold Optimization
**Module**: [`fast_train_xgb.py`](file:///home/rover/Amazon%20ML%20challenge%20-%20Copy/Team%20Noir_submission/code/business_entity_resolution/src/fast_train_xgb.py)

- **Model**: Gradient Boosted Decision Tree (`xgb.XGBClassifier`) with histogram tree building (`tree_method='hist'`) for GPU/CPU acceleration.
- **Grouped Entity-Level Splitting**: Source 1 entities are partitioned into strict train and validation sets, preventing data leakage across candidate pairs.
- **Automated $F_{0.5}$ Sweeping**: Evaluates probability thresholds from $0.05$ to $0.95$ using the official macro $F_{0.5}$ metric.
- **Singleton Handling**: Low-confidence pairs are aggressively pruned, ensuring reference entities with zero true matches remain empty, yielding a perfect $1.0$ score on those rows.

---

### Stage 5: Verification & Diagnostics
**Modules**: [`validate_submission.py`](file:///home/rover/Amazon%20ML%20challenge%20-%20Copy/_student_resource/student_resource/utils/validate_submission.py), [`eval_recall.py`](file:///home/rover/Amazon%20ML%20challenge%20-%20Copy/Team%20Noir_submission/code/business_entity_resolution/src/eval_recall.py), [`find_fn.py`](file:///home/rover/Amazon%20ML%20challenge%20-%20Copy/Team%20Noir_submission/code/business_entity_resolution/src/find_fn.py), [`analyze_reasons.py`](file:///home/rover/Amazon%20ML%20challenge%20-%20Copy/Team%20Noir_submission/code/business_entity_resolution/src/analyze_reasons.py)

- **Offline Recall Verification**: Streams predictions against ground truth to confirm blocking recall ceiling without loading multi-gigabyte outputs into RAM.
- **False Negative Analysis**: Isolates missed matches and categorizes them into script mismatches, heavy typos, or near misses (`fn_analysis.json`).
- **Submission Validator**: Checks format conformance, non-empty row coverage, absence of self-matches, and valid ID references before submission.

---

## 📐 Evaluation Metric ($F_{0.5}$)

Submissions are evaluated on **Macro-Averaged $F_{0.5}$**:

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Characteristics
- **Precision Weighting**: Precision is weighted $2\times$ higher than Recall. Merging two distinct businesses (false positive) degrades commercial entity graphs far worse than failing to link a business (false negative).
- **Macro Averaging**: $F_{0.5}$ is calculated independently for each Source 1 entity, then averaged across all Source 1 entities in the evaluation split.
- **Singleton Credit**: A Source 1 entity with no true matches receives a score of **$1.0$** if the model predicts an empty list, and **$0.0$** if even a single false match is predicted.

---

## 📂 Repository Structure

```
.
├── .gitignore                                      # Comprehensive exclusions for large data/caches/outputs
├── README.md                                       # Main project documentation & architecture guide
│
├── _student_resource/                              # Official competition resources
│   └── student_resource/
│       ├── README.md                               # Official problem statement & guidelines
│       ├── Documentation_template.md               # Technical methodology template
│       ├── utils/
│       │   └── validate_submission.py              # Zero-dependency submission format validator
│       └── dataset/                                # (Ignored by git) Raw data files
│           ├── train/                              # train_source1/2/3.tsv, train_ground_truth.tsv
│           └── test/                               # test_source1/2/3.tsv
│
└── Team Noir_submission/                           # Competition submission package
    ├── code/
    │   └── business_entity_resolution/
    │       ├── audit_report.md                     # Quantitative preprocessing verification report
    │       ├── README.md                           # Reproducibility run guide
    │       └── src/                                # Core pipeline source code
    │           ├── preprocess.py                   # Stage 1: Multilingual text & address normalization
    │           ├── audit_preprocessing.py          # Preprocessing validation & change audit
    │           ├── blocking2.py                    # Stage 2: Two-phase GPU/CPU TF-IDF blocking
    │           ├── blocking.py                     # Streaming cuML / NearestNeighbors blocking
    │           ├── fast_blocking.py                # Lightweight country/prefix hash blocking
    │           ├── lsh_blocking.py                 # Locality-sensitive hashing candidate generator
    │           ├── feature_engineering.py          # Stage 3: 25+ pairwise similarity features
    │           ├── fast_train_xgb.py               # Stage 4: XGBoost training & F0.5 optimizer
    │           ├── eval_recall.py                  # Streaming recall evaluation against ground truth
    │           ├── find_fn.py                      # False negative extractor for error analysis
    │           ├── analyze_reasons.py              # Root-cause failure analysis
    │           ├── fn_analysis.json                # Quantitative failure breakdown results
    │           ├── ranking_diagnostic.py           # Ranking calibration diagnostic
    │           ├── lang_check.py                   # Script and language distribution diagnostic
    │           ├── detect_language_dist.py         # Multi-language token distribution analysis
    │           ├── extract_foreign.py              # Non-ASCII / international token parser
    │           ├── test_cupy.py                    # CUDA / CuPy environment test
    │           ├── test_cupy_dot.py                # Sparse dot product benchmark
    │           └── requirements.txt                # Pinned production dependencies
    └── output/                                     # (Ignored by git) Generated predictions & cache
        ├── matching_results.tsv                    # Final submitted predictions
        ├── candidate_pairs.tsv                     # Blocking candidate pairs
        └── blocking2_cache/                        # Serialized sparse matrices & vectorizers
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup

Clone the repository and install the dependencies in a virtual environment:

```bash
# Clone the repository
git clone <your-repository-url>
cd "Amazon ML challenge - Copy"

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r "Team Noir_submission/code/business_entity_resolution/src/requirements.txt"
```

> **Hardware Recommendations**:
> - **GPU Mode**: NVIDIA GPU with 6GB+ VRAM (CUDA 11.8+ or 12.x, `cupy`, `cuml`).
> - **CPU Mode**: 8+ CPU cores with 16GB+ RAM (accelerated using `sparse_dot_topn` multithreading).

---

### 2. Preprocessing

Standardize raw text, clean addresses, normalize countries, and extract postal codes:

```bash
cd "Team Noir_submission/code/business_entity_resolution/src"

# Preprocess training data
python preprocess.py \
    --input-dir "../../../../_student_resource/student_resource/dataset/train" \
    --output-dir "../../../../_student_resource/student_resource/dataset/train/Preprocessed"

# Preprocess test data
python preprocess.py \
    --input-dir "../../../../_student_resource/student_resource/dataset/test" \
    --output-dir "../../../../_student_resource/student_resource/dataset/test/Preprocessed"
```

Verify preprocessing data integrity:
```bash
python audit_preprocessing.py \
    --raw "../../../../_student_resource/student_resource/dataset/train/train_source1.tsv" \
    --clean "../../../../_student_resource/student_resource/dataset/train/Preprocessed/train_source1_clean.tsv" \
    --sample 5000
```

---

### 3. Candidate Generation (Blocking)

Generate high-recall candidate pairs from $S_2$ and $S_3$ for each $S_1$ entity:

```bash
python blocking2.py \
    --s1 "../../../../_student_resource/student_resource/dataset/train/Preprocessed/train_source1_clean.tsv" \
    --s2 "../../../../_student_resource/student_resource/dataset/train/Preprocessed/train_source2_clean.tsv" \
    --s3 "../../../../_student_resource/student_resource/dataset/train/Preprocessed/train_source3_clean.tsv" \
    --output "../../../output/candidate_pairs.tsv" \
    --cache-dir "../../../output/blocking2_cache" \
    --max-k 50
```

Evaluate blocking recall against ground truth:
```bash
python eval_recall.py
```

---

### 4. Model Training & Inference

Extract features, train the XGBoost classifier, optimize the decision threshold for macro $F_{0.5}$, and generate predictions:

```bash
python fast_train_xgb.py \
    --s1 "../../../../_student_resource/student_resource/dataset/train/Preprocessed/train_source1_clean.tsv" \
    --s2 "../../../../_student_resource/student_resource/dataset/train/Preprocessed/train_source2_clean.tsv" \
    --s3 "../../../../_student_resource/student_resource/dataset/train/Preprocessed/train_source3_clean.tsv" \
    --cands "../../../output/candidate_pairs.tsv" \
    --gt "../../../../_student_resource/student_resource/dataset/train/train_ground_truth.tsv" \
    --output "../../../output/matching_results.tsv" \
    --device cuda   # or cpu
```

---

### 5. Submission Validation

Before uploading to the leaderboard or packaging for final submission, validate that both output files satisfy all competition rules:

```bash
cd "../../../../_student_resource/student_resource"

python3 utils/validate_submission.py \
    --matching "../../Team Noir_submission/output/matching_results.tsv" \
    --candidate "../../Team Noir_submission/output/candidate_pairs.tsv" \
    --test-dir "dataset/test"
```

Expected output:
```
PASS
All formatting and consistency checks passed successfully.
```

---

## 📊 Failure Analysis & Diagnostics

Through offline error audits ([`fn_analysis.json`](file:///home/rover/Amazon%20ML%20challenge%20-%20Copy/Team%20Noir_submission/code/business_entity_resolution/src/fn_analysis.json)), false negatives in candidate generation were categorized as follows:

| Failure Category | % of False Negatives | Root Cause | Mitigation in Pipeline |
| :--- | :---: | :--- | :--- |
| **Near Misses** | 40.18% | Boundary rank cutoffs ($K > 50$) | Dual-channel inverted index + threshold relaxation |
| **Language / Script Mismatch** | 35.48% | Non-Latin transliterations vs Latin script | NFKD decomposition + cross-lingual character n-grams |
| **Heavy Typos** | 21.11% | Extreme edit distance in business names | Street-number + postal exact fallback bucket |
| **Completely Different** | 3.23% | Pure trade name (DBA) differences | Address token Jaccard & spatial matching |
| **Missing Data** | 0.00% | Completely empty name and address | Addressed via graceful null imputations |

---

## ⚖️ Compliance & Constraints

- **Zero External Lookups**: In strict adherence to competition rules, no commercial APIs, external databases, geocoding endpoints, or web-scraping augmentations were utilized.
- **Model Size Constraints**: All models are under 8 Billion parameters and licensed under permissive open-source licenses (Apache 2.0 / MIT).
- **Zero-Shot Transferability**: Country features are modeled relationally (`same_country`), ensuring generalization to France (`FR`) without country-specific bias.

---

## 👥 Authors & Acknowledgments

Developed by **Team Noir**:
- **Ved Thakkar**
- **Jival Dhingra**

Organized by **Amazon** in partnership with **Unstop** for the **Amazon ML Challenge 2026**.
