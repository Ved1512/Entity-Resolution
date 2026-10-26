# Business Entity Resolution Pipeline

This directory contains the self-contained source code for the **Business Entity Resolution** solution developed by **Team Noir** for the **Amazon ML Challenge 2026**.

## Directory Structure

```
business_entity_resolution/
├── audit_report.md             # Quantitative preprocessing verification report
├── README.md                   # This reproduction document
├── requirements.txt            # Python dependencies (also in src/)
└── src/                        # Full pipeline implementation
    ├── preprocess.py           # Stage 1: Text & address standardization
    ├── audit_preprocessing.py  # Verification tool for preprocessing
    ├── blocking2.py            # Stage 2: Two-phase candidate blocking
    ├── blocking.py             # Streaming GPU candidate blocking
    ├── fast_blocking.py        # Prefix & country hash blocking
    ├── lsh_blocking.py         # LSH candidate retrieval
    ├── feature_engineering.py  # Stage 3: 25+ pairwise similarity features
    ├── fast_train_xgb.py       # Stage 4: XGBoost classifier & F0.5 optimizer
    ├── eval_recall.py          # Recall evaluation tool
    ├── find_fn.py              # False negative analyzer
    ├── analyze_reasons.py      # Failure diagnosis
    ├── fn_analysis.json        # Failure breakdown metrics
    ├── ranking_diagnostic.py   # Prediction rank diagnostics
    └── requirements.txt        # Pinned requirements
```

## Quick Reproduction

From within the `src/` directory:

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Stage 1 — Preprocess Data**:
   ```bash
   python preprocess.py \
       --input-dir "../../../../_student_resource/student_resource/dataset/train" \
       --output-dir "../../../../_student_resource/student_resource/dataset/train/Preprocessed"
   ```

3. **Stage 2 — Candidate Generation (Blocking)**:
   ```bash
   python blocking2.py \
       --s1 "../../../../_student_resource/student_resource/dataset/train/Preprocessed/train_source1_clean.tsv" \
       --s2 "../../../../_student_resource/student_resource/dataset/train/Preprocessed/train_source2_clean.tsv" \
       --s3 "../../../../_student_resource/student_resource/dataset/train/Preprocessed/train_source3_clean.tsv" \
       --output "../../../output/candidate_pairs.tsv" \
       --cache-dir "../../../output/blocking2_cache" \
       --max-k 50
   ```

4. **Stage 3 & 4 — Feature Extraction, Training & Inference**:
   ```bash
   python fast_train_xgb.py \
       --s1 "../../../../_student_resource/student_resource/dataset/train/Preprocessed/train_source1_clean.tsv" \
       --s2 "../../../../_student_resource/student_resource/dataset/train/Preprocessed/train_source2_clean.tsv" \
       --s3 "../../../../_student_resource/student_resource/dataset/train/Preprocessed/train_source3_clean.tsv" \
       --cands "../../../output/candidate_pairs.tsv" \
       --gt "../../../../_student_resource/student_resource/dataset/train/train_ground_truth.tsv" \
       --output "../../../output/matching_results.tsv" \
       --device cuda
   ```

5. **Stage 5 — Validation**:
   ```bash
   python3 ../../../../_student_resource/student_resource/utils/validate_submission.py \
       --matching "../../../output/matching_results.tsv" \
       --candidate "../../../output/candidate_pairs.tsv" \
       --test-dir "../../../../_student_resource/student_resource/dataset/test"
   ```
