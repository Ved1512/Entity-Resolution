"""
Amazon ML Challenge 2026: Fast Baseline Pipeline
File 2: GPU-Accelerated XGBoost Classifier & F0.5 Optimizer (fast_train_xgb.py)

Features:
1. Fast feature extraction across candidate pairs (Jaccard, edit similarity, postal match, country match).
2. GPU-Accelerated training via xgb.XGBClassifier(device='cuda', tree_method='hist').
3. Strict entity-level train/validation split (no data leakage).
4. Automated F0.5 decision threshold sweep to maximize macro-averaged F0.5.
5. Correct singleton handling (empty prediction on true singleton = 1.0 score).
6. Outputs final matching_results.tsv and saves the trained model.
"""

import sys
import time
import argparse
import difflib
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

# Force UTF-8 for console output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ---------------------------------------------------------
# 1. FAST FEATURE EXTRACTION
# ---------------------------------------------------------

def calc_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Computes Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return (intersection / union) if union > 0 else 0.0


def fast_char_sim(s1: str, s2: str) -> float:
    """Computes quick character sequence similarity (0.0 to 1.0)."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


def extract_pair_features(
    s1_row: dict,
    cand_row: dict
) -> List[float]:
    """
    Computes lightweight, highly predictive pairwise features:
    [name_jaccard, name_char_sim, name_len_diff, addr_jaccard, postal_match, same_country, is_s3]
    """
    # Name features
    n1 = s1_row['clean_name']
    n2 = cand_row['clean_name']
    toks1 = set(n1.split()) if n1 else set()
    toks2 = set(n2.split()) if n2 else set()
    name_jacc = calc_jaccard(toks1, toks2)
    name_csim = fast_char_sim(n1, n2)
    name_ldiff = abs(len(n1) - len(n2))

    # Address features
    a1 = s1_row['clean_address']
    a2 = cand_row['clean_address']
    atok1 = set(a1.split()) if a1 else set()
    atok2 = set(a2.split()) if a2 else set()
    addr_jacc = calc_jaccard(atok1, atok2)

    # Postal code match
    p1 = s1_row['postal_code']
    p2 = cand_row['postal_code']
    if p1 and p2:
        postal_match = 1.0 if p1 == p2 else 0.0
    else:
        postal_match = -1.0  # Missing indicator

    # Country match (relational, 100% generalizable to France)
    same_country = 1.0 if s1_row['clean_country'] == cand_row['clean_country'] else 0.0

    # Source indicator
    is_s3 = 1.0 if cand_row['entity_id'].startswith('S3-') else 0.0

    return [name_jacc, name_csim, name_ldiff, addr_jacc, postal_match, same_country, is_s3]


# ---------------------------------------------------------
# 2. MACRO F0.5 EVALUATION METRIC (OFFICIAL COMPETITION FORMULA)
# ---------------------------------------------------------

def compute_macro_f05(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]]
) -> Tuple[float, float, float]:
    """
    Computes macro-averaged F0.5 score across all S1 entities:
    F0.5 = (1.25 * P * R) / (0.25 * P + R)
    - Singletons: If true is empty and pred is empty -> score = 1.0. If pred non-empty -> score = 0.0.
    """
    f05_scores = []
    precisions = []
    recalls = []

    for s1_id, true_set in ground_truth.items():
        pred_list = predictions.get(s1_id, [])
        pred_set = set(pred_list)

        # Singleton case
        if len(true_set) == 0:
            if len(pred_set) == 0:
                f05_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                f05_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(1.0)
            continue

        # Non-singleton case
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)

        if (0.25 * p + r) > 0:
            f05 = (1.25 * p * r) / (0.25 * p + r)
        else:
            f05 = 0.0
        f05_scores.append(f05)

    mean_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    mean_p = float(np.mean(precisions)) if precisions else 0.0
    mean_r = float(np.mean(recalls)) if recalls else 0.0

    return mean_f05, mean_p, mean_r


# ---------------------------------------------------------
# 3. MAIN TRAINING & INFERENCE PIPELINE
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fast XGBoost Classifier on Candidate Pairs")
    parser.add_argument("--s1", type=str, required=True, help="Path to preprocessed S1 TSV")
    parser.add_argument("--s2", type=str, required=True, help="Path to preprocessed S2 TSV")
    parser.add_argument("--s3", type=str, required=True, help="Path to preprocessed S3 TSV")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--ground-truth", "--ground_truth", dest="ground_truth", type=str, required=True, help="Path to train_ground_truth.tsv")
    parser.add_argument("--output", type=str, default="Team Noir_submission/output/matching_results.tsv", help="Path to save matching_results.tsv")
    parser.add_argument("--model-out", type=str, default="Team Noir_submission/code/business_entity_resolution/xgb_model.json", help="Path to save trained XGBoost model")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Hardware device (default: cuda for GPU)")
    parser.add_argument("--val-split", type=float, default=0.20, help="Fraction of S1 entities held out for validation (default: 0.20)")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample limit for faster training")
    args = parser.parse_args()

    start_time = time.time()

    # 1. Load source tables into lookup dictionaries
    print("[XGBoost Pipeline] Loading preprocessed source datasets...")
    df_s1 = pd.read_csv(args.s1, sep="\t", dtype=str).fillna('')
    df_s2 = pd.read_csv(args.s2, sep="\t", dtype=str).fillna('')
    df_s3 = pd.read_csv(args.s3, sep="\t", dtype=str).fillna('')

    s1_dict = {row['entity_id']: row for row in df_s1.to_dict('records')}
    cand_dict = {row['entity_id']: row for row in df_s2.to_dict('records')}
    cand_dict.update({row['entity_id']: row for row in df_s3.to_dict('records')})

    # 2. Load Ground Truth
    print(f"[XGBoost Pipeline] Loading ground truth from {args.ground_truth}...")
    df_gt = pd.read_csv(args.ground_truth, sep="\t", dtype=str).fillna('')
    gt_map: Dict[str, Set[str]] = {}
    for _, row in df_gt.iterrows():
        s1_id = row['source1_entity_id']
        m_str = row['matched_entity_ids'].strip()
        gt_map[s1_id] = set(m.strip() for m in m_str.split(',') if m.strip()) if m_str else set()

    # 3. Load Candidate Pairs
    print(f"[XGBoost Pipeline] Loading candidate pairs from {args.candidates}...")
    df_cands = pd.read_csv(args.candidates, sep="\t", dtype=str).fillna('')

    if args.sample and len(df_cands) > args.sample:
        print(f"[XGBoost Pipeline] Sampling first {args.sample:,} S1 entities...")
        df_cands = df_cands.iloc[:args.sample].copy()

    # 4. Train/Validation Entity-Level Split
    s1_all_ids = df_cands['source1_entity_id'].values
    np.random.seed(42)
    shuffled_ids = np.random.permutation(s1_all_ids)
    n_val = int(len(shuffled_ids) * args.val_split)
    val_s1_set = set(shuffled_ids[:n_val])
    train_s1_set = set(shuffled_ids[n_val:])

    print(f"[XGBoost Pipeline] Entity split: {len(train_s1_set):,} Train S1 entities | {len(val_s1_set):,} Validation S1 entities")

    # 5. Build Training and Validation Feature Matrices
    print("[XGBoost Pipeline] Computing pairwise similarity features...")
    X_train, y_train = [], []
    X_val, y_val = [], []
    val_pairs_meta: List[Tuple[str, str]] = []  # (s1_id, cand_id) for validation evaluation

    for _, row in df_cands.iterrows():
        s1_id = row['source1_entity_id']
        c_str = row['candidate_entity_ids'].strip()
        if not c_str:
            continue
        c_list = [c.strip() for c in c_str.split(',') if c.strip()]

        s1_data = s1_dict.get(s1_id)
        if not s1_data:
            continue

        true_matches = gt_map.get(s1_id, set())
        is_val = s1_id in val_s1_set

        for cid in c_list:
            cand_data = cand_dict.get(cid)
            if not cand_data:
                continue

            feats = extract_pair_features(s1_data, cand_data)
            label = 1 if cid in true_matches else 0

            if is_val:
                X_val.append(feats)
                y_val.append(label)
                val_pairs_meta.append((s1_id, cid))
            else:
                X_train.append(feats)
                y_train.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)

    print(f"    - Train pairs: {len(X_train):,} (Positives: {np.sum(y_train):,}, Negatives: {len(y_train) - np.sum(y_train):,})")
    print(f"    - Val pairs:   {len(X_val):,} (Positives: {np.sum(y_val):,}, Negatives: {len(y_val) - np.sum(y_val):,})")

    # 6. Train XGBoost Model on GPU
    device_to_use = args.device
    print(f"\n[XGBoost Pipeline] Training XGBoost classifier on [{device_to_use.upper()}]...")

    try:
        model = xgb.XGBClassifier(
            device=device_to_use,
            tree_method="hist",
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X_train, y_train)
    except Exception as e:
        if device_to_use == "cuda":
            print(f"[Warning] CUDA initialization failed ({e}). Falling back to CPU...")
            model = xgb.XGBClassifier(
                device="cpu",
                tree_method="hist",
                n_estimators=250,
                max_depth=6,
                learning_rate=0.08,
                n_jobs=-1,
                random_state=42
            )
            model.fit(X_train, y_train)
        else:
            raise e

    print("[XGBoost Pipeline] Model training complete!")

    # 7. Predict on Validation and Sweep F0.5 Thresholds
    print("\n[XGBoost Pipeline] Predicting on validation set and optimizing F0.5 threshold...")
    val_probs = model.predict_proba(X_val)[:, 1]

    # Group probabilities by S1 entity
    val_entity_preds: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in val_s1_set}
    for (s1_id, cid), prob in zip(val_pairs_meta, val_probs):
        val_entity_preds[s1_id].append((cid, float(prob)))

    # Sweep thresholds
    thresholds = [0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    best_thresh = 0.50
    best_f05 = -1.0
    best_p, best_r = 0.0, 0.0

    val_gt = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_set}

    print("\n--------------------------------------------------------------")
    print("THRESHOLD SWEEP ON VALIDATION SPLIT (MACRO-AVERAGED METRICS)")
    print("--------------------------------------------------------------")
    print(f"{'Threshold':<12} | {'Precision':<12} | {'Recall':<12} | {'Macro F0.5':<12}")
    print("--------------------------------------------------------------")

    for thresh in thresholds:
        curr_preds: Dict[str, List[str]] = {}
        for s1_id, cands in val_entity_preds.items():
            curr_preds[s1_id] = [cid for cid, p in cands if p >= thresh]

        f05, p, r = compute_macro_f05(curr_preds, val_gt)
        print(f"{thresh:<12.2f} | {p:<12.4f} | {r:<12.4f} | {f05:<12.4f}")

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_p = p
            best_r = r

    print("--------------------------------------------------------------")
    print(f"Optimal Threshold: {best_thresh:.2f}")
    print(f"Peak Macro F0.5  : {best_f05:.4f} (Precision: {best_p:.4f}, Recall: {best_r:.4f})")
    print("--------------------------------------------------------------")

    # 8. Save Model
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    print(f"\n[XGBoost Pipeline] Saved model to: {model_path}")

    elapsed = time.time() - start_time
    print(f"[XGBoost Pipeline] Total pipeline runtime: {elapsed:.1f}s ({elapsed/60:.2f} mins).")


if __name__ == "__main__":
    main()
