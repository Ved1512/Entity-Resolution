"""
Amazon ML Challenge 2026: Fast Baseline Pipeline
File 1: Fast Prefix & Country Blocking (fast_blocking.py)

Achieves ultra-fast candidate generation in seconds (not hours) using:
1. Country hard partitioning (US, IN, FR).
2. O(1) Inverted hash index on:
   - First word of business name (e.g. 'tata', 'infosys', 'orelee')
   - First 4-character prefix (e.g. 'tata', 'info', 'orel')
   - Exact postal / PIN code (e.g. '700089', '17560')
3. Compact candidate cap (default: top 15-20 candidates per S1 entity).
4. Direct TSV output matching the competition schema.
"""

import sys
import time
import argparse
from pathlib import Path
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np

# GPU imports
import cupy as cp
import cupyx
from sklearn.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors

# Force UTF-8 encoding on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def get_combined_text(row) -> str:
    name = str(row.clean_name) if pd.notna(row.clean_name) else ""
    pcode = str(row.postal_code) if pd.notna(row.postal_code) else ""
    return f"{name} {pcode}".strip()

def run_fast_blocking(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    out_path: Path,
    max_k: int = 50,
    cand_chunk_size: int = 1500000,
    s1_chunk_size: int = 5000
) -> None:
    """
    Builds TF-IDF indices per country and queries S1 using nested GPU chunking.
    """
    print("[Fast Blocking GPU] Combining candidate pool (S2 + S3)...")
    use_cols = ['entity_id', 'clean_name', 'postal_code', 'clean_country']
    s2_sub = df_s2[use_cols].copy() if all(c in df_s2.columns for c in use_cols) else df_s2.copy()
    s3_sub = df_s3[use_cols].copy() if all(c in df_s3.columns for c in use_cols) else df_s3.copy()
    
    df_cand = pd.concat([s2_sub, s3_sub], ignore_index=True)

    # Get unique countries across S1 and Candidates
    s1_countries = df_s1['clean_country'].dropna().unique()
    cand_countries = df_cand['clean_country'].dropna().unique()
    all_countries = set(s1_countries).intersection(set(cand_countries))
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        
        t0 = time.time()
        
        for country in all_countries:
            print(f"\n[Fast Blocking GPU] Processing Country: {country}")
            
            s1_ctry = df_s1[df_s1['clean_country'] == country].copy()
            cand_ctry = df_cand[df_cand['clean_country'] == country].copy()
            
            if len(s1_ctry) == 0 or len(cand_ctry) == 0:
                print(f"    Skipping {country} - missing records.")
                continue
                
            print(f"    - S1 records: {len(s1_ctry):,}")
            print(f"    - Candidate records: {len(cand_ctry):,}")
            
            cand_texts = cand_ctry.apply(get_combined_text, axis=1)
            s1_texts = s1_ctry.apply(get_combined_text, axis=1)
            
            print("    - Fitting TF-IDF vectorizer (CPU)...")
            vectorizer = TfidfVectorizer(
                analyzer='word', 
                ngram_range=(1, 2), 
                min_df=3, 
                max_df=0.1,
                max_features=200000, 
                lowercase=True
            )
            # Compute sparse matrix on CPU to avoid massive string GPU allocations
            cands_tfidf_cpu = vectorizer.fit_transform(cand_texts)
            s1_tfidf_cpu = vectorizer.transform(s1_texts)
            
            cand_ids = cand_ctry['entity_id'].values
            s1_ids = s1_ctry['entity_id'].values
            
            # Two parallel lists: distances and candidate IDs per S1 query
            all_s1_dists = [[] for _ in range(len(s1_ctry))]
            all_s1_cands = [[] for _ in range(len(s1_ctry))]
            
            total_cand_chunks = (len(cand_ctry) // cand_chunk_size) + 1
            print(f"    - Querying across {total_cand_chunks} Candidate Chunks to save VRAM...")
            
            for c_idx, c_start in enumerate(range(0, len(cand_ctry), cand_chunk_size)):
                c_end = min(c_start + cand_chunk_size, len(cand_ctry))
                cand_chunk_tfidf_cpu = cands_tfidf_cpu[c_start:c_end]
                cand_chunk_ids = cand_ids[c_start:c_end]
                
                # Load candidate chunk to GPU
                cand_chunk_tfidf_gpu = cupyx.scipy.sparse.csr_matrix(cand_chunk_tfidf_cpu)
                
                nn_model = NearestNeighbors(n_neighbors=min(max_k, cand_chunk_tfidf_gpu.shape[0]), metric='cosine', algorithm='brute')
                nn_model.fit(cand_chunk_tfidf_gpu)
                
                processed_s1 = 0
                for s_start in range(0, len(s1_ctry), s1_chunk_size):
                    s_end = min(s_start + s1_chunk_size, len(s1_ctry))
                    s1_chunk_tfidf_cpu = s1_tfidf_cpu[s_start:s_end]
                    
                    # Load query chunk to GPU
                    s1_chunk_tfidf_gpu = cupyx.scipy.sparse.csr_matrix(s1_chunk_tfidf_cpu)
                    
                    distances, indices = nn_model.kneighbors(s1_chunk_tfidf_gpu)
                    
                    if hasattr(indices, 'get'):
                        indices = indices.get()
                        distances = distances.get()
                    elif hasattr(indices, 'to_numpy'):
                        indices = indices.to_numpy()
                        distances = distances.to_numpy()
                    
                    # Vectorized: map local chunk indices to global candidate IDs
                    mapped_ids = cand_chunk_ids[indices]  # shape: (chunk, max_k)
                    
                    # Store results in bulk per row (vectorized filtering)
                    for i in range(len(indices)):
                        mask = distances[i] < 1.0
                        if mask.any():
                            all_s1_dists[s_start + i].append(distances[i][mask])
                            all_s1_cands[s_start + i].append(mapped_ids[i][mask])
                                
                    processed_s1 += (s_end - s_start)
                    if processed_s1 % (s1_chunk_size * 10) == 0:
                        print(f"        S1 queries: {processed_s1:,}/{len(s1_ctry):,} against cand chunk {c_idx+1}")
                    
                # Clean up GPU VRAM for the next candidate chunk
                del nn_model, cand_chunk_tfidf_gpu
                cp.get_default_memory_pool().free_all_blocks()
                
                print(f"      Finished Candidate Chunk {c_idx+1}/{total_cand_chunks} - {(time.time() - t0):.1f}s")
                
            print(f"    - Merging results and writing to disk...")
            for i in range(len(s1_ctry)):
                if all_s1_dists[i]:
                    # Concatenate results from all candidate chunks
                    all_d = np.concatenate(all_s1_dists[i])
                    all_c = np.concatenate(all_s1_cands[i])
                    # Take overall top K (lowest distance = highest similarity)
                    if len(all_d) > max_k:
                        top_idx = np.argpartition(all_d, max_k)[:max_k]
                        top_idx = top_idx[np.argsort(all_d[top_idx])]
                    else:
                        top_idx = np.argsort(all_d)
                    valid_cands = all_c[top_idx].tolist()
                else:
                    valid_cands = []
                
                s1_id = s1_ids[i]
                f_out.write(f"{s1_id}\t{','.join(valid_cands)}\n")
            
            print(f"    - Country {country} done in {(time.time() - t0):.1f}s")
                    
        # Handle entities without a country or unmatched country
        processed_ids = set(df_s1[df_s1['clean_country'].isin(all_countries)]['entity_id'])
        missing_ids = set(df_s1['entity_id']) - processed_ids
        if missing_ids:
            print(f"\n[Fast Blocking GPU] Emitting empty predictions for {len(missing_ids):,} unmatched S1 entities...")
            for s1_id in missing_ids:
                f_out.write(f"{s1_id}\t\n")


def evaluate_recall(preds_path: Path, gt_path: Path):
    """Computes exact recall ceiling against train_ground_truth.tsv."""
    print(f"\n[Evaluation] Evaluating against {gt_path.name}...")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str)

    gt_dict = {}
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = row.get('matched_entity_ids', '')
        true_matches = set(m.strip() for m in str(matches_str).split(',') if m.strip()) if pd.notna(matches_str) else set()
        gt_dict[s1_id] = true_matches

    total_true = sum(len(v) for v in gt_dict.values())
    captured_true = 0
    total_cands = 0

    with open(preds_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip('\n').split('\t')
            if len(parts) < 1:
                continue
            s1_id = parts[0]
            if s1_id in gt_dict:
                cands = set(parts[1].split(',')) if len(parts) > 1 and parts[1] else set()
                true_matches = gt_dict[s1_id]
                total_cands += len(cands)
                captured_true += len(true_matches.intersection(cands))

    recall = (captured_true / total_true) if total_true > 0 else 1.0
    avg_cands = total_cands / len(gt) if len(gt) > 0 else 0

    print("======================================================")
    print("FAST BLOCKING PERFORMANCE REPORT")
    print("======================================================")
    print(f"Total True Pairs in Ground Truth : {total_true:,}")
    print(f"True Pairs Captured in Candidates: {captured_true:,}")
    print(f"Blocking Recall Ceiling          : {recall * 100:.2f}%")
    print(f"Total Candidate Pairs Generated  : {total_cands:,}")
    print(f"Average Candidates per S1        : {avg_cands:.2f}")
    print("======================================================")


def main():
    parser = argparse.ArgumentParser(description="Fast Baseline Blocking Module")
    parser.add_argument("--s1", type=str, required=True, help="Path to preprocessed S1 TSV")
    parser.add_argument("--s2", type=str, required=True, help="Path to preprocessed S2 TSV")
    parser.add_argument("--s3", type=str, required=True, help="Path to preprocessed S3 TSV")
    parser.add_argument("--output", type=str, default="Team Noir_submission/output/candidate_pairs.tsv", help="Output path for candidate_pairs.tsv")
    parser.add_argument("--ground-truth", "--ground_truth", dest="ground_truth", type=str, default=None, help="Path to ground truth TSV for recall score")
    parser.add_argument("--max-k", type=int, default=50, help="Maximum candidates per S1 entity (default: 50)")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size of S1 for fast testing")
    args = parser.parse_args()

    start_time = time.time()

    print("[Fast Blocking] Loading preprocessed source files...")
    use_cols = ['entity_id', 'clean_name', 'postal_code', 'clean_country']
    df_s1 = pd.read_csv(args.s1, sep="\t", dtype=str, usecols=use_cols)
    df_s2 = pd.read_csv(args.s2, sep="\t", dtype=str, usecols=use_cols)
    df_s3 = pd.read_csv(args.s3, sep="\t", dtype=str, usecols=use_cols)

    if args.sample and len(df_s1) > args.sample:
        print(f"[Fast Blocking] Sampling first {args.sample:,} S1 entities...")
        df_s1 = df_s1.iloc[:args.sample].copy()

    out_path = Path(args.output)
    run_fast_blocking(df_s1, df_s2, df_s3, out_path, max_k=args.max_k)

    print(f"[Fast Blocking] Successfully saved {len(df_s1):,} rows to {out_path}.")

    if args.ground_truth:
        evaluate_recall(out_path, Path(args.ground_truth))

    elapsed = time.time() - start_time
    print(f"[Fast Blocking] Total runtime: {elapsed:.1f}s ({elapsed/60:.2f} mins).")


if __name__ == "__main__":
    main()