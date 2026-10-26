"""
Rank Diagnostic: finds the TRUE recall-vs-K curve for the blocking stage.

For every (S1, matched_entity_id) pair in the ground truth, this script determines
one of three outcomes:
  1. FOUND_IN_POOL_AT_RANK_R -- the true match was in the candidate pool for that
     country, and appeared at raw similarity rank R (1 = best).
  2. MISSING_FROM_POOL       -- the true match's entity_id was NEVER in the S2+S3
     candidate pool loaded for that S1's country partition. No value of K can fix
     this -- it means the true match either has a different/blank country value,
     or a data-cleaning mismatch. This is checked FIRST because if this fraction
     is large, your recall problem is NOT about K, sim_floor, or the vectorizer.
  3. BEYOND_PROBE_DEPTH      -- present in the pool, but ranked worse than
     --top-k-probe, so we can't say exactly where it landed.

Run this alongside your existing blocking.py (same directory), since it reuses
ScalableBlocker.load_candidates_for_country() and get_blocking_name() to stay
consistent with the real pipeline.

Usage:
  python3 rank_diagnostic.py --s1 S1.tsv --s2 S2.tsv --s3 S3.tsv \
      --ground-truth train_ground_truth.tsv --sample 3000 --top-k-probe 150 --use-gpu
"""

import sys
import time
import argparse
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Reuse the exact same helpers/config as the real pipeline so this diagnostic
# reflects reality, not a re-implementation that might behave differently.
from blocking import ScalableBlocker, get_blocking_name


def load_ground_truth(gt_path: Path) -> Dict[str, Set[str]]:
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
    gt_map: Dict[str, Set[str]] = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = row.get('matched_entity_ids', '')
        if pd.notna(matches_str) and str(matches_str).strip():
            gt_map[s1_id] = set(m.strip() for m in str(matches_str).split(',') if m.strip())
    return {k: v for k, v in gt_map.items() if v}  # keep only S1 entities that actually have matches


def load_s1_fields(s1_path: Path) -> pd.DataFrame:
    header = pd.read_csv(s1_path, sep="\t", nrows=0).columns.tolist()
    ctry_col = 'clean_country' if 'clean_country' in header else 'country'
    name_col = 'clean_name' if 'clean_name' in header else 'business_name'
    df = pd.read_csv(s1_path, sep="\t", usecols=['entity_id', ctry_col, name_col], dtype=str)
    df = df.rename(columns={ctry_col: 'country', name_col: 'name'})
    df['country'] = df['country'].fillna('').str.strip().str.upper()
    return df


def main():
    ap = argparse.ArgumentParser(description="Rank diagnostic for blocking recall-vs-K")
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--ground-truth", "--ground_truth", dest="ground_truth", required=True)
    ap.add_argument("--sample", type=int, default=3000, help="Number of GT S1 entities to probe (default 3000)")
    ap.add_argument("--top-k-probe", type=int, default=150, help="How deep to rank before giving up (default 150)")
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--max-df", type=float, default=0.08)
    ap.add_argument("--ngram-min", type=int, default=4)
    ap.add_argument("--ngram-max", type=int, default=4)
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    t_start = time.time()

    print("[Diagnostic] Loading ground truth...")
    gt_map = load_ground_truth(Path(args.ground_truth))
    print(f"[Diagnostic] {len(gt_map):,} S1 entities have at least one ground-truth match.")

    print("[Diagnostic] Loading S1 fields...")
    s1_df = load_s1_fields(Path(args.s1))
    s1_df = s1_df[s1_df['entity_id'].isin(gt_map.keys())].reset_index(drop=True)

    if len(s1_df) > args.sample:
        s1_df = s1_df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
    print(f"[Diagnostic] Probing {len(s1_df):,} S1 entities (with GT matches) across "
          f"{s1_df['country'].nunique()} countries.")

    blocker = ScalableBlocker(
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(args.ngram_min, args.ngram_max),
        use_gpu=args.use_gpu
    )

    # Result records: one per (s1_id, matched_id) pair actually probed
    # status in {"found", "beyond_probe", "missing_from_pool"}
    records: List[Tuple[str, str, str, int]] = []  # (s1_id, matched_id, status, rank_or_-1)

    cand_sources = [Path(args.s2), Path(args.s3)]

    for country, group in s1_df.groupby('country'):
        ctry_label = country if country else "[Missing/Empty]"
        print(f"\n--- Country: {ctry_label} ({len(group):,} probe entities) ---")

        cand_ids, cand_texts, _, _ = blocker.load_candidates_for_country(cand_sources, country)
        n_cand = len(cand_ids)
        print(f"    Candidate pool size: {n_cand:,}")
        if n_cand == 0:
            for _, row in group.iterrows():
                for m in gt_map[row['entity_id']]:
                    records.append((row['entity_id'], m, "missing_from_pool", -1))
            continue

        cand_id_to_idx = {cid: i for i, cid in enumerate(cand_ids)}

        # Check pool membership FIRST -- this is free (no vectorization needed) and
        # tells us immediately how much of the recall gap is a K problem vs a data problem.
        s1_ids_this_country = group['entity_id'].tolist()
        present_pairs = []   # (s1_id, matched_id) whose matched_id IS in this country's pool
        for s1_id in s1_ids_this_country:
            for m in gt_map[s1_id]:
                if m in cand_id_to_idx:
                    present_pairs.append((s1_id, m))
                else:
                    records.append((s1_id, m, "missing_from_pool", -1))

        if not present_pairs:
            print("    All ground-truth matches for this country are MISSING from the candidate pool.")
            continue

        # Vectorize candidates + the probe queries exactly like the real pipeline
        vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=blocker.ngram_range,
            min_df=blocker.min_df, max_df=blocker.max_df,
            sublinear_tf=True, dtype=np.float32
        )
        sample_size = min(300_000, n_cand)
        vectorizer.fit(cand_texts[:sample_size])

        cand_sub = []
        for c_start in range(0, n_cand, 300_000):
            c_end = min(c_start + 300_000, n_cand)
            cand_sub.append(vectorizer.transform(cand_texts[c_start:c_end]))
        cand_matrix = sparse.vstack(cand_sub, format="csr")

        probe_s1_ids = sorted(set(p[0] for p in present_pairs))
        name_lookup = dict(zip(group['entity_id'], group['name']))
        probe_names = np.array([get_blocking_name(name_lookup[sid]) for sid in probe_s1_ids], dtype=object)
        probe_matrix = vectorizer.transform(probe_names)

        top_k_probe = min(args.top_k_probe, n_cand)
        rank_of: Dict[str, np.ndarray] = {}   # s1_id -> array of candidate indices, best first
        rank_pos: Dict[str, Dict[int, int]] = {}  # s1_id -> {cand_idx: rank(1-based)}

        if args.use_gpu:
            import cupyx.scipy.sparse as cupy_sparse
            from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
            gpu_cand = cupy_sparse.csr_matrix(cand_matrix.astype('float32'))
            nn_model = cuNearestNeighbors(n_neighbors=top_k_probe, metric='cosine',
                                           algorithm='brute', output_type='numpy')
            nn_model.fit(gpu_cand)
            gpu_probe = cupy_sparse.csr_matrix(probe_matrix.astype('float32'))
            _dist, idx = nn_model.kneighbors(gpu_probe)
            for row_i, sid in enumerate(probe_s1_ids):
                rank_pos[sid] = {int(c_idx): r + 1 for r, c_idx in enumerate(idx[row_i])}
        else:
            from sparse_dot_topn import sp_matmul_topn
            sims = sp_matmul_topn(probe_matrix, cand_matrix.T, top_n=top_k_probe,
                                   threshold=0.0, sort=True, n_threads=4)
            for row_i, sid in enumerate(probe_s1_ids):
                r_start, r_end = sims.indptr[row_i], sims.indptr[row_i + 1]
                cols = sims.indices[r_start:r_end]
                rank_pos[sid] = {int(c_idx): r + 1 for r, c_idx in enumerate(cols)}

        for s1_id, matched_id in present_pairs:
            c_idx = cand_id_to_idx[matched_id]
            r = rank_pos.get(s1_id, {}).get(c_idx)
            if r is None:
                records.append((s1_id, matched_id, "beyond_probe", -1))
            else:
                records.append((s1_id, matched_id, "found", r))

    # ---- Aggregate & report ----
    total_pairs = len(records)
    missing = [r for r in records if r[2] == "missing_from_pool"]
    beyond = [r for r in records if r[2] == "beyond_probe"]
    found = [r for r in records if r[2] == "found"]
    found_ranks = sorted(r[3] for r in found)

    print("\n" + "=" * 60)
    print("RANK DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"Total GT pairs probed              : {total_pairs:,}")
    print(f"  Missing from candidate pool       : {len(missing):,} ({len(missing)/total_pairs*100:.1f}%)  <- NOT fixable by K")
    print(f"  Beyond top-{args.top_k_probe} probe depth   : {len(beyond):,} ({len(beyond)/total_pairs*100:.1f}%)")
    print(f"  Found within probe depth          : {len(found):,} ({len(found)/total_pairs*100:.1f}%)")

    if len(missing) / total_pairs > 0.15:
        print("\n  *** WARNING: >15% of true matches are missing from the candidate pool")
        print("      entirely. This is a country/id-mapping data issue upstream of blocking.")
        print("      Fixing K will NOT meaningfully improve recall until this is addressed. ***")

    if found_ranks:
        import statistics as st
        print(f"\nRank stats (pairs actually found, n={len(found_ranks):,}):")
        print(f"  median rank = {st.median(found_ranks)}")
        for p in (50, 75, 90, 95, 99):
            idx = min(int(len(found_ranks) * p / 100), len(found_ranks) - 1)
            print(f"  p{p} rank = {found_ranks[idx]}")

        print(f"\nCumulative recall-vs-K (share of ALL {total_pairs:,} probed pairs captured by K):")
        print(f"  {'K':>5} | {'pairs captured':>15} | {'cumulative recall':>18}")
        for K in [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 150]:
            if K > args.top_k_probe:
                continue
            captured = sum(1 for r in found_ranks if r <= K)
            print(f"  {K:>5} | {captured:>15,} | {captured/total_pairs*100:>17.2f}%")

    print("=" * 60)
    print(f"[Diagnostic] Done in {time.time()-t_start:.1f}s.")


if __name__ == "__main__":
    main()