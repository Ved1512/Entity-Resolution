"""
Amazon ML Challenge 2026: Business Entity Resolution
Stage 2: Scalable Streaming Blocking & Candidate Generation Module (High-Efficiency & Rock-Solid Stability)

Reduces the O(N * M) search space down to a compact, high-recall candidate set
per Source 1 entity, writing output/candidate_pairs.tsv streamingly.

GPU OPTIMIZATION NOTE (this version):
- The hand-rolled cuSPARSE chunk-multiply + manual argpartition/argsort/hstack top-k merge
  has been replaced with a single cuML `NearestNeighbors(algorithm='brute', metric='cosine')`
  index, fit ONCE per country on the whole candidate matrix. Each S1 batch is answered with
  one fused `.kneighbors()` call that does the similarity computation AND top-k selection as
  one GPU-native operation, instead of many small Python-orchestrated kernel launches. This
  removes per-chunk merge overhead entirely and lets batch size scale up much further.
"""

import sys
import os
import re
import gc
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
# cuml.neighbors.NearestNeighbors is imported lazily inside the GPU code path below
# (same pattern as the existing cupy/cupyx imports), so CPU-only runs never require it.

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


STREET_NUM_RE = re.compile(r'^\s*(\d+)\b')

def get_street_number(addr: str) -> str:
    if not addr or pd.isna(addr):
        return ""
    m = STREET_NUM_RE.search(addr)
    return m.group(1) if m else ""


def extract_street_numbers(series: pd.Series) -> pd.Series:
    return series.fillna('').astype(str).str.extract(r'^\s*(\d+)', expand=False).fillna('')


LEGAL_STOPWORDS_RE = re.compile(
    r'\b(?:pvt|ltd|llc|llp|inc|corp|co|ent|services|intl|assoc|sarl|sas|sa|private|limited|incorporated|corporation|company)\b',
    re.IGNORECASE
)

def get_blocking_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    clean = ' '.join(LEGAL_STOPWORDS_RE.sub(' ', name).split())
    return clean if len(clean) >= 2 else name.strip()


def discover_country_partitions(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    chunk_size: int = 1_000_000
) -> List[str]:
    countries = set()
    has_empty = False

    for path in (s1_path, s2_path, s3_path):
        if not path.is_file():
            continue
        header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
        ctry_col = 'clean_country' if 'clean_country' in header else ('country' if 'country' in header else None)
        if not ctry_col:
            continue
        for chunk in pd.read_csv(path, sep="\t", usecols=[ctry_col], chunksize=chunk_size, dtype=str):
            vals = chunk[ctry_col].unique()
            for v in vals:
                if pd.isna(v) or not str(v).strip():
                    has_empty = True
                else:
                    countries.add(str(v).strip().upper())

    result = sorted(list(countries))
    if has_empty:
        result.append('')
    return result


class ScalableBlocker:
    def __init__(
        self,
        min_k: int = 30,
        max_k: int = 60,
        sim_floor: float = 0.08,
        batch_size: int = 200,
        s1_chunk_size: int = 50000,
        cand_chunk_size: int = 500000,
        max_df: float = 0.08,
        min_df: int = 5,
        ngram_range: Tuple[int, int] = (4, 4),
        n_jobs: int = 4,
        use_gpu: bool = False
    ):
        self.min_k = min_k
        self.max_k = max_k
        self.sim_floor = sim_floor
        self.batch_size = batch_size
        self.s1_chunk_size = s1_chunk_size
        self.cand_chunk_size = cand_chunk_size
        self.max_df = max_df
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.n_jobs = (os.cpu_count() or 4) if (n_jobs is None or n_jobs < 1) else n_jobs
        self.use_gpu = use_gpu

    def load_candidates_for_country(
        self,
        candidate_paths: List[Path],
        country: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        all_ids = []
        all_texts = []
        all_pcodes = []
        all_snums = []

        for src_path in candidate_paths:
            header = pd.read_csv(src_path, sep="\t", nrows=0).columns.tolist()
            ctry_col = 'clean_country' if 'clean_country' in header else 'country'
            name_col = 'clean_name' if 'clean_name' in header else 'business_name'
            addr_col = 'clean_address' if 'clean_address' in header else 'business_address'
            pcode_col = 'postal_code' if 'postal_code' in header else None
            snum_col = 'street_num' if 'street_num' in header else None

            cols_to_use = ['entity_id', ctry_col, name_col]
            if pcode_col:
                cols_to_use.append(pcode_col)
            if snum_col:
                cols_to_use.append(snum_col)
            elif addr_col:
                cols_to_use.append(addr_col)

            for chunk in pd.read_csv(src_path, sep="\t", usecols=cols_to_use, chunksize=self.cand_chunk_size, dtype=str):
                chunk_ctry = chunk[ctry_col].fillna('').str.strip().str.upper()
                if country:
                    mask = chunk_ctry.isin([country, ''])
                else:
                    mask = (chunk_ctry == '')

                sub = chunk[mask]
                if len(sub) == 0:
                    continue

                ids = sub['entity_id'].fillna('').to_numpy()
                raw_names = sub[name_col].fillna('').to_numpy()
                names = np.array([get_blocking_name(n) for n in raw_names], dtype=object)

                if pcode_col:
                    pcodes = sub[pcode_col].fillna('').to_numpy()
                else:
                    pcodes = np.full(len(sub), '', dtype=object)

                if snum_col:
                    snums = sub[snum_col].fillna('').to_numpy()
                elif addr_col:
                    snums = extract_street_numbers(sub[addr_col]).to_numpy()
                else:
                    snums = np.full(len(sub), '', dtype=object)

                all_ids.append(ids)
                all_texts.append(names)
                all_pcodes.append(pcodes)
                all_snums.append(snums)

        if all_ids:
            cand_ids = np.concatenate(all_ids)
            cand_texts = np.concatenate(all_texts)
            cand_pcodes = np.concatenate(all_pcodes)
            cand_snums = np.concatenate(all_snums)
        else:
            cand_ids = np.array([], dtype=object)
            cand_texts = np.array([], dtype=object)
            cand_pcodes = np.array([], dtype=object)
            cand_snums = np.array([], dtype=object)

        return cand_ids, cand_texts, cand_pcodes, cand_snums

    def process_and_flush_country(
        self,
        country: str,
        s1_path: Path,
        candidate_paths: List[Path],
        output_file_handle,
        s1_sample_limit: Optional[int] = None,
        total_s1_processed_ref: Optional[List[int]] = None,
        country_s1_total: Optional[int] = None
    ) -> int:
        ctry_label = country if country else "[Missing/Empty]"
        print(f"\n--- Partition: Country = {ctry_label} ---")

        print(f"    - Loading candidate pool (S2+S3) for {ctry_label}...")
        cand_ids, cand_texts, cand_pcodes, cand_snums = self.load_candidates_for_country(candidate_paths, country)
        n_cand = len(cand_ids)
        print(f"    - Loaded {n_cand:,} candidates for country {ctry_label}.")

        if n_cand == 0:
            print(f"    - No candidates found for country {ctry_label}. Writing empty candidates for matching S1.")
            return self._flush_empty_s1(s1_path, country, output_file_handle, s1_sample_limit, total_s1_processed_ref)

        print("    - Building postal code & street number fallback buckets...")
        postal_bucket: Dict[str, List[int]] = {}
        for idx in range(n_cand):
            p_code = cand_pcodes[idx]
            street_num = cand_snums[idx]
            if p_code and street_num:
                key = f"{p_code}_{street_num}"
                if key not in postal_bucket:
                    postal_bucket[key] = [idx]
                else:
                    postal_bucket[key].append(idx)

        print(f"    - Fitting character n-gram TF-IDF Vectorizer {self.ngram_range} on discriminative business names...")
        vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=True,
            dtype=np.float32
        )

        sample_size = min(300_000, n_cand)
        vectorizer.fit(cand_texts[:sample_size])

        print(f"    - Transforming {n_cand:,} candidates in streaming chunks...")
        cand_sub_matrices = []
        tf_chunk_sz = 300_000
        for c_start in range(0, n_cand, tf_chunk_sz):
            c_end = min(c_start + tf_chunk_sz, n_cand)
            cand_sub_matrices.append(vectorizer.transform(cand_texts[c_start:c_end]))

        cand_matrix = sparse.vstack(cand_sub_matrices, format="csr")
        del cand_sub_matrices, cand_texts
        gc.collect()

        print(f"    - Vocabulary size: {len(vectorizer.vocabulary_):,} character n-grams")
        print(f"    - Candidate matrix non-zeros: {cand_matrix.nnz:,}")

        # CPU path needs the transposed (vocab x n_cand) matrix for sp_matmul_topn.
        # GPU path no longer needs this at all (see chunk-building below), which
        # avoids holding/transposing it when running on GPU.
        cand_matrix_t = None if self.use_gpu else cand_matrix.T

        print(f"    - Streaming S1 queries (batch size: {self.batch_size:,})...")
        header = pd.read_csv(s1_path, sep="\t", nrows=0).columns.tolist()
        ctry_col = 'clean_country' if 'clean_country' in header else 'country'
        name_col = 'clean_name' if 'clean_name' in header else 'business_name'
        addr_col = 'clean_address' if 'clean_address' in header else 'business_address'
        pcode_col = 'postal_code' if 'postal_code' in header else None
        snum_col = 'street_num' if 'street_num' in header else None

        cols_to_use = ['entity_id', ctry_col, name_col]
        if pcode_col:
            cols_to_use.append(pcode_col)
        if snum_col:
            cols_to_use.append(snum_col)
        elif addr_col:
            cols_to_use.append(addr_col)

        country_s1_processed = 0
        last_printed_count = 0
        t0_queries = time.time()

        if self.use_gpu:
            print("    - Uploading candidate matrix to GPU and building cuML NearestNeighbors index...")
            import cupy
            import cupyx.scipy.sparse as cupy_sparse
            from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

            gpu_cand_matrix = cupy_sparse.csr_matrix(cand_matrix.astype('float32'))
            nn_model = cuNearestNeighbors(
                n_neighbors=self.max_k,
                metric='cosine',
                algorithm='brute',
                output_type='numpy'   # kneighbors() returns plain numpy arrays, no .get() needed
            )
            nn_model.fit(gpu_cand_matrix)
        else:
            nn_model = None

        for s1_chunk_df in pd.read_csv(s1_path, sep="\t", usecols=cols_to_use, chunksize=self.s1_chunk_size, dtype=str):
            s1_ctry = s1_chunk_df[ctry_col].fillna('').str.strip().str.upper()
            if country:
                mask = (s1_ctry == country)
            else:
                mask = (s1_ctry == '')

            sub_s1 = s1_chunk_df[mask]
            if len(sub_s1) == 0:
                continue

            if s1_sample_limit is not None:
                current_total = total_s1_processed_ref[0] if total_s1_processed_ref else 0
                remaining = s1_sample_limit - current_total
                if remaining <= 0:
                    break
                if len(sub_s1) > remaining:
                    sub_s1 = sub_s1.iloc[:remaining]

            s1_ids = sub_s1['entity_id'].fillna('').to_numpy()
            raw_s1_names = sub_s1[name_col].fillna('').to_numpy()
            s1_names = np.array([get_blocking_name(n) for n in raw_s1_names], dtype=object)

            if pcode_col:
                s1_pcodes = sub_s1[pcode_col].fillna('').to_numpy()
            else:
                s1_pcodes = np.full(len(sub_s1), '', dtype=object)

            if snum_col:
                s1_snums = sub_s1[snum_col].fillna('').to_numpy()
            elif addr_col:
                s1_snums = extract_street_numbers(sub_s1[addr_col]).to_numpy()
            else:
                s1_snums = np.full(len(sub_s1), '', dtype=object)

            s1_block_mat = vectorizer.transform(s1_names)
            n_block_s1 = len(s1_ids)

            if self.use_gpu:
                s1_block_mat_f32 = s1_block_mat.astype('float32')
                gpu_s1_block = cupy_sparse.csr_matrix(s1_block_mat_f32)
            else:
                gpu_s1_block = None

            batch_intervals = [
                (b_start, min(b_start + self.batch_size, n_block_s1))
                for b_start in range(0, n_block_s1, self.batch_size)
            ]

            def process_sub_batch(interval: Tuple[int, int]) -> List[str]:
                b_start, b_end = interval
                batch_len = b_end - b_start
                min_cutoff = min(0.04, self.sim_floor)

                if self.use_gpu:
                    # One fused GPU call: similarity computation + top-k selection, no manual
                    # chunk-merge, no argpartition/argsort orchestration in Python.
                    query_block = gpu_s1_block[b_start:b_end]
                    cosine_dist, nbr_idx = nn_model.kneighbors(query_block)

                    # cosine distance -> cosine similarity
                    cpu_scores = 1.0 - cosine_dist
                    cpu_indices = nbr_idx
                else:
                    sub_chunk = s1_block_mat[b_start:b_end]
                    chunk_sims = sp_matmul_topn(
                        sub_chunk,
                        cand_matrix_t,
                        top_n=self.max_k,
                        threshold=min_cutoff,
                        sort=True,
                        n_threads=1
                    )

                batch_lines = []

                for i in range(batch_len):
                    s1_id = s1_ids[b_start + i]

                    selected_cand_ids: List[str] = []
                    seen_cand_ids: Set[str] = set()

                    s1_pcode = s1_pcodes[b_start + i]
                    s1_snum = s1_snums[b_start + i]
                    if s1_pcode and s1_snum:
                        key = f"{s1_pcode}_{s1_snum}"
                        if key in postal_bucket:
                            for c_idx in postal_bucket[key][:10]:
                                cid = cand_ids[c_idx]
                                if cid not in seen_cand_ids:
                                    seen_cand_ids.add(cid)
                                    selected_cand_ids.append(cid)

                    if self.use_gpu:
                        cand_scores = cpu_scores[i]
                        cand_indices = cpu_indices[i]
                        for score, c_idx in zip(cand_scores, cand_indices):
                            if score < min_cutoff:
                                continue
                            cid = cand_ids[c_idx]
                            if cid in seen_cand_ids:
                                continue
                            if len(selected_cand_ids) < self.min_k:
                                if score > 0.0:
                                    seen_cand_ids.add(cid)
                                    selected_cand_ids.append(cid)
                            elif len(selected_cand_ids) < self.max_k:
                                if score >= self.sim_floor:
                                    seen_cand_ids.add(cid)
                                    selected_cand_ids.append(cid)
                                else:
                                    break
                            else:
                                break
                    else:
                        r_start = chunk_sims.indptr[i]
                        r_end = chunk_sims.indptr[i + 1]
                        n_scores = r_end - r_start
                        if n_scores > 0:
                            cand_scores = chunk_sims.data[r_start:r_end]
                            cand_indices = chunk_sims.indices[r_start:r_end]
                            top_order = range(n_scores)
                            for cand_pos in top_order:
                                score = cand_scores[cand_pos]
                                c_idx = cand_indices[cand_pos]
                                cid = cand_ids[c_idx]
                                if cid in seen_cand_ids:
                                    continue
                                if len(selected_cand_ids) < self.min_k:
                                    if score > 0.0:
                                        seen_cand_ids.add(cid)
                                        selected_cand_ids.append(cid)
                                elif len(selected_cand_ids) < self.max_k:
                                    if score >= self.sim_floor:
                                        seen_cand_ids.add(cid)
                                        selected_cand_ids.append(cid)
                                    else:
                                        break
                                else:
                                    break

                    batch_lines.append(f"{s1_id}\t{','.join(selected_cand_ids)}\n")

                if not self.use_gpu:
                    del chunk_sims, sub_chunk
                return batch_lines

            if self.n_jobs > 1 and len(batch_intervals) > 1 and not self.use_gpu:
                # NOTE: thread-parallelizing GPU sub-batches provides no benefit (CUDA calls
                # serialize on the device and risk stacking memory allocations); only used for CPU.
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    window_sz = self.n_jobs * 2
                    for w_start in range(0, len(batch_intervals), window_sz):
                        window = batch_intervals[w_start:w_start + window_sz]
                        futures = [executor.submit(process_sub_batch, iv) for iv in window]
                        for f in futures:
                            lines = f.result()
                            output_file_handle.writelines(lines)
                            country_s1_processed += len(lines)
                            if total_s1_processed_ref is not None:
                                total_s1_processed_ref[0] += len(lines)

                            if country_s1_processed - last_printed_count >= 1000:
                                last_printed_count = country_s1_processed
                                elapsed_q = time.time() - t0_queries
                                rate = country_s1_processed / elapsed_q if elapsed_q > 0 else 0
                                total_str = f" / {country_s1_total:,}" if country_s1_total else ""
                                print(f"      Processed {country_s1_processed:,}{total_str} S1 queries for country {ctry_label} ({rate:.1f} q/s)...", flush=True)
            else:
                for iv in batch_intervals:
                    lines = process_sub_batch(iv)
                    output_file_handle.writelines(lines)
                    country_s1_processed += len(lines)
                    if total_s1_processed_ref is not None:
                        total_s1_processed_ref[0] += len(lines)

                    if country_s1_processed - last_printed_count >= 1000:
                        last_printed_count = country_s1_processed
                        elapsed_q = time.time() - t0_queries
                        rate = country_s1_processed / elapsed_q if elapsed_q > 0 else 0
                        total_str = f" / {country_s1_total:,}" if country_s1_total else ""
                        print(f"      Processed {country_s1_processed:,}{total_str} S1 queries for country {ctry_label} ({rate:.1f} q/s)...", flush=True)

            del s1_block_mat, sub_s1
            gc.collect()

            if s1_sample_limit is not None and total_s1_processed_ref and total_s1_processed_ref[0] >= s1_sample_limit:
                break

        del cand_matrix, postal_bucket
        if cand_matrix_t is not None:
            del cand_matrix_t
        gc.collect()

        return country_s1_processed

    def _flush_empty_s1(
        self,
        s1_path: Path,
        country: str,
        output_file_handle,
        s1_sample_limit: Optional[int] = None,
        total_s1_processed_ref: Optional[List[int]] = None
    ) -> int:
        header = pd.read_csv(s1_path, sep="\t", nrows=0).columns.tolist()
        ctry_col = 'clean_country' if 'clean_country' in header else 'country'
        processed = 0

        for chunk in pd.read_csv(s1_path, sep="\t", usecols=['entity_id', ctry_col], chunksize=self.s1_chunk_size, dtype=str):
            s1_ctry = chunk[ctry_col].fillna('').str.strip().str.upper()
            if country:
                mask = (s1_ctry == country)
            else:
                mask = (s1_ctry == '')

            sub = chunk[mask]
            if len(sub) == 0:
                continue

            for s1_id in sub['entity_id'].values:
                output_file_handle.write(f"{s1_id}\t\n")
                processed += 1
                if total_s1_processed_ref is not None:
                    total_s1_processed_ref[0] += 1
                if s1_sample_limit is not None and total_s1_processed_ref and total_s1_processed_ref[0] >= s1_sample_limit:
                    return processed

        return processed

    def run_blocking_stream(
        self,
        s1_path: Union[str, Path],
        s2_path: Union[str, Path],
        s3_path: Union[str, Path],
        output_path: Union[str, Path],
        sample: Optional[int] = None
    ) -> Path:
        s1_p = Path(s1_path)
        s2_p = Path(s2_path)
        s3_p = Path(s3_path)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        print("[Stage 2] Discovering country partitions across sources...")
        all_countries = discover_country_partitions(s1_p, s2_p, s3_p)
        print(f"[Stage 2] Detected {len(all_countries)} country partitions: {all_countries}")

        print("[Stage 2] Pre-calculating total S1 queries per country partition...")
        s1_country_counts = {c: 0 for c in all_countries}

        try:
            header = pd.read_csv(s1_p, sep="\t", nrows=0).columns.tolist()
            ctry_col = 'clean_country' if 'clean_country' in header else ('country' if 'country' in header else None)
            if ctry_col:
                for chunk in pd.read_csv(s1_p, sep="\t", usecols=[ctry_col], chunksize=500_000, dtype=str):
                    counts = chunk[ctry_col].fillna('').str.strip().str.upper().value_counts()
                    for c, count in counts.items():
                        if not c: c = ''
                        if c in s1_country_counts:
                            s1_country_counts[c] += count
        except Exception as e:
            print(f"[Stage 2] Warning: Could not precompute counts: {e}")

        with open(out_p, "w", encoding="utf-8") as out_f:
            out_f.write("source1_entity_id\tcandidate_entity_ids\n")

        total_processed = [0]
        cand_sources = [s2_p, s3_p]

        with open(out_p, "a", encoding="utf-8") as out_f:
            for ctry in all_countries:
                if sample is not None and total_processed[0] >= sample:
                    print(f"[Stage 2] Sample limit of {sample:,} reached.")
                    break

                ctry_total = s1_country_counts.get(ctry, 0)

                self.process_and_flush_country(
                    country=ctry,
                    s1_path=s1_p,
                    candidate_paths=cand_sources,
                    output_file_handle=out_f,
                    s1_sample_limit=sample,
                    total_s1_processed_ref=total_processed,
                    country_s1_total=ctry_total
                )

        print(f"\n[Stage 2] Streaming blocking completed. Total S1 queries emitted: {total_processed[0]:,}.")
        return out_p


def evaluate_blocking_recall(
    candidate_file: Path,
    ground_truth_path: Path
) -> Tuple[float, float, float]:
    print(f"\n[Stage 2 Evaluation] Evaluating candidates against ground truth: {ground_truth_path.name}...")

    gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str)
    gt_map: Dict[str, Set[str]] = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = row.get('matched_entity_ids', '')
        if pd.notna(matches_str) and str(matches_str).strip():
            gt_map[s1_id] = set(m.strip() for m in str(matches_str).split(',') if m.strip())
        else:
            gt_map[s1_id] = set()

    total_true_pairs = sum(len(m) for m in gt_map.values())
    captured_true_pairs = 0
    total_candidates = 0
    total_queries = 0

    with open(candidate_file, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0].strip()
            cands_str = parts[1].strip() if len(parts) > 1 else ""

            if cands_str:
                cands = set(c.strip() for c in cands_str.split(",") if c.strip())
            else:
                cands = set()

            total_candidates += len(cands)
            total_queries += 1

            if s1_id in gt_map:
                true_matches = gt_map[s1_id]
                captured_true_pairs += len(true_matches.intersection(cands))

    recall = (captured_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    avg_cand = (total_candidates / total_queries) if total_queries > 0 else 0.0

    print("======================================================")
    print("STAGE 2: BLOCKING PERFORMANCE REPORT")
    print("======================================================")
    print(f"Total True Match Pairs in GT  : {total_true_pairs:,}")
    print(f"True Match Pairs Captured     : {captured_true_pairs:,}")
    print(f"Blocking Recall Ceiling       : {recall * 100:.2f}%")
    print(f"Total Candidate Pairs Emitted : {total_candidates:,}")
    print(f"Total S1 Entities Evaluated   : {total_queries:,}")
    print(f"Average Candidates per S1     : {avg_cand:.2f}")
    print("======================================================")

    return recall, avg_cand, float(total_candidates)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Scalable Streaming Blocking / Candidate Generation")
    parser.add_argument("--s1", type=str, required=True, help="Path to preprocessed S1 TSV")
    parser.add_argument("--s2", type=str, required=True, help="Path to preprocessed S2 TSV")
    parser.add_argument("--s3", type=str, required=True, help="Path to preprocessed S3 TSV")
    parser.add_argument("--output", type=str, default="Team Noir_submission/output/candidate_pairs.tsv", help="Path to save candidate_pairs.tsv")
    parser.add_argument("--ground-truth", "--ground_truth", dest="ground_truth", type=str, default=None, help="Optional path to train_ground_truth.tsv for recall evaluation")
    parser.add_argument("--min-k", type=int, default=10, help="Minimum candidate count per S1 (default: 30)")
    parser.add_argument("--max-k", type=int, default=70, help="Maximum candidate count per S1 (default: 60)")
    parser.add_argument("--floor", type=float, default=0.08, help="Similarity floor to expand between min-k and max-k (default: 0.12)")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size of S1 queries per dot-product (default: 200)")
    parser.add_argument("--s1-chunk-size", type=int, default=50000, help="Number of S1 queries streamed per block (default: 50,000)")
    parser.add_argument("--cand-chunk-size", type=int, default=500000, help="Candidate chunk size for streaming load (default: 500,000)")
    parser.add_argument("--max-df", type=float, default=0.08, help="Max document frequency for TF-IDF to filter generic stopword trigrams (default: 0.08)")
    parser.add_argument("--min-df", type=int, default=5, help="Min document frequency for TF-IDF (default: 5)")
    parser.add_argument("--ngram-min", type=int, default=4, help="Min character n-gram length (default: 4)")
    parser.add_argument("--ngram-max", type=int, default=4, help="Max character n-gram length (default: 4)")
    parser.add_argument("--n-jobs", type=int, default=4, help="Number of workers (default: 4 for high throughput)")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample of S1 to process (useful for quick validation)")
    parser.add_argument("--use-gpu", action="store_true", help="Use CuPy for GPU-accelerated sparse matrix multiplication")
    args = parser.parse_args()

    start_time = time.time()

    blocker = ScalableBlocker(
        min_k=args.min_k,
        max_k=args.max_k,
        sim_floor=args.floor,
        batch_size=args.batch_size,
        s1_chunk_size=args.s1_chunk_size,
        cand_chunk_size=args.cand_chunk_size,
        max_df=args.max_df,
        min_df=args.min_df,
        ngram_range=(args.ngram_min, args.ngram_max),
        n_jobs=args.n_jobs,
        use_gpu=args.use_gpu
    )

    out_path = blocker.run_blocking_stream(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        output_path=args.output,
        sample=args.sample
    )

    if args.ground_truth:
        evaluate_blocking_recall(out_path, Path(args.ground_truth))

    elapsed = time.time() - start_time
    print(f"\n[Stage 2] Done in {elapsed:.1f}s ({elapsed/60:.2f} mins).")


if __name__ == "__main__":
    main()