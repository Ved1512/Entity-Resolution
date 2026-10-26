"""
Amazon ML Challenge 2026 – Business Entity Resolution
blocking2.py – Two-Phase GPU-Accelerated Blocking

PHASE A  (offline, runs once, results cached to disk):
  1. Fit ONE global TF-IDF vectorizer from a cross-country sample of S2+S3.
     No per-country refit – this alone eliminates 2/3 of the fit overhead.
  2. Per country: transform named candidates → save .npz on disk;
                  build city-token bucket for NaN-name (address-only) candidates.
  3. Pre-split S1 by country → per-country TSV files (eliminates 3× S1 rescans).

PHASE B  (online, ~7–10 min after Phase A is cached):
  1. Load vectorizer + per-country .npz from cache.
  2. Transform ALL S1 names for the country in ONE sklearn call (no chunking).
  3. cuPy SpGEMM (sparse × sparse, chunked) – 6 GB VRAM safe.
       Peak VRAM ≈ batch_size × cand_chunk × 4 bytes
       Default   ≈ 3 000 × 50 000 × 4  =  600 MB
  4. Three candidate channels:
       a) Postal + street-number exact bucket   (named candidates)
       b) TF-IDF cosine top-k via GPU SpGEMM   (named candidates)
       c) City-token lookup                     (NaN-name / address-only candidates)
  5. Stream output to candidate_pairs.tsv.

NOTE on "common words":
  clean_name is used as-is from the preprocessed file.  No legal-suffix stripping
  is applied here; IDF naturally down-weights high-frequency terms while keeping
  them in the vocabulary so the downstream model can use them as features.
  max_df defaults to 0.95 (very permissive) to retain legal-form n-grams.

Usage:
    # First run (builds Phase A cache, then runs Phase B):
    python blocking2.py --s1 ... --s2 ... --s3 ... --output ...

    # Subsequent runs (Phase A already cached – only Phase B runs):
    python blocking2.py --s1 ... --s2 ... --s3 ... --output ...

    # Force full Phase A rebuild:
    python blocking2.py ... --rebuild
"""

import sys
import gc
import re
import time
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

CACHE_VERSION = "b2v1"    # bump this string to force a full Phase A rebuild
GLOBAL_SAMPLE = 500_000   # rows drawn from ALL S2+S3 to fit the one vectorizer
ADDR_K_DEFAULT = 20       # max address-channel candidates added per S1 entity

_POSTAL_RE = re.compile(r'\b(\d{5}(?:-\d{4})?|\d{6})\b')
_STREET_RE  = re.compile(r'^\s*(\d+)\b')


# ──────────────────────────────────────────────────────────────────────────────
# SMALL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _postal(addr: str) -> str:
    if not addr or not isinstance(addr, str):
        return ""
    m = _POSTAL_RE.search(addr)
    return m.group(1).split('-')[0] if m else ""


def _street_num(addr: str) -> str:
    if not addr or not isinstance(addr, str):
        return ""
    m = _STREET_RE.search(addr)
    return m.group(1) if m else ""


def _city_token(addr: str) -> str:
    """First purely-alphabetic word ≥5 chars in the address (coarse locality key)."""
    if not addr or not isinstance(addr, str):
        return ""
    for w in addr.split():
        if w.isalpha() and len(w) >= 5:
            return w.lower()
    return ""


def _ctry_col(cols: List[str]) -> str:
    return 'clean_country' if 'clean_country' in cols else 'country'


def _name_col(cols: List[str]) -> str:
    return 'clean_name' if 'clean_name' in cols else 'business_name'


def _addr_col(cols: List[str]) -> str:
    return 'clean_address' if 'clean_address' in cols else 'business_address'


def _discover_countries(paths: List[Path]) -> List[str]:
    countries: Set[str] = set()
    has_empty = False
    for p in paths:
        if not p.is_file():
            continue
        cols = pd.read_csv(p, sep='\t', nrows=0).columns.tolist()
        col  = _ctry_col(cols)
        for chunk in pd.read_csv(p, sep='\t', usecols=[col],
                                 chunksize=500_000, dtype=str):
            for v in chunk[col].fillna('').str.strip().str.upper().unique():
                if v:
                    countries.add(v)
                else:
                    has_empty = True
    result = sorted(countries)
    if has_empty:
        result.append('')
    return result


# ──────────────────────────────────────────────────────────────────────────────
# PHASE A: BUILD GLOBAL INDEX CACHE
# ──────────────────────────────────────────────────────────────────────────────

def phase_a(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    cache_dir: Path,
    ngram_range: Tuple[int, int],
    min_df: int,
    max_df: float,
) -> List[str]:
    """
    Runs once.  Builds and saves to cache_dir:
      • vectorizer_{VER}.pkl           – global TF-IDF
      • cand_{country}_{VER}.npz       – per-country sparse candidate matrix
      • meta_{country}_{VER}.pkl       – ids, postal/city buckets, NaN-name bucket
      • s1_{country}_{VER}.tsv         – per-country S1 split
      • countries_{VER}.pkl            – ordered list of countries
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cand_paths = [p for p in [s2_path, s3_path] if p.is_file()]

    # ── 1. Discover countries ──────────────────────────────────────────────
    print("[Phase A] Discovering countries …")
    countries = _discover_countries([s1_path] + cand_paths)
    print(f"[Phase A] Countries: {countries}")

    # ── 2. Sample across ALL countries for a single global vectorizer ──────
    print(f"[Phase A] Sampling {GLOBAL_SAMPLE:,} rows from S2+S3 (cross-country) …")
    sample_texts: List[str] = []
    per_path = max(1, GLOBAL_SAMPLE // len(cand_paths))
    for p in cand_paths:
        cols  = pd.read_csv(p, sep='\t', nrows=0).columns.tolist()
        n_col = _name_col(cols)
        df_s  = pd.read_csv(p, sep='\t', usecols=[n_col],
                            nrows=per_path, dtype=str)
        sample_texts.extend(df_s[n_col].fillna('').tolist())
        del df_s

    print(f"[Phase A] Fitting global TF-IDF on {len(sample_texts):,} samples "
          f"(ngram={ngram_range}, min_df={min_df}, max_df={max_df}) …")
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        dtype=np.float32,
        norm='l2',          # L2-normalised → dot-product = cosine similarity
    )
    vectorizer.fit(sample_texts)
    del sample_texts
    vocab_size = len(vectorizer.vocabulary_)
    print(f"[Phase A] Vocabulary: {vocab_size:,} char n-grams")

    vec_path = cache_dir / f"vectorizer_{CACHE_VERSION}.pkl"
    with open(vec_path, 'wb') as fh:
        pickle.dump(vectorizer, fh, protocol=4)
    print(f"[Phase A] Vectorizer saved → {vec_path.name}")

    # ── 3. Per-country candidate index ────────────────────────────────────
    for ctry in countries:
        ctry_key  = ctry if ctry else '_empty_'
        cand_npz  = cache_dir / f"cand_{ctry_key}_{CACHE_VERSION}.npz"
        meta_path = cache_dir / f"meta_{ctry_key}_{CACHE_VERSION}.pkl"

        if cand_npz.exists() and meta_path.exists():
            print(f"[Phase A] '{ctry_key}' – cache exists, skipping.")
            continue

        print(f"\n[Phase A] Building index for country: '{ctry_key}' …")

        named_ids: List[str]   = []
        named_names: List[str] = []
        named_pcodes: List[str] = []
        named_snums: List[str]  = []
        named_addrs: List[str]  = []
        nan_ids: List[str]   = []
        nan_addrs: List[str] = []

        for p in cand_paths:
            cols  = pd.read_csv(p, sep='\t', nrows=0).columns.tolist()
            c_col = _ctry_col(cols)
            n_col = _name_col(cols)
            a_col = _addr_col(cols)

            for chunk in pd.read_csv(
                p, sep='\t',
                usecols=['entity_id', c_col, n_col, a_col],
                chunksize=200_000, dtype=str,
            ):
                ctry_vals = chunk[c_col].fillna('').str.strip().str.upper()
                mask = (ctry_vals == ctry) if ctry else (ctry_vals == '')
                sub  = chunk[mask]
                if sub.empty:
                    continue

                has_name = sub[n_col].fillna('').str.strip() != ''

                named = sub[has_name]
                if not named.empty:
                    addrs_s = named[a_col].fillna('')
                    named_ids.extend(named['entity_id'].tolist())
                    named_names.extend(named[n_col].fillna('').tolist())
                    named_pcodes.extend(addrs_s.apply(_postal).tolist())
                    named_snums.extend(addrs_s.apply(_street_num).tolist())
                    named_addrs.extend(addrs_s.tolist())

                unnamed = sub[~has_name]
                if not unnamed.empty:
                    nan_ids.extend(unnamed['entity_id'].tolist())
                    nan_addrs.extend(unnamed[a_col].fillna('').tolist())

        n_named = len(named_ids)
        print(f"  Named candidates  : {n_named:,}")
        print(f"  Address-only      : {len(nan_ids):,}")

        # Transform named candidates in CPU-RAM-safe chunks
        if n_named > 0:
            TRANS_CHUNK = 300_000
            mats = []
            for c0 in range(0, n_named, TRANS_CHUNK):
                c1 = min(c0 + TRANS_CHUNK, n_named)
                mats.append(vectorizer.transform(named_names[c0:c1]))
                print(f"  Transformed {c1:,}/{n_named:,} …")
            cand_mat = sp.vstack(mats, format='csr').astype(np.float32)
            del mats, named_names
            sp.save_npz(str(cand_npz), cand_mat)
            print(f"  Saved {cand_npz.name}  "
                  f"shape={cand_mat.shape}  nnz={cand_mat.nnz:,}")
            del cand_mat
        else:
            # Empty placeholder so Phase B can detect the file
            sp.save_npz(
                str(cand_npz),
                sp.csr_matrix((0, vocab_size), dtype=np.float32),
            )

        # Postal + street-number bucket (named candidates only)
        postal_bucket: Dict[str, List[int]] = {}
        for i, (pc, sn) in enumerate(zip(named_pcodes, named_snums)):
            if pc and sn:
                postal_bucket.setdefault(f"{pc}_{sn}", []).append(i)

        # City-token bucket for NaN-name candidates  →  maps token → [entity_id …]
        nan_city_bucket: Dict[str, List[str]] = {}
        for eid, addr in zip(nan_ids, nan_addrs):
            city = _city_token(addr)
            if city:
                nan_city_bucket.setdefault(city, []).append(eid)

        meta = {
            'ids':             named_ids,
            'pcodes':          named_pcodes,
            'snums':           named_snums,
            'postal_bucket':   postal_bucket,
            'nan_city_bucket': nan_city_bucket,
        }
        with open(meta_path, 'wb') as fh:
            pickle.dump(meta, fh, protocol=4)
        print(f"  Meta saved → {meta_path.name}")

    # ── 4. Pre-split S1 by country (one scan, no per-country rescan) ───────
    print("\n[Phase A] Pre-splitting S1 by country …")
    cols  = pd.read_csv(s1_path, sep='\t', nrows=0).columns.tolist()
    c_col = _ctry_col(cols)

    s1_written: Set[str] = set()
    for chunk in pd.read_csv(s1_path, sep='\t', chunksize=200_000, dtype=str):
        ctry_vals = chunk[c_col].fillna('').str.strip().str.upper()
        for ctry in countries:
            mask = (ctry_vals == ctry) if ctry else (ctry_vals == '')
            sub  = chunk[mask]
            if sub.empty:
                continue
            ctry_key = ctry if ctry else '_empty_'
            out_path = cache_dir / f"s1_{ctry_key}_{CACHE_VERSION}.tsv"
            mode = 'a' if ctry_key in s1_written else 'w'
            sub.to_csv(out_path, sep='\t', index=False,
                       mode=mode, header=(ctry_key not in s1_written))
            s1_written.add(ctry_key)

    print(f"[Phase A] S1 split into {len(s1_written)} per-country files.")

    # Save country list
    with open(cache_dir / f"countries_{CACHE_VERSION}.pkl", 'wb') as fh:
        pickle.dump(countries, fh)

    print(f"\n[Phase A] Complete.  Cache → {cache_dir}\n")
    return countries



# ──────────────────────────────────────────────────────────────────────────────
# CPU TOP-K  (sparse_dot_topn, chunked multi-threaded)
# ──────────────────────────────────────────────────────────────────────────────

def _cpu_topk(
    q_scipy: "sp.csr_matrix",       # (batch × vocab)  scipy CSR float32
    cand_scipy_t: "sp.csr_matrix",   # (vocab × n_cand) scipy CSR float32 (TRANSPOSED)
    max_k: int,
    sim_floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    from sparse_dot_topn import sp_matmul_topn
    
    # We do a direct sparse matmul top-N. It is highly optimized C++ for exactly this.
    res = sp_matmul_topn(
        q_scipy,
        cand_scipy_t,
        top_n=max_k,
        threshold=sim_floor,
        sort=True,
        n_threads=8  # use threads for batch
    )
    
    batch = q_scipy.shape[0]
    scores = np.zeros((batch, max_k), dtype=np.float32)
    indices = np.full((batch, max_k), -1, dtype=np.int32)
    
    for i in range(batch):
        r_start = res.indptr[i]
        r_end = res.indptr[i+1]
        length = min(max_k, r_end - r_start)
        if length > 0:
            scores[i, :length] = res.data[r_start:r_start+length]
            indices[i, :length] = res.indices[r_start:r_start+length]
            
    return scores, indices


# ──────────────────────────────────────────────────────────────────────────────
# PHASE B: CPU BLOCKING (No VRAM limits, C++ accelerated)
# ──────────────────────────────────────────────────────────────────────────────


def phase_b(
    cache_dir: Path,
    output_path: Path,
    query_batch_size: int,
    cand_chunk_size: int,
    max_k: int,
    min_k: int,
    sim_floor: float,
    addr_k: int,
) -> Path:
    """
    For each country:
      1. Load pre-split S1 TSV + transform ALL names in one sklearn call.
      2. Load cached candidate .npz (stays on CPU RAM; chunks uploaded to GPU).
      3. GPU SpGEMM in query batches → merge top-k on CPU.
      4. Apply postal and city-token fallback channels.
      5. Stream results to output_path.
    """
    print("[Phase B] Loading cache …")
    with open(cache_dir / f"countries_{CACHE_VERSION}.pkl", 'rb') as fh:
        countries: List[str] = pickle.load(fh)
    with open(cache_dir / f"vectorizer_{CACHE_VERSION}.pkl", 'rb') as fh:
        vectorizer: TfidfVectorizer = pickle.load(fh)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")

    total_written = 0
    t_global = time.time()

    with open(output_path, 'a', encoding='utf-8') as out_f:
        for ctry in countries:
            ctry_key = ctry if ctry else '_empty_'
            print(f"\n{'='*60}")
            print(f"[Phase B] Country: '{ctry_key}'")

            # ── Load S1 for this country ───────────────────────────────────
            s1_tsv = cache_dir / f"s1_{ctry_key}_{CACHE_VERSION}.tsv"
            if not s1_tsv.exists():
                print(f"  No S1 file found for '{ctry_key}' – skipping.")
                continue

            s1_df   = pd.read_csv(s1_tsv, sep='\t', dtype=str)
            cols    = s1_df.columns.tolist()
            n_col   = _name_col(cols)
            a_col   = _addr_col(cols) if _addr_col(cols) in cols else None

            s1_ids   = s1_df['entity_id'].fillna('').tolist()
            s1_names = s1_df[n_col].fillna('').tolist()
            s1_addrs = (s1_df[a_col].fillna('').tolist()
                        if a_col else [''] * len(s1_ids))
            n_s1 = len(s1_ids)
            print(f"  S1 entities : {n_s1:,}")

            # ── Load candidate meta ────────────────────────────────────────
            meta_path = cache_dir / f"meta_{ctry_key}_{CACHE_VERSION}.pkl"
            if not meta_path.exists():
                print(f"  No candidate meta for '{ctry_key}' – writing empty lines.")
                out_f.writelines(f"{sid}\t\n" for sid in s1_ids)
                total_written += n_s1
                del s1_df
                continue

            with open(meta_path, 'rb') as fh:
                meta = pickle.load(fh)

            cand_ids        = meta['ids']            # list[str]
            cand_pcodes     = meta['pcodes']         # list[str]
            cand_snums      = meta['snums']          # list[str]
            postal_bucket   = meta['postal_bucket']  # {key: [idx, …]}
            nan_city_bucket = meta['nan_city_bucket'] # {city: [eid, …]}
            n_named         = len(cand_ids)
            print(f"  Named cands : {n_named:,}")

            # ── Load candidate sparse matrix (stays in CPU RAM) ────────────
            cand_npz = cache_dir / f"cand_{ctry_key}_{CACHE_VERSION}.npz"
            cand_mat: Optional[sp.csr_matrix] = None
            cand_mat_t: Optional[sp.csr_matrix] = None
            if cand_npz.exists() and n_named > 0:
                print(f"  Loading {cand_npz.name} …")
                cand_mat = sp.load_npz(str(cand_npz))  # scipy CSR float32
                print(f"  Cand matrix : {cand_mat.shape}  nnz={cand_mat.nnz:,}")
                print(f"  Transposing candidate matrix …")
                cand_mat_t = cand_mat.T.tocsr()

            # ── Transform ALL S1 names in a single sklearn call ────────────
            print(f"  Transforming {n_s1:,} S1 names …")
            t0 = time.time()
            s1_mat = vectorizer.transform(s1_names).astype(np.float32)
            print(f"  S1 transform done in {time.time() - t0:.1f}s")

            # ── GPU batched blocking ───────────────────────────────────────
            print(f"  GPU SpGEMM  "
                  f"batch={query_batch_size}  chunk={cand_chunk_size}  max_k={max_k}")
            t0 = time.time()

            for b_start in range(0, n_s1, query_batch_size):
                b_end = min(b_start + query_batch_size, n_s1)
                blen  = b_end - b_start

                # Pre-compute S1 address features for this batch (CPU)
                b_addrs  = s1_addrs[b_start:b_end]
                b_pcodes = [_postal(a)     for a in b_addrs]
                b_snums  = [_street_num(a) for a in b_addrs]
                b_cities = [_city_token(a) for a in b_addrs]

                # CPU top-k from TF-IDF channel using sparse_dot_topn
                if cand_mat_t is not None and n_named > 0:
                    scores_arr, indices_arr = _cpu_topk(
                        s1_mat[b_start:b_end],   # scipy CSR slice (no copy)
                        cand_mat_t,
                        max_k,
                        sim_floor,
                    )
                else:
                    scores_arr  = np.zeros((blen, 0), np.float32)
                    indices_arr = np.zeros((blen, 0), np.int32)

                # Build output lines for this batch
                lines: List[str] = []
                for i in range(blen):
                    seen: Set[str]      = set()
                    selected: List[str] = []

                    # ── Channel 1: postal + street-number exact match ──────
                    pc, sn = b_pcodes[i], b_snums[i]
                    if pc and sn:
                        key = f"{pc}_{sn}"
                        for c_idx in postal_bucket.get(key, [])[:10]:
                            cid = cand_ids[c_idx]
                            if cid not in seen:
                                seen.add(cid)
                                selected.append(cid)

                    # ── Channel 2: TF-IDF cosine top-k ────────────────────
                    n_scores = scores_arr.shape[1]
                    for rank in range(n_scores):
                        score = float(scores_arr[i, rank])
                        c_idx = int(indices_arr[i, rank])

                        if c_idx < 0 or score <= 0.0:
                            # -1 = unfilled slot; 0 = below floor (zeroed in _gpu_topk)
                            continue

                        cid = cand_ids[c_idx]
                        if cid in seen:
                            continue

                        if len(selected) < min_k:
                            # Always fill up to min_k regardless of score
                            seen.add(cid)
                            selected.append(cid)
                        elif len(selected) < max_k and score >= sim_floor:
                            # Expand up to max_k only if score clears the floor
                            seen.add(cid)
                            selected.append(cid)
                        else:
                            # Scores are descending; nothing after this passes
                            break

                        if len(selected) >= max_k:
                            break

                    # ── Channel 3: city-token lookup (NaN-name candidates) ─
                    city = b_cities[i]
                    if city and len(selected) < max_k:
                        for cid in nan_city_bucket.get(city, [])[:addr_k]:
                            if cid not in seen and len(selected) < max_k:
                                seen.add(cid)
                                selected.append(cid)

                    lines.append(
                        f"{s1_ids[b_start + i]}\t{','.join(selected)}\n"
                    )

                out_f.writelines(lines)
                total_written += blen

                # Progress every 20 batches
                batch_num = b_start // query_batch_size
                if batch_num > 0 and batch_num % 20 == 0:
                    elapsed = time.time() - t0
                    rate    = b_start / elapsed if elapsed > 0 else 0
                    print(f"  {b_start + blen:,}/{n_s1:,}  "
                          f"({rate:.0f} q/s, {elapsed:.0f}s elapsed)")

            elapsed = time.time() - t0
            rate    = n_s1 / elapsed if elapsed > 0 else 0
            print(f"  Done {n_s1:,} entities in {elapsed:.1f}s  ({rate:.0f} q/s)")

            # Cleanup before next country
            del s1_df, s1_mat, cand_mat
            gc.collect()
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    total_elapsed = time.time() - t_global
    print(f"\n[Phase B] Done.  {total_written:,} S1 entities → {output_path}")
    print(f"[Phase B] Wall time: {total_elapsed:.1f}s  ({total_elapsed / 60:.1f} min)")
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_recall(candidate_file: Path, gt_path: Path) -> float:
    print(f"\n[Eval] {candidate_file.name}  vs  {gt_path.name}")
    gt = pd.read_csv(gt_path, sep='\t', dtype=str)

    gt_map: Dict[str, Set[str]] = {}
    for _, row in gt.iterrows():
        ms = row.get('matched_entity_ids', '')
        gt_map[row['source1_entity_id']] = (
            set(m.strip() for m in str(ms).split(',') if m.strip())
            if pd.notna(ms) else set()
        )

    total_true = sum(len(v) for v in gt_map.values())
    captured = total_cands = total_q = 0

    with open(candidate_file, 'r', encoding='utf-8') as fh:
        fh.readline()   # skip header
        for line in fh:
            parts = line.strip().split('\t')
            if not parts:
                continue
            s1_id = parts[0]
            raw   = parts[1] if len(parts) > 1 else ''
            cands = set(c.strip() for c in raw.split(',') if c.strip())
            total_cands += len(cands)
            total_q     += 1
            if s1_id in gt_map:
                captured += len(gt_map[s1_id] & cands)

    recall = captured / total_true if total_true else 1.0
    print("=" * 56)
    print(f"  Recall        : {recall * 100:.2f}%")
    print(f"  Captured      : {captured:,} / {total_true:,} true pairs")
    print(f"  Avg cands/S1  : {total_cands / total_q:.1f}")
    print("=" * 56)
    return recall


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="blocking2: Two-Phase GPU-Accelerated Candidate Blocking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument("--s1",           required=True,
                        help="Path to preprocessed S1 TSV")
    parser.add_argument("--s2",           required=True,
                        help="Path to preprocessed S2 TSV")
    parser.add_argument("--s3",           required=True,
                        help="Path to preprocessed S3 TSV")
    parser.add_argument("--output",       default="output/candidate_pairs.tsv")
    parser.add_argument("--cache-dir",    default="output/blocking2_cache",
                        help="Directory for Phase A artefacts")
    parser.add_argument("--ground-truth", default=None,
                        help="Optional GT TSV for recall evaluation after blocking")

    # Blocking quality
    parser.add_argument("--min-k",    type=int,   default=50,
                        help="Minimum candidates per S1 entity")
    parser.add_argument("--max-k",    type=int,   default=100,
                        help="Maximum candidates per S1 entity")
    parser.add_argument("--floor",    type=float, default=0.08,
                        help="Similarity floor: expand beyond min-k only if score ≥ floor")
    parser.add_argument("--addr-k",   type=int,   default=ADDR_K_DEFAULT,
                        help="Max address-channel (NaN-name) candidates per S1 entity")

    # VRAM tuning  ─  peak VRAM ≈ batch-size × cand-chunk × 4 bytes
    parser.add_argument("--batch-size",  type=int, default=3000,
                        help="S1 query batch size.  3000×50k×4 ≈ 600 MB peak VRAM")
    parser.add_argument("--cand-chunk",  type=int, default=50_000,
                        help="Candidate chunk size uploaded per GPU call")

    # Vectorizer  (Phase A only; ignored if cache exists)
    parser.add_argument("--ngram-min", type=int,   default=3)
    parser.add_argument("--ngram-max", type=int,   default=4)
    parser.add_argument("--min-df",    type=int,   default=2,
                        help="min_df for TF-IDF")
    parser.add_argument("--max-df",    type=float, default=0.95,
                        help="max_df for TF-IDF.  0.95 retains legal-form n-grams")

    parser.add_argument("--rebuild",   action='store_true',
                        help="Force a full Phase A rebuild even if cache exists")

    args = parser.parse_args()

    s1_p  = Path(args.s1)
    s2_p  = Path(args.s2)
    s3_p  = Path(args.s3)
    cache = Path(args.cache_dir)
    out_p = Path(args.output)

    t_start = time.time()

    # ── Phase A ──────────────────────────────────────────────────────────────
    vec_cache  = cache / f"vectorizer_{CACHE_VERSION}.pkl"
    ctry_cache = cache / f"countries_{CACHE_VERSION}.pkl"

    if args.rebuild or not vec_cache.exists() or not ctry_cache.exists():
        print("\n" + "=" * 60)
        print(" PHASE A: Building global index cache")
        print("=" * 60)
        t_a = time.time()
        phase_a(
            s1_path=s1_p, s2_path=s2_p, s3_path=s3_p,
            cache_dir=cache,
            ngram_range=(args.ngram_min, args.ngram_max),
            min_df=args.min_df,
            max_df=args.max_df,
        )
        print(f"[Phase A] Completed in {(time.time() - t_a) / 60:.1f} min")
    else:
        print(f"[Phase A] Cache found at '{cache}' – skipping.  "
              f"Use --rebuild to force a full rebuild.")

    # ── Phase B ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" PHASE B: GPU blocking")
    print("=" * 60)
    t_b = time.time()
    phase_b(
        cache_dir=cache,
        output_path=out_p,
        query_batch_size=args.batch_size,
        cand_chunk_size=args.cand_chunk,
        max_k=args.max_k,
        min_k=args.min_k,
        sim_floor=args.floor,
        addr_k=args.addr_k,
    )
    print(f"[Phase B] Completed in {(time.time() - t_b) / 60:.1f} min")

    total = time.time() - t_start
    print(f"\nTotal wall time: {total:.1f}s  ({total / 60:.1f} min)")

    if args.ground_truth:
        evaluate_recall(out_p, Path(args.ground_truth))


if __name__ == "__main__":
    main()
