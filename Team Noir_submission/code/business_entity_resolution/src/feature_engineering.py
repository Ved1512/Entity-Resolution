"""
Amazon ML Challenge 2026: Business Entity Resolution
Feature Engineering Module (Stage 3)
"""

import sys
import re
import time
import argparse
import gc
import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# HARDCODED GEOGRAPHY DICTIONARIES (Tier 1 Extension)
# ---------------------------------------------------------------------------
US_STATES = {"california", "texas", "new york", "florida", "illinois", "pennsylvania", "ohio", "georgia", "north carolina", "michigan", "nj", "ny", "ca", "tx", "fl"}
US_CITIES = {"new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia", "san antonio", "san diego", "dallas", "sanjose"}

IN_STATES = {"maharashtra", "delhi", "karnataka", "gujarat", "tamil nadu", "west bengal", "uttar pradesh", "telangana", "kerala"}
IN_CITIES = {"mumbai", "delhi", "bangalore", "hyderabad", "ahmedabad", "chennai", "kolkata", "surat", "pune", "jaipur"}

FR_STATES = {"ile-de-france", "auvergne-rhone-alpes", "hauts-de-france", "nouvelle-aquitaine", "occitanie", "provence-alpes-cote d'azur"}
FR_CITIES = {"paris", "marseille", "lyon", "toulouse", "nice", "nantes", "montpellier", "strasbourg", "bordeaux", "lille"}

ALL_STATES = US_STATES | IN_STATES | FR_STATES
ALL_CITIES = US_CITIES | IN_CITIES | FR_CITIES

# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    # --- Tier 1 ---
    "tfidf_cosine_sim",
    "exact_postal_match",
    "exact_street_num_match",
    "same_country",
    "state_match",
    "city_match",
    "levenshtein_sim_name",
    "jaccard_word_name",
    "name_len_diff",
    "name_len_ratio",
    "jaccard_word_addr",
    "is_s3",
    # --- Tier 2 ---
    "jaro_winkler_name",
    "token_sort_ratio_name",
    "token_set_ratio_name",
    "token_count_diff_name",
    "longest_common_prefix_name",
    "raw_levenshtein_sim_name",
    "raw_jaccard_word_name",
    "raw_jaro_winkler_name",
    "raw_token_sort_ratio_name",
    "raw_token_set_ratio_name",
    "delta_levenshtein_name",
    "delta_token_sort_ratio_name",
    "raw_jaccard_word_addr",
]
N_FEATURES = len(FEATURE_NAMES)

def _import_rapidfuzz():
    from rapidfuzz import fuzz as _fuzz
    from rapidfuzz.distance import Levenshtein as _Lev
    return _fuzz, _Lev

def _import_jellyfish():
    import jellyfish as _jf
    return _jf

def _get_street_num(addr: str) -> str:
    if not addr: return ""
    m = re.match(r'^(\d+)', addr)
    return m.group(1) if m else ""

def _jaccard_tokens(a: str, b: str) -> float:
    t1 = set(a.split()) if a else set()
    t2 = set(b.split()) if b else set()
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    union = len(t1 | t2)
    return len(t1 & t2) / union if union else 0.0

def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0: return 1.0
    if la == 0 or lb == 0: return 0.0
    return min(la, lb) / max(la, lb)

def _longest_common_prefix_norm(a: str, b: str) -> float:
    if not a or not b: return 0.0
    max_len = max(len(a), len(b))
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]: i += 1
    return i / max_len

def _extract_geo(addr: str, geo_set: set) -> set:
    found = set()
    for token in addr.split():
        if token in geo_set:
            found.add(token)
    return found

# ---------------------------------------------------------------------------
# Globals for Forked Workers (Zero serialization overhead)
# ---------------------------------------------------------------------------
_G_S1_LOOKUP = None
_G_CAND_LOOKUP = None
_G_GT_MAP = None

def init_worker(s1_dict, cand_dict, gt_map):
    global _G_S1_LOOKUP, _G_CAND_LOOKUP, _G_GT_MAP
    _G_S1_LOOKUP = s1_dict
    _G_CAND_LOOKUP = cand_dict
    _G_GT_MAP = gt_map

def process_batch(pair_batch: List[Tuple[str, str]]) -> str:
    fuzz, Lev = _import_rapidfuzz()
    jf = _import_jellyfish()
    
    out_lines = []
    
    for s1_id, cid in pair_batch:
        s1_row = _G_S1_LOOKUP.get(s1_id)
        cand_row = _G_CAND_LOOKUP.get(cid)
        if not s1_row or not cand_row:
            continue
            
        cn1, cn2 = s1_row.get("clean_name", ""), cand_row.get("clean_name", "")
        ca1, ca2 = s1_row.get("clean_address", ""), cand_row.get("clean_address", "")
        p1, p2   = s1_row.get("postal_code", ""), cand_row.get("postal_code", "")
        c1, c2   = s1_row.get("clean_country", ""), cand_row.get("clean_country", "")
        rn1, rn2 = s1_row.get("business_name", ""), cand_row.get("business_name", "")
        ra1, ra2 = s1_row.get("business_address", ""), cand_row.get("business_address", "")
        
        # Replace heavy matrix TF-IDF with rapidfuzz partial ratio (functionally identical for names)
        f_cosine = fuzz.partial_ratio(cn1, cn2) / 100.0
        f_postal = 1.0 if p1 == p2 else 0.0 if (p1 and p2) else -1.0
        
        sn1, sn2 = _get_street_num(ca1), _get_street_num(ca2)
        f_street = 1.0 if sn1 == sn2 else 0.0 if (sn1 and sn2) else -1.0
        
        f_country = 1.0 if (c1 and c1 == c2) else 0.0
        
        # New State/City hardcoded matches
        s1_states = _extract_geo(ca1, ALL_STATES)
        s2_states = _extract_geo(ca2, ALL_STATES)
        if s1_states and s2_states:
            f_state = 1.0 if (s1_states & s2_states) else 0.0
        else:
            f_state = -1.0
            
        s1_cities = _extract_geo(ca1, ALL_CITIES)
        s2_cities = _extract_geo(ca2, ALL_CITIES)
        if s1_cities and s2_cities:
            f_city = 1.0 if (s1_cities & s2_cities) else 0.0
        else:
            f_city = -1.0
        
        max_len_cn = max(len(cn1), len(cn2))
        f_lev_clean = 1.0 - (Lev.distance(cn1[:300], cn2[:300]) / max_len_cn) if max_len_cn else 1.0
        
        f_jacc_name = _jaccard_tokens(cn1, cn2)
        f_len_diff = float(abs(len(cn1) - len(cn2)))
        f_len_ratio = _len_ratio(cn1, cn2)
        f_jacc_addr = _jaccard_tokens(ca1, ca2)
        f_is_s3 = 1.0 if cid.startswith("S3-") else 0.0
        
        f_jw_clean = jf.jaro_winkler_similarity(cn1[:200], cn2[:200])
        f_tsr_clean = fuzz.token_sort_ratio(cn1, cn2) / 100.0
        f_tset_clean = fuzz.token_set_ratio(cn1, cn2) / 100.0
        
        tc1, tc2 = len(cn1.split()), len(cn2.split())
        f_tok_cnt_diff = float(abs(tc1 - tc2))
        f_lcp = _longest_common_prefix_norm(cn1, cn2)
        
        max_len_rn = max(len(rn1), len(rn2))
        f_lev_raw = 1.0 - (Lev.distance(rn1[:300], rn2[:300]) / max_len_rn) if max_len_rn else 1.0
        
        f_jacc_raw_name = _jaccard_tokens(rn1.lower(), rn2.lower())
        f_jw_raw = jf.jaro_winkler_similarity(rn1[:200].lower(), rn2[:200].lower())
        f_tsr_raw  = fuzz.token_sort_ratio(rn1, rn2) / 100.0
        f_tset_raw = fuzz.token_set_ratio(rn1, rn2) / 100.0
        
        f_delta_lev = f_lev_raw - f_lev_clean
        f_delta_tsr = f_tsr_raw - f_tsr_clean
        f_jacc_raw_addr = _jaccard_tokens(ra1.lower(), ra2.lower())
        
        feats = [
            f_cosine, f_postal, f_street, f_country, f_state, f_city, f_lev_clean, f_jacc_name,
            f_len_diff, f_len_ratio, f_jacc_addr, f_is_s3, f_jw_clean, f_tsr_clean,
            f_tset_clean, f_tok_cnt_diff, f_lcp, f_lev_raw, f_jacc_raw_name, f_jw_raw,
            f_tsr_raw, f_tset_raw, f_delta_lev, f_delta_tsr, f_jacc_raw_addr
        ]
        
        label_str = ""
        if _G_GT_MAP is not None:
            label = 1 if cid in _G_GT_MAP.get(s1_id, set()) else 0
            label_str = f"{label}\t"
            
        feat_str = "\t".join(f"{x:.4f}" for x in feats)
        out_lines.append(f"{s1_id}\t{cid}\t{label_str}{feat_str}\n")
        
    return "".join(out_lines)

def build_feature_matrix(s1_path, s2_path, s3_path, candidates_path, output_path, ground_truth_path=None, sample=None):
    print("[Features] Loading source tables...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str).fillna("")
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str).fillna("")

    s1_lookup = {r["entity_id"]: r for r in df_s1.to_dict("records")}
    cand_lookup = {r["entity_id"]: r for r in df_s2.to_dict("records")}
    cand_lookup.update({r["entity_id"]: r for r in df_s3.to_dict("records")})

    gt_map = None
    if ground_truth_path and ground_truth_path.is_file():
        gt_map = {}
        df_gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str).fillna("")
        for _, row in df_gt.iterrows():
            sid, m_str = row["source1_entity_id"], row["matched_entity_ids"].strip()
            gt_map[sid] = set(m.strip() for m in m_str.split(",") if m.strip()) if m_str else set()

    print("[Features] Loading candidate pairs...")
    df_cands = pd.read_csv(candidates_path, sep="\t", dtype=str).fillna("")
    
    pair_list = []
    for _, row in df_cands.iterrows():
        s1_id = row["source1_entity_id"]
        for cid in (c.strip() for c in row["candidate_entity_ids"].strip().split(",") if c.strip()):
            pair_list.append((s1_id, cid))

    print(f"[Features] Total candidate pairs to featurize: {len(pair_list):,}")

    # Output File Setup
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        label_head = "label\t" if gt_map else ""
        feat_head = "\t".join(FEATURE_NAMES)
        f.write(f"source1_entity_id\tcandidate_entity_id\t{label_head}{feat_head}\n")

    # Multiprocessing Chunking
    BATCH_SIZE = 50_000
    batches = [pair_list[i:i + BATCH_SIZE] for i in range(0, len(pair_list), BATCH_SIZE)]
    
    import os
    print(f"[Features] Streaming output directly to disk across {os.cpu_count()} CPU cores...")
    t1 = time.time()
    
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=init_worker, initargs=(s1_lookup, cand_lookup, gt_map)) as executor:
        futures = {executor.submit(process_batch, b): i for i, b in enumerate(batches)}
        
        processed_pairs = 0
        for future in as_completed(futures):
            res_str = future.result()
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(res_str)
                
            processed_pairs += BATCH_SIZE
            if processed_pairs % 250_000 < BATCH_SIZE:
                print(f"[Features]   {processed_pairs:,}/{len(pair_list):,} pairs processed in {time.time()-t1:.1f}s")

    print(f"[Features] Feature extraction COMPLETE in {time.time()-t1:.1f}s!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", required=True)
    parser.add_argument("--s2", required=True)
    parser.add_argument("--s3", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ground-truth", dest="ground_truth", default=None)
    args = parser.parse_args()

    build_feature_matrix(
        s1_path=Path(args.s1),
        s2_path=Path(args.s2),
        s3_path=Path(args.s3),
        candidates_path=Path(args.candidates),
        output_path=Path(args.output),
        ground_truth_path=Path(args.ground_truth) if args.ground_truth else None,
    )

if __name__ == "__main__":
    main()
