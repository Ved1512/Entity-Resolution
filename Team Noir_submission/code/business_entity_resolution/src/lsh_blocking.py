import os
import time
import gc
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import concurrent.futures
import mmh3

def process_chunk(texts, n_gram_size=2):
    """
    Runs in a separate CPU process. Bypasses GIL.
    Converts text to n-grams, hashes them, and pads arrays to perfectly fit GPU tensors.
    """
    hashed_list = []
    max_len = 0
    
    for text in texts:
        text = str(text).lower()
        if len(text) < n_gram_size:
            if not text:
                hashed_list.append([0])
            else:
                hashed_list.append([mmh3.hash(text, seed=42, signed=False)])
            continue
            
        # Generate Character N-grams
        ngrams = [text[i:i+n_gram_size] for i in range(len(text) - n_gram_size + 1)]
        
        hashed = [mmh3.hash(ng, seed=42, signed=False) for ng in set(ngrams)]
        if len(hashed) > max_len:
            max_len = len(hashed)
        hashed_list.append(hashed)
        
    padded = np.zeros((len(texts), max_len), dtype=np.uint32)
    for i, h in enumerate(hashed_list):
        padded[i, :len(h)] = h
        
    return padded

# Numba optimized MinHash for GPU/CPU fallback
try:
    from numba import cuda
    HAS_CUDA = cuda.is_available()
    if HAS_CUDA:
        @cuda.jit
        def _gpu_minhash_kernel(data, a, b, prime, out):
            row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if row < data.shape[0]:
                for k in range(a.shape[0]):
                    min_h = 4294967295
                    a_k = a[k]
                    b_k = b[k]
                    for col in range(data.shape[1]):
                        val = data[row, col]
                        if val == 0 and col > 0:
                            break
                        h = (a_k * val + b_k) % prime
                        if h < min_h:
                            min_h = h
                    out[row, k] = min_h
except ImportError:
    HAS_CUDA = False

def gpu_minhash(data_padded, a, b, prime):
    K = len(a)
    num_rows = data_padded.shape[0]
    out = np.empty((num_rows, K), dtype=np.uint32)
    
    if HAS_CUDA:
        try:
            from numba import cuda
            threadsperblock = 256
            blockspergrid = (num_rows + (threadsperblock - 1)) // threadsperblock
            
            d_data = cuda.to_device(data_padded)
            d_a = cuda.to_device(a)
            d_b = cuda.to_device(b)
            d_out = cuda.device_array((num_rows, K), dtype=np.uint32)
            
            _gpu_minhash_kernel[blockspergrid, threadsperblock](d_data, d_a, d_b, prime, d_out)
            
            out = d_out.copy_to_host()
            return out
        except Exception as e:
            print(f"CUDA Failed: {e}. Falling back to CPU...")
            
    for k in range(K):
        hashes = (data_padded.astype(np.uint64) * a[k] + b[k]) % prime
        hashes[data_padded == 0] = 4294967295
        if data_padded.shape[1] > 0:
            out[:, k] = np.min(hashes, axis=1)
        else:
            out[:, k] = 4294967295
            
    return out

try:
    from numba import njit
    @njit
    def sliding_window_bipartite(sorted_indices, source_tags, window_size):
        n = len(sorted_indices)
        est_pairs = n * window_size
        out_s1 = np.empty(est_pairs, dtype=np.int32)
        out_cand = np.empty(est_pairs, dtype=np.int32)
        count = 0
        
        for i in range(n):
            idx_i = sorted_indices[i]
            tag_i = source_tags[idx_i]
            
            end_j = min(i + window_size, n)
            for j in range(i + 1, end_j):
                idx_j = sorted_indices[j]
                tag_j = source_tags[idx_j]
                
                if tag_i != tag_j:
                    if tag_i == 1:
                        out_s1[count] = idx_i
                        out_cand[count] = idx_j
                        count += 1
                    elif tag_j == 1:
                        out_s1[count] = idx_j
                        out_cand[count] = idx_i
                        count += 1
                        
        return out_s1[:count], out_cand[:count]
except ImportError:
    def sliding_window_bipartite(sorted_indices, source_tags, window_size):
        pass

if __name__ == "__main__":
    DATA_DIR = Path(r"/home/rover/Amazon ML challenge - Copy/_student_resource/student_resource/dataset/train/Preprocessed") 
    OUTPUT_DIR = Path(r"../../output")
    
    TEXT_COLUMN = "clean_name"
    ADDR_COLUMN = "clean_address"
    CHUNK_SIZE = 15_000
    K = 200
    WINDOW_SIZE = 15
    
    print("Loading datasets efficiently to prevent RAM spikes...")
    cols_to_load = ["entity_id", TEXT_COLUMN, ADDR_COLUMN]
    texts_list, tags_list, ids_list = [], [], []
    all_s1_ids = None
    
    for tag_val, fname in [(1, "train_source1_clean.tsv"), (2, "train_source2_clean.tsv"), (3, "train_source3_clean.tsv")]:
        file_path = DATA_DIR / fname
        print(f"  Reading {fname}...")
        df_temp = pd.read_csv(file_path, sep='\t', usecols=cols_to_load)
        
        composite_text = df_temp[TEXT_COLUMN].fillna("").astype(str) + " " + df_temp[ADDR_COLUMN].fillna("").astype(str)
        texts_list.append(composite_text.values)
        ids_list.append(df_temp['entity_id'].values)
        tags_list.append(np.full(len(df_temp), tag_val, dtype=np.int8))
        if tag_val == 1:
            all_s1_ids = df_temp['entity_id'].values.copy()
        del df_temp
        gc.collect()

    print("Concatenating lightweight arrays...")
    texts = np.concatenate(texts_list)
    source_tags = np.concatenate(tags_list)
    entity_ids = np.concatenate(ids_list)
    num_rows = len(texts)
    del texts_list, tags_list, ids_list
    gc.collect()
    
    print(f"Total Combined Rows: {num_rows}")
    np.random.seed(42)
    prime = 4294967291
    a = np.random.randint(1, prime, size=K, dtype=np.uint32)
    b = np.random.randint(0, prime, size=K, dtype=np.uint32)
    
    mmap_path = Path(tempfile.gettempdir()) / "lsh_sigs.mmap"
    final_signatures = np.memmap(mmap_path, dtype=np.uint32, mode='w+', shape=(num_rows, K))
    start_time = time.time()
    
    chunks = [texts[i:i+CHUNK_SIZE] for i in range(0, num_rows, CHUNK_SIZE)]
    print(f"Processing {num_rows} rows in {len(chunks)} chunks across CPU/GPU...")
    MAX_WORKERS = os.cpu_count() or 4
    MAX_ACTIVE = MAX_WORKERS * 2
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        active_futures = {}
        chunk_iter = iter(enumerate(chunks))
        
        for _ in range(MAX_ACTIVE):
            try:
                idx, chunk_data = next(chunk_iter)
                active_futures[executor.submit(process_chunk, chunk_data)] = idx
            except StopIteration:
                break
                
        processed_count = 0
        while active_futures:
            done, _ = concurrent.futures.wait(active_futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                idx = active_futures.pop(future)
                padded_chunk = future.result()
                processed_count += 1
                print(f"  Sending Chunk {processed_count}/{len(chunks)} to GPU...")
                
                gpu_sigs = gpu_minhash(padded_chunk, a, b, prime)
                start_row = idx * CHUNK_SIZE
                final_signatures[start_row : start_row + gpu_sigs.shape[0]] = gpu_sigs
                del padded_chunk, gpu_sigs
                
                try:
                    next_idx, next_chunk_data = next(chunk_iter)
                    active_futures[executor.submit(process_chunk, next_chunk_data)] = next_idx
                except StopIteration:
                    pass
            
    print(f"Signature Generation Complete in {time.time() - start_time:.2f} seconds.")
    del texts, chunks
    gc.collect()
    
    BANDS = 40
    ROWS_PER_BAND = 5
    NUM_BUCKETS = 10
    bucket_paths = [Path(tempfile.gettempdir()) / f"pairs_bucket_{i}.bin" for i in range(NUM_BUCKETS)]
    
    for p in bucket_paths:
        if p.exists(): p.unlink()
        
    print(f"Starting {BANDS} LSH Bands (Rows per band: {ROWS_PER_BAND}, Window: {WINDOW_SIZE})...")
    lsh_start_time = time.time()
    
    for band in range(BANDS):
        band_start = time.time()
        start_col = band * ROWS_PER_BAND
        end_col = start_col + ROWS_PER_BAND
        
        sort_keys = tuple(final_signatures[:, i] for i in range(end_col - 1, start_col - 1, -1))
        sorted_indices = np.lexsort(sort_keys).astype(np.int32)
        idx_s1, idx_cand = sliding_window_bipartite(sorted_indices, source_tags, WINDOW_SIZE)
        
        pass_pairs = (idx_s1.astype(np.uint64) << 32) | idx_cand.astype(np.uint64)
        pass_pairs = np.unique(pass_pairs)
        
        s1_indices = (pass_pairs >> 32).astype(np.uint32)
        bucket_assignments = s1_indices % NUM_BUCKETS
        
        for i in range(NUM_BUCKETS):
            mask = (bucket_assignments == i)
            if np.any(mask):
                bucket_pairs = pass_pairs[mask]
                with open(bucket_paths[i], "ab") as f:
                    f.write(bucket_pairs.tobytes())
                    
        del sort_keys, sorted_indices, idx_s1, idx_cand, pass_pairs, s1_indices, bucket_assignments
        gc.collect()
            
        print(f"  Band {band+1}/{BANDS} processed in {time.time() - band_start:.2f}s.")

    del final_signatures
    if mmap_path.exists():
        mmap_path.unlink()
    gc.collect()
    
    print(f"All {BANDS} Bands Complete in {time.time() - lsh_start_time:.2f} seconds.")
    print("Formatting output directly to disk to preserve RAM...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "candidate_pairs.tsv"
    
    with open(out_path, "w", encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        
    processed_s1_idx = set()
    total_unique_pairs = 0
    
    for i in range(NUM_BUCKETS):
        if not bucket_paths[i].exists():
            continue
            
        bucket_pairs = np.fromfile(bucket_paths[i], dtype=np.uint64)
        bucket_pairs = np.unique(bucket_pairs)
        total_unique_pairs += len(bucket_pairs)
        
        if len(bucket_pairs) == 0:
            bucket_paths[i].unlink()
            continue
            
        s1_array = (bucket_pairs >> 32).astype(np.int32)
        cand_array = (bucket_pairs & 0xFFFFFFFF).astype(np.int32)
        del bucket_pairs
        
        u_s1, boundaries = np.unique(s1_array, return_index=True)
        num_unique_s1 = len(u_s1)
        
        out_rows = []
        for j in range(num_unique_s1):
            start = boundaries[j]
            end = boundaries[j+1] if j + 1 < num_unique_s1 else len(s1_array)
            s1_idx = u_s1[j]
            processed_s1_idx.add(s1_idx)
            
            cands_indices = cand_array[start:end]
            s1_str = entity_ids[s1_idx]
            cands_str = ",".join(entity_ids[cands_indices])
            out_rows.append(f"{s1_str}\t{cands_str}\n")
            
        with open(out_path, "a", encoding='utf-8') as f_out:
            f_out.writelines(out_rows)
            
        bucket_paths[i].unlink()
        del s1_array, cand_array, u_s1, boundaries, out_rows
        gc.collect()
        
    print(f"Total Unique Pairs Found: {total_unique_pairs}")
    print("Writing singletons...")
    singleton_rows = []
    
    for idx, s1_str in enumerate(all_s1_ids):
        if idx not in processed_s1_idx:
            singleton_rows.append(f"{s1_str}\t\n")
        if len(singleton_rows) >= 1_000_000:
            with open(out_path, "a", encoding='utf-8') as f_out:
                f_out.writelines(singleton_rows)
            singleton_rows = []
            
    if singleton_rows:
        with open(out_path, "a", encoding='utf-8') as f_out:
            f_out.writelines(singleton_rows)
    
    print(f"Total Pipeline Runtime: {time.time() - start_time:.2f} seconds.")
    print(f"Saved exactly formatted challenge submission to: {out_path}")
    
    import subprocess
    print("\nTriggering memory-safe evaluation script...")
    try:
        subprocess.run(["python3", "eval_recall.py"], check=True)
    except Exception as e:
        print(f"Failed to run eval_recall.py automatically: {e}")