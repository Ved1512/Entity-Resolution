import cupy as cp
import numpy as np

batch_len = 5
chunk_cols = 1000
chunk_sims_dense = cp.random.rand(batch_len, chunk_cols)

k = 60
partitioned_idx = cp.argpartition(-chunk_sims_dense, k-1, axis=1)[:, :k]
row_idx = cp.arange(batch_len)[:, None]
partitioned_scores = chunk_sims_dense[row_idx, partitioned_idx]

sort_order = cp.argsort(-partitioned_scores, axis=1)
top_idx = partitioned_idx[row_idx, sort_order]
top_scores = partitioned_scores[row_idx, sort_order]

print(top_scores.shape)
print(top_idx.shape)
print(top_scores[0, :5])

