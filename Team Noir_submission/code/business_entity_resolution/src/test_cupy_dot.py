import cupy as cp
import cupyx.scipy.sparse as cupy_sparse
import numpy as np
import scipy.sparse as sp

A = cp.random.rand(5, 10).astype(cp.float32)
B_cpu = sp.random(10, 20, density=0.1, format='csr', dtype=np.float32)
B = cupy_sparse.csr_matrix(B_cpu)

try:
    C = A @ B
    print("A @ B Works. shape:", C.shape)
except Exception as e:
    print("A @ B Failed:", e)

try:
    C = B.T.dot(A.T).T
    print("B.T.dot(A.T).T Works. shape:", C.shape)
except Exception as e:
    print("B.T.dot(A.T).T Failed:", e)

