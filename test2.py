import pcgsolver
import numpy as np
import time
from scipy.io import mmread
from scipy.sparse import csr_matrix

# =========================
# Load Matrix and Vector
# =========================
A = mmread("../Matrix.mtx").tocsr()
b = mmread("../b.mtx").ravel()


print("Matrix and vector loaded.")
print(f"A shape = {A.shape}, nnz = {A.nnz}")

solver = pcgsolver.PCG()

# =========================
# Jacobi PCG
# =========================
t0 = time.perf_counter()
x_jacobi = solver.solve_csr(
    A.indptr,
    A.indices,
    A.data,
    b,
    max_iter=10000,
    tol=1e-12,
    precond="jacobi"
)
t1 = time.perf_counter()

print("Jacobi PCG solution:", np.array(x_jacobi))
print(f"Jacobi PCG time: {t1 - t0:.6f} seconds")

# =========================
# SSOR PCG
# =========================
t0 = time.perf_counter()
x_ssor = solver.solve_csr(
    A.indptr,
    A.indices,
    A.data,
    b,
    max_iter=2000,
    tol=1e-12,
    precond="ssor",
    omega=1.3
)
t1 = time.perf_counter()

print("SSOR PCG solution:", np.array(x_ssor))
print(f"SSOR PCG time: {t1 - t0:.6f} seconds")

# =========================
# IC(0) PCG
# =========================
t0 = time.perf_counter()
x_ic0 = solver.solve_csr(
    A.indptr,
    A.indices,
    A.data,
    b,
    max_iter=2000,
    tol=1e-12,
    precond="ic0"
)
t1 = time.perf_counter()

print("IC(0) PCG solution:", np.array(x_ic0))
print(f"IC(0) PCG time: {t1 - t0:.6f} seconds")
