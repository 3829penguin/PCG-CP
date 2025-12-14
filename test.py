import pcgsolver
import numpy as np
import time

solver = pcgsolver.PCG()

# 讀取矩陣與向量
solver.parse_A("../Matrix.mtx")
solver.parse_b("../b.mtx")
solver.check_symmetry()
print("Matrix and vector loaded.")

# =========================
# Jacobi PCG
# =========================
t0 = time.perf_counter()
x_jacobi = solver.solve(
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
x_ssor = solver.solve(
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
x_ic0 = solver.solve(
    max_iter=2000,
    tol=1e-12,
    precond="ic0"
)
t1 = time.perf_counter()

print("IC(0) PCG solution:", np.array(x_ic0))
print(f"IC(0) PCG time: {t1 - t0:.6f} seconds")
