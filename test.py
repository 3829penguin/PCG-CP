import pcgsolver
import numpy as np

solver = pcgsolver.PCG()

# 讀取矩陣與向量
solver.parse_A("../Matrix.mtx")
solver.parse_b("../b.mtx")
solver.check_symmetry()
print("Matrix and vector loaded.")

# Jacobi PCG
x_jacobi = solver.solve(
    max_iter=2000,
    tol=1e-10,
    precond="jacobi"
)
print("Jacobi PCG:", np.array(x_jacobi))

# SSOR (需要 omega)
x_ssor = solver.solve(
    max_iter=2000,
    tol=1e-10,
    precond="ssor",
    omega=1.3
)
print("SSOR PCG:", np.array(x_ssor))

# IC(0) PCG
x_ic0 = solver.solve(
    max_iter=2000,
    tol=1e-12,
    precond="ic0"
)
print("IC(0) PCG:", np.array(x_ic0))
