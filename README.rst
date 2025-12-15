PCG-CP: PCG Solver with Configurable Preconditioners
===================================================

Basic Information
=================

**PCG-CP (Preconditioned Conjugate Gradient – Configurable Preconditioners)**
is a C++ library designed to efficiently solve large-scale sparse linear systems
that typically arise from finite-difference modeling of thermal problems in
advanced IC and package designs.

It provides a unified interface for the Conjugate Gradient (CG) solver enhanced
with configurable preconditioners, enabling faster convergence on
**symmetric positive definite (SPD)** systems.

Repository
----------

https://github.com/3829penguin/PCG-CP

Problem Statement
=================

In advanced process nodes and 2.5D/3D integrated packages, the rapid increase in
power density gives rise to severe local hotspots and intricate inter-layer heat
transfer phenomena. These effects must be captured using high-resolution
finite-difference (FDM) models to ensure thermal accuracy.

However, FDM generates extremely large grid systems, leading to very large sparse
linear equations. Classical iterative solvers such as Conjugate Gradient (CG)
suffer from long runtimes and slow convergence, especially on ill-conditioned
systems.

To address this challenge, this project implements the
**Preconditioned Conjugate Gradient (PCG)** method, which applies preconditioning
techniques to CG to significantly accelerate convergence and reduce
computational cost.

Target Users
============

This project targets:

- Chip and system designers requiring accurate and efficient thermal verification
- Engineers performing early design exploration and sign-off analysis
- Users who need to solve large sparse linear systems efficiently to ensure
  thermal reliability in advanced integrated packages

System Overview
===============

The solver is designed to handle sparse linear systems with matrix sizes ranging
from **10⁵ × 10⁵ to 10⁷ × 10⁷**, corresponding to several hundred thousand to
several million unknowns.

Inputs
------

1. Sparse matrix :math:`A`
2. Right-hand side vector :math:`b`
3. Preconditioner type
4. Maximum number of iterations
5. Convergence tolerance

Output
------

- Approximate solution vector :math:`x`

Constraints
-----------

- The matrix :math:`A` must be **Symmetric Positive Definite (SPD)**

Input File Format
=================

The input matrix must be provided in **Matrix Market coordinate format**
(``.mtx``). The solver expects a **real-valued sparse matrix** stored in
COO (coordinate) form.

File Structure
--------------

::

    %%MatrixMarket matrix coordinate real general
    % Optional comments
    <row> <col> <nnz>
    <i> <j> <value>
    <i> <j> <value>
    ...

Header
------

::

    %%MatrixMarket matrix coordinate real general

- ``matrix`` : matrix data
- ``coordinate`` : coordinate (COO) storage format
- ``real`` : real-valued entries
- ``general`` : general matrix format

Matrix Size and Nonzeros
------------------------

::

    570000 570000 3947200

- First number: number of rows
- Second number: number of columns
- Third number: number of nonzero entries

Matrix Entries
--------------

Each subsequent line represents a **nonzero entry** of the matrix:

::

    row_index  column_index  value

Example
~~~~~~~

::

    1 1 0.000325554
    1 2 -3e-06
    1 101 -3e-06
    1 10001 -0.0003

- Indices are **1-based** (Matrix Market convention)
- Values are floating-point numbers
- Each row typically contains nonzero entries corresponding to
  **locally coupled neighboring variables**, common in grid-based
  finite-difference discretizations

Notes
-----

- The matrix is expected to be **large and sparse**
- The sparsity pattern typically reflects **local coupling** between
  neighboring degrees of freedom
- Such matrices are well suited for
  **Preconditioned Conjugate Gradient (PCG)**

Right-Hand Side Vector
----------------------

The right-hand side vector :math:`b` must also be provided in Matrix Market
format:

::

    %%MatrixMarket matrix array real general
    <row>
    <value>
    <value>
    ...

Python API Description
======================

Import
------

::

    import pcgsolver

Parsing CSR Matrix
------------------

The PCG solver provides functions to parse matrices and vectors stored in
Matrix Market format and convert them into CSR format.

::

    solver = pcgsolver.PCG()
    solver.parse_A("Matrix.mtx")
    solver.parse_b("b.mtx")

Solving Linear Systems
---------------------

The linear system :math:`Ax = b` can be solved using the
Preconditioned Conjugate Gradient (PCG) method.

::

    x = solver.solve(
        max_iter=2000,
        tol=1e-10,
        precond="ic0"
    )

Preconditioners
---------------

The following preconditioners are supported:

- ``jacobi``  
  Diagonal Jacobi preconditioner

- ``ssor``  
  Symmetric Successive Over-Relaxation (requires ``omega``)

- ``ic0``  
  Incomplete Cholesky factorization with zero fill-in

solve() API
-----------

::

    solve(max_iter=2000, tol=1e-10, precond="jacobi", omega=None)

Solve the linear system :math:`Ax = b` using PCG.

- ``max_iter`` (int): Maximum number of iterations
- ``tol`` (float): Convergence tolerance
- ``precond`` (str): Preconditioner type (``jacobi``, ``ssor``, ``ic0``)
- ``omega`` (float): Relaxation parameter for SSOR

solve_csr() API
---------------------
::

    solve_csr(row_ptr, col_idx, values, b, max_iter=2000, tol=1e-10, precond="jacobi", omega=None)
    
Solve the linear system :math:`Ax = b` using PCG with CSR matrix input.
- ``row_ptr`` (numpy.ndarray): CSR row pointer array
- ``col_idx`` (numpy.ndarray): CSR column indices array
- ``values`` (numpy.ndarray): CSR nonzero values array
- ``b`` (numpy.ndarray): Right-hand side vector
- ``max_iter`` (int): Maximum number of iterations
- ``tol`` (float): Convergence tolerance
- ``precond`` (str): Preconditioner type (``jacobi``, ``ssor``, ``ic0``)
- ``omega`` (float): Relaxation parameter for SSOR

Return Value
------------

- Solution vector :math:`x` (``numpy.ndarray``)

Engineering Infrastructure
==========================

- **Build System:** CMake
- **Version Control:** GitHub
- **Testing Framework:** Google Test (C++)
- **Documentation:** README.rst

Development Schedule
====================

- **Week 1 (10/6):** Implement CSR matrix parser and verify correctness of CSR
  arrays (``row_ptr``, ``col_idx``, ``values``). Set up Google Test.
- **Week 2 (10/13):** Implement Jacobi preconditioner and verify correctness.
- **Week 3 (10/20):** Implement Conjugate Gradient solver and verify results
  against a direct solver.
- **Week 4 (10/27):** Integrate Jacobi-PCG and add convergence and residual tests.
- **Week 5 (11/03):** Add OpenMP/MKL acceleration and performance regression tests.
- **Week 6 (11/10):** Implement Gauss–Seidel preconditioner and compare performance.
- **Week 7 (11/17):** Support multiple preconditioners via enum and add tests.
- **Week 8 (11/24):** Final testing, documentation, and demo materials.

References
==========

1. Saad, Y. *Iterative Methods for Sparse Linear Systems.* SIAM, 2003.
