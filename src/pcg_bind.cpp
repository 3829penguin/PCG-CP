#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "PCG_CP.h"

namespace py = pybind11;

py::array_t<double> solve_pcg_wrapper(
    PCG_CP &solver,
    int max_iter,
    double tol,
    std::string precond,
    py::object omega_obj)
{
    int n = solver.b.size();
    std::vector<double> x(n, 0.0);

    double omega = 1.0;
    if (!omega_obj.is_none()) {
        omega = omega_obj.cast<double>();
    }

    if (precond == "jacobi")
    {
        solver.jacobi_pcg(solver.A, solver.b, x, max_iter, tol);
    }
    else if (precond == "ssor")
    {
        solver.pcg_ssor(solver.A, solver.b, x, omega, max_iter, tol);
    }
    else if (precond == "ic0")
    {
        CsrMatrix L, LT;
        solver.build_ic0(solver.A, L, LT);
        solver.pcg_ic0(solver.A, solver.b, x, max_iter, tol);
    }
    else
    {
        throw std::runtime_error("Unknown preconditioner: " + precond);
    }
    // ¦^¶Ç numpy array
    return py::array_t<double>(x.size(), x.data());
}
py::array_t<double> solve_pcg_csr_wrapper(
    PCG_CP &solver,
    py::array_t<int> row_ptr,
    py::array_t<int> col_indices,
    py::array_t<double> values,
    py::array_t<double> b,
    int max_iter,
    double tol,
    std::string precond,
    py::object omega_obj)
{
    // ---------- build CSR matrix ----------
    solver.A = CsrMatrix();
    solver.A.num_rows = static_cast<int>(row_ptr.size()) - 1;

    solver.A.row_ptr.assign(row_ptr.data(),
                            row_ptr.data() + row_ptr.size());

    solver.A.col_indices.assign(col_indices.data(),
                                col_indices.data() + col_indices.size());

    solver.A.values.assign(values.data(),
                            values.data() + values.size());

    // ---------- build RHS ----------
    solver.b.assign(b.data(), b.data() + b.size());

    int n = solver.b.size();
    std::vector<double> x(n, 0.0);

    // ---------- omega ----------
    double omega = 1.0;
    if (!omega_obj.is_none())
        omega = omega_obj.cast<double>();

    // ---------- PCG ----------
    if (precond == "jacobi")
    {
        solver.jacobi_pcg(solver.A, solver.b, x, max_iter, tol);
    }
    else if (precond == "ssor")
    {
        solver.pcg_ssor(solver.A, solver.b, x, omega, max_iter, tol);
    }
    else if (precond == "ic0")
    {
        CsrMatrix L, LT;
        solver.build_ic0(solver.A, L, LT);
        solver.pcg_ic0(solver.A, solver.b, x, max_iter, tol);
    }
    else
    {
        throw std::runtime_error("Unknown preconditioner: " + precond);
    }

    return py::array_t<double>(x.size(), x.data());
}

PYBIND11_MODULE(pcgsolver, m)
{
    m.doc() = "PCG solver Python binding";

    py::class_<PCG_CP>(m, "PCG")
        .def(py::init<>())

        // -------------------------------
        // File-based MatrixMarket parsers
        // -------------------------------
        .def("parse_A", &PCG_CP::parsing_A_from_mtx_and_convert2csr,
             "Parse MatrixMarket A into CSR format.")
        .def("parse_b", &PCG_CP::parsing_b_from_mtx,
             "Parse RHS vector b from MatrixMarket format.")
        .def("check_symmetry", &PCG_CP::check_A_symmetry,
             "Check if matrix A is symmetric.")

        // -------------------------------
        // Unified solve() using parsed A,b
        // -------------------------------
        .def("solve", &solve_pcg_wrapper,
             py::arg("max_iter") = 2000,
             py::arg("tol") = 1e-10,
             py::arg("precond") = "jacobi",
             py::arg("omega") = py::none(),
             R"pbdoc(
Solve the linear system Ax=b using PCG.

Parameters
----------
max_iter : int
    Maximum number of iterations.
tol : float
    Convergence tolerance.
precond : str
    Preconditioner type ("jacobi", "ssor", "ic0").
omega : float, optional
    Relaxation parameter for SSOR.

Notes
-----
This method assumes that A and b have already been loaded
using parse_A() and parse_b().
             )pbdoc")

        // ---------------------------------------
        // NEW: solve using CSR matrix from Python
        // ---------------------------------------
        .def("solve_csr", &solve_pcg_csr_wrapper,
             py::arg("row_ptr"),
             py::arg("col_indices"),
             py::arg("values"),
             py::arg("b"),
             py::arg("max_iter") = 2000,
             py::arg("tol") = 1e-10,
             py::arg("precond") = "jacobi",
             py::arg("omega") = py::none(),
             R"pbdoc(
Solve the linear system Ax=b using PCG with CSR matrix input.

Parameters
----------
row_ptr : numpy.ndarray (int)
    CSR row pointer array (size = n+1).
col_indices : numpy.ndarray (int)
    CSR column indices.
values : numpy.ndarray (float)
    CSR nonzero values.
b : numpy.ndarray (float)
    Right-hand side vector.
max_iter : int
    Maximum number of iterations.
tol : float
    Convergence tolerance.
precond : str
    Preconditioner type ("jacobi", "ssor", "ic0").
omega : float, optional
    Relaxation parameter for SSOR.

Notes
-----
This method allows direct in-memory CSR input from Python,
making it suitable for integration with scipy.sparse.csr_matrix.
             )pbdoc");
}

