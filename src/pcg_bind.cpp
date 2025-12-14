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
PYBIND11_MODULE(pcgsolver, m)
{
    m.doc() = "PCG solver Python binding";

    py::class_<PCG_CP>(m, "PCG")
        .def(py::init<>())

        // expose C++ matrix parsers
        .def("parse_A", &PCG_CP::parsing_A_from_mtx_and_convert2csr,
             "Parse MatrixMarket A into CSR format.")
        .def("parse_b", &PCG_CP::parsing_b_from_mtx,
             "Parse RHS vector b from MatrixMarket format.")
        .def("check_symmetry", &PCG_CP::check_A_symmetry,
             "Check if matrix A is symmetric.")
        // Unified solve()
        .def("solve", &solve_pcg_wrapper,
             py::arg("max_iter") = 2000,
             py::arg("tol") = 1e-10,
             py::arg("precond") = "jacobi",
             py::arg("omega") = py::none(),
             R"pbdoc(
Solve the linear system Ax=b using PCG.

precond options:
 - "jacobi"
 - "ssor"  (requires omega)
 - "ic0"

omega: relaxation parameter only used when precond="ssor".
             )pbdoc");
}
