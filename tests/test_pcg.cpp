#include <gtest/gtest.h>
#include "PCG_CP.h"
#include "MatrixComputation.h"

TEST(PCGTest, CheckAgainstPardisoGolden)
{
    CsrMatrix A;
    A.num_rows = 4;
    A.row_ptr = {0, 2, 5, 8, 10};
    A.col_indices = {0,1, 0,1,2, 1,2,3, 2,3};
    A.values = {2,-1, -1,2,-1, -1,2,-1, -1,2};
    std::vector<double> b = {1, 0, 0, 1};

    std::vector<double> x_pcg_J(4, 0.0);
    std::vector<double> x_pcg_ic0(4, 0.0);
    std::vector<double> x_pardiso(4, 0.0);

    PCG_CP solver;

    // ---- Solve with PCG ----
    solver.jacobi_pcg(A, b, x_pcg_J, 1000, 1e-12);

    // ---- Solve with MKL PARDISO ----
    MKL_Pardiso_csr(4, A.row_ptr, A.col_indices, A.values, x_pardiso, b);

    // ---- Compare ----
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(x_pcg_J[i], x_pardiso[i], 1e-8);
    }
    
    solver.build_ic0(A, solver.L_ic0, solver.LT_ic0);
    solver.pcg_ic0(A, b, x_pcg_ic0, 1000, 1e-12);
    // ---- Compare ----
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(x_pcg_ic0[i], x_pardiso[i], 1e-8);
    }
}
