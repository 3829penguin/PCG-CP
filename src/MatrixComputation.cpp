#include "MatrixComputation.h"
#include <chrono>
void convertToPardisoCSR_vector(
    std::vector<int>& row_ptr,
    std::vector<int>& col_indices,
    std::vector<double>& values,
    std::vector<double>& a,
    std::vector<MKL_INT>& ia,
    std::vector<MKL_INT>& ja,
    int numProcs
)
{
    

    ia.assign(row_ptr.size(), 0); 
    a.resize(values.size()); 
    ja.resize(values.size()); 
    ia[0] = 1; 
#pragma omp parallel for num_threads(numProcs)
    for (int i = 1; i < row_ptr.size(); i++) {
        ia[i] = row_ptr[i] + 1;
    }
#pragma omp parallel for num_threads(numProcs)
    for (int i = 0; i < col_indices.size(); i++)
    {
        a[i] = values[i];
        ja[i] = col_indices[i] + 1;
    }
    return;
}
void MKL_Pardiso_csr
(
    int nRow,
    std::vector<int>& row_ptr,
    std::vector<int>& col_indices,
    std::vector<double>& values,
    vector<double>& x,
    vector<double>& b
)
{
    //A->b.resize(A->nROW);
    double t1 = 0, t2 = 0 , Totaltime = 0;
    MKL_INT threads = mkl_get_max_threads();
    mkl_set_num_threads(threads);
    MKL_INT n_nodes = b.size();

    std::vector<double> a;
    std::vector<MKL_INT> ia;
    std::vector<MKL_INT> ja;
    int NZT = 0;

    convertToPardisoCSR_vector
    (
        row_ptr, col_indices, values, a, ia, ja, threads
    );

    /* for (int i = 0; i < a.size(); i++) {
        std::cout << "a[" << i << "]: " << a[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < ia.size(); i++) {
        std::cout << "ia[" << i << "]: " << ia[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < ja.size(); i++) {
        std::cout << "ja[" << i << "]: " << ja[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < b.size(); i++) {
        std::cout << "b[" << i << "]: " << b[i] << " ";
    }
    std::cout << std::endl;*/
    std::cout << "ia.size()=" << ia.size()
        << "  ja.size()=" << ja.size()
        << "  ia.back()=" << ia.back() << std::endl;

    //std::cout  << "**********" << mkl_domain_get_max_threads(MKL_DOMAIN_ALL)<<"**********\n";

    MKL_INT mtype = 11;       /* Real symmetric matrix */
    /* RHS and solution vectors. */
    //x = std::vector<double>(n_nodes);
    //double b[8], x[8];
    MKL_INT nrhs = 1;     /* Number of right hand sides. */
    /* Internal solver memory pointer pt, */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
    /* or void *pt[64] should be OK on both architectures */
    int pt[64];
    /* Pardiso control parameters. */
    MKL_INT iparm[64];
    MKL_INT maxfct, mnum, phase, error, msglvl;
    /* Auxiliary variables. */
    MKL_INT i;
    double ddum;          /* Double dummy */
    MKL_INT idum;         /* Integer dummy. */
    /* -------------------------------------------------------------------- */
    /* .. Setup Pardiso control parameters. */
    /* -------------------------------------------------------------------- */
    for (i = 0; i < 64; i++)
    {
        iparm[i] = 0;
    }
    iparm[0] = 1;         /* No solver default */
    iparm[1] = 2;         /* Fill-in reordering from METIS */
    iparm[2] = 1;         // test
    iparm[3] = 0;         /* No iterative-direct algorithm */
    iparm[4] = 0;         /* No user fill-in reducing permutation */
    iparm[5] = 0;         /* Write solution into x */
    iparm[6] = 0;         /* Not in use */
    iparm[7] = 2;         /* Max numbers of iterative refinement steps */
    iparm[8] = 0;         /* Not in use */
    iparm[9] = 8;        /* Perturb the pivot elements with 1E-13 */
    iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
    iparm[11] = 0;        /* Not in use */
    iparm[12] = 0;        /* Maximum weighted matching algorithm is switched-off (default for symmetric). Try iparm[12] = 1 in case of inappropriate accuracy */
    iparm[13] = 0;        /* Output: Number of perturbed pivots */
    iparm[14] = 0;        /* Not in use */
    iparm[15] = 0;        /* Not in use */
    iparm[16] = 0;        /* Not in use */
    iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
    iparm[18] = -1;       /* Output: Mflops for LU factorization */
    iparm[19] = 0;        /* Output: Numbers of CG Iterations */
    //iparm[34] = 1;  // zero-based indexing
    maxfct = 1;           /* Maximum number of numerical factorizations. */
    mnum = 1;         /* Which factorization to use. */
    msglvl = 0;// 1;           /* Print statistical information in file */
    error = 0;            /* Initialize error flag */
    //std::cout << error << std::endl;
    /* -------------------------------------------------------------------- */
    /* .. Initialize the internal solver memory pointer. This is only */
    /* necessary for the FIRST call of the PARDISO solver. */
    /* -------------------------------------------------------------------- */
    for (i = 0; i < 64; i++)
    {
        pt[i] = 0;
    }
    /* -------------------------------------------------------------------- */
    /* .. Reordering and Symbolic Factorization. This step also allocates */
    /* all memory that is necessary for the factorization. */
    /* -------------------------------------------------------------------- */
    std::cout << "Check CSR..." << std::endl;
    std::cout << "nRow = " << nRow << ", nnz = " << NZT << std::endl;
    std::cout << "Check CSR consistency...\n";
    std::cout << "ia.size() = " << ia.size() << " (should be " << nRow + 1 << ")\n";
    std::cout << "ia[0] = " << ia[0] << ", ia[nRow] = " << ia[nRow] << "\n";
    std::cout << "nnz (from ia) = " << ia[nRow] - ia[0] << ", nnz (from ja) = " << ja.size() << "\n";
    for (int i = 0; i < nRow; i++) {
        if (ia[i] > ia[i + 1]) {
            std::cerr << "Error: ia not non-decreasing at row " << i << std::endl;
            exit(1);
        }
        for (int k = ia[i]; k < ia[i + 1]; k++) {
            int col = ja[k - 1]; // ¦pªG¬O1-based
            if (col < 1 || col > nRow) {
                std::cerr << "Error: invalid column index at row " << i
                    << " ¡÷ col = " << col << std::endl;
                exit(1);
            }
        }
    }
    for (size_t k = 0; k < a.size(); k++) {
        if (std::isnan(a[k]) || std::isinf(a[k])) {
            std::cerr << "Invalid a[" << k << "] = " << a[k] << std::endl;
        }
    }
    phase = 11;
    t1 = dsecnd();
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
        &n_nodes, a.data(), ia.data(), ja.data(), &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    t2 = dsecnd();
    Totaltime += (t2 - t1);
    /*std::cout << "Analysis...\n" << std::setprecision(4)
        << (double)(std::clock() - temp) / CLOCKS_PER_SEC << " sec.\n";*/
        /*std::cout << error << std::endl;*/
    if (error != 0)
    {
        printf("\nERROR during symbolic factorization: %d", error);
        exit(1);
    }
    //printf("\nReordering completed ... ");
    //printf("\nNumber of nonzeros in factors = %d", iparm[17]);
    //printf("\nNumber of factorization MFLOPS = %d", iparm[18]);
    /* -------------------------------------------------------------------- */
    /* .. Numerical factorization. */
    /* -------------------------------------------------------------------- */
    phase = 22;
    t1 = dsecnd();
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
        &n_nodes, a.data(), ia.data(), ja.data(), &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    /*std::cout << "LU...\n" << std::setprecision(4)
        << (double)(std::clock() - temp) / CLOCKS_PER_SEC << " sec.\n";*/
    t2 = dsecnd();
    Totaltime += (t2 - t1);
    if (error != 0)
    {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    //printf("\nFactorization completed ... ");
    /* -------------------------------------------------------------------- */
    /* .. Back substitution and iterative refinement. */
    /* -------------------------------------------------------------------- */
    phase = 33;
    //iparm[7] = 2;         /* Max numbers of iterative refinement steps. */
    t1 = dsecnd();
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
        &n_nodes, a.data(), ia.data(), ja.data(), &idum, &nrhs, iparm, &msglvl, b.data(), x.data(), &error);
    t2 = dsecnd();
    Totaltime += (t2 - t1);
    /*std::cout << "Backward and forward...\n" << std::setprecision(4)
        << (double)(std::clock() - temp) / CLOCKS_PER_SEC << " sec.\n";*/
    if (error != 0)
    {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }


    /* -------------------------------------------------------------------- */
    /* .. Termination and release of memory. */
    /* -------------------------------------------------------------------- */
    //std::cout <<"sample sol:"<< solution[0];
    phase = -1;           /* Release internal memory. */
    /*PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
        &n_nodes, &ddum, ia.data(), ja.data(), &idum, &nrhs,
        iparm, &msglvl, &ddum, &ddum, &error);*/
    std::cout << "Finished Pardiso solving...\n" << std::setprecision(4)
        << Totaltime << " sec.\n";
    //task_runtime->clock_summary("Direct solve");
    /*for (int i = 0; i < x.size(); i++) {
        std::cout << "x[" << i << "]: " << x[i] << " ";
    }
    std::cout << std::endl;*/

}

double DENSEvector_dot_product(int dimension, std::vector<double>& A, std::vector<double>& B) {
    double result = cblas_ddot(dimension, A.data(), 1, B.data(), 1);
    return result;
}