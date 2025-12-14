#ifndef PCG_CP_H
#define PCG_CP_H
#include <iostream>
#include <vector>
#include <cmath>      // for std::sqrt, std::abs
#include <numeric>    // for std::inner_product
#include <stdexcept>  // for std::runtime_error
#include <fstream>
#include <sstream>
#include <omp.h>
#include <atomic>    
#include <iomanip>
#include <chrono>
using namespace std;
//======================= Memory Tracking =========================
extern std::atomic<size_t> g_bytes_used;
extern std::atomic<size_t> g_allocated;
extern std::atomic<size_t> g_deallocated;

void print_memory_stats(const char* where = nullptr);

class CsrMatrix 
{
public:
    CsrMatrix() = default;
    int num_rows;
    std::vector<double> values;     // 非零元素的值
    std::vector<int> col_indices; // 對應的列索引
    std::vector<int> row_ptr;     // 每行在 values/col_indices 中的起始位置
};
class PCG_CP 
{
public:
    //==============================Golden Ans====================
    vector<double> Golden_x;
    vector<double> Error;
    //==============================Golden Ans====================
    PCG_CP() = default;
    /*PCG_CP(bool mode,int max_iter,double tolerance)*/
    int mode = 0;
    int max_iter = 0;
    double tolerance = 0;
    int count_iter = 0;
    CsrMatrix A;
    vector<double> b;
    vector<double> x;
    double omega = 1.3; // relaxation parameter for SSOR: 0 < ω < 2
    void parsing_A_from_mtx_and_convert2csr(const std::string& filename);
    void parsing_b_from_mtx(const std::string& filename);
    void check_A_symmetry();
    void resizing_x();
    double dot_product(const std::vector<double>& a, const std::vector<double>& b);
    double vector_norm(const std::vector<double>& v);
    void axpy(double a, const std::vector<double>& x, std::vector<double>& y);
    void spmv_csr(const CsrMatrix& A, const std::vector<double>& x, std::vector<double>& y);
    void apply_jacobi_preconditioner(const std::vector<double>& diag_inv, const std::vector<double>& r, std::vector<double>& z);
    int jacobi_pcg(const CsrMatrix& A,
        const std::vector<double>& b,
        std::vector<double>& x,
        int max_iter,
        double tolerance);
    void apply_sgs_preconditioner(const CsrMatrix& A,
        const std::vector<double>& r,
        std::vector<double>& z,
        int sweeps);
    void apply_ssor_preconditioner(
        const CsrMatrix& A,
        const std::vector<double>& r,
        std::vector<double>& z,
        double omega,    // relaxation parameter: 0 < ω < 2
        int sweeps);
    int pcg_sgs(const CsrMatrix& A,
        const std::vector<double>& b,
        std::vector<double>& x,
        int max_iter,
        double tolerance);
    int pcg_ssor(const CsrMatrix& A,
        const std::vector<double>& b,
        std::vector<double>& x,
        double omega,
        int max_iter,
        double tolerance);

    CsrMatrix L_ic0;
    CsrMatrix LT_ic0;
    void build_ic0(const CsrMatrix& A, CsrMatrix& L, CsrMatrix& LT);

    // 用 IC(0) 做 M^{-1} r
    void apply_ic0_preconditioner(
        const CsrMatrix& L,
        const CsrMatrix& LT,
        const std::vector<double>& r,
        std::vector<double>& z);
    int pcg_ic0(const CsrMatrix& A,
        const std::vector<double>& b,
        std::vector<double>& x,
        int max_iter,
        double tolerance);
    void PCG();
    void DirectSolver_Golden_Gen();
    void ErrorCompare();
};


//Parser helper
void _SplitString
(
    std::vector<std::string>& _token,
    std::string          _str,
    std::string          _delimiter
);
void indexSortCSR(
    std::vector<double>& values,
    std::vector<int>& col_indices,
    const std::vector<int>& row_ptr);
#endif // PCG_CP_H