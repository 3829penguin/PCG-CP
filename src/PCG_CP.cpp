// jacobi_pcg.cpp
#include "PCG_CP.h"
#include "MatrixComputation.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <utility>
using namespace std;

// // ======================= Memory Tracking (定義) =========================
std::atomic<size_t> g_bytes_used{0};
std::atomic<size_t> g_allocated{0};
std::atomic<size_t> g_deallocated{0};
// 全域 operator new / delete：攔截所有 heap allocation
void* operator new(std::size_t size) {
    g_allocated += size;
    g_bytes_used += size;
    if (void* p = std::malloc(size)) {
        return p;
    }
    throw std::bad_alloc{};
}

void operator delete(void* p, std::size_t size) noexcept {
    if (!p) return;
    g_deallocated += size;
    g_bytes_used -= size;
    std::free(p);
}

// 方便在關鍵位置印出統計資訊
void print_memory_stats(const char* where) {
    std::cout << "\n===== Memory stats";
    if (where) std::cout << " [" << where << "]";
    std::cout << " =====\n";
    std::cout << "  allocated:   " << g_allocated.load()   << " bytes\n";
    std::cout << "  deallocated: " << g_deallocated.load() << " bytes\n";
    std::cout << "  in use:      " << g_bytes_used.load()  << " bytes\n";
    std::cout << "=========================================\n";
}


void indexSortCSR(
    std::vector<double>& values,
    std::vector<int>& col_indices,
    const std::vector<int>& row_ptr)
{
    // 檢查 row_ptr 陣列的有效性
    if (row_ptr.empty()) {
        return;
    }

    // 獲取總行數 N。對於 CSR，row_ptr 陣列的大小是 N + 1。
    size_t N = row_ptr.size() - 1;

    // 遍歷每一行 (i = 0 到 N-1)
    for (size_t i = 0; i < N; ++i)
    {
        // 獲取當前行 i 的起始索引 (start) 和結束索引 (end)。
        // 由於 CSR 索引通常是 0-based 或 1-based，這裡使用 int 轉換。
        int start = row_ptr[i];
        int end = row_ptr[i + 1];

        // 檢查該行是否有非零元素
        if (start >= end) {
            continue; // 該行是空行
        }

        // ----------------------------------------------------
        // 1. 將當前行 i 的 (col_index, value) 數據打包成 pair
        // ----------------------------------------------------
        // 計算當前行的非零元素數量 (nnz_row)
        int nnz_row = end - start;

        // 使用 vector of pairs 進行排序：pair.first 是列索引，pair.second 是值
        std::vector<std::pair<int, double>> row_data;
        row_data.reserve(nnz_row);

        for (int k = 0; k < nnz_row; ++k)
        {
            // 注意：這裡使用 k + start 作為 values/col_indices 中的實際索引
            row_data.emplace_back(col_indices[start + k], values[start + k]);
        }

        // ----------------------------------------------------
        // 2. 依據列索引 (pair.first) 進行排序
        // ----------------------------------------------------
        std::sort(row_data.begin(), row_data.end(),
            [](const auto& a, const auto& b) {
                // 升序排列：col_index 較小的排在前面
                return a.first < b.first;
            });

        // ----------------------------------------------------
        // 3. 將排序後的結果寫回原來的 values 和 col_indices 陣列
        // ----------------------------------------------------
        for (int k = 0; k < nnz_row; ++k)
        {
            values[start + k] = row_data[k].second;
            col_indices[start + k] = row_data[k].first;
        }
    }
}
void PCG_CP::parsing_A_from_mtx_and_convert2csr(const string& filename) {
    // Parse Matrix Market file to build CSR
    cout << "[Start Parsing A Matrix and Convert to CSV Format]" << endl;
    ifstream fin(filename);
    if (!fin.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    string line;
    while (getline(fin, line)) {
        if (line.empty() || line[0] == '%') continue;
        else break;
    }
    int n, nnz;
    {
        stringstream ss(line);
        ss >> n >> n >> nnz;
    }
    A.num_rows = n;
    vector<tuple<int, int, double>> triples; // (row,col,value)
    triples.reserve(nnz);

    int i, j;
    double val;
    while (fin >> i >> j >> val)
        triples.emplace_back(i - 1, j - 1, val); // MatrixMarket 是 1-based

    fin.close();

    A.row_ptr.assign(A.num_rows + 1, 0);
    int now_index = 1;
    for (const auto& t : triples) {
        int r = get<0>(t);
        A.row_ptr[r + 1] = now_index++;
    }
    int nnz_total = triples.size();
    A.col_indices.resize(nnz_total);
    A.values.resize(nnz_total);

    for (int r = 0; r < A.num_rows; ++r) {
        int row_start = A.row_ptr[r];
        int row_end = A.row_ptr[r + 1];
        for (int pos = row_start; pos < row_end; ++pos) {
            A.col_indices[pos] = get<1>(triples[pos]);
            A.values[pos] = get<2>(triples[pos]);
        }
    }
    cout << "[End Parsing A Matrix and Convert to CSV Format]" << endl;
    // print_memory_stats("after parsing A");
    return;
}
void PCG_CP::check_A_symmetry()
{
    cout << "[Start Checking A Matrix Symmetry]" << endl;
    indexSortCSR(A.values, A.col_indices, A.row_ptr);
    const int n = A.num_rows;
    const int nnz = A.values.size();

    /* ===============================
     * Step 1: build transpose AT
     * =============================== */
    CsrMatrix AT;
    AT.num_rows = n;
    AT.row_ptr.resize(n + 1, 0);
    AT.col_indices.resize(nnz);
    AT.values.resize(nnz);

    // count nnz per column (-> row of AT)
    for (int p = 0; p < nnz; ++p) {
        AT.row_ptr[A.col_indices[p] + 1]++;
    }

    // prefix sum
    for (int i = 0; i < n; ++i) {
        AT.row_ptr[i + 1] += AT.row_ptr[i];
    }

    // temp cursor
    std::vector<int> cur = AT.row_ptr;

    // fill AT
    for (int i = 0; i < n; ++i) {
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            int j = A.col_indices[p];
            int dest = cur[j]++;
            AT.col_indices[dest] = i;
            AT.values[dest] = A.values[p];
        }
    }

    /* ===============================
     * Step 2: symmetry check
     * =============================== */
    for (int i = 0; i < n; ++i) {
        int a_pos  = A.row_ptr[i];
        int a_end  = A.row_ptr[i + 1];
        int at_pos = AT.row_ptr[i];
        int at_end = AT.row_ptr[i + 1];

        while (a_pos < a_end && at_pos < at_end) {
            int a_col  = A.col_indices[a_pos];
            int at_col = AT.col_indices[at_pos];

            if (a_col == at_col) {
                double diff = std::fabs(A.values[a_pos] - AT.values[at_pos]);
                if (diff > 1e-10) {
                    cout << "Asymmetry value mismatch at ("
                         << i << ", " << a_col << "): "
                         << A.values[a_pos] << " vs "
                         << AT.values[at_pos] << endl;
                }
                ++a_pos;
                ++at_pos;
            }
            else if (a_col < at_col) {
                cout << "Missing transpose entry: A("
                     << i << ", " << a_col << ")" << endl;
                ++a_pos;
            }
            else {
                cout << "Extra transpose entry: AT("
                     << i << ", " << at_col << ")" << endl;
                ++at_pos;
            }
        }

        while (a_pos < a_end) {
            cout << "Missing transpose entry: A("
                 << i << ", " << A.col_indices[a_pos] << ")" << endl;
            ++a_pos;
        }

        while (at_pos < at_end) {
            cout << "Extra transpose entry: AT("
                 << i << ", " << AT.col_indices[at_pos] << ")" << endl;
            ++at_pos;
        }
    }

    cout << "[End Checking A Matrix Symmetry]" << endl;
}

void PCG_CP::parsing_b_from_mtx(const std::string& filename)
{
    cout << "[Start Parsing b vector]" << endl;
    ifstream fin(filename);
    if (!fin.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }
    string line;
    while (getline(fin, line)) {
        if (line.empty() || line[0] == '%') continue;
        else break;
    }
    int n, is_one;
    {
        stringstream ss(line);
        ss >> n >> is_one;
    }
    int index;
    double value;
    b.resize(n);
    while (fin >> index >> value)
    {
        b[index - 1] = value;
    }
    fin.close();
    cout << "[End Parsing b vector]" << endl;
    // print_memory_stats("after parsing b");
    return;
}
void PCG_CP::resizing_x()
{
    x.resize(b.size(), 0);
    return;
}
double PCG_CP::dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    // C++17 寫法
    // return std::transform_reduce(a.begin(), a.end(), b.begin(), 0.0);

    // C++11 寫法
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) 
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
double PCG_CP::vector_norm(const std::vector<double>& v) {
    return std::sqrt(dot_product(v, v));
}

// y = y + alpha * x
void PCG_CP::axpy(double a, const std::vector<double>& x, std::vector<double>& y) {
    #pragma omp parallel for
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] += a * x[i];
    }
}

// Jacobi preconditioner: z = M^{-1} r,  M = diag(A)  (with safe inverse)
// Symmetric Gauss-Seidel preconditioner:
// Solve approximately M z = r,  where M ≈ A,
// by doing 'sweeps' 次 前向 + 後向 Gauss-Seidel。

void PCG_CP::spmv_csr(const CsrMatrix& A, const std::vector<double>& x, std::vector<double>& y) {
    std::fill(y.begin(), y.end(), 0.0); // y 初始化為 0
    #pragma omp parallel for
    for (int i = 0; i < A.num_rows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
}
//============================= Jacobi Preconditioner ============================
void PCG_CP::apply_jacobi_preconditioner(const std::vector<double>& diag_inv, const std::vector<double>& r, std::vector<double>& z) {
    #pragma omp parallel for
    for (size_t i = 0; i < r.size(); ++i) {
        z[i] = diag_inv[i] * r[i];
    }
}
int PCG_CP::jacobi_pcg(const CsrMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    int max_iter,
    double tolerance) 
{
    resizing_x();
    count_iter = 0;
    int n = A.num_rows;
    if (b.size() != n || x.size() != n) {
        throw std::runtime_error("Vector and matrix dimensions do not match.");
    }

    // 1. 提取 A 的對角線元素的倒數 (M_inv)
    std::vector<double> diag_inv(n);
    for (int i = 0; i < n; ++i) {
        bool found_diag = false;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_indices[j] == i) {
                diag_inv[i] = 1.0 / A.values[j];
                found_diag = true;
                break;
            }
        }
        if (!found_diag) {
            throw std::runtime_error("Matrix diagonal element missing.");
        }
    }

    // 2. 初始化
    std::vector<double> r(n);
    std::vector<double> p(n);
    std::vector<double> z(n);
    std::vector<double> Ap(n);

    // r = b - A*x_0 (如果 x_0 非零)
    spmv_csr(A, x, r); // r = A*x_0
    for (int i = 0; i < n; ++i) r[i] = b[i] - r[i]; // r = b - A*x_0
    /*ofstream check_r("./r.txt");
    for (int i = 0; i < n; ++i)
    {
        if(r[i]!= 0)
        { 
            check_r << r[i] << endl;
        }
        
    }
    exit(0);*/
    double initial_residual_norm = vector_norm(r);
    //if (initial_residual_norm < 1e-10) { // 初始猜測已經很準
    //    cout << "initial guess is good." << endl;
    //    return 0;
    //}

    // z = M^{-1} * r
    apply_jacobi_preconditioner(diag_inv, r, z);

    // p = z
    p = z;

    // rs_old = r^T * z
    double rs_old = dot_product(r, z);

    // 3. 迭代
    // int count_iter = 0;
    for (int k = 0; k < max_iter; ++k) {
        // Ap = A * p
        spmv_csr(A, p, Ap);

        // alpha = (r^T * z) / (p^T * A * p)
        double p_dot_Ap = dot_product(p, Ap);
        double alpha = rs_old / p_dot_Ap;

        // x = x + alpha * p
        axpy(alpha, p, x);

        // r = r - alpha * A * p
        axpy(-alpha, Ap, r);

        // 檢查收斂
        double residual_norm = vector_norm(r);
        if (count_iter % 200 == 0)
        {
            cout <<" iter : "<< count_iter << " = " << residual_norm << endl;
        }
        if (residual_norm / initial_residual_norm < tolerance) {
            std::cout << "PCG converged in " << k + 1 << " iterations." << std::endl;
            std::cout << "Final relative residual: " << (residual_norm / initial_residual_norm) << std::endl;
            return k + 1;
        }

        // z_new = M^{-1} * r_new
        apply_jacobi_preconditioner(diag_inv, r, z);

        // rs_new = r_new^T * z_new
        double rs_new = dot_product(r, z);

        // beta = rs_new / rs_old
        double beta = rs_new / rs_old;

        // p = z_new + beta * p
        for (int i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }

        // rs_old = rs_new
        rs_old = rs_new;
        count_iter++;
    }

    std::cerr << "PCG did not converge after " << max_iter << " iterations." << std::endl;
    return count_iter;
}
//============================= End of Jacobi Preconditioner ============================

//============================= SGS Preconditioner ============================
void PCG_CP::apply_sgs_preconditioner(const CsrMatrix& A,
    const std::vector<double>& r,
    std::vector<double>& z,
    int sweeps)
{
    const int n = A.num_rows;

    // 初始：每次當成從 0 開始，也可以改 z 保留上一次的值
    std::fill(z.begin(), z.end(), 0.0);

    for (int s = 0; s < sweeps; ++s)
    {
        // ------- Forward sweep -------
        for (int i = 0; i < n; ++i)
        {
            double diag = 0.0;
            double sum = 0.0;

            for (int jj = A.row_ptr[i]; jj < A.row_ptr[i + 1]; ++jj)
            {
                int col = A.col_indices[jj];
                double val = A.values[jj];

                if (col == i)
                    diag = val;
                else
                    sum += val * z[col];   // 對已更新/舊的 z 做 GS 更新
            }

            if (std::fabs(diag) < 1e-30)
                throw std::runtime_error("Zero diagonal in Gauss-Seidel preconditioner.");

            z[i] = (r[i] - sum) / diag;
        }

        // ------- Backward sweep -------
        for (int i = n - 1; i >= 0; --i)
        {
            double diag = 0.0;
            double sum = 0.0;

            for (int jj = A.row_ptr[i]; jj < A.row_ptr[i + 1]; ++jj)
            {
                int col = A.col_indices[jj];
                double val = A.values[jj];

                if (col == i)
                    diag = val;
                else
                    sum += val * z[col];
            }

            if (std::fabs(diag) < 1e-30)
                throw std::runtime_error("Zero diagonal in Gauss-Seidel preconditioner.");

            z[i] = (r[i] - sum) / diag;
        }
    }
}
int PCG_CP::pcg_sgs(const CsrMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    int max_iter,
    double tolerance)
{
    resizing_x();
    count_iter = 0;
    int n = A.num_rows;
    if (b.size() != n || x.size() != n) {
        throw std::runtime_error("Vector and matrix dimensions do not match.");
    }

    // 不需要 diag_inv，因為 SGS 直接用 A
    std::vector<double> r(n), p(n), z(n), Ap(n);

    // r = b - A*x0
    spmv_csr(A, x, r);
    for (int i = 0; i < n; ++i)
        r[i] = b[i] - r[i];

    double initial_residual_norm = vector_norm(r);

    // z = M^{-1} r  (用 SGS)
    apply_sgs_preconditioner(A, r, z, 1);  // sweeps = 1

    // p = z
    p = z;

    double rs_old = dot_product(r, z);

    // int count_iter = 0;
    for (int k = 0; k < max_iter; ++k)
    {
        // Ap = A * p
        spmv_csr(A, p, Ap);

        double p_dot_Ap = dot_product(p, Ap);
        double alpha = rs_old / p_dot_Ap;

        // x = x + alpha * p
        axpy(alpha, p, x);

        // r = r - alpha * Ap
        axpy(-alpha, Ap, r);

        double residual_norm = vector_norm(r);
        if (count_iter % 200 == 0) {
            cout <<" iter : "<< count_iter << " = " << residual_norm << endl;
        }
        if (residual_norm / initial_residual_norm < tolerance) {
            std::cout << "PCG (SGS) converged in " << k + 1 << " iterations.\n";
            std::cout << "Final relative residual: "
                << (residual_norm / initial_residual_norm) << std::endl;
            return k + 1;
        }

        // z = M^{-1} r (再做一次 SGS)
        apply_sgs_preconditioner(A, r, z, 1);

        double rs_new = dot_product(r, z);
        double beta = rs_new / rs_old;

        for (int i = 0; i < n; ++i)
            p[i] = z[i] + beta * p[i];

        rs_old = rs_new;
        count_iter++;
    }

    std::cerr << "PCG (SGS) did not converge after "
        << max_iter << " iterations.\n";
    return count_iter;
}
//============================= End of SGS Preconditioner ============================

//============================= SSOR Preconditioner ============================
void PCG_CP::apply_ssor_preconditioner(
    const CsrMatrix& A,
    const std::vector<double>& r,
    std::vector<double>& z,
    double omega,    // relaxation parameter: 0 < ω < 2
    int sweeps)
{
    const int n = A.num_rows;

    // 每次預設從 0 開始
    std::fill(z.begin(), z.end(), 0.0);

    for (int s = 0; s < sweeps; ++s)
    {
        // ---------- Forward SOR ----------
        for (int i = 0; i < n; ++i)
        {
            double diag = 0.0;
            double sum = 0.0;

            for (int jj = A.row_ptr[i]; jj < A.row_ptr[i + 1]; ++jj)
            {
                int col = A.col_indices[jj];
                double val = A.values[jj];

                if (col == i)
                    diag = val;
                else
                    sum += val * z[col];
            }

            if (std::fabs(diag) < 1e-30)
                throw std::runtime_error("Zero diagonal in SSOR preconditioner.");

            double gs_value = (r[i] - sum) / diag;

            // SOR update
            z[i] = (1.0 - omega) * z[i] + omega * gs_value;
        }

        // ---------- Backward SOR ----------
        for (int i = n - 1; i >= 0; --i)
        {
            double diag = 0.0;
            double sum = 0.0;

            for (int jj = A.row_ptr[i]; jj < A.row_ptr[i + 1]; ++jj)
            {
                int col = A.col_indices[jj];
                double val = A.values[jj];

                if (col == i)
                    diag = val;
                else
                    sum += val * z[col];
            }

            if (std::fabs(diag) < 1e-30)
                throw std::runtime_error("Zero diagonal in SSOR preconditioner.");

            double gs_value = (r[i] - sum) / diag;

            // SOR update
            z[i] = (1.0 - omega) * z[i] + omega * gs_value;
        }
    }
}
int PCG_CP::pcg_ssor(const CsrMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    double omega,
    int max_iter,
    double tolerance)
{
    int n = A.num_rows;
    resizing_x();
    if (b.size() != n || x.size() != n) {
        throw std::runtime_error("Vector and matrix dimensions do not match.");
    }
    count_iter = 0;
    // 不需要 diag_inv，因為 SGS 直接用 A
    std::vector<double> r(n), p(n), z(n), Ap(n);

    // r = b - A*x0
    spmv_csr(A, x, r);
    for (int i = 0; i < n; ++i)
        r[i] = b[i] - r[i];

    double initial_residual_norm = vector_norm(r);

    // z = M^{-1} r  (用 SGS)
    apply_ssor_preconditioner(A, r, z, omega, 1);  // sweeps = 1

    // p = z
    p = z;

    double rs_old = dot_product(r, z);

    // int count_iter = 0;
    for (int k = 0; k < max_iter; ++k)
    {
        // Ap = A * p
        spmv_csr(A, p, Ap);

        double p_dot_Ap = dot_product(p, Ap);
        double alpha = rs_old / p_dot_Ap;

        // x = x + alpha * p
        axpy(alpha, p, x);

        // r = r - alpha * Ap
        axpy(-alpha, Ap, r);

        double residual_norm = vector_norm(r);
        if (count_iter % 200 == 0) {
            cout <<" iter : "<< count_iter << " = " << residual_norm << endl;
        }
        if (residual_norm / initial_residual_norm < tolerance) {
            std::cout << "PCG (SGS) converged in " << k + 1 << " iterations.\n";
            std::cout << "Final relative residual: "
                << (residual_norm / initial_residual_norm) << std::endl;
            return k + 1;
        }

        // z = M^{-1} r (再做一次 SGS)
        apply_ssor_preconditioner(A, r, z, omega, 1);

        double rs_new = dot_product(r, z);
        double beta = rs_new / rs_old;

        for (int i = 0; i < n; ++i)
            p[i] = z[i] + beta * p[i];

        rs_old = rs_new;
        count_iter++;
    }

    std::cerr << "PCG (SGS) did not converge after "
        << max_iter << " iterations.\n";
    return count_iter;
}
//============================= End of SSOR Preconditioner ============================
//============================= Multi-Color Gauss–Seidel ============================
// std::vector<int> greedy_coloring(const CsrMatrix& A) {
//     int n = A.num_rows;
//     std::vector<int> colors(n, -1);

//     std::vector<bool> used;

//     for (int i = 0; i < n; i++) {
//         used.assign(n, false);

//         // 找所有鄰居的顏色
//         for (int jj = A.row_ptr[i]; jj < A.row_ptr[i+1]; jj++) {
//             int j = A.col_indices[jj];
//             if (colors[j] != -1)
//                 used[colors[j]] = true;
//         }

//         // 找第一個沒被用的顏色
//         int c = 0;
//         while (c < n && used[c]) c++;
//         colors[i] = c;
//     }

//     return colors;
// }
// void PCG_CP::apply_mcgs_preconditioner(
//     const CsrMatrix& A,
//     const std::vector<double>& r,
//     std::vector<double>& z,
//     const std::vector<int>& colors,
//     int num_colors
// ) {
//     int n = A.num_rows;
//     std::fill(z.begin(), z.end(), 0.0);

//     // Forward sweep
//     for (int c = 0; c < num_colors; c++) {
//         #pragma omp parallel for schedule(static)
//         for (int i = 0; i < n; i++) {
//             if (colors[i] != c) continue;

//             double diag = 0.0, sum = 0.0;
//             for (int jj = A.row_ptr[i]; jj < A.row_ptr[i+1]; jj++) {
//                 int j = A.col_indices[jj];
//                 double val = A.values[jj];

//                 if (j == i) diag = val;
//                 else sum += val * z[j];
//             }

//             z[i] = (r[i] - sum) / diag;
//         }
//     }

//     // Backward sweep (SGS)
//     for (int c = num_colors - 1; c >= 0; c--) {
//         #pragma omp parallel for schedule(static)
//         for (int i = 0; i < n; i++) {
//             if (colors[i] != c) continue;

//             double diag = 0.0, sum = 0.0;
//             for (int jj = A.row_ptr[i]; jj < A.row_ptr[i+1]; jj++) {
//                 int j = A.col_indices[jj];
//                 double val = A.values[jj];

//                 if (j == i) diag = val;
//                 else sum += val * z[j];
//             }

//             z[i] = (r[i] - sum) / diag;
//         }
//     }
// }
//============================= End of Multi-Color Gauss–Seidel ============================
// 取得 A(i,j)，找不到就回 0.0
static double get_A_ij(const CsrMatrix& A, int i, int j)
{
    for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
        if (A.col_indices[p] == j)
            return A.values[p];
    }
    return 0.0;
}

// CSR 轉置：AT = A^T
static void csr_transpose(const CsrMatrix& A, CsrMatrix& AT)
{
    int n = A.num_rows;
    int nnz = (int)A.values.size();

    AT.num_rows = n;
    AT.row_ptr.assign(n + 1, 0);
    AT.col_indices.assign(nnz, 0);
    AT.values.assign(nnz, 0.0);

    // 計算每個 column 的 nnz，變成 AT 的 row_ptr 前綴和
    for (int i = 0; i < n; ++i) {
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            int col = A.col_indices[p];
            ++AT.row_ptr[col + 1];
        }
    }
    for (int i = 0; i < n; ++i) {
        AT.row_ptr[i + 1] += AT.row_ptr[i];
    }

    // 暫存當前寫入位置
    std::vector<int> offset = AT.row_ptr;

    // 寫入 AT
    for (int i = 0; i < n; ++i) {
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            int col = A.col_indices[p];
            double val = A.values[p];

            int dest = offset[col]++;
            AT.col_indices[dest] = i;   // 行列交換
            AT.values[dest]       = val;
        }
    }
}

void PCG_CP::build_ic0(const CsrMatrix& A, CsrMatrix& L, CsrMatrix& LT)
{
    const int n = A.num_rows;
    L.num_rows = n;
    L.row_ptr.assign(n + 1, 0);

    // 先決定 sparsity：只保留 A 的下三角 (j <= i)
    std::vector<int> cols;
    cols.reserve(A.values.size());

    for (int i = 0; i < n; ++i) {
        L.row_ptr[i] = (int)cols.size();
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            int j = A.col_indices[p];
            if (j <= i) {
                cols.push_back(j);
            }
        }
    }
    L.row_ptr[n] = (int)cols.size();

    int nnzL = (int)cols.size();
    L.col_indices.resize(nnzL);
    L.values.assign(nnzL, 0.0);

    for (int k = 0; k < nnzL; ++k) {
        L.col_indices[k] = cols[k];
    }

    const double eps   = 1e-14;
    const double shift = 1e-10;

    // IC(0) factorization
    for (int i = 0; i < n; ++i)
    {
        int row_start = L.row_ptr[i];
        int row_end   = L.row_ptr[i + 1];

        // 先處理 off-diagonal L(i,j), j < i
        for (int idx = row_start; idx < row_end; ++idx)
        {
            int j = L.col_indices[idx];
            if (j == i) continue;   // 對角之後再處理

            // A(i,j)
            double A_ij = get_A_ij(A, i, j);

            // sum_{k < j} L(i,k) * L(j,k)
            double sum = 0.0;

            int p_i = row_start;
            int p_j = L.row_ptr[j];
            int end_i = row_end;
            int end_j = L.row_ptr[j + 1];

            while (p_i < end_i && p_j < end_j) {
                int col_i = L.col_indices[p_i];
                int col_j = L.col_indices[p_j];

                if (col_i < col_j) {
                    ++p_i;
                } else if (col_j < col_i) {
                    ++p_j;
                } else {
                    int k = col_i;
                    if (k < j) {
                        sum += L.values[p_i] * L.values[p_j];
                    }
                    ++p_i;
                    ++p_j;
                }
            }

            // 找 L(j,j)
            double L_jj = 0.0;
            for (int p = L.row_ptr[j]; p < L.row_ptr[j + 1]; ++p) {
                if (L.col_indices[p] == j) {
                    L_jj = L.values[p];
                    break;
                }
            }
            if (std::fabs(L_jj) < eps) {
                // 若還未算好，給一個安全的值（理論上在 SPD 下 j<i 時已算出）
                double A_jj = get_A_ij(A, j, j);
                double diag_val = std::max(A_jj, shift);
                L_jj = std::sqrt(diag_val);
                // 寫回對角
                for (int p = L.row_ptr[j]; p < L.row_ptr[j + 1]; ++p) {
                    if (L.col_indices[p] == j) {
                        L.values[p] = L_jj;
                        break;
                    }
                }
            }

            L.values[idx] = (A_ij - sum) / L_jj;
        }

        // 然後處理 L(i,i)
        double A_ii = get_A_ij(A, i, i);
        double sum_diag = 0.0;
        for (int idx = row_start; idx < row_end; ++idx) {
            int col = L.col_indices[idx];
            if (col < i) {
                double Lik = L.values[idx];
                sum_diag += Lik * Lik;
            }
        }

        double diag_val = A_ii - sum_diag;
        if (diag_val <= eps) {
            diag_val += shift;
        }
        double L_ii = std::sqrt(diag_val);

        bool written = false;
        for (int idx = row_start; idx < row_end; ++idx) {
            if (L.col_indices[idx] == i) {
                L.values[idx] = L_ii;
                written = true;
                break;
            }
        }
        if (!written) {
            throw std::runtime_error("IC(0): diagonal entry not found in pattern.");
        }
    }

    // 同時產生 L^T 方便做 backward solve
    csr_transpose(L, LT);

    std::cout << "[IC(0)] build done. n = " << n
              << ", nnz(L) = " << nnzL << std::endl;
}
void PCG_CP::apply_ic0_preconditioner(
    const CsrMatrix& L,
    const CsrMatrix& LT,
    const std::vector<double>& r,
    std::vector<double>& z)
{
    int n = L.num_rows;
    if ((int)r.size() != n) {
        throw std::runtime_error("IC(0) preconditioner: size mismatch.");
    }

    if ((int)z.size() != n)
        z.assign(n, 0.0);

    std::vector<double> y(n, 0.0);

    // ---------- Forward solve: L y = r ----------
    for (int i = 0; i < n; ++i)
    {
        double sum = r[i];
        double L_ii = 0.0;

        for (int p = L.row_ptr[i]; p < L.row_ptr[i + 1]; ++p) {
            int col = L.col_indices[p];
            double val = L.values[p];

            if (col < i) {
                sum -= val * y[col];
            } else if (col == i) {
                L_ii = val;
            }
        }
        if (std::fabs(L_ii) < 1e-30) {
            throw std::runtime_error("IC(0) preconditioner: zero diagonal in L (forward).");
        }
        y[i] = sum / L_ii;
    }

    // ---------- Backward solve: L^T z = y ----------
    for (int i = n - 1; i >= 0; --i)
    {
        double sum = y[i];
        double L_ii = 0.0;

        // LT 的第 i 列 = L 的第 i 欄
        for (int p = LT.row_ptr[i]; p < LT.row_ptr[i + 1]; ++p) {
            int row = LT.col_indices[p]; // 這是原來 L 的 row index
            double val = LT.values[p];

            if (row > i) {
                // 對應到 L(row, i) * z[row]
                sum -= val * z[row];
            } else if (row == i) {
                L_ii = val;
            }
        }

        if (std::fabs(L_ii) < 1e-30) {
            throw std::runtime_error("IC(0) preconditioner: zero diagonal in L (backward).");
        }

        z[i] = sum / L_ii;
    }
}
int PCG_CP::pcg_ic0(const CsrMatrix& A,
        const std::vector<double>& b,
        std::vector<double>& x,
        int max_iter,
        double tolerance)
{
    resizing_x();
    int n = A.num_rows;
    if (b.size() != n || x.size() != n) {
        throw std::runtime_error("Vector and matrix dimensions do not match.");
    }
    count_iter = 0;
    build_ic0(A, L_ic0, LT_ic0);
    // 1. 提取 A 的對角線元素的倒數 (M_inv)
    std::vector<double> diag_inv(n);
    for (int i = 0; i < n; ++i) {
        bool found_diag = false;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_indices[j] == i) {
                diag_inv[i] = 1.0 / A.values[j];
                found_diag = true;
                break;
            }
        }
        if (!found_diag) {
            throw std::runtime_error("Matrix diagonal element missing.");
        }
    }

    // 2. 初始化
    std::vector<double> r(n);
    std::vector<double> p(n);
    std::vector<double> z(n);
    std::vector<double> Ap(n);

    // r = b - A*x_0 (如果 x_0 非零)
    spmv_csr(A, x, r); // r = A*x_0
    for (int i = 0; i < n; ++i) r[i] = b[i] - r[i]; // r = b - A*x_0
    /*ofstream check_r("./r.txt");
    for (int i = 0; i < n; ++i)
    {
        if(r[i]!= 0)
        { 
            check_r << r[i] << endl;
        }
        
    }
    exit(0);*/
    double initial_residual_norm = vector_norm(r);
    //if (initial_residual_norm < 1e-10) { // 初始猜測已經很準
    //    cout << "initial guess is good." << endl;
    //    return 0;
    //}

    // z = M^{-1} * r
    apply_ic0_preconditioner(L_ic0, LT_ic0, r, z);

    // p = z
    p = z;

    // rs_old = r^T * z
    double rs_old = dot_product(r, z);

    // 3. 迭代
    // int count_iter = 0;
    for (int k = 0; k < max_iter; ++k) {
        // Ap = A * p
        spmv_csr(A, p, Ap);

        // alpha = (r^T * z) / (p^T * A * p)
        double p_dot_Ap = dot_product(p, Ap);
        double alpha = rs_old / p_dot_Ap;

        // x = x + alpha * p
        axpy(alpha, p, x);

        // r = r - alpha * A * p
        axpy(-alpha, Ap, r);

        // 檢查收斂
        double residual_norm = vector_norm(r);
        if (count_iter % 200 == 0)
        {
            cout <<" iter : "<< count_iter << " = " << residual_norm << endl;
        }
        if (residual_norm / initial_residual_norm < tolerance) {
            std::cout << "PCG converged in " << k + 1 << " iterations." << std::endl;
            std::cout << "Final relative residual: " << (residual_norm / initial_residual_norm) << std::endl;
            return k + 1;
        }

        // z_new = M^{-1} * r_new
        apply_ic0_preconditioner(L_ic0, LT_ic0, r, z);

        // rs_new = r_new^T * z_new
        double rs_new = dot_product(r, z);

        // beta = rs_new / rs_old
        double beta = rs_new / rs_old;

        // p = z_new + beta * p
        for (int i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }

        // rs_old = rs_new
        rs_old = rs_new;
        count_iter++;
    }

    std::cerr << "PCG did not converge after " << max_iter << " iterations." << std::endl;
    return count_iter;
}
void PCG_CP::PCG()
{
    // indexSortCSR(A.values, A.col_indices, A.row_ptr);
    print_memory_stats("before PCG");
    if (mode == 0)
    {
        auto start = std::chrono::steady_clock::now();
        cout << "[Start Jacobi PCG]" << endl;
        jacobi_pcg(A, b, x, max_iter, tolerance);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Jacobi PCG Solving Time: " << elapsed_seconds.count() << " Sec\n";
        std::cout << "Final iteration count: " << count_iter << std::endl;
    }
    else if(mode == 1)
    {
        cout << "GS PCG" << endl;
        auto start = std::chrono::steady_clock::now();
        cout << "[Start GS PCG]" << endl;
        pcg_ssor(A, b, x, omega, max_iter, tolerance);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "GS PCG Solving Time: " << elapsed_seconds.count() << " Sec\n";
        std::cout << "Final iteration count: " << count_iter << std::endl;
    }
    else
    {
        auto start = std::chrono::steady_clock::now();
        cout << "[Start IC(0) PCG]" << endl;
        pcg_ic0(A, b, x, max_iter, tolerance);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "IC(0) PCG Solving Time: " << elapsed_seconds.count() << " Sec\n";
        std::cout << "Final iteration count: " << count_iter << std::endl;
    }
    // auto start = std::chrono::steady_clock::now();
    // cout << "[Start GS PCG]" << endl;
    // pcg_ssor(A, b, x, 1.3,max_iter, tolerance);
    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;
    // std::cout << "SSOR GS PCG Solving Time: " << elapsed_seconds.count() << " Sec\n";
    // std::cout << "Final iteration count: " << count_iter << std::endl;
    print_memory_stats("after PCG");
}
void PCG_CP::DirectSolver_Golden_Gen()
{
    // print_memory_stats("before DirectSolver");
    Golden_x.resize(A.num_rows,0);
    MKL_Pardiso_csr(this->A.num_rows, this->A.row_ptr, this->A.col_indices, this->A.values, Golden_x, this->b);
    ErrorCompare();
    // print_memory_stats("after DirectSolver");
}
void PCG_CP::ErrorCompare()
{
    Error.resize(A.num_rows, 0);
    double EPS = 1e-8;
    for (int i = 0; i < A.num_rows; i++)
    {
        Error[i] = abs(x[i] - Golden_x[i]);
        // if (Error[i] > EPS)
        // {
        //     cout << "Have Error" << endl;
        // }
    }
}
void _SplitString
(
    std::vector<std::string>& _token,
    std::string          _str,
    std::string          _delimiter
)
{
    int start = _str.find_first_not_of(_delimiter);
    int end = start;
    _token.clear();
    while (start != std::string::npos)
    {
        end = _str.find_first_of(_delimiter, start + 1);
        if (end == std::string::npos)
        {
            end = _str.length();
        }
        _token.push_back(_str.substr(start, end - start));
        start = _str.find_first_not_of(_delimiter, end + 1);
    }
    return;
}