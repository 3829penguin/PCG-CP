#include "PCG_CP.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include "omp.h"
using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <mode> <MatrixA.mtx> <VectorB.mtx> <iter> <tol> <omega(if mode = 1)>\n";
        return -1;
    }
    int nthreads = omp_get_max_threads();
    omp_set_num_threads(nthreads);
    cout << "Using " << nthreads << " threads for OpenMP parallel regions." << endl;
    PCG_CP solver;
    solver.mode = stoi(argv[1]);
    cout << "Mode =" << solver.mode << endl;
    std::string A ;
    std::string b ;
    if(solver.mode == 1)
    {
        A = argv[2];  
        b = argv[3];
        solver.max_iter = stoi(argv[4]);
        solver.tolerance = stod(argv[5]);
        solver.omega = stod(argv[6]);
    }
    else
    {
        A = argv[2];  
        b = argv[3];
        solver.max_iter = stoi(argv[4]);
        solver.tolerance = stod(argv[5]);
    }
    solver.parsing_A_from_mtx_and_convert2csr(A);
    solver.parsing_b_from_mtx(b);
    solver.check_A_symmetry();
    /*ofstream b_out("./b_out.txt");
    for (int i = 0; i < solver.b.size(); i++)
    {
        if (solver.b[i] != 0)
        {
            b_out << solver.b[i] << endl;
        }
    }
    b_out.close();*/
    //parsing build RHS b and initial guess x
    //pair<int, double> result = jacobi_pcg(A, b, x, /*tol=*/1e-10, /*max_iters=*/1000);
    // int final_iter = 0;
    // 執行求解器
    // if (mode == 0)
    // {
    //     auto start = std::chrono::steady_clock::now();
    //     cout << "[Start Jacobi PCG]" << endl;
    //     final_iter = solver.jacobi_pcg(solver.A, solver.b, solver.x, max_iter, tol);
    //     auto end = std::chrono::steady_clock::now();
    //     std::chrono::duration<double> elapsed_seconds = end - start;
    //     std::cout << "Jacobi PCG Solving Time: " << elapsed_seconds.count() << " 秒\n";
    //     std::cout << "Final iteration count: " << final_iter << std::endl;
    // }
    // else
    // {
    //     auto start = std::chrono::steady_clock::now();
    //     cout << "[Start GS PCG]" << endl;
    //     final_iter = solver.pcg_sgs(solver.A, solver.b, solver.x, max_iter, tol);
    //     auto end = std::chrono::steady_clock::now();
    //     std::chrono::duration<double> elapsed_seconds = end - start;
    //     std::cout << "GS PCG Solving Time: " << elapsed_seconds.count() << " 秒\n";
    //     std::cout << "Final iteration count: " << final_iter << std::endl;
    // }
    // solver.PCG();
    //===========================TESTING===============================
    solver.DirectSolver_Golden_Gen();
    //final_iter = solver.jacobi_pcg(solver.A, solver.b, solver.x, max_iter, tol);
    // 打印結果
    ofstream output_x("./Output_x.txt");
    output_x << "Solution x:" << std::endl;
    double max = INT64_MIN;
    double min = INT64_MAX;
    for (size_t i = 0; i < solver.x.size(); ++i) {
        output_x << "x[" << i << "] = " << solver.x[i] << std::endl;
        if (solver.x[i] < min)
        {
            min = solver.x[i];
        }
        if (solver.x[i] > max)
        {
            max = solver.x[i];
        }
    }
    ofstream output_Golden_x("./Output_Golden_x.txt");
    output_Golden_x << "Solution x:" << std::endl;
    double Golden_max = INT64_MIN;
    double Golden_min = INT64_MAX;
    for (size_t i = 0; i < solver.Golden_x.size(); ++i)
    {
        output_Golden_x << "x[" << i << "] = " << solver.Golden_x[i] << std::endl;
        if (solver.Golden_x[i] < Golden_min)
        {
            Golden_min = solver.Golden_x[i];
        }
        if (solver.Golden_x[i] > Golden_max)
        {
            Golden_max = solver.Golden_x[i];
        }
    }

    ofstream output_Error("./Output_Error.txt");
    output_Error << "Error:" << std::endl;
    double Error_max = INT64_MIN;
    double Error_min = INT64_MAX;
    for (size_t i = 0; i < solver.Error.size(); ++i)
    {
        output_Error << "Error[" << i << "] = " << solver.Error[i] << std::endl;
        if (solver.Error[i] < Error_min)
        {
            Error_min = solver.Error[i];
        }
        if (solver.Error[i] > Error_max)
        {
            Error_max = solver.Error[i];
        }
    }
    cout << "Max = " << max << " " << "min = " << min << endl;
    cout << "Golden_Max = " << Golden_max << " " << "Golden_min = " << Golden_min << endl;
    cout << "Error_Max = " << Error_max << " " << "Error_min = " << Error_min << endl;
    return 0;
}