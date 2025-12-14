#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <mkl.h> // 引入MKL库
#include <omp.h>
#include <stdexcept>
#include <unordered_map>
#include <stdio.h>
#include "mkl_rci.h"
#include "mkl_blas.h"
#include "mkl_spblas.h"
#include "mkl_service.h"
#include <map>
using namespace std;
void convertToPardisoCSR_vector(
    std::vector<int>& row_ptr,
    std::vector<int>& col_indices,
    std::vector<double>& values,
    std::vector<double>& a,
    std::vector<MKL_INT>& ia,
    std::vector<MKL_INT>& ja,
    int numProcs
);

void MKL_Pardiso_csr
(
    int nRow,
    std::vector<int>& row_ptr,
    std::vector<int>& col_indices,
    std::vector<double>& values,
    vector<double>& x,
    vector<double>& b
);