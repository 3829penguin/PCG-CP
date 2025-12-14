cmake_minimum_required(VERSION 3.16)
project(PCG_CP LANGUAGES CXX)

# ==========================
# Global compile flags
# ==========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -Wall -Wextra")

enable_testing()

# ==========================
# Include folder
# ==========================
include_directories(${CMAKE_SOURCE_DIR}/include)

# ==========================
# GoogleTest
# ==========================
add_subdirectory(googletest)

# ==========================
# PCG Library
# ==========================
add_library(pcg_lib
    src/PCG_CP.cpp
    src/MatrixComputation.cpp
)

set_target_properties(pcg_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(pcg_lib PUBLIC pthread)

# ==========================
# MKL
# ==========================
set(MKLROOT "/opt/intel/oneapi/mkl/latest")
set(IOMPROOT "/opt/intel/oneapi/compiler/latest")

target_include_directories(pcg_lib PUBLIC
    ${MKLROOT}/include
    ${IOMPROOT}/include
)

target_link_libraries(pcg_lib PUBLIC
    mkl_intel_lp64
    mkl_intel_thread
    mkl_core
    iomp5 pthread m dl
)

# ==========================
# Main Executable
# ==========================
add_executable(pcg_exec src/main.cpp)
target_link_libraries(pcg_exec PRIVATE pcg_lib)

# ==========================
# Test Executable
# ==========================
add_executable(test_exec tests/test_pcg.cpp)
target_link_libraries(test_exec PRIVATE pcg_lib gtest gtest_main)
add_test(NAME PCGTests COMMAND test_exec)


# ===========================================================
#                  PYBIND11 Python Module
# ===========================================================
find_package(pybind11 REQUIRED)

pybind11_add_module(pcgsolver src/pcg_bind.cpp)

target_include_directories(pcgsolver PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(pcgsolver PRIVATE pcg_lib)

