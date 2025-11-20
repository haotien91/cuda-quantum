cmake_minimum_required(VERSION 3.20)
project(formotensor_simple)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# CUTENSORNET_ROOT=/home/tim911004/miniconda3/envs/cuda-quantum/lib/python3.10/site-packages/cuquantum
# Find required packages
find_package(CUDAToolkit REQUIRED)
# Find cuTensorNet manually
# Prefer versioned .so to avoid missing unversioned symlink
find_library(CUTENSORNET_LIB
    NAMES   libcutensornet.so.2 libcutensornet.so cutensornet
    HINTS   
        ${CUTENSORNET_ROOT}/lib
        /home/ubuntu/cuda-quantum/cuquantum/lib
)

find_path(CUTENSORNET_INC_DIR
    NAMES   cutensornet.h
    HINTS   
        ${CUTENSORNET_ROOT}/include      
        /home/ubuntu/cuda-quantum/cuquantum/include
)

if(NOT CUTENSORNET_LIB OR NOT CUTENSORNET_INC_DIR)
  message(FATAL_ERROR "\nUnable to find the cutensornet installation. Please ensure it is correctly installed and define CUTENSORNET_ROOT if necessary (currently set to: ${CUTENSORNET_ROOT}).")
endif()
get_filename_component(CUTENSORNET_LIB_DIR ${CUTENSORNET_LIB} DIRECTORY)
message(STATUS "CUTENSORNET_LIB: ${CUTENSORNET_LIB}")
message(STATUS "CUTENSORNET_INC: ${CUTENSORNET_INC_DIR}/cutensornet.h")

# Ensure linker can locate the found library directory
link_directories(${CUTENSORNET_LIB_DIR})

# Include directories
# Add repo root so includes like "common/FmtCore.h" resolve
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tpls/fmt/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tpls/eigen
    /home/ubuntu/cuda-quantum/cuquantum/include
)

# Add fmt as header-only library
add_library(fmt_header_only INTERFACE)
target_include_directories(fmt_header_only INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../../tpls/fmt/include)
target_compile_definitions(fmt_header_only INTERFACE FMT_HEADER_ONLY)

# Create a simple test executable
add_executable(test_batch_simple simple_test.cpp)
add_executable(tensor_batch_concept_test tensor_batch_concept_test.cpp)
add_executable(test_shared_params_batch test_shared_params_batch.cpp)

# Link libraries
target_link_libraries(test_batch_simple
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${CUTENSORNET_LIB}
    fmt_header_only
)

# Ensure the per-batch params test sees cuTensorNet headers
target_include_directories(test_shared_params_batch
    PRIVATE
    ${CUTENSORNET_INC_DIR}
)

target_link_libraries(tensor_batch_concept_test
    PRIVATE
    CUDA::cudart
)

target_link_directories(test_shared_params_batch PRIVATE ${CUTENSORNET_LIB_DIR})

target_link_libraries(test_shared_params_batch
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${CUTENSORNET_LIB}
    fmt_header_only
)

# Link required implementation units for utilities used by the test
target_sources(test_shared_params_batch
    PRIVATE
    formotensor_utils.cpp
)

# Compile definitions
target_compile_definitions(test_batch_simple PRIVATE CUDAQ_SIMULATION_SCALAR_FP64)
target_compile_definitions(test_shared_params_batch PRIVATE CUDAQ_SIMULATION_SCALAR_FP64 FORMOTENSOR_SIMPLE_TEST)

# GHZ Batch Test
add_executable(test_ghz_batch test_ghz_batch.cpp)
target_include_directories(test_ghz_batch PRIVATE ${CUTENSORNET_INC_DIR})
target_link_directories(test_ghz_batch PRIVATE ${CUTENSORNET_LIB_DIR})
target_link_libraries(test_ghz_batch
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${CUTENSORNET_LIB}
    fmt_header_only
)
target_sources(test_ghz_batch PRIVATE formotensor_utils.cpp)
target_compile_definitions(test_ghz_batch PRIVATE CUDAQ_SIMULATION_SCALAR_FP64 FORMOTENSOR_SIMPLE_TEST)

# Benchmark
add_executable(benchmark_shared_batch benchmark_shared_batch.cpp)
target_include_directories(benchmark_shared_batch PRIVATE ${CUTENSORNET_INC_DIR})
target_link_directories(benchmark_shared_batch PRIVATE ${CUTENSORNET_LIB_DIR})
target_link_libraries(benchmark_shared_batch
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${CUTENSORNET_LIB}
    fmt_header_only
)
target_sources(benchmark_shared_batch PRIVATE formotensor_utils.cpp)
target_compile_definitions(benchmark_shared_batch PRIVATE CUDAQ_SIMULATION_SCALAR_FP64 FORMOTENSOR_SIMPLE_TEST)
