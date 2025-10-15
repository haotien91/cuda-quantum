cmake_minimum_required(VERSION 3.20)
project(formotensor_simple)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
# Find cuTensorNet manually
find_library(CUTENSORNET_LIB
    NAMES   cutensornet libcutensornet.so.2
    HINTS   
        ${CUTENSORNET_ROOT}/lib
        /home/ubuntu/cuda-quantum/cuquantum/lib
)

find_file(CUTENSORNET_INC
    NAMES   cutensornet.h
    HINTS   
        ${CUTENSORNET_ROOT}/include      
        /home/ubuntu/cuda-quantum/cuquantum/include
)

if(NOT CUTENSORNET_LIB OR NOT CUTENSORNET_INC)
  message(FATAL_ERROR "\nUnable to find the cutensornet installation. Please ensure it is correctly installed and define CUTENSORNET_ROOT if necessary (currently set to: ${CUTENSORNET_ROOT}).")
endif()
message(STATUS "CUTENSORNET_LIB: ${CUTENSORNET_LIB}")
message(STATUS "CUTENSORNET_INC: ${CUTENSORNET_INC}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tpls/fmt/include
    /home/ubuntu/cuda-quantum/cuquantum/include
)

# Add fmt as header-only library
add_library(fmt_header_only INTERFACE)
target_include_directories(fmt_header_only INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../../tpls/fmt/include)

# Create a simple test executable
add_executable(test_batch_simple simple_test.cpp)
add_executable(tensor_batch_concept_test tensor_batch_concept_test.cpp)

# Link libraries
target_link_libraries(test_batch_simple
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${CUTENSORNET_LIB}
    fmt_header_only
)

target_link_libraries(tensor_batch_concept_test
    PRIVATE
    CUDA::cudart
)

# Compile definitions
target_compile_definitions(test_batch_simple PRIVATE CUDAQ_SIMULATION_SCALAR_FP64)
