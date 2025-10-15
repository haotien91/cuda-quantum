/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cutensornet.h"
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace nvqir {

/// @brief Error handling macro for CUDA API calls
#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error %s in line %d\n", cudaGetErrorString(err), __LINE__); \
      fflush(stdout);                                                          \
      std::abort();                                                            \
    }                                                                          \
  }

/// @brief Error handling macro for cuTensorNet API calls
#define HANDLE_CUTN_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUTENSORNET_STATUS_SUCCESS) {                                   \
      printf("cuTensorNet error %s in line %d\n",                              \
             cutensornetGetErrorString(err), __LINE__);                        \
      fflush(stdout);                                                          \
      std::abort();                                                            \
    }                                                                          \
  }

/// @brief Track whether the tensor state is default initialized vs already has gates applied
constexpr std::int64_t InvalidTensorIndexValue = -1;

/// @brief Allocate gate matrix on device and copy data
template <typename T>
void *allocateGateMatrix(const std::vector<std::complex<T>> &gateMatrix) {
  void *d_gate = nullptr;
  const std::size_t gateSize = gateMatrix.size() * sizeof(std::complex<T>);
  
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gate, gateSize));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gate, gateMatrix.data(), gateSize,
                                cudaMemcpyHostToDevice));
  
  return d_gate;
}

/// @brief Struct to allocate and clean up device memory scratch space.
/// This is used for cuTensorNet workspace allocation.
struct ScratchDeviceMem {
  // Device pointer to scratch buffer
  void *d_scratch = nullptr;
  // Actual size in bytes
  std::size_t scratchSize = 0;
  // Ratio to current free memory size to allocate the scratch size.
  // Note: the actual allocation size may slightly be different due to alignment
  // consideration.
  // The default ratio if not otherwise specified.
  static inline constexpr double defaultFreeMemRatio = 0.5;
  double freeMemRatio = defaultFreeMemRatio;

  ScratchDeviceMem();
  
  // Compute the scratch size to allocate.
  void computeScratchSize();

  // Allocate scratch device memory based on available memory
  void allocate();

  ~ScratchDeviceMem();
};

/// @brief Generate an array of random values in the range (0.0, max)
inline std::vector<double> randomValues(uint64_t num_samples, double max_value,
                                        std::mt19937 &randomEngine) {
  std::vector<double> result(num_samples);
  std::uniform_real_distribution<double> dist(0.0, max_value);
  for (uint64_t i = 0; i < num_samples; i++) {
    result[i] = dist(randomEngine);
  }
  return result;
}

} // namespace nvqir
