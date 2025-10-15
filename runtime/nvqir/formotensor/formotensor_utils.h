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

/// @brief Allocate and initialize device memory according to the input host data
template <typename T>
inline void *allocateGateMatrix(const std::vector<std::complex<T>> &gateMatHost) {
  void *d_gate{nullptr};
  const auto sizeBytes = gateMatHost.size() * sizeof(std::complex<T>);
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gate, sizeBytes));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gate, gateMatHost.data(), sizeBytes,
                               cudaMemcpyHostToDevice));
  return d_gate;
}

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
