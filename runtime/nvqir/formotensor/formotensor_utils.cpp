/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "formotensor_utils.h"
#include "common/Logger.h"
#include <fmt/format.h>
#include <cstdlib>

namespace nvqir {

ScratchDeviceMem::ScratchDeviceMem() {
  if (auto *scratchSizePercent =
          std::getenv("CUDAQ_FORMOTENSOR_SCRATCH_SIZE_PERCENTAGE")) {
    auto envIntVal = atoi(scratchSizePercent);
    // Bound the allowed value between 5% and 95%.
    // Note: values near these limits, while allowed, may still exhibit
    // instability, e.g., not enough workspace (if the scratch size is too
    // small).
    constexpr int minScratchSizePercent = 5;
    constexpr int maxScratchSizePercent = 95;

    if (envIntVal < minScratchSizePercent || envIntVal > maxScratchSizePercent)
      throw std::runtime_error(fmt::format(
          "Invalid CUDAQ_FORMOTENSOR_SCRATCH_SIZE_PERCENTAGE environment "
          "variable setting. Expecting a "
          "positive integer value between {} and {}, got '{}'.",
          minScratchSizePercent, maxScratchSizePercent, scratchSizePercent));

    freeMemRatio = static_cast<double>(envIntVal) / 100.0;
    CUDAQ_INFO("Setting FormoTensor scratch size ratio to {}.", freeMemRatio);
  }
}

// Compute the scratch size to allocate.
void ScratchDeviceMem::computeScratchSize() {
  // Query the free memory on Device
  std::size_t freeSize{0}, totalSize{0};
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
  scratchSize =
      (freeSize - (freeSize % 4096)) *
      freeMemRatio; // use a set proportion available memory with alignment
}

// Allocate scratch device memory based on available memory
void ScratchDeviceMem::allocate() {
  if (d_scratch)
    throw std::runtime_error(
        "Multiple scratch device memory allocations is not allowed.");

  computeScratchSize();
  // Try allocate device memory
  auto errCode = cudaMalloc(&d_scratch, scratchSize);
  if (errCode == cudaErrorMemoryAllocation) {
    // This indicates race condition whereby other GPU code is allocating
    // memory while we are calling cudaMemGetInfo.
    // Attempt to redo the allocation with an updated cudaMemGetInfo data.
    computeScratchSize();
    HANDLE_CUDA_ERROR(cudaMalloc(&d_scratch, scratchSize));
  } else {
    HANDLE_CUDA_ERROR(errCode);
  }
}

ScratchDeviceMem::~ScratchDeviceMem() {
  if (scratchSize > 0)
    HANDLE_CUDA_ERROR(cudaFree(d_scratch));
}

} // namespace nvqir



// 只在 simple_build 測試裡啟用這些 stub
namespace cudaq {
  namespace details {
  
  bool should_log(const LogLevel) {
    // 測試時可以乾脆全部關掉 log
    return false;
  }
  
  void trace(const std::string_view) {}
  void info(const std::string_view) {}
  void debug(const std::string_view) {}
  void warn(const std::string_view) {}
  
  std::string pathToFileName(const std::string_view fullFilePath) {
    // 簡單版：直接轉成 std::string 回傳
    return std::string(fullFilePath);
  }
  
  } // namespace details
  } // namespace cudaq