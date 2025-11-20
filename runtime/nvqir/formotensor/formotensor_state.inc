/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <bitset>
#include <cassert>
#include <fmt/format.h>
#include "../common/EigenDense.h"

namespace nvqir {

// Define the static member for numHyperSamples
template <typename ScalarType>
std::int32_t FormoTensorState<ScalarType>::numHyperSamples = []() {
  constexpr int32_t defaultNumHyperSamples = 8;
  if (auto envVal = std::getenv("CUDAQ_TENSORNET_NUM_HYPER_SAMPLES")) {
    int32_t specifiedNumHyperSamples = 0;
    try {
      specifiedNumHyperSamples = std::stoi(envVal);
      if (specifiedNumHyperSamples > 0)
        return specifiedNumHyperSamples;
    } catch (...) {
    }
  }
  return defaultNumHyperSamples;
}();

template <typename ScalarType>
FormoTensorState<ScalarType>::FormoTensorState(std::size_t numQubits,
                                               ScratchDeviceMem &inScratchPad,
                                               cutensornetHandle_t handle,
                                               std::mt19937 &randomEngine)
    : m_numQubitsPerSample(numQubits), m_batchSize(1), m_isBatched(false),
      m_cutnHandle(handle), scratchPad(inScratchPad), m_randomEngine(randomEngine) {
  
  m_quantumState = nullptr;
  setZeroState();
}

template <typename ScalarType>
FormoTensorState<ScalarType>::FormoTensorState(std::size_t numQubitsPerSample,
                                               std::size_t batchSize,
                                               ScratchDeviceMem &inScratchPad,
                                               cutensornetHandle_t handle,
                                               std::mt19937 &randomEngine)
    : m_numQubitsPerSample(numQubitsPerSample), m_batchSize(batchSize), m_isBatched(true),
      m_cutnHandle(handle), scratchPad(inScratchPad), m_randomEngine(randomEngine) {
  
  m_quantumState = nullptr;
  setZeroState();
}

template <typename ScalarType>
FormoTensorState<ScalarType>::~FormoTensorState() {
  // Destroy the quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetDestroyState(m_quantumState));
  for (auto *ptr : m_tempDevicePtrs)
    HANDLE_CUDA_ERROR(cudaFree(ptr));
}

template <typename ScalarType>
std::unique_ptr<FormoTensorState<ScalarType>>
FormoTensorState<ScalarType>::createFromStateVector(
    std::span<std::complex<ScalarType>> stateVec,
    ScratchDeviceMem &inScratchPad, cutensornetHandle_t handle,
    std::mt19937 &randomEngine) {
  
  const std::size_t numQubits = std::log2(stateVec.size());
  auto state = std::make_unique<FormoTensorState>(numQubits, inScratchPad, handle, randomEngine);

  // Support initializing the tensor network in a specific state vector state.
  // Reverse the qubit order to match cutensornet convention
  auto newStateVec = reverseQubitOrder(stateVec);
  auto ket = Eigen::Map<Eigen::Vector<std::complex<ScalarType>, Eigen::Dynamic>>(
      newStateVec.data(), newStateVec.size());
  Eigen::Vector<std::complex<ScalarType>, Eigen::Dynamic> initState =
      Eigen::Vector<std::complex<ScalarType>, Eigen::Dynamic>::Zero(stateVec.size());
  initState(0) = std::complex<ScalarType>{1.0, 0.0};
  Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic, Eigen::Dynamic>
      stateVecProj = ket * initState.transpose();
  assert(static_cast<std::size_t>(stateVecProj.size()) ==
         stateVec.size() * stateVec.size());
  stateVecProj.transposeInPlace();
  
  void *d_proj{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_proj, stateVecProj.size() *
                                        sizeof(std::complex<ScalarType>)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(d_proj, stateVecProj.data(),
                 stateVecProj.size() * sizeof(std::complex<ScalarType>),
                 cudaMemcpyHostToDevice));

  std::vector<int32_t> qubitIdx(numQubits);
  std::iota(qubitIdx.begin(), qubitIdx.end(), 0);
  // Project the state to the input state.
  state->applyQubitProjector(d_proj, qubitIdx);
  state->m_tempDevicePtrs.emplace_back(d_proj);
  return state;
}

template <typename ScalarType>
std::unique_ptr<FormoTensorState<ScalarType>>
FormoTensorState<ScalarType>::createFromBatchStateVector(
    std::size_t numQubitsPerSample, std::size_t batchSize,
    std::span<std::complex<ScalarType>> batchStateVec,
    ScratchDeviceMem &inScratchPad, cutensornetHandle_t handle,
    std::mt19937 &randomEngine) {
  
  auto state = std::make_unique<FormoTensorState>(
      numQubitsPerSample, batchSize, inScratchPad, handle, randomEngine);
  
  // Initialize batch data
  state->initializeBatchData(batchStateVec);
  
  return state;
}

template <typename ScalarType>
void FormoTensorState<ScalarType>::initializeBatchData(
    std::span<std::complex<ScalarType>> batchStateVec) {
  
  const std::size_t stateVecSize = 1ULL << m_numQubitsPerSample;
  const std::size_t totalSize = m_batchSize * stateVecSize;
  
  if (batchStateVec.size() != totalSize) {
    throw std::runtime_error(fmt::format(
        "Batch state vector size mismatch: expected {} elements, got {}",
        totalSize, batchStateVec.size()));
  }
  
  // Allocate device memory for batch data
  void *d_batchData{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_batchData, totalSize * sizeof(std::complex<ScalarType>)));
  
  // Copy batch data to device
  HANDLE_CUDA_ERROR(cudaMemcpy(d_batchData, batchStateVec.data(),
                               totalSize * sizeof(std::complex<ScalarType>),
                               cudaMemcpyHostToDevice));
  
  // Create projector to set the batch state
  // For batch mode, we need to project each sample independently
  std::vector<int32_t> allModes(m_numQubitsPerSample + 1);
  std::iota(allModes.begin(), allModes.end(), 0);
  
  // Apply the batch state projector
  applyQubitProjector(d_batchData, allModes);
  m_tempDevicePtrs.emplace_back(d_batchData);
}

template <typename ScalarType>
void FormoTensorState<ScalarType>::applyGate(
    const std::vector<int32_t> &controlQubits,
    const std::vector<int32_t> &targetQubits, void *gateDeviceMem,
    bool adjoint) {
  
  if (m_isBatched) {
    // With Batch Mode as Last Mode (ID = N), Qubit IDs (0..N-1) map directly to Mode IDs.
    // No index adjustment needed.
    
    if (controlQubits.empty()) {
      HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
          m_cutnHandle, m_quantumState, targetQubits.size(), targetQubits.data(),
          gateDeviceMem, nullptr, /*immutable*/ 1,
          /*adjoint*/ static_cast<int32_t>(adjoint), /*unitary*/ 1, &m_tensorId));
    } else {
      HANDLE_CUTN_ERROR(cutensornetStateApplyControlledTensorOperator(
          m_cutnHandle, m_quantumState, /*numControlModes=*/controlQubits.size(),
          /*stateControlModes=*/controlQubits.data(),
          /*stateControlValues=*/nullptr,
          /*numTargetModes*/ targetQubits.size(),
          /*stateTargetModes*/ targetQubits.data(), gateDeviceMem, nullptr,
          /*immutable*/ 1,
          /*adjoint*/ static_cast<int32_t>(adjoint), /*unitary*/ 1, &m_tensorId));
    }
    
    // No temporary memory to clean up (gate memory managed by caller/cache)
  } else {
    // Single state mode - use original logic
    if (controlQubits.empty()) {
      HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
          m_cutnHandle, m_quantumState, targetQubits.size(), targetQubits.data(),
          gateDeviceMem, nullptr, /*immutable*/ 1,
          /*adjoint*/ static_cast<int32_t>(adjoint), /*unitary*/ 1, &m_tensorId));
    } else {
      HANDLE_CUTN_ERROR(cutensornetStateApplyControlledTensorOperator(
          m_cutnHandle, m_quantumState, /*numControlModes=*/controlQubits.size(),
          /*stateControlModes=*/controlQubits.data(),
          /*stateControlValues=*/nullptr,
          /*numTargetModes*/ targetQubits.size(),
          /*stateTargetModes*/ targetQubits.data(), gateDeviceMem, nullptr,
          /*immutable*/ 1,
          /*adjoint*/ static_cast<int32_t>(adjoint), /*unitary*/ 1, &m_tensorId));
    }
  }
  
  m_tensorOps.emplace_back(AppliedTensorOp{gateDeviceMem, targetQubits,
                                           controlQubits, adjoint, true});
}

// Removed duplicate createBatchGateTensor definition. The implementation is provided later in file.

template <typename ScalarType>
void FormoTensorState<ScalarType>::applyQubitProjector(
    void *proj_d, const std::vector<int32_t> &qubitIdx) {
  
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
      m_cutnHandle, m_quantumState, qubitIdx.size(), qubitIdx.data(), proj_d,
      nullptr, /*immutable*/ 1, /*adjoint*/ 0, /*unitary*/ 0, &m_tensorId));
  
  m_tensorOps.emplace_back(AppliedTensorOp{proj_d, qubitIdx, {}, false, false});
}

template <typename ScalarType>
std::vector<std::complex<ScalarType>>
FormoTensorState<ScalarType>::getStateVector() {
  if (m_isBatched) {
    throw std::runtime_error("Cannot get single state vector from batch state. Use getBatchStateVectors() instead.");
  }
  
  // For single state, use the standard contraction
  auto [d_sv, svDim] = contractStateVectorInternal({}, {});
  std::vector<std::complex<ScalarType>> h_sv(svDim);
  HANDLE_CUDA_ERROR(cudaMemcpy(h_sv.data(), d_sv,
                               svDim * sizeof(std::complex<ScalarType>),
                               cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaFree(d_sv));
  return h_sv;
}

template <typename ScalarType>
std::vector<std::vector<std::complex<ScalarType>>>
FormoTensorState<ScalarType>::getBatchStateVectors() {
  if (!m_isBatched) {
    throw std::runtime_error("Cannot get batch state vectors from single state. Use getStateVector() instead.");
  }
  
  std::vector<std::vector<std::complex<ScalarType>>> batchStates(m_batchSize);
  
  const std::size_t stateVecSize = 1ULL << m_numQubitsPerSample;
  const std::size_t totalElements = m_batchSize * stateVecSize;
  
  // Perform full contraction without projecting the batch dimension
  // Empty projectedModes means calculate the full tensor (including batch dim)
  auto [d_sv, svDim] = contractStateVectorInternal({}, {});
  
  if (svDim != totalElements) {
    // Clean up before throwing
    HANDLE_CUDA_ERROR(cudaFree(d_sv));
    throw std::runtime_error("Contraction dimension mismatch: expected " + 
                             std::to_string(totalElements) + ", got " + 
                             std::to_string(svDim));
  }
  
  std::vector<std::complex<ScalarType>> h_fullSv(totalElements);
  HANDLE_CUDA_ERROR(cudaMemcpy(h_fullSv.data(), d_sv,
                               totalElements * sizeof(std::complex<ScalarType>),
                               cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaFree(d_sv));
  
  // Slice the full state vector into batch vectors
  for (std::size_t b = 0; b < m_batchSize; ++b) {
    auto startIter = h_fullSv.begin() + (b * stateVecSize);
    auto endIter = startIter + stateVecSize;
    batchStates[b].assign(startIter, endIter);
  }
  
  return batchStates;
}

template <typename ScalarType>
std::pair<void *, std::size_t>
FormoTensorState<ScalarType>::contractStateVectorInternal(
    const std::vector<int32_t> &projectedModes,
    const std::vector<int64_t> &in_projectedModeValues) {
  
  // Make sure that we don't overflow the memory size calculation.
  // If batched, the total dimension is B * 2^n. We need to account for the batch dimension
  // when calculating effective qubits or total dimension if not projected out.
  
  // effectiveQubits here represents the number of modes in the state tensor.
  // Single state: N modes (qubits)
  // Batch state: N+1 modes (Batch + qubits)
  std::size_t totalModes = m_isBatched ? m_numQubitsPerSample + 1 : m_numQubitsPerSample;
  std::size_t resultRank = totalModes - projectedModes.size();
  
  // Calculate result dimension
  // Note: This simple 1<<rank logic only works if all dimensions are 2.
  // But batch dim size is m_batchSize.
  // We need to check if batch dim (mode 0) is projected or not.
  
  uint64_t resultNumElements = 1;
  
  if (m_isBatched) {
    bool batchProjected = false;
    for (auto mode : projectedModes) {
      if (mode == static_cast<int32_t>(m_numQubitsPerSample)) {
        batchProjected = true;
        break;
      }
    }
    
    if (!batchProjected) {
      resultNumElements *= m_batchSize;
    }
    
    // All other N modes are size 2
    // Number of qubit modes remaining = resultRank - (batchProjected ? 0 : 1)
    std::size_t qubitModesRemaining = resultRank - (batchProjected ? 0 : 1);
    resultNumElements *= (1ULL << qubitModesRemaining);
    
  } else {
    resultNumElements = (1ULL << resultRank);
  }

  if (resultRank > 64 ||
      resultNumElements > std::numeric_limits<uint64_t>::max() / sizeof(std::complex<ScalarType>))
    throw std::runtime_error(
        "Too many qubits are requested for full state vector contraction.");
  
  void *d_sv{nullptr};
  HANDLE_CUDA_ERROR(
      cudaMalloc(&d_sv, resultNumElements * sizeof(std::complex<ScalarType>)));
  
  // Create the quantum state amplitudes accessor
  cutensornetStateAccessor_t accessor;
  HANDLE_CUTN_ERROR(cutensornetCreateAccessor(
      m_cutnHandle, m_quantumState, projectedModes.size(),
      projectedModes.data(), nullptr, &accessor));
  
  // Configure the accessor (align with 2.9.1)
  HANDLE_CUTN_ERROR(cutensornetAccessorConfigure(
      m_cutnHandle, accessor, CUTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES,
      &numHyperSamples, sizeof(numHyperSamples)));
  
  // Prepare the accessor
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  
  int64_t requiredWorkspaceSize = 0;
  HANDLE_CUTN_ERROR(cutensornetAccessorPrepare(
      m_cutnHandle, accessor, scratchPad.scratchSize, workDesc, 0));
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
      m_cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
      CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH,
      &requiredWorkspaceSize));
  
  assert(requiredWorkspaceSize > 0);
  assert(static_cast<std::size_t>(requiredWorkspaceSize) <= scratchPad.scratchSize);
  HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
      m_cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
      CUTENSORNET_WORKSPACE_SCRATCH, scratchPad.d_scratch, requiredWorkspaceSize));
  
  // Compute the state vector
  std::vector<int64_t> projectedModeValues =
      in_projectedModeValues.empty()
          ? std::vector<int64_t>(projectedModes.size(), 0)
          : in_projectedModeValues;
  std::complex<ScalarType> stateNorm{0.0, 0.0};
  HANDLE_CUTN_ERROR(cutensornetAccessorCompute(
      m_cutnHandle, accessor, projectedModeValues.data(), workDesc, d_sv,
      static_cast<void *>(&stateNorm), 0));
  
  // Clean up
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  HANDLE_CUTN_ERROR(cutensornetDestroyAccessor(accessor));
  
  return {d_sv, resultNumElements};
}

template <typename ScalarType>
std::vector<std::complex<ScalarType>>
FormoTensorState<ScalarType>::reverseQubitOrder(
    std::span<std::complex<ScalarType>> stateVec) {
  
  const std::size_t numQubits = std::log2(stateVec.size());
  std::vector<std::complex<ScalarType>> reversedStateVec(stateVec.size());
  
  for (std::size_t i = 0; i < stateVec.size(); i++) {
    std::bitset<64> bits(i);
    std::bitset<64> reversedBits;
    
    for (std::size_t j = 0; j < numQubits; j++) {
      reversedBits[j] = bits[numQubits - 1 - j];
    }
    
    reversedStateVec[reversedBits.to_ullong()] = stateVec[i];
  }
  
  return reversedStateVec;
}

template <typename ScalarType>
void FormoTensorState<ScalarType>::addQubits(std::size_t numQubits) {
  if (m_isBatched) {
    throw std::runtime_error("Cannot add qubits to batch state dynamically");
  }
  
  // For simplicity, we don't support dynamic qubit addition in batch mode
  // This would require reconstructing the entire state
  throw std::runtime_error("addQubits not yet implemented for FormoTensorState");
}

template <typename ScalarType>
void FormoTensorState<ScalarType>::setZeroState() {
  // Reset the state to |0...0>
  // This is done by destroying and recreating the state
  if (m_quantumState) {
      HANDLE_CUTN_ERROR(cutensornetDestroyState(m_quantumState));
      m_quantumState = nullptr;
  }
  
  if (m_isBatched) {
    // CHANGE: Put Batch Dimension LAST to ensure C-style layout [B, 2^n]
    // when using cuTensorNet's default F-order output.
    // Modes: 0, 1, ..., n-1 (Qubits), n (Batch)
    std::vector<int64_t> qubitDims(m_numQubitsPerSample + 1);
    for (std::size_t i = 0; i < m_numQubitsPerSample; i++) {
      qubitDims[i] = 2;
    }
    qubitDims[m_numQubitsPerSample] = m_batchSize;

    HANDLE_CUTN_ERROR(cutensornetCreateState(
        m_cutnHandle, CUTENSORNET_STATE_PURITY_PURE, m_numQubitsPerSample + 1,
        qubitDims.data(), cudaDataType, &m_quantumState));
    
    // Initialize Batch Mode: Broadcast |0>_B to sum |b>_B
    // We apply a B x B operator on Mode N (Batch Mode).
    // Op[b, 0] = 1.0 (First column is 1s).
    
    std::vector<std::complex<ScalarType>> batchOp(m_batchSize * m_batchSize, {0.0, 0.0});
    for(size_t b=0; b<m_batchSize; ++b) {
        batchOp[b * m_batchSize] = {1.0, 0.0};
    }
    
    void* d_op = nullptr;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_op, batchOp.size() * sizeof(std::complex<ScalarType>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_op, batchOp.data(), batchOp.size() * sizeof(std::complex<ScalarType>), cudaMemcpyHostToDevice));
    
    int32_t mode = static_cast<int32_t>(m_numQubitsPerSample); // Batch mode is Last
    int64_t tensorId; // Local id
    HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
        m_cutnHandle, m_quantumState, 1, &mode, d_op, nullptr, 
        1, 0, 0, &tensorId));
        
    m_tempDevicePtrs.push_back(d_op);

  } else {
    const std::vector<int64_t> qubitDims(m_numQubitsPerSample, 2);
    HANDLE_CUTN_ERROR(cutensornetCreateState(
        m_cutnHandle, CUTENSORNET_STATE_PURITY_PURE, m_numQubitsPerSample,
        qubitDims.data(), cudaDataType, &m_quantumState));
  }
  
  m_tensorId = InvalidTensorIndexValue;
  m_tensorOps.clear();
}

template <typename ScalarType>
void FormoTensorState<ScalarType>::applyUnitaryChannel(
    const std::vector<int32_t> &qubits, const std::vector<void *> &krausOps,
    const std::vector<double> &probabilities) {
  
  // Qubit indices map directly to Mode indices (Batch Mode is Last)
  // No adjustment needed.

  // Create batch-aware Kraus operators if in batch mode
  std::vector<void *> deviceOps;
  deviceOps.reserve(krausOps.size());
  
  if (m_isBatched) {
    for (auto op : krausOps) {
      // Create batch-aware tensor: [2^k, 2^k] -> [1, 2^k, 2^k]
      // NOTE: createBatchGateTensor needs update if it assumes specific layout?
      // It adds a dimension of size 1. Where?
      // If we use broadcasting, we might not need createBatchGateTensor at all?
      // But Channels might require matching rank? 
      // Let's stick to broadcasting if possible.
      // cuTensorNet supports broadcasting for channel operators too if modes match.
      // So we can use original ops!
      deviceOps.push_back(op);
    }
  } else {
    deviceOps = krausOps;
  }

  HANDLE_CUTN_ERROR(cutensornetStateApplyUnitaryChannel(
      m_cutnHandle, m_quantumState,
      /*numStateModes=*/static_cast<int32_t>(qubits.size()),
      /*stateModes=*/qubits.data(),
      /*numTensors=*/static_cast<int32_t>(deviceOps.size()),
      /*tensorData=*/const_cast<void **>(deviceOps.data()),
      /*tensorModeStrides=*/nullptr,
      /*probabilities=*/probabilities.data(), &m_tensorId));
  
  m_hasNoiseChannel = true;
}

template <typename ScalarType>
void FormoTensorState<ScalarType>::applyGeneralChannel(
    const std::vector<int32_t> &qubits, const std::vector<void *> &krausOps) {
  
  // Qubit indices map directly to Mode indices.

  // Create batch-aware Kraus operators if in batch mode
  std::vector<void *> deviceOps;
  deviceOps.reserve(krausOps.size());
  
  if (m_isBatched) {
    for (auto op : krausOps) {
       // Use broadcasting
       deviceOps.push_back(op);
    }
  } else {
    deviceOps = krausOps;
  }

  HANDLE_CUTN_ERROR(cutensornetStateApplyGeneralChannel(
      m_cutnHandle, m_quantumState,
      /*numStateModes=*/static_cast<int32_t>(qubits.size()),
      /*stateModes=*/qubits.data(),
      /*numTensors=*/static_cast<int32_t>(deviceOps.size()),
      /*tensorData=*/const_cast<void **>(deviceOps.data()),
      /*tensorModeStrides=*/nullptr, &m_tensorId));
  
  m_hasNoiseChannel = true;
}

template <typename ScalarType>
std::unordered_map<std::string, size_t>
FormoTensorState<ScalarType>::executeSample(
    cutensornetStateSampler_t &sampler,
    cutensornetWorkspaceDescriptor_t &workDesc,
    const std::vector<int32_t> &measuredBitIds, int32_t shots,
    bool enableCacheWorkspace) {
  
  if (m_isBatched) {
    throw std::runtime_error(
        "executeSample is for single state only. Use sampleBatch() for batch mode.");
  }

  // This simplified version ignores cache workspace and hyper-sampling.
  // In production, you should allocate cache memory as in cutensornet.
  std::unordered_map<std::string, size_t> counts;

  // Perform sampling in chunks. If noise channels have been applied,
  // cuTensorNet requires sampling one shot at a time (trajectory simulation).
  const bool trajectory = m_hasNoiseChannel;
  const int64_t maxShotsPerRun = trajectory ? 1 : shots;
  int64_t shotsRemaining = shots;
  
  while (shotsRemaining > 0) {
    const int64_t numShots = std::min(shotsRemaining, maxShotsPerRun);
    std::vector<int64_t> samples(measuredBitIds.size() * numShots);
    
    // Generate a random seed for the sampler. This ensures that
    // consecutive calls produce different shot sets while remaining deterministic.
    const int32_t rndSeed = m_randomEngine();
    HANDLE_CUTN_ERROR(cutensornetSamplerConfigure(
        m_cutnHandle, sampler, CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC,
        &rndSeed, sizeof(rndSeed)));
    
    // Perform the sampling.
    HANDLE_CUTN_ERROR(cutensornetSamplerSample(
        m_cutnHandle, sampler, numShots, workDesc, samples.data(), 0));
    
    // Convert bit values into bitstrings and accumulate counts.
    const std::size_t numMeasured = measuredBitIds.size();
    std::string bitstr(numMeasured, '0');
    constexpr char digits[2] = {'0', '1'};
    
    for (int64_t i = 0; i < numShots; ++i) {
      for (std::size_t j = 0; j < numMeasured; ++j) {
        bitstr[j] = digits[samples[i * numMeasured + j]];
      }
      counts[bitstr] += 1;
    }
    
    shotsRemaining -= numShots;
  }
  
  return counts;
}

template <typename ScalarType>
std::vector<std::unordered_map<std::string, std::size_t>>
FormoTensorState<ScalarType>::sampleBatch(
    const std::vector<int32_t> &measuredQubits, int32_t shots) {
  
  if (!m_isBatched) {
    throw std::runtime_error("sampleBatch only valid in batch mode.");
  }

  const std::size_t numMeasured = measuredQubits.size();
  const std::size_t stateDim = 1ULL << m_numQubitsPerSample;

  // Get state vectors for each batch sample
  auto batchStates = getBatchStateVectors();
  std::vector<std::unordered_map<std::string, std::size_t>> results(m_batchSize);

  // Precompute bit shifts for measurement
  std::vector<int32_t> shifts(numMeasured);
  for (std::size_t i = 0; i < numMeasured; ++i) {
    shifts[i] = measuredQubits[i];
  }

  // Sample each batch independently
  for (std::size_t b = 0; b < m_batchSize; ++b) {
    const auto &stateVec = batchStates[b];
    
    // Compute probability distribution for measurement outcomes
    std::unordered_map<std::string, double> probMap;
    probMap.reserve(1ULL << numMeasured);
    
    for (std::size_t idx = 0; idx < stateDim; ++idx) {
      // Extract measured qubit bits from index
      std::string bitstr(numMeasured, '0');
      for (std::size_t k = 0; k < numMeasured; ++k) {
        std::size_t bitVal = (idx >> shifts[k]) & 0x1;
        bitstr[numMeasured - 1 - k] = bitVal ? '1' : '0';
      }
      double amp2 = std::norm(stateVec[idx]);
      probMap[bitstr] += amp2;
    }

    // Normalize probabilities
    double totalProb = 0.0;
    for (const auto &[_, p] : probMap) {
      totalProb += p;
    }
    for (auto &kv : probMap) {
      kv.second /= totalProb;
    }

    // Create probability vector and key vector for sampling
    std::vector<double> probs;
    std::vector<std::string> keys;
    probs.reserve(probMap.size());
    keys.reserve(probMap.size());
    for (const auto &kv : probMap) {
      keys.push_back(kv.first);
      probs.push_back(kv.second);
    }

    // Sample using discrete distribution
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    std::unordered_map<std::string, std::size_t> count;
    count.reserve(1ULL << numMeasured);
    
    for (int i = 0; i < shots; ++i) {
      int sampleIdx = dist(m_randomEngine);
      const std::string &bitstring = keys[sampleIdx];
      count[bitstring] += 1;
    }
    
    results[b] = std::move(count);
  }
  
  return results;
}

template <typename ScalarType>
std::vector<std::unordered_map<std::string, std::size_t>>
FormoTensorState<ScalarType>::sampleBatchGpu(
    const std::vector<int32_t> &measuredQubits, int32_t shots) {
  if (!m_isBatched) {
    throw std::runtime_error("sampleBatchGpu only valid in batch mode");
  }

  std::vector<std::unordered_map<std::string, std::size_t>> results(m_batchSize);

  for (std::size_t b = 0; b < m_batchSize; ++b) {
    const std::vector<int32_t> projectedModes{static_cast<int32_t>(m_numQubitsPerSample)};
    const std::vector<int64_t> projectedVals{static_cast<int64_t>(b)};

    auto [d_sv, svDim] = contractStateVectorInternal(projectedModes, projectedVals);

    std::vector<std::complex<ScalarType>> hostState(svDim);
    HANDLE_CUDA_ERROR(cudaMemcpy(hostState.data(), d_sv,
                                 svDim * sizeof(std::complex<ScalarType>),
                                 cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR(cudaFree(d_sv));

    std::span<std::complex<ScalarType>> stateSpan(hostState.data(), hostState.size());
    auto sampleState = FormoTensorState<ScalarType>::createFromStateVector(
        stateSpan, scratchPad, m_cutnHandle, m_randomEngine);

    cutensornetStateSampler_t sampler;
    HANDLE_CUTN_ERROR(cutensornetCreateSampler(
        m_cutnHandle, sampleState->m_quantumState,
        static_cast<int32_t>(measuredQubits.size()), measuredQubits.data(), &sampler));

    HANDLE_CUTN_ERROR(cutensornetSamplerConfigure(
        m_cutnHandle, sampler, CUTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES,
        &numHyperSamples, sizeof(numHyperSamples)));

    cutensornetWorkspaceDescriptor_t workDesc;
    HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
    HANDLE_CUTN_ERROR(cutensornetSamplerPrepare(
        m_cutnHandle, sampler, scratchPad.scratchSize, workDesc, 0));

    int64_t worksize = 0;
    HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
        m_cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
        CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, &worksize));

    if (worksize > 0) {
      if (static_cast<std::size_t>(worksize) > scratchPad.scratchSize) {
        HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
        HANDLE_CUTN_ERROR(cutensornetDestroySampler(sampler));
        throw std::runtime_error("Insufficient device workspace for sampler.");
      }
      HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
          m_cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
          CUTENSORNET_WORKSPACE_SCRATCH, scratchPad.d_scratch, worksize));
    }

    auto counts = sampleState->executeSample(
        sampler, workDesc, measuredQubits, shots, /*enableCacheWorkspace=*/false);

    HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
    HANDLE_CUTN_ERROR(cutensornetDestroySampler(sampler));

    results[b] = std::move(counts);
  }

  return results;
}

template <typename ScalarType>
void FormoTensorState<ScalarType>::applyBatchParametrizedGate(
    const std::vector<int32_t> &targetQubits,
    const std::string &gateName,
    const std::vector<std::vector<double>> &batchParams) {
  
  if (!m_isBatched) {
    throw std::runtime_error(
        "applyBatchParametrizedGate only valid in batch mode");
  }

  // Current implementation of cuTensorNet State API does not support
  // block-diagonal operators over the batch dimension in a single contraction
  // without explicit network construction.
  // To ensure mathematical correctness ("soundness"), we disable this feature
  // until a Tensor Network API based implementation is available.
  // Users should rely on "Shared Parameter Gate + Batch State" for acceleration.
  
  throw std::runtime_error(
      "applyBatchParametrizedGate is not currently supported with the single-contraction "
      "State API backend. Please use shared parameters (standard applyGate) for batch acceleration, "
      "or wait for future updates enabling per-batch parameters via Network API.");
  
  /* 
   * Previous implementation (disabled for correctness):
   * Attempts to use [B, 2^k, 2^k] tensor with mode 0 were numerically incorrect
   * as they treated batch mode as a physical coupling dimension.
   */
}

template <typename ScalarType>
std::vector<std::complex<ScalarType>>
FormoTensorState<ScalarType>::generateGateMatrix(
    const std::string &gateName,
    const std::vector<double> &params) {
  
  if (params.empty()) {
    throw std::runtime_error("Gate parameters cannot be empty");
  }
  
  const double theta = params[0];
  
  if (gateName == "ry") {
    // Ry(θ) = [[cos(θ/2), -sin(θ/2)],
    //          [sin(θ/2),  cos(θ/2)]]
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    return {
      {c, 0.0}, {-s, 0.0},
      {s, 0.0}, {c, 0.0}
    };
  } 
  else if (gateName == "rz") {
    // Rz(θ) = [[e^(-iθ/2), 0],
    //          [0, e^(iθ/2)]]
    const double half_theta = theta / 2.0;
    const double c = std::cos(half_theta);
    const double s = std::sin(half_theta);
    return {
      {c, -s}, {0.0, 0.0},
      {0.0, 0.0}, {c, s}
    };
  } 
  else if (gateName == "rx") {
    // Rx(θ) = [[cos(θ/2), -i*sin(θ/2)],
    //          [-i*sin(θ/2), cos(θ/2)]]
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    return {
      {c, 0.0}, {0.0, -s},
      {0.0, -s}, {c, 0.0}
    };
  }
  
  throw std::runtime_error("Unsupported gate type: " + gateName + 
                          ". Supported: ry, rz, rx");
}

template <typename ScalarType>
void *FormoTensorState<ScalarType>::createBatchGateTensor(
    void *gateDeviceMem, std::size_t numTargets) {
  
  if (!m_isBatched) {
    return gateDeviceMem; // No modification needed for single state
  }
  
  // Gate matrix size: (2^numTargets) x (2^numTargets)
  const std::size_t gateDim = 1ULL << numTargets;
  const std::size_t gateSize = gateDim * gateDim;
  const std::size_t elemSize = sizeof(std::complex<ScalarType>);
  
  // Allocate batch gate tensor: [1, gateDim, gateDim]
  // Total size = 1 * gateSize
  void *batchGate_d = nullptr;
  HANDLE_CUDA_ERROR(cudaMalloc(&batchGate_d, gateSize * elemSize));
  
  // Copy the gate matrix (broadcasting will be handled by cuTensorNet)
  HANDLE_CUDA_ERROR(cudaMemcpy(batchGate_d, gateDeviceMem, 
                                gateSize * elemSize,
                                cudaMemcpyDeviceToDevice));
  
  return batchGate_d;
}

template <typename ScalarType>
std::unique_ptr<FormoTensorState<ScalarType>>
FormoTensorState<ScalarType>::clone() const {
  // Create a new state with the same parameters
  auto clonedState = std::make_unique<FormoTensorState>(
      m_numQubitsPerSample, m_batchSize, scratchPad, m_cutnHandle, m_randomEngine);
  
  // Copy the tensor operations
  clonedState->m_tensorOps = m_tensorOps;
  clonedState->m_hasNoiseChannel = m_hasNoiseChannel;
  
  // Reapply all tensor operations
  int64_t tensorId = 0;
  for (const auto &op : m_tensorOps) {
    if (op.deviceData) {
      if (op.controlQubitIds.empty()) {
        HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
            m_cutnHandle, clonedState->m_quantumState, op.targetQubitIds.size(),
            op.targetQubitIds.data(), op.deviceData, nullptr, /*immutable*/ 1,
            /*adjoint*/ static_cast<int32_t>(op.isAdjoint),
            /*unitary*/ static_cast<int32_t>(op.isUnitary), &tensorId));
      } else {
        HANDLE_CUTN_ERROR(cutensornetStateApplyControlledTensorOperator(
            m_cutnHandle, clonedState->m_quantumState,
            /*numControlModes=*/op.controlQubitIds.size(),
            /*stateControlModes=*/op.controlQubitIds.data(),
            /*stateControlValues=*/nullptr,
            /*numTargetModes*/ op.targetQubitIds.size(),
            /*stateTargetModes*/ op.targetQubitIds.data(), op.deviceData,
            nullptr, /*immutable*/ 1,
            /*adjoint*/ static_cast<int32_t>(op.isAdjoint),
            /*unitary*/ static_cast<int32_t>(op.isUnitary), &tensorId));
      }
    }
  }
  
  return clonedState;
}

} // namespace nvqir
