/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "../common/EigenDense.h"
#include <algorithm>
#include <bitset>
#include <cassert>

namespace nvqir {

template <typename ScalarType>
FormoTensorState<ScalarType>::FormoTensorState(std::size_t numQubits,
                                               ScratchDeviceMem &inScratchPad,
                                               cutensornetHandle_t handle,
                                               std::mt19937 &randomEngine)
    : m_numQubitsPerSample(numQubits), m_batchSize(1), m_isBatched(false),
      m_cutnHandle(handle), scratchPad(inScratchPad), m_randomEngine(randomEngine) {
  
  // Create single state: dims = [2, 2, ..., 2] (n qubits)
  const std::vector<int64_t> qubitDims(m_numQubitsPerSample, 2);
  HANDLE_CUTN_ERROR(cutensornetCreateState(
      m_cutnHandle, CUTENSORNET_STATE_PURITY_PURE, m_numQubitsPerSample,
      qubitDims.data(), cudaDataType, &m_quantumState));
}

template <typename ScalarType>
FormoTensorState<ScalarType>::FormoTensorState(std::size_t numQubitsPerSample,
                                               std::size_t batchSize,
                                               ScratchDeviceMem &inScratchPad,
                                               cutensornetHandle_t handle,
                                               std::mt19937 &randomEngine)
    : m_numQubitsPerSample(numQubitsPerSample), m_batchSize(batchSize), m_isBatched(true),
      m_cutnHandle(handle), scratchPad(inScratchPad), m_randomEngine(randomEngine) {
  
  // Create batch state: dims = [d, 2, 2, ..., 2] (1 batch dim + n qubit dims)
  std::vector<int64_t> qubitDims(m_numQubitsPerSample + 1);
  qubitDims[0] = m_batchSize;  // batch dimension
  for (std::size_t i = 1; i <= m_numQubitsPerSample; i++) {
    qubitDims[i] = 2;  // qubit dimensions
  }
  
  HANDLE_CUTN_ERROR(cutensornetCreateState(
      m_cutnHandle, CUTENSORNET_STATE_PURITY_PURE, m_numQubitsPerSample + 1,
      qubitDims.data(), cudaDataType, &m_quantumState));
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
    // For batch mode, adjust target qubit indices (+1 because batch occupies index 0)
    std::vector<int32_t> adjustedTargets;
    for (auto target : targetQubits) {
      adjustedTargets.push_back(target + 1);
    }
    
    // Create batch-aware gate tensor
    void *batchGateMem = createBatchGateTensor(gateDeviceMem, targetQubits.size());
    
    if (controlQubits.empty()) {
      HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
          m_cutnHandle, m_quantumState, adjustedTargets.size(), adjustedTargets.data(),
          batchGateMem, nullptr, /*immutable*/ 1,
          /*adjoint*/ static_cast<int32_t>(adjoint), /*unitary*/ 1, &m_tensorId));
    } else {
      // Adjust control qubit indices as well
      std::vector<int32_t> adjustedControls;
      for (auto control : controlQubits) {
        adjustedControls.push_back(control + 1);
      }
      
      HANDLE_CUTN_ERROR(cutensornetStateApplyControlledTensorOperator(
          m_cutnHandle, m_quantumState, /*numControlModes=*/adjustedControls.size(),
          /*stateControlModes=*/adjustedControls.data(),
          /*stateControlValues=*/nullptr,
          /*numTargetModes*/ adjustedTargets.size(),
          /*stateTargetModes*/ adjustedTargets.data(), batchGateMem, nullptr,
          /*immutable*/ 1,
          /*adjoint*/ static_cast<int32_t>(adjoint), /*unitary*/ 1, &m_tensorId));
    }
    
    // Clean up temporary gate memory
    cudaFree(batchGateMem);
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

template <typename ScalarType>
void *FormoTensorState<ScalarType>::createBatchGateTensor(void *gateDeviceMem,
                                                          std::size_t numTargets) {
  // For batch mode, we need to create a gate tensor with an additional batch dimension
  // Original gate: [2, 2, ..., 2] (2*numTargets dimensions)
  // Batch gate: [1, 2, 2, ..., 2] (1 + 2*numTargets dimensions)
  
  const std::size_t gateSize = 1ULL << (2 * numTargets);
  const std::size_t batchGateSize = 1 * gateSize;  // batch dimension = 1
  
  void *batchGateMem{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&batchGateMem, batchGateSize * sizeof(std::complex<ScalarType>)));
  
  // Copy the original gate data to the batch gate tensor
  // The batch dimension is broadcast (size 1), so we just copy the original gate
  HANDLE_CUDA_ERROR(cudaMemcpy(batchGateMem, gateDeviceMem,
                               gateSize * sizeof(std::complex<ScalarType>),
                               cudaMemcpyDeviceToDevice));
  
  return batchGateMem;
}

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
  
  std::vector<std::vector<std::complex<ScalarType>>> batchStates;
  batchStates.reserve(m_batchSize);
  
  const std::size_t stateVecSize = 1ULL << m_numQubitsPerSample;
  
  // For each batch sample, contract the state vector
  for (std::size_t batchIdx = 0; batchIdx < m_batchSize; batchIdx++) {
    // Project to specific batch index
    std::vector<int32_t> projectedModes = {0};  // batch mode
    std::vector<int64_t> projectedValues = {static_cast<int64_t>(batchIdx)};
    
    auto [d_sv, svDim] = contractStateVectorInternal(projectedModes, projectedValues);
    std::vector<std::complex<ScalarType>> h_sv(svDim);
    HANDLE_CUDA_ERROR(cudaMemcpy(h_sv.data(), d_sv,
                                 svDim * sizeof(std::complex<ScalarType>),
                                 cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR(cudaFree(d_sv));
    
    batchStates.emplace_back(std::move(h_sv));
  }
  
  return batchStates;
}

template <typename ScalarType>
std::pair<void *, std::size_t>
FormoTensorState<ScalarType>::contractStateVectorInternal(
    const std::vector<int32_t> &projectedModes,
    const std::vector<int64_t> &in_projectedModeValues) {
  
  // Make sure that we don't overflow the memory size calculation.
  std::size_t effectiveQubits = m_isBatched ? m_numQubitsPerSample : m_numQubitsPerSample;
  if ((effectiveQubits - projectedModes.size()) > 64 ||
      (1ull << (effectiveQubits - projectedModes.size())) >
          std::numeric_limits<uint64_t>::max() /
              sizeof(std::complex<ScalarType>))
    throw std::runtime_error(
        "Too many qubits are requested for full state vector contraction.");
  
  void *d_sv{nullptr};
  const uint64_t svDim = 1ull << (effectiveQubits - projectedModes.size());
  HANDLE_CUDA_ERROR(
      cudaMalloc(&d_sv, svDim * sizeof(std::complex<ScalarType>)));
  
  // Create the quantum state amplitudes accessor
  cutensornetStateAccessor_t accessor;
  HANDLE_CUTN_ERROR(cutensornetCreateAccessor(
      m_cutnHandle, m_quantumState, projectedModes.size(),
      projectedModes.data(), in_projectedModeValues.data(), &accessor));
  
  // Configure the accessor
  HANDLE_CUTN_ERROR(cutensornetAccessorConfigure(
      m_cutnHandle, accessor, CUTENSORNET_ACCESSOR_OPT_NUM_HYPER_SAMPLES,
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
  HANDLE_CUTN_ERROR(cutensornetAccessorCompute(
      m_cutnHandle, accessor, nullptr, d_sv, workDesc, 0));
  
  // Clean up
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  HANDLE_CUTN_ERROR(cutensornetDestroyAccessor(accessor));
  
  return {d_sv, svDim};
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
