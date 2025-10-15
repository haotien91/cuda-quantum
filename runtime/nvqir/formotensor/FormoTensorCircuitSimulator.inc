/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq.h"
#include "cutensornet.h"
#include "formotensor_utils.h"

namespace nvqir {

// Constructor
template <typename ScalarType>
SimulatorFormoTensor<ScalarType>::SimulatorFormoTensor()
    : m_randomEngine(std::random_device()()) {
  int numDevices{0};
  HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
  
  // Select device (support MPI if available)
  const int deviceId = cudaq::mpi::is_initialized() 
                          ? cudaq::mpi::rank() % numDevices 
                          : 0;
  HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
  
  // Initialize cuTensorNet
  HANDLE_CUTN_ERROR(cutensornetCreate(&m_cutnHandle));
  
  // Allocate scratch pad after device selection
  scratchPad.allocate();
  
  // Initialize MPI communication if available
  if (cudaq::mpi::is_initialized()) {
    initCuTensornetComm(m_cutnHandle);
  }
  
  CUDAQ_INFO("FormoTensor simulator initialized with batch processing support");
}

// Destructor
template <typename ScalarType>
SimulatorFormoTensor<ScalarType>::~SimulatorFormoTensor() {
  if (m_cutnHandle) {
    HANDLE_CUTN_ERROR(cutensornetDestroy(m_cutnHandle));
  }
}

// Add qubits to state
template <typename ScalarType>
void SimulatorFormoTensor<ScalarType>::addQubitsToState(std::size_t numQubits, 
                                                        const void *ptr) {
  if (m_isBatchMode) {
    throw std::runtime_error(
        "Cannot add qubits in batch mode. Use initializeBatchState() instead.");
  }
  
  if (!m_state) {
    if (!ptr) {
      // Create zero state
      m_state = std::make_unique<FormoTensorState<ScalarType>>(
          numQubits, scratchPad, m_cutnHandle, m_randomEngine);
    } else {
      // Create from state vector
      auto *casted = reinterpret_cast<const std::complex<ScalarType> *>(ptr);
      std::span<std::complex<ScalarType>> stateVec(casted, 1ULL << numQubits);
      m_state = FormoTensorState<ScalarType>::createFromStateVector(
          stateVec, scratchPad, m_cutnHandle, m_randomEngine);
    }
    m_numQubitsPerSample = numQubits;
  } else {
    // Add more qubits to existing state
    m_state->addQubits(numQubits);
    m_numQubitsPerSample += numQubits;
  }
}

// Initialize batch state
template <typename ScalarType>
void SimulatorFormoTensor<ScalarType>::initializeBatchState(
    std::size_t numQubits, std::size_t batchSize, const void *data) {
  if (m_state) {
    throw std::runtime_error("State already initialized. Call resetState() first.");
  }
  
  if (batchSize == 0) {
    throw std::runtime_error("Batch size must be greater than 0");
  }
  
  auto *casted = reinterpret_cast<const std::complex<ScalarType> *>(data);
  const std::size_t stateVecSize = 1ULL << numQubits;
  const std::size_t totalSize = batchSize * stateVecSize;
  
  std::span<std::complex<ScalarType>> batchStateVec(casted, totalSize);
  
  m_state = FormoTensorState<ScalarType>::createFromBatchStateVector(
      numQubits, batchSize, batchStateVec, scratchPad, m_cutnHandle, m_randomEngine);
  
  m_isBatchMode = true;
  m_batchSize = batchSize;
  m_numQubitsPerSample = numQubits;
  
  CUDAQ_INFO("Initialized batch state: {} qubits per sample, {} samples", 
             numQubits, batchSize);
}

// Apply gate with caching (similar to cutensornet approach)
template <typename ScalarType>
void SimulatorFormoTensor<ScalarType>::applyGate(const GateApplicationTask &task) {
  if (!m_state) {
    throw std::runtime_error("No quantum state initialized");
  }
  
  const auto &controls = task.controls;
  const auto &targets = task.targets;
  
  // Create cache key for gate
  const std::string gateKey = task.operationName + "_" + [&]() {
    std::stringstream paramsSs;
    for (const auto &param : task.parameters) {
      paramsSs << param << "_";
    }
    // Simple hash for matrix
    std::size_t matHash = task.matrix.size();
    for (const auto &val : task.matrix) {
      matHash ^= std::hash<ScalarType>{}(val.real()) + 
                 std::hash<ScalarType>{}(val.imag());
    }
    return paramsSs.str() + "__" + std::to_string(matHash);
  }();
  
  // Handle controlled gates
  if (controls.size() <= m_maxControlledRankForFullTensorExpansion) {
    // Expand full tensor for small number of controls
    std::vector<std::int32_t> qubitOperands(controls.begin(), controls.end());
    qubitOperands.insert(qubitOperands.end(), targets.begin(), targets.end());
    
    const auto expandedMatKey = 
        gateKey + "_c(" + std::to_string(controls.size()) + ")";
    
    const auto iter = m_gateDeviceMemCache.find(expandedMatKey);
    if (iter != m_gateDeviceMemCache.end()) {
      m_state->applyGate(/*controlQubits=*/{}, qubitOperands, iter->second);
    } else {
      // Generate expanded gate tensor
      const auto expandedGateMat = 
          generateFullGateTensor(controls.size(), task.matrix);
      void *dMem = allocateGateMatrix(expandedGateMat);
      m_gateDeviceMemCache[expandedMatKey] = dMem;
      m_state->applyGate(/*controlQubits=*/{}, qubitOperands, dMem);
    }
  } else {
    // Propagate control qubits to cutensornet
    const auto iter = m_gateDeviceMemCache.find(gateKey);
    if (iter == m_gateDeviceMemCache.end()) {
      void *dMem = allocateGateMatrix(task.matrix);
      m_gateDeviceMemCache[gateKey] = dMem;
    }
    
    const std::vector<std::int32_t> ctrlQubits(controls.begin(), controls.end());
    const std::vector<std::int32_t> targetQubits(targets.begin(), targets.end());
    m_state->applyGate(ctrlQubits, targetQubits, m_gateDeviceMemCache[gateKey]);
  }
}

// Generate full gate tensor with controls
template <typename ScalarType>
std::vector<std::complex<ScalarType>>
SimulatorFormoTensor<ScalarType>::generateFullGateTensor(
    std::size_t num_control_qubits,
    const std::vector<std::complex<ScalarType>> &target_gate) {
  
  const auto mat_size = target_gate.size();
  assert(std::ceil(std::sqrt(mat_size)) == std::floor(std::sqrt(mat_size)) &&
         "Input matrix is not a square matrix.");
  
  const std::size_t target_gate_dim = std::ceil(std::sqrt(mat_size));
  const std::size_t num_target_qubits = std::bit_width(target_gate_dim) - 1;
  
  if (num_control_qubits == 0)
    return target_gate;
  
  const std::size_t full_dim = (1UL << (num_control_qubits + num_target_qubits));
  std::vector<std::complex<ScalarType>> gate_tensor(full_dim * full_dim, {0.0, 0.0});
  
  std::size_t offset = 0;
  // Set diagonal elements (identity for control qubits)
  for (int i = 0; i < static_cast<int>(full_dim - target_gate_dim); ++i) {
    gate_tensor[offset] = {1.0, 0.0};
    offset += (full_dim + 1);
  }
  
  // Set target gate matrix in lower-right block
  for (std::size_t row = 0; row < target_gate_dim; ++row) {
    for (std::size_t col = 0; col < target_gate_dim; ++col) {
      const auto org_idx = row * target_gate_dim + col;
      const auto block_anchor = full_dim - target_gate_dim;
      const auto block_row_idx = block_anchor + row;
      const auto block_col_idx = block_anchor + col;
      const auto expanded_idx = block_row_idx * full_dim + block_col_idx;
      gate_tensor[expanded_idx] = target_gate[org_idx];
    }
  }
  
  return gate_tensor;
}

// Noise channel application
template <typename ScalarType>
void SimulatorFormoTensor<ScalarType>::applyNoiseChannel(
    const std::string_view gateName,
    const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &targets,
    const std::vector<double> &params) {
  
  // If no state is initialised or no execution context/noise model exists,
  // there is nothing to do.
  if (!m_state || !this->executionContext || !this->executionContext->noiseModel)
    return;

  // Gather all qubit indices (controls followed by targets) as 32-bit ints.
  std::vector<int32_t> qubits;
  qubits.reserve(controls.size() + targets.size());
  for (auto c : controls) qubits.push_back(static_cast<int32_t>(c));
  for (auto t : targets) qubits.push_back(static_cast<int32_t>(t));

  // Use the noise model to obtain Kraus channels for this gate.
  // This API mirrors the call used in the cutensornet backend.
  std::string gName(gateName);
  auto krausChannels = this->executionContext->noiseModel->get_channels(
      gName, targets, controls, params);
  if (krausChannels.empty()) return;

  // For each channel, build device pointers for the Kraus operators and
  // delegate to the state. We reuse the gate matrix cache to avoid
  // repeatedly allocating device memory for identical matrices.
  for (const auto &channel : krausChannels) {
    if (channel.is_unitary_mixture()) {
      // Channel is a unitary mixture: each entry in unitary_ops is a
      // square matrix, and probabilities holds the mixing coefficients.
      std::vector<void *> dOps;
      dOps.reserve(channel.unitary_ops.size());
      for (const auto &mat : channel.unitary_ops) {
        // Convert to std::complex<ScalarType> vector as required by
        // allocateGateMatrix(). In a real implementation, you should
        // implement getOrCacheMat() similar to cutensornet to reuse cached
        // allocations. Here we call allocateGateMatrix() directly for clarity.
        std::vector<std::complex<ScalarType>> casted(mat.begin(), mat.end());
        void *d_op = allocateGateMatrix(casted);
        dOps.push_back(d_op);
      }
      // Delegate to state: apply the unitary channel on the specified
      // qubits. The probabilities vector tells cuTensorNet how to mix
      // these unitaries.
      m_state->applyUnitaryChannel(qubits, dOps, channel.probabilities);
    } else {
      // General channel: a list of Kraus operators with no weights.
      std::vector<void *> dOps;
      dOps.reserve(channel.get_ops().size());
      for (const auto &op : channel.get_ops()) {
        // op.data is in the precision dictated by the noise model.
        // For simplicity we cast it to ScalarType; production code
        // should handle mixed precision as cutensornet does.
        std::vector<std::complex<ScalarType>> casted(op.data.begin(),
                                                     op.data.end());
        void *d_op = allocateGateMatrix(casted);
        dOps.push_back(d_op);
      }
      m_state->applyGeneralChannel(qubits, dOps);
    }
  }
}

template <typename ScalarType>
bool SimulatorFormoTensor<ScalarType>::isValidNoiseChannel(
    const cudaq::noise_model_type &type) const {
  // Support amplitude damping, phase damping, and depolarization
  using namespace cudaq;
  return type == noise_model_type::amplitude_damping ||
         type == noise_model_type::phase_flip ||
         type == noise_model_type::bit_flip ||
         type == noise_model_type::depolarization_channel;
}

// Sampling (aggregated)
template <typename ScalarType>
std::unordered_map<std::string, size_t>
SimulatorFormoTensor<ScalarType>::sample(
    const std::vector<std::size_t> &measuredBitIds, int32_t shots) {
  
  if (!m_state) {
    throw std::runtime_error("No quantum state initialized");
  }
  
  // Convert to int32_t
  std::vector<int32_t> measuredQubits;
  measuredQubits.reserve(measuredBitIds.size());
  for (auto q : measuredBitIds) {
    measuredQubits.push_back(static_cast<int32_t>(q));
  }
  
  if (m_isBatchMode) {
    // Call batch sampling and aggregate results across all samples
    auto batchCounts = m_state->sampleBatch(measuredQubits, shots);
    
    // Aggregate all batch results into a single count map
    std::unordered_map<std::string, std::size_t> aggregated;
    for (const auto &sampleCounts : batchCounts) {
      for (const auto &[bitstring, count] : sampleCounts) {
        aggregated[bitstring] += count;
      }
    }
    return aggregated;
  } else {
    // Single state sampling - not yet implemented
    throw std::runtime_error("Single state sampling not yet fully implemented");
  }
}

// Sampling (per-batch-element, CPU-based)
template <typename ScalarType>
std::vector<std::unordered_map<std::string, std::size_t>>
SimulatorFormoTensor<ScalarType>::sampleBatch(
    const std::vector<std::size_t> &measuredBitIds, int32_t shots) {
  
  if (!m_state) {
    throw std::runtime_error("No quantum state initialized");
  }
  
  if (!m_isBatchMode) {
    throw std::runtime_error(
        "sampleBatch only valid in batch mode. Use sample() for single state.");
  }
  
  // Convert to int32_t
  std::vector<int32_t> measuredQubits;
  measuredQubits.reserve(measuredBitIds.size());
  for (auto q : measuredBitIds) {
    measuredQubits.push_back(static_cast<int32_t>(q));
  }
  
  return m_state->sampleBatch(measuredQubits, shots);
}

// Sampling (per-batch-element, GPU-accelerated)
template <typename ScalarType>
std::vector<std::unordered_map<std::string, std::size_t>>
SimulatorFormoTensor<ScalarType>::sampleBatchGpu(
    const std::vector<std::size_t> &measuredBitIds, int32_t shots) {
  
  if (!m_state) {
    throw std::runtime_error("No quantum state initialized");
  }
  
  if (!m_isBatchMode) {
    throw std::runtime_error(
        "sampleBatchGpu only valid in batch mode. Use sample() for single state.");
  }
  
  // Convert to int32_t
  std::vector<int32_t> measuredQubits;
  measuredQubits.reserve(measuredBitIds.size());
  for (auto q : measuredBitIds) {
    measuredQubits.push_back(static_cast<int32_t>(q));
  }
  
  return m_state->sampleBatchGpu(measuredQubits, shots);
}

// Get state vector
template <typename ScalarType>
std::vector<std::complex<ScalarType>>
SimulatorFormoTensor<ScalarType>::getStateVector() {
  if (!m_state) {
    throw std::runtime_error("No quantum state initialized");
  }
  
  if (m_isBatchMode) {
    throw std::runtime_error(
        "Cannot get single state vector from batch state. Use getBatchStateVectors() instead.");
  }
  
  return m_state->getStateVector();
}

// Get batch state vectors
template <typename ScalarType>
std::vector<std::vector<std::complex<ScalarType>>>
SimulatorFormoTensor<ScalarType>::getBatchStateVectors() {
  if (!m_state) {
    throw std::runtime_error("No quantum state initialized");
  }
  
  if (!m_isBatchMode) {
    throw std::runtime_error(
        "State is not in batch mode. Use getStateVector() instead.");
  }
  
  return m_state->getBatchStateVectors();
}

// Reset state
template <typename ScalarType>
void SimulatorFormoTensor<ScalarType>::resetState() {
  m_state.reset();
  m_isBatchMode = false;
  m_batchSize = 1;
  m_numQubitsPerSample = 0;
}

// Get simulator name
template <typename ScalarType>
std::string SimulatorFormoTensor<ScalarType>::name() const {
#ifdef FORMOTENSOR_FP32
  return "formotensor-fp32";
#else
  return "formotensor";
#endif
}

// Initialize cuTensorNet communication for distributed processing
template <typename ScalarType>
void SimulatorFormoTensor<ScalarType>::initCuTensornetComm(
    cutensornetHandle_t handle) {
  CUDAQ_INFO("Initializing cuTensorNet communication for distributed processing");
  // TODO: Implement MPI support similar to cutensornet backend if needed
}

// Explicit template instantiations
template class SimulatorFormoTensor<double>;
template class SimulatorFormoTensor<float>;

} // namespace nvqir
