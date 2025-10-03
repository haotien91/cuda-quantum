/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "FormoTensorCircuitSimulator.h"
#include "formotensor_state.inc"
#include "../common/Logger.h"
#include "../common/ScratchDeviceMem.h"
#include "../../cudaq.h"
#include <fmt/format.h>

namespace nvqir {

template <typename ScalarType>
SimulatorFormoTensor<ScalarType>::SimulatorFormoTensor()
    : m_randomEngine(std::random_device()()) {
  
  int numDevices{0};
  HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
  
  // Select device (support MPI if available)
  const int deviceId = cudaq::mpi::is_initialized() ? cudaq::mpi::rank() % numDevices : 0;
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

template <typename ScalarType>
SimulatorFormoTensor<ScalarType>::~SimulatorFormoTensor() {
  if (m_cutnHandle) {
    HANDLE_CUTN_ERROR(cutensornetDestroy(m_cutnHandle));
  }
}

template <typename ScalarType>
void SimulatorFormoTensor<ScalarType>::addQubitsToState(std::size_t numQubits, const void *ptr) {
  if (m_isBatchMode) {
    throw std::runtime_error("Cannot add qubits to batch state. Use initializeBatchState() instead.");
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

template <typename ScalarType>
void SimulatorFormoTensor<ScalarType>::initializeBatchState(std::size_t numQubits, 
                                                           std::size_t batchSize, 
                                                           const void *data) {
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
  
  CUDAQ_INFO("Initialized batch state with {} qubits per sample and {} samples", 
             numQubits, batchSize);
}

template <typename ScalarType>
void SimulatorFormoTensor<ScalarType>::applyGate(const GateApplicationTask &task) {
  if (!m_state) {
    throw std::runtime_error("No quantum state initialized");
  }
  
  if (m_isBatchMode) {
    applyBatchGate(task);
  } else {
    // Single state mode - use standard gate application
    const auto &[gateName, parameters, controls, targets, op] = task;
    
    // Convert gate to device memory
    void *gateDeviceMem = nullptr;
    const std::size_t gateSize = 1ULL << (2 * targets.size());
    HANDLE_CUDA_ERROR(cudaMalloc(&gateDeviceMem, gateSize * sizeof(std::complex<ScalarType>)));
    
    // Copy gate matrix to device
    HANDLE_CUDA_ERROR(cudaMemcpy(gateDeviceMem, op.data(),
                                 gateSize * sizeof(std::complex<ScalarType>),
                                 cudaMemcpyHostToDevice));
    
    // Apply gate
    m_state->applyGate(controls, targets, gateDeviceMem, false);
    
    // Clean up
    cudaFree(gateDeviceMem);
  }
}

template <typename ScalarType>
void SimulatorFormoTensor<ScalarType>::applyBatchGate(const GateApplicationTask &task) {
  const auto &[gateName, parameters, controls, targets, op] = task;
  
  // Convert gate to device memory
  void *gateDeviceMem = nullptr;
  const std::size_t gateSize = 1ULL << (2 * targets.size());
  HANDLE_CUDA_ERROR(cudaMalloc(&gateDeviceMem, gateSize * sizeof(std::complex<ScalarType>)));
  
  // Copy gate matrix to device
  HANDLE_CUDA_ERROR(cudaMemcpy(gateDeviceMem, op.data(),
                               gateSize * sizeof(std::complex<ScalarType>),
                               cudaMemcpyHostToDevice));
  
  // Apply gate in batch mode
  m_state->applyGate(controls, targets, gateDeviceMem, false);
  
  // Clean up
  cudaFree(gateDeviceMem);
}

template <typename ScalarType>
void SimulatorFormoTensor<ScalarType>::applyNoiseChannel(
    const std::string_view gateName,
    const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &targets,
    const std::vector<double> &params) {
  
  if (!m_state) {
    throw std::runtime_error("No quantum state initialized");
  }
  
  // For now, we don't support noise channels in batch mode
  if (m_isBatchMode) {
    throw std::runtime_error("Noise channels not yet supported in batch mode");
  }
  
  // Standard noise channel implementation would go here
  throw std::runtime_error("Noise channels not yet implemented in FormoTensor backend");
}

template <typename ScalarType>
bool SimulatorFormoTensor<ScalarType>::isValidNoiseChannel(
    const cudaq::noise_model_type &type) const {
  // For now, we don't support noise channels
  return false;
}

template <typename ScalarType>
std::unordered_map<std::string, size_t>
SimulatorFormoTensor<ScalarType>::sample(const std::vector<std::size_t> &measuredBitIds,
                                         int32_t shots) {
  if (!m_state) {
    throw std::runtime_error("No quantum state initialized");
  }
  
  if (m_isBatchMode) {
    throw std::runtime_error("Sampling not yet supported in batch mode");
  }
  
  // Standard sampling implementation would go here
  throw std::runtime_error("Sampling not yet implemented in FormoTensor backend");
}

template <typename ScalarType>
std::vector<std::complex<ScalarType>>
SimulatorFormoTensor<ScalarType>::getStateVector() {
  if (!m_state) {
    throw std::runtime_error("No quantum state initialized");
  }
  
  if (m_isBatchMode) {
    throw std::runtime_error("Cannot get single state vector from batch state. Use getBatchStateVectors() instead.");
  }
  
  return m_state->getStateVector();
}

template <typename ScalarType>
void SimulatorFormoTensor<ScalarType>::resetState() {
  m_state.reset();
  m_isBatchMode = false;
  m_batchSize = 1;
  m_numQubitsPerSample = 0;
}

template <typename ScalarType>
std::string SimulatorFormoTensor<ScalarType>::name() const {
#ifdef FORMOTENSOR_FP32
  return "formotensor-fp32";
#else
  return "formotensor";
#endif
}

template <typename ScalarType>
void SimulatorFormoTensor<ScalarType>::initCuTensornetComm(cutensornetHandle_t handle) {
  // Initialize cuTensorNet communication for distributed processing
  // This would be similar to the implementation in cutensornet backend
  CUDAQ_INFO("Initializing cuTensorNet communication for distributed processing");
}

// Explicit template instantiations
template class SimulatorFormoTensor<double>;
template class SimulatorFormoTensor<float>;

// C API for registration
extern "C" nvqir::CircuitSimulator *getCircuitSimulator_formotensor() {
  return new SimulatorFormoTensor<double>();
}

} // namespace nvqir
