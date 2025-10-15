/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "../CircuitSimulator.h"
#include "cutensornet.h"
#include "formotensor_state.h"

namespace nvqir {

/// @brief FormoTensor simulator backend with batch processing support
/// This backend extends cuTensorNet to support batch processing of quantum states
/// for Quantum-ML applications, enabling tensor parallelism across multiple samples.
template <typename ScalarType = double>
class SimulatorFormoTensor : public nvqir::CircuitSimulatorBase<ScalarType> {
public:
  using DataType = std::complex<ScalarType>;
  static constexpr cudaDataType_t cudaDataType =
      std::is_same_v<ScalarType, float> ? CUDA_C_32F : CUDA_C_64F;
  using GateApplicationTask =
      typename nvqir::CircuitSimulatorBase<ScalarType>::GateApplicationTask;

  SimulatorFormoTensor();
  SimulatorFormoTensor(const SimulatorFormoTensor &another) = delete;
  SimulatorFormoTensor &operator=(const SimulatorFormoTensor &another) = delete;
  SimulatorFormoTensor(SimulatorFormoTensor &&another) noexcept = delete;
  SimulatorFormoTensor &operator=(SimulatorFormoTensor &&another) noexcept = delete;

  virtual ~SimulatorFormoTensor();

  /// @brief Apply quantum gate
  void applyGate(const GateApplicationTask &task) override;

  /// @brief Apply a noise channel
  void applyNoiseChannel(const std::string_view gateName,
                         const std::vector<std::size_t> &controls,
                         const std::vector<std::size_t> &targets,
                         const std::vector<double> &params) override;

  bool isValidNoiseChannel(const cudaq::noise_model_type &type) const override;

  /// @brief Allocate qubits to the quantum state
  void addQubitsToState(std::size_t numQubits, const void *ptr) override;

  /// @brief Initialize batch state for parallel processing
  /// @param numQubits Number of qubits per sample
  /// @param batchSize Number of samples in the batch
  /// @param data Pointer to batch data (shape: [batchSize, 2^numQubits])
  void initializeBatchState(std::size_t numQubits, std::size_t batchSize, 
                           const void *data);

  /// @brief Check if currently in batch mode
  bool isBatchMode() const { return m_isBatchMode; }

  /// @brief Get current batch size
  std::size_t getBatchSize() const { return m_batchSize; }

  /// @brief Get number of qubits per sample
  std::size_t getNumQubitsPerSample() const { return m_numQubitsPerSample; }

  /// @brief Sample the quantum state (aggregates batch results if in batch mode)
  std::unordered_map<std::string, size_t>
  sample(const std::vector<std::size_t> &measuredBitIds,
         int32_t shots) override;

  /// @brief Sample each batch element independently (only valid in batch mode)
  /// @param measuredBitIds Qubit indices to measure
  /// @param shots Number of shots per batch element
  /// @return Vector of count maps, one per batch element
  std::vector<std::unordered_map<std::string, std::size_t>>
  sampleBatch(const std::vector<std::size_t> &measuredBitIds, int32_t shots);

  /// @brief Get the state vector
  std::vector<std::complex<ScalarType>> getStateVector() override;

  /// @brief Get batch state vectors (only valid in batch mode)
  std::vector<std::vector<std::complex<ScalarType>>> getBatchStateVectors();

  /// @brief Reset the quantum state
  void resetState() override;

  /// @brief Get the name of this simulator
  virtual std::string name() const override;

protected:
  cutensornetHandle_t m_cutnHandle;
  std::unique_ptr<FormoTensorState<ScalarType>> m_state;
  std::unordered_map<std::string, void *> m_gateDeviceMemCache;
  ScratchDeviceMem scratchPad;
  std::mt19937 m_randomEngine;
  
  // Batch processing state
  bool m_isBatchMode = false;
  std::size_t m_batchSize = 1;
  std::size_t m_numQubitsPerSample = 0;

  // Maximum controlled rank for full tensor expansion
  std::size_t m_maxControlledRankForFullTensorExpansion = 2;

  /// @brief Initialize cuTensorNet communication for distributed processing
  void initCuTensornetComm(cutensornetHandle_t handle);

  /// @brief Generate full gate tensor with control qubits
  std::vector<std::complex<ScalarType>>
  generateFullGateTensor(std::size_t num_control_qubits,
                         const std::vector<std::complex<ScalarType>> &target_gate);
};

// Forward declarations for registration
extern "C" nvqir::CircuitSimulator *getCircuitSimulator_formotensor();

} // namespace nvqir

#include "FormoTensorCircuitSimulator.inc"

