/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cutensornet.h"
#include "formotensor_utils.h"
#include <memory>
#include <vector>
#include <random>
#include <span>
#include <unordered_map>

namespace nvqir {

/// @brief Track gate tensors that were appended to the tensor network.
struct AppliedTensorOp {
  void *deviceData = nullptr;
  std::vector<int32_t> targetQubitIds;
  std::vector<int32_t> controlQubitIds;
  bool isAdjoint;
  bool isUnitary;
  
  AppliedTensorOp(void *dataPtr, const std::vector<int32_t> &targetQubits,
                  const std::vector<int32_t> &controlQubits, bool adjoint,
                  bool unitary)
      : deviceData(dataPtr), targetQubitIds(targetQubits),
        controlQubitIds(controlQubits), isAdjoint(adjoint), isUnitary(unitary) {}
};

/// @brief FormoTensor state management with batch processing support
/// This class wraps cutensornetState_t to provide batch-aware APIs for CUDA-Q
/// simulator implementation, supporting d × 2^n state space.
template <typename ScalarType = double>
class FormoTensorState {
  using DataType = std::complex<ScalarType>;
  static constexpr cudaDataType_t cudaDataType =
      std::is_same_v<ScalarType, float> ? CUDA_C_32F : CUDA_C_64F;

protected:
  std::size_t m_numQubitsPerSample;  // n: qubits per sample
  std::size_t m_batchSize;           // d: number of samples
  bool m_isBatched;                  // whether in batch mode
  
  cutensornetHandle_t m_cutnHandle;
  cutensornetState_t m_quantumState;
  
  /// Track id of gate tensors that are applied to the state tensors.
  std::int64_t m_tensorId = InvalidTensorIndexValue;
  
  // Device memory pointers to be cleaned up.
  std::vector<void *> m_tempDevicePtrs;
  
  // Tensor ops that have been applied to the state.
  std::vector<AppliedTensorOp> m_tensorOps;
  
  ScratchDeviceMem &scratchPad;
  std::mt19937 &m_randomEngine;
  bool m_hasNoiseChannel = false;

public:
  /// @brief Number of hyper samples used in tensor network contraction path finder
  static std::int32_t numHyperSamples;
  /// @brief Constructor for single state
  FormoTensorState(std::size_t numQubits, ScratchDeviceMem &inScratchPad,
                   cutensornetHandle_t handle, std::mt19937 &randomEngine);

  /// @brief Constructor for batch state
  FormoTensorState(std::size_t numQubitsPerSample, std::size_t batchSize,
                   ScratchDeviceMem &inScratchPad, cutensornetHandle_t handle,
                   std::mt19937 &randomEngine);

  /// @brief Destructor
  ~FormoTensorState();

  /// @brief Create state from single state vector
  static std::unique_ptr<FormoTensorState>
  createFromStateVector(std::span<std::complex<ScalarType>> stateVec,
                        ScratchDeviceMem &inScratchPad,
                        cutensornetHandle_t handle, std::mt19937 &randomEngine);

  /// @brief Create state from batch state vectors
  /// @param batchStateVec Batch data with shape [batchSize, 2^numQubits]
  static std::unique_ptr<FormoTensorState>
  createFromBatchStateVector(std::size_t numQubitsPerSample, std::size_t batchSize,
                             std::span<std::complex<ScalarType>> batchStateVec,
                             ScratchDeviceMem &inScratchPad,
                             cutensornetHandle_t handle, std::mt19937 &randomEngine);

  /// @brief Apply a unitary gate
  /// @param controlQubits Controlled qubit operands
  /// @param targetQubits Target qubit operands
  /// @param gateDeviceMem Gate unitary matrix in device memory
  /// @param adjoint Apply the adjoint of gate matrix if true
  void applyGate(const std::vector<int32_t> &controlQubits,
                 const std::vector<int32_t> &targetQubits, void *gateDeviceMem,
                 bool adjoint = false);

  /// @brief Apply a noise channel
  void applyUnitaryChannel(const std::vector<int32_t> &qubits,
                           const std::vector<void *> &krausOps,
                           const std::vector<double> &probabilities);

  /// @brief Apply a general channel
  void applyGeneralChannel(const std::vector<int32_t> &qubits,
                           const std::vector<void *> &krausOps);

  /// @brief Apply qubit projector
  void applyQubitProjector(void *proj_d, const std::vector<int32_t> &qubitIdx);

  /// @brief Add more qubits to the state
  void addQubits(std::size_t numQubits);

  /// @brief Set the state to zero
  void setZeroState();

  /// @brief Get the state vector
  std::vector<std::complex<ScalarType>> getStateVector();

  /// @brief Get the batch state vectors
  /// @return Vector of state vectors, each of size 2^numQubitsPerSample
  std::vector<std::vector<std::complex<ScalarType>>> getBatchStateVectors();

  /// @brief Sample the quantum state (single state mode)
  std::unordered_map<std::string, size_t>
  executeSample(cutensornetStateSampler_t &sampler,
                cutensornetWorkspaceDescriptor_t &workDesc,
                const std::vector<int32_t> &measuredBitIds, int32_t shots,
                bool enableCacheWorkspace = true);

  /// @brief Sample the quantum state in batch mode (CPU-based)
  /// @param measuredQubits Qubit ids to measure (0-indexed)
  /// @param shots Number of shots per sample
  /// @return Vector of count maps, one per batch sample
  std::vector<std::unordered_map<std::string, std::size_t>>
  sampleBatch(const std::vector<int32_t> &measuredQubits, int32_t shots);

  /// @brief Sample the quantum state in batch mode using GPU acceleration
  /// @param measuredQubits Qubit ids to measure (0-indexed)
  /// @param shots Number of shots per sample
  /// @return Vector of count maps, one per batch sample
  /// @note This method uses cuTensorNet sampler API with projection to keep
  ///       sampling on GPU. More efficient for large qubit counts.
  std::vector<std::unordered_map<std::string, std::size_t>>
  sampleBatchGpu(const std::vector<int32_t> &measuredQubits, int32_t shots);

  /// @brief Apply parametrized gate with different parameters per batch sample
  /// @param targetQubits Target qubit indices (0-indexed)
  /// @param gateName Gate type ("ry", "rz", "rx")
  /// @param batchParams Parameters for each batch sample, shape [B, numParams]
  /// @note This is the core method for QML batch training
  void applyBatchParametrizedGate(
      const std::vector<int32_t> &targetQubits,
      const std::string &gateName,
      const std::vector<std::vector<double>> &batchParams);

  /// @brief Get the number of qubits per sample
  std::size_t getNumQubitsPerSample() const { return m_numQubitsPerSample; }

  /// @brief Get the batch size
  std::size_t getBatchSize() const { return m_batchSize; }

  /// @brief Check if in batch mode
  bool isBatched() const { return m_isBatched; }

  /// @brief Get applied tensor operations
  const std::vector<AppliedTensorOp> &getTensorOps() const { return m_tensorOps; }

  /// @brief Clone this state
  std::unique_ptr<FormoTensorState> clone() const;

private:
  /// @brief Helper: Generate gate matrix for given gate type and parameters
  /// @param gateName Gate type ("ry", "rz", "rx")
  /// @param params Gate parameters (e.g., rotation angle)
  /// @return Gate matrix as flat vector
  std::vector<std::complex<ScalarType>>
  generateGateMatrix(const std::string &gateName,
                    const std::vector<double> &params);

  /// @brief Initialize batch data from input vectors
  void initializeBatchData(std::span<std::complex<ScalarType>> batchStateVec);

  /// @brief Create batch-aware gate tensor
  void *createBatchGateTensor(void *gateDeviceMem, std::size_t numTargets);

  /// @brief Contract state vector internally
  std::pair<void *, std::size_t>
  contractStateVectorInternal(const std::vector<int32_t> &projectedModes,
                              const std::vector<int64_t> &in_projectedModeValues);
};

} // namespace nvqir
