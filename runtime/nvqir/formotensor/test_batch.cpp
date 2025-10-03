/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "FormoTensorCircuitSimulator.h"
#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <random>
#include <cassert>
#include <cmath>

using namespace nvqir;

// Test data generation
std::vector<std::complex<double>> generateTestState(int numQubits, int seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dis(-1.0, 1.0);
  
  const int size = 1 << numQubits;
  std::vector<std::complex<double>> state(size);
  
  double norm = 0.0;
  for (int i = 0; i < size; i++) {
    state[i] = std::complex<double>(dis(gen), dis(gen));
    norm += std::norm(state[i]);
  }
  
  // Normalize
  norm = std::sqrt(norm);
  for (int i = 0; i < size; i++) {
    state[i] /= norm;
  }
  
  return state;
}

std::vector<std::complex<double>> generateBatchData(int numQubits, int batchSize) {
  const int stateSize = 1 << numQubits;
  const int totalSize = batchSize * stateSize;
  std::vector<std::complex<double>> batchData(totalSize);
  
  for (int b = 0; b < batchSize; b++) {
    auto state = generateTestState(numQubits, 42 + b);
    for (int i = 0; i < stateSize; i++) {
      batchData[b * stateSize + i] = state[i];
    }
  }
  
  return batchData;
}

// Simple Hadamard gate
std::vector<std::complex<double>> getHadamardGate() {
  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  return {
    inv_sqrt2,  inv_sqrt2,
    inv_sqrt2, -inv_sqrt2
  };
}

// Simple CNOT gate
std::vector<std::complex<double>> getCNOTGate() {
  return {
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 0.0
  };
}

// Apply Hadamard gate to a state vector (CPU reference implementation)
std::vector<std::complex<double>> applyHadamardCPU(const std::vector<std::complex<double>>& state, int qubit) {
  const int size = state.size();
  const int qubitMask = 1 << qubit;
  std::vector<std::complex<double>> result(size);
  
  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  
  for (int i = 0; i < size; i++) {
    int j = i ^ qubitMask;  // Flip the qubit bit
    result[i] = inv_sqrt2 * (state[i] + state[j]);
  }
  
  return result;
}

// Apply CNOT gate to a state vector (CPU reference implementation)
std::vector<std::complex<double>> applyCNOTCPU(const std::vector<std::complex<double>>& state, int control, int target) {
  const int size = state.size();
  std::vector<std::complex<double>> result = state;
  
  const int controlMask = 1 << control;
  const int targetMask = 1 << target;
  
  for (int i = 0; i < size; i++) {
    if (i & controlMask) {  // If control qubit is |1>
      int j = i ^ targetMask;  // Flip target qubit
      std::swap(result[i], result[j]);
    }
  }
  
  return result;
}

void testBatchStateInitialization() {
  std::cout << "=== Testing Batch State Initialization ===" << std::endl;
  
  SimulatorFormoTensor<double> simulator;
  
  const int numQubits = 2;
  const int batchSize = 4;
  
  // Generate batch data
  auto batchData = generateBatchData(numQubits, batchSize);
  
  // Initialize batch state
  simulator.initializeBatchState(numQubits, batchSize, batchData.data());
  
  std::cout << "✓ Batch state initialized with " << batchSize << " samples of " 
            << numQubits << " qubits each" << std::endl;
  
  // Verify batch mode
  assert(simulator.isBatchMode());
  assert(simulator.getBatchSize() == batchSize);
  assert(simulator.getNumQubitsPerSample() == numQubits);
  
  std::cout << "✓ Batch mode verification passed" << std::endl;
  
  // Test state vector retrieval
  try {
    auto batchStates = simulator.getBatchStateVectors();
    assert(batchStates.size() == batchSize);
    
    for (int b = 0; b < batchSize; b++) {
      assert(batchStates[b].size() == (1 << numQubits));
    }
    
    std::cout << "✓ Batch state vector retrieval successful" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "✗ Batch state vector retrieval failed: " << e.what() << std::endl;
    throw;
  }
}

void testBatchCorrectness() {
  std::cout << "=== Testing Batch Correctness ===" << std::endl;
  
  const int numQubits = 2;
  const int batchSize = 3;
  
  // Generate batch data
  auto batchData = generateBatchData(numQubits, batchSize);
  
  // Create batch simulator
  SimulatorFormoTensor<double> batchSimulator;
  batchSimulator.initializeBatchState(numQubits, batchSize, batchData.data());
  
  // Create individual simulators for comparison
  std::vector<SimulatorFormoTensor<double>> individualSimulators(batchSize);
  for (int b = 0; b < batchSize; b++) {
    const int stateSize = 1 << numQubits;
    std::vector<std::complex<double>> singleState(
      batchData.begin() + b * stateSize,
      batchData.begin() + (b + 1) * stateSize
    );
    individualSimulators[b].addQubitsToState(numQubits, singleState.data());
  }
  
  std::cout << "✓ Created " << batchSize << " individual simulators for comparison" << std::endl;
  
  // Apply the same sequence of gates to both batch and individual simulators
  // Note: This is a conceptual test - actual gate application would need proper GateApplicationTask
  
  // For now, we'll test that the initial states match
  auto batchStates = batchSimulator.getBatchStateVectors();
  
  for (int b = 0; b < batchSize; b++) {
    auto individualState = individualSimulators[b].getStateVector();
    
    // Compare states (with some tolerance for floating point errors)
    const double tolerance = 1e-10;
    for (int i = 0; i < (1 << numQubits); i++) {
      double diff = std::abs(batchStates[b][i] - individualState[i]);
      if (diff > tolerance) {
        std::cout << "✗ State mismatch at batch " << b << ", element " << i 
                  << ": batch=" << batchStates[b][i] 
                  << ", individual=" << individualState[i] << std::endl;
        throw std::runtime_error("State mismatch detected");
      }
    }
  }
  
  std::cout << "✓ Batch states match individual states within tolerance" << std::endl;
}

void testPerformanceComparison() {
  std::cout << "=== Performance Comparison Test ===" << std::endl;
  
  const int numQubits = 3;
  const int batchSize = 8;
  const int numIterations = 100;  // Number of iterations for timing
  
  // Generate test data
  auto batchData = generateBatchData(numQubits, batchSize);
  
  // Test batch processing
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int iter = 0; iter < numIterations; iter++) {
    SimulatorFormoTensor<double> batchSimulator;
    batchSimulator.initializeBatchState(numQubits, batchSize, batchData.data());
    
    // Simulate gate applications (conceptual - would need proper implementation)
    // For now, we're just testing initialization overhead
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto batchTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  std::cout << "✓ Batch processing time (" << numIterations << " iterations): " 
            << batchTime.count() << " μs" << std::endl;
  
  // Test sequential processing
  start = std::chrono::high_resolution_clock::now();
  
  for (int iter = 0; iter < numIterations; iter++) {
    for (int b = 0; b < batchSize; b++) {
      SimulatorFormoTensor<double> singleSimulator;
      const int stateSize = 1 << numQubits;
      std::vector<std::complex<double>> singleState(
        batchData.begin() + b * stateSize,
        batchData.begin() + (b + 1) * stateSize
      );
      singleSimulator.addQubitsToState(numQubits, singleState.data());
    }
  }
  
  end = std::chrono::high_resolution_clock::now();
  auto sequentialTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  std::cout << "✓ Sequential processing time (" << numIterations << " iterations): " 
            << sequentialTime.count() << " μs" << std::endl;
  
  double speedup = static_cast<double>(sequentialTime.count()) / batchTime.count();
  std::cout << "✓ Initialization speedup: " << speedup << "x" << std::endl;
  
  // Note: This only tests initialization overhead, not actual gate application performance
  std::cout << "⚠ Note: This only tests initialization overhead, not gate application performance" << std::endl;
}

void testErrorHandling() {
  std::cout << "=== Testing Error Handling ===" << std::endl;
  
  SimulatorFormoTensor<double> simulator;
  
  // Test invalid batch size
  try {
    simulator.initializeBatchState(2, 0, nullptr);
    std::cout << "✗ Should have thrown error for batch size 0" << std::endl;
    assert(false);
  } catch (const std::runtime_error &e) {
    std::cout << "✓ Correctly caught error: " << e.what() << std::endl;
  }
  
  // Test double initialization
  auto batchData = generateBatchData(2, 4);
  simulator.initializeBatchState(2, 4, batchData.data());
  
  try {
    simulator.initializeBatchState(2, 4, batchData.data());
    std::cout << "✗ Should have thrown error for double initialization" << std::endl;
    assert(false);
  } catch (const std::runtime_error &e) {
    std::cout << "✓ Correctly caught error: " << e.what() << std::endl;
  }
  
  // Test invalid state vector access
  try {
    auto singleState = simulator.getStateVector();
    std::cout << "✗ Should have thrown error for single state access in batch mode" << std::endl;
    assert(false);
  } catch (const std::runtime_error &e) {
    std::cout << "✓ Correctly caught error: " << e.what() << std::endl;
  }
  
  std::cout << "✓ Error handling test completed" << std::endl;
}

void testStateVectorOperations() {
  std::cout << "=== Testing State Vector Operations ===" << std::endl;
  
  // Test single state
  SimulatorFormoTensor<double> singleSimulator;
  auto singleState = generateTestState(2);
  singleSimulator.addQubitsToState(2, singleState.data());
  
  auto retrievedState = singleSimulator.getStateVector();
  assert(retrievedState.size() == singleState.size());
  
  const double tolerance = 1e-10;
  for (int i = 0; i < singleState.size(); i++) {
    double diff = std::abs(retrievedState[i] - singleState[i]);
    if (diff > tolerance) {
      std::cout << "✗ Single state mismatch at element " << i << std::endl;
      throw std::runtime_error("Single state mismatch");
    }
  }
  
  std::cout << "✓ Single state operations work correctly" << std::endl;
  
  // Test batch state
  const int batchSize = 3;
  auto batchData = generateBatchData(2, batchSize);
  
  SimulatorFormoTensor<double> batchSimulator;
  batchSimulator.initializeBatchState(2, batchSize, batchData.data());
  
  auto batchStates = batchSimulator.getBatchStateVectors();
  assert(batchStates.size() == batchSize);
  
  for (int b = 0; b < batchSize; b++) {
    assert(batchStates[b].size() == 4);  // 2^2 = 4
    
    // Compare with original data
    for (int i = 0; i < 4; i++) {
      double diff = std::abs(batchStates[b][i] - batchData[b * 4 + i]);
      if (diff > tolerance) {
        std::cout << "✗ Batch state mismatch at batch " << b << ", element " << i << std::endl;
        throw std::runtime_error("Batch state mismatch");
      }
    }
  }
  
  std::cout << "✓ Batch state operations work correctly" << std::endl;
}

int main() {
  std::cout << "FormoTensor Batch Processing Test Suite (Fixed Version)" << std::endl;
  std::cout << "========================================================" << std::endl;
  
  try {
    testBatchStateInitialization();
    std::cout << std::endl;
    
    testStateVectorOperations();
    std::cout << std::endl;
    
    testBatchCorrectness();
    std::cout << std::endl;
    
    testPerformanceComparison();
    std::cout << std::endl;
    
    testErrorHandling();
    std::cout << std::endl;
    
    std::cout << "🎉 All tests passed!" << std::endl;
    std::cout << std::endl;
    std::cout << "Note: This test suite validates:" << std::endl;
    std::cout << "  ✓ Batch state initialization" << std::endl;
    std::cout << "  ✓ State vector retrieval and correctness" << std::endl;
    std::cout << "  ✓ Error handling" << std::endl;
    std::cout << "  ✓ Basic performance comparison" << std::endl;
    std::cout << std::endl;
    std::cout << "⚠ Missing: Actual gate application testing (requires proper GateApplicationTask)" << std::endl;
    
  } catch (const std::exception &e) {
    std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}
