/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/**
 * @file test_per_batch_params.cpp
 * @brief Test suite for per-batch parametrized gates in FormoTensor
 * 
 * This file tests the core QML functionality: applying different gate
 * parameters to each batch sample.
 */

#include "FormoTensorCircuitSimulator.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <random>
#include <iomanip>

using namespace nvqir;

// Helper: Check if two complex numbers are approximately equal
template <typename T>
bool approxEqual(const std::complex<T> &a, const std::complex<T> &b, 
                 T tolerance = 1e-10) {
  return std::abs(a - b) < tolerance;
}

// Helper: Print complex number
template <typename T>
void printComplex(const std::complex<T> &c) {
  std::cout << std::fixed << std::setprecision(6);
  if (c.imag() >= 0)
    std::cout << c.real() << "+" << c.imag() << "i";
  else
    std::cout << c.real() << c.imag() << "i";
}

// Test 1: Basic per-batch parametrization with Ry
void test_basic_per_batch_ry() {
  std::cout << "\n=== Test 1: Basic Per-Batch Ry Gate ===" << std::endl;
  
  const std::size_t n = 2;
  const std::size_t B = 4;
  
  auto *sim = new SimulatorFormoTensor<double>();
  sim->initializeBatchStateFromZero(n, B);
  
  // Apply different Ry angles to each sample on qubit 0
  std::vector<double> angles = {0.0, M_PI/2, M_PI, 3*M_PI/2};
  sim->ryBatch(angles, 0);
  
  auto results = sim->getBatchStateVectors();
  
  const double sqrt2_inv = 1.0 / std::sqrt(2.0);
  
  std::cout << "\nVerifying results:" << std::endl;
  
  // Sample 0 (θ=0): Ry(0)|00⟩ = |00⟩
  std::cout << "  Sample 0 (θ=0): ";
  printComplex(results[0][0]);
  std::cout << " (expected 1.0)" << std::endl;
  assert(approxEqual(results[0][0], std::complex<double>(1.0, 0.0)));
  
  // Sample 1 (θ=π/2): Ry(π/2)|00⟩ = (|00⟩+|10⟩)/√2
  std::cout << "  Sample 1 (θ=π/2): |00⟩ = ";
  printComplex(results[1][0]);
  std::cout << ", |10⟩ = ";
  printComplex(results[1][2]);
  std::cout << " (expected " << sqrt2_inv << ")" << std::endl;
  assert(approxEqual(results[1][0], std::complex<double>(sqrt2_inv, 0.0)));
  assert(approxEqual(results[1][2], std::complex<double>(sqrt2_inv, 0.0)));
  
  // Sample 2 (θ=π): Ry(π)|00⟩ = |10⟩
  std::cout << "  Sample 2 (θ=π): |10⟩ = ";
  printComplex(results[2][2]);
  std::cout << " (expected 1.0)" << std::endl;
  assert(approxEqual(results[2][2], std::complex<double>(1.0, 0.0)));
  
  // Sample 3 (θ=3π/2): Ry(3π/2)|00⟩ = (|00⟩-|10⟩)/√2
  std::cout << "  Sample 3 (θ=3π/2): |00⟩ = ";
  printComplex(results[3][0]);
  std::cout << ", |10⟩ = ";
  printComplex(results[3][2]);
  std::cout << " (expected ±" << sqrt2_inv << ")" << std::endl;
  assert(approxEqual(results[3][0], std::complex<double>(sqrt2_inv, 0.0)));
  assert(approxEqual(results[3][2], std::complex<double>(-sqrt2_inv, 0.0)));
  
  delete sim;
  std::cout << "✅ Test 1 PASSED" << std::endl;
}

// Test 2: All three rotation gates (Rx, Ry, Rz)
void test_all_rotation_gates() {
  std::cout << "\n=== Test 2: All Rotation Gates (Rx, Ry, Rz) ===" << std::endl;
  
  const std::size_t n = 3;
  const std::size_t B = 3;
  
  auto *sim = new SimulatorFormoTensor<double>();
  sim->initializeBatchStateFromZero(n, B);
  
  // Apply different gates to different qubits
  std::vector<double> angles = {M_PI/4, M_PI/3, M_PI/6};
  
  std::cout << "  Applying rxBatch with angles: ";
  for (auto a : angles) std::cout << a << " ";
  std::cout << std::endl;
  sim->rxBatch(angles, 0);
  
  std::cout << "  Applying ryBatch with angles: ";
  for (auto a : angles) std::cout << a << " ";
  std::cout << std::endl;
  sim->ryBatch(angles, 1);
  
  std::cout << "  Applying rzBatch with angles: ";
  for (auto a : angles) std::cout << a << " ";
  std::cout << std::endl;
  sim->rzBatch(angles, 2);
  
  auto results = sim->getBatchStateVectors();
  
  // Verify that all states are normalized
  std::cout << "\nVerifying normalization:" << std::endl;
  for (std::size_t b = 0; b < B; ++b) {
    double norm = 0.0;
    for (const auto &amp : results[b]) {
      norm += std::norm(amp);
    }
    std::cout << "  Sample " << b << ": norm = " << norm << std::endl;
    assert(std::abs(norm - 1.0) < 1e-10);
  }
  
  delete sim;
  std::cout << "✅ Test 2 PASSED" << std::endl;
}

// Test 3: Numerical equivalence with for-loop
void test_numerical_equivalence() {
  std::cout << "\n=== Test 3: Numerical Equivalence (Batch vs For-Loop) ===" << std::endl;
  
  const std::size_t n = 4;
  const std::size_t B = 8;
  const std::size_t depth = 5;
  
  std::cout << "  Configuration: n=" << n << ", B=" << B << ", depth=" << depth << std::endl;
  
  // Generate random parameters
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 2 * M_PI);
  
  std::vector<std::vector<double>> batchParams(B);
  for (std::size_t b = 0; b < B; ++b) {
    for (std::size_t d = 0; d < depth; ++d) {
      batchParams[b].push_back(dist(rng));
    }
  }
  
  // For-loop version (baseline)
  std::cout << "  Running for-loop version..." << std::endl;
  std::vector<std::vector<std::complex<double>>> forloopResults;
  for (std::size_t b = 0; b < B; ++b) {
    auto *sim = new SimulatorFormoTensor<double>();
    sim->addQubitsToState(n, nullptr);
    for (std::size_t d = 0; d < depth; ++d) {
      sim->ry(batchParams[b][d], d % n);
    }
    forloopResults.push_back(sim->getRawStateVector());
    delete sim;
  }
  
  // Batch version
  std::cout << "  Running batch version..." << std::endl;
  auto *batchSim = new SimulatorFormoTensor<double>();
  batchSim->initializeBatchStateFromZero(n, B);
  for (std::size_t d = 0; d < depth; ++d) {
    std::vector<double> angles(B);
    for (std::size_t b = 0; b < B; ++b) {
      angles[b] = batchParams[b][d];
    }
    batchSim->ryBatch(angles, d % n);
  }
  auto batchResults = batchSim->getBatchStateVectors();
  
  // Verify numerical equivalence
  std::cout << "  Comparing results..." << std::endl;
  const std::size_t stateDim = 1ULL << n;
  double maxDiff = 0.0;
  bool allMatch = true;
  
  for (std::size_t b = 0; b < B; ++b) {
    for (std::size_t i = 0; i < stateDim; ++i) {
      double diff = std::abs(forloopResults[b][i] - batchResults[b][i]);
      maxDiff = std::max(maxDiff, diff);
      if (diff > 1e-10) {
        std::cerr << "  ❌ Mismatch at sample " << b << ", index " << i 
                  << ": diff = " << diff << std::endl;
        allMatch = false;
      }
    }
  }
  
  if (allMatch) {
    std::cout << "  ✓ All " << B << " samples match (max diff: " 
              << std::scientific << maxDiff << ")" << std::endl;
  }
  
  assert(allMatch);
  delete batchSim;
  std::cout << "✅ Test 3 PASSED" << std::endl;
}

// Test 4: Error handling
void test_error_handling() {
  std::cout << "\n=== Test 4: Error Handling ===" << std::endl;
  
  const std::size_t n = 2;
  const std::size_t B = 4;
  
  auto *sim = new SimulatorFormoTensor<double>();
  sim->initializeBatchStateFromZero(n, B);
  
  // Test 4.1: Wrong number of angles
  std::cout << "  Testing wrong angle count..." << std::endl;
  try {
    std::vector<double> wrongSize = {0.1, 0.2, 0.3}; // Only 3, but B=4
    sim->ryBatch(wrongSize, 0);
    std::cerr << "  ❌ Should have thrown exception for wrong angle count" << std::endl;
    assert(false);
  } catch (const std::runtime_error &e) {
    std::cout << "  ✓ Correctly caught error: " << e.what() << std::endl;
  }
  
  // Test 4.2: Calling batch method in non-batch mode
  std::cout << "  Testing batch method in non-batch mode..." << std::endl;
  auto *singleSim = new SimulatorFormoTensor<double>();
  singleSim->addQubitsToState(n, nullptr);
  
  try {
    std::vector<double> angles = {0.1, 0.2};
    singleSim->ryBatch(angles, 0);
    std::cerr << "  ❌ Should have thrown exception in non-batch mode" << std::endl;
    assert(false);
  } catch (const std::runtime_error &e) {
    std::cout << "  ✓ Correctly caught error: " << e.what() << std::endl;
  }
  
  delete sim;
  delete singleSim;
  std::cout << "✅ Test 4 PASSED" << std::endl;
}

// Test 5: QML-like multi-layer circuit
void test_qml_circuit() {
  std::cout << "\n=== Test 5: QML-like Multi-Layer Circuit ===" << std::endl;
  
  const std::size_t n = 4;
  const std::size_t B = 16;
  const std::size_t numLayers = 3;
  
  std::cout << "  Configuration: n=" << n << ", B=" << B 
            << ", layers=" << numLayers << std::endl;
  
  auto *sim = new SimulatorFormoTensor<double>();
  sim->initializeBatchStateFromZero(n, B);
  
  std::mt19937 rng(123);
  std::uniform_real_distribution<double> dist(0.0, 2 * M_PI);
  
  // Feature encoding layer (per-batch parameters)
  std::cout << "  Applying feature encoding layer..." << std::endl;
  for (std::size_t q = 0; q < n; ++q) {
    std::vector<double> encodingAngles(B);
    for (std::size_t b = 0; b < B; ++b) {
      encodingAngles[b] = dist(rng);
    }
    sim->ryBatch(encodingAngles, q);
  }
  
  // Variational layers (shared parameters)
  std::cout << "  Applying " << numLayers << " variational layers..." << std::endl;
  for (std::size_t layer = 0; layer < numLayers; ++layer) {
    for (std::size_t q = 0; q < n; ++q) {
      double sharedAngle = dist(rng);
      sim->ry(sharedAngle, q); // Same angle for all samples
    }
    // Entangling layer
    for (std::size_t q = 0; q < n - 1; ++q) {
      sim->x({q}, q + 1); // CNOT
    }
  }
  
  // Verify all states are normalized
  std::cout << "  Verifying normalization..." << std::endl;
  auto results = sim->getBatchStateVectors();
  for (std::size_t b = 0; b < B; ++b) {
    double norm = 0.0;
    for (const auto &amp : results[b]) {
      norm += std::norm(amp);
    }
    if (std::abs(norm - 1.0) > 1e-10) {
      std::cerr << "  ❌ Sample " << b << " not normalized: norm = " << norm << std::endl;
      assert(false);
    }
  }
  
  std::cout << "  ✓ QML circuit executed successfully" << std::endl;
  std::cout << "  ✓ All " << B << " batch samples are normalized" << std::endl;
  
  delete sim;
  std::cout << "✅ Test 5 PASSED" << std::endl;
}

int main() {
  std::cout << "\n╔════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║  FormoTensor Per-Batch Parametrization Test Suite     ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
  
  try {
    test_basic_per_batch_ry();
    test_all_rotation_gates();
    test_numerical_equivalence();
    test_error_handling();
    test_qml_circuit();
    
    std::cout << "\n╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║           ✅ ALL TESTS PASSED ✅                       ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝\n" << std::endl;
    
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
}