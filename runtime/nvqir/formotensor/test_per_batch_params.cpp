// Include standard library headers BEFORE namespace-internal headers
#include <iostream>
#include <cmath>
#include <cassert>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include <cutensornet.h>

#include "formotensor_state.h"
#include "formotensor_state.inc"
#include "formotensor_utils.h"

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

// Helper: Create cuTensorNet handle and scratch
struct TestBackend {
  cutensornetHandle_t handle{nullptr};
  ScratchDeviceMem scratch;
  std::mt19937 rng{123};

  TestBackend() {
    HANDLE_CUTN_ERROR(cutensornetCreate(&handle));
    scratch.allocate();
  }
  ~TestBackend() {
    if (handle) cutensornetDestroy(handle);
  }
};

// Helper: Gate generators
template <typename T>
std::vector<std::complex<T>> rx_gate(double theta) {
  const double c = std::cos(theta / 2.0);
  const double s = std::sin(theta / 2.0);
  return { {c,0.0}, {0.0,-s}, {0.0,-s}, {c,0.0} };
}

template <typename T>
std::vector<std::complex<T>> ry_gate(double theta) {
  const double c = std::cos(theta / 2.0);
  const double s = std::sin(theta / 2.0);
  return { {c,0.0}, {-s,0.0}, {s,0.0}, {c,0.0} };
}

template <typename T>
std::vector<std::complex<T>> rz_gate(double theta) {
  const double half = theta / 2.0;
  return { {std::cos(half), -std::sin(half)}, {0.0,0.0}, {0.0,0.0}, {std::cos(half), std::sin(half)} };
}

template <typename T>
std::vector<std::complex<T>> x_gate() {
  return { {0.0,0.0}, {1.0,0.0}, {1.0,0.0}, {0.0,0.0} };
}

// Apply single-qubit gate on single-state
template <typename T>
void apply_single_gate(FormoTensorState<T> &state, const std::vector<std::complex<T>> &gate, int target) {
  void *d_gate = allocateGateMatrix<T>(gate);
  state.applyGate(/*controls*/{}, {static_cast<int32_t>(target)}, d_gate, /*adjoint*/false);
  cudaFree(d_gate);
}

// Build for-loop baseline result for a sequence of Ry on targets (depth applies cyclic qubit indices)
template <typename T>
std::vector<std::complex<T>> run_forloop_ry(std::size_t n, const std::vector<double> &angles, TestBackend &env) {
  FormoTensorState<T> single(static_cast<std::size_t>(n), env.scratch, env.handle, env.rng);
  single.setZeroState();
  for (std::size_t d = 0; d < angles.size(); ++d) {
    int tgt = static_cast<int>(d % n);
    apply_single_gate<T>(single, ry_gate<T>(angles[d]), tgt);
  }
  return single.getStateVector();
}

// Run batch Ry with per-sample angles on a zero batch state
template <typename T>
std::vector<std::vector<std::complex<T>>> run_batch_ry(std::size_t n, std::size_t B, const std::vector<double> &angles, int target, TestBackend &env) {
  FormoTensorState<T> batched(n, B, env.scratch, env.handle, env.rng);
  batched.setZeroState();
  std::vector<std::vector<double>> batchParams(B);
  for (std::size_t b = 0; b < B; ++b) batchParams[b] = {angles[b]};
  batched.applyBatchParametrizedGate({static_cast<int32_t>(target)}, "ry", batchParams);
  return batched.getBatchStateVectors();
}

// Test 1: Basic per-batch parametrization with Ry
void test_basic_per_batch_ry() {
  std::cout << "\n=== Test 1: Basic Per-Batch Ry Gate ===" << std::endl;
  const std::size_t n = 2;
  const std::size_t B = 4;
  TestBackend env;
  std::vector<double> angles = {0.0, M_PI/2, M_PI, 3*M_PI/2};
  auto results = run_batch_ry<double>(n, B, angles, /*target=*/0, env);
  const double sqrt2_inv = 1.0 / std::sqrt(2.0);
  std::cout << "\nVerifying results:" << std::endl;
  std::cout << "  Sample 0 (θ=0): ";
  printComplex(results[0][0]);
  std::cout << " (expected 1.0)" << std::endl;
  assert(approxEqual(results[0][0], std::complex<double>(1.0, 0.0)));
  std::cout << "  Sample 1 (θ=π/2): |00⟩ = ";
  printComplex(results[1][0]);
  std::cout << ", |10⟩ = ";
  printComplex(results[1][2]);
  std::cout << " (expected " << sqrt2_inv << ")" << std::endl;
  assert(approxEqual(results[1][0], std::complex<double>(sqrt2_inv, 0.0)));
  assert(approxEqual(results[1][2], std::complex<double>(sqrt2_inv, 0.0)));
  std::cout << "  Sample 2 (θ=π): |10⟩ = ";
  printComplex(results[2][2]);
  std::cout << " (expected 1.0)" << std::endl;
  assert(approxEqual(results[2][2], std::complex<double>(1.0, 0.0)));
  std::cout << "  Sample 3 (θ=3π/2): |00⟩ = ";
  printComplex(results[3][0]);
  std::cout << ", |10⟩ = ";
  printComplex(results[3][2]);
  std::cout << " (expected ±" << sqrt2_inv << ")" << std::endl;
  assert(approxEqual(results[3][0], std::complex<double>(sqrt2_inv, 0.0)));
  assert(approxEqual(results[3][2], std::complex<double>(-sqrt2_inv, 0.0)));
  std::cout << "✅ Test 1 PASSED" << std::endl;
}

// Test 2: All three rotation gates (Rx, Ry, Rz)
void test_all_rotation_gates() {
  std::cout << "\n=== Test 2: All Rotation Gates (Rx, Ry, Rz) ===" << std::endl;
  const std::size_t n = 3;
  const std::size_t B = 3;
  TestBackend env;
  FormoTensorState<double> batched(n, B, env.scratch, env.handle, env.rng);
  batched.setZeroState();
  std::vector<double> angles = {M_PI/4, M_PI/3, M_PI/6};
  std::vector<std::vector<double>> params(B);
  for (std::size_t b = 0; b < B; ++b) params[b] = {angles[b]};
  std::cout << "  Applying rxBatch with angles: "; for (auto a:angles) std::cout<<a<<" "; std::cout<<std::endl;
  batched.applyBatchParametrizedGate({0}, "rx", params);
  std::cout << "  Applying ryBatch with angles: "; for (auto a:angles) std::cout<<a<<" "; std::cout<<std::endl;
  batched.applyBatchParametrizedGate({1}, "ry", params);
  std::cout << "  Applying rzBatch with angles: "; for (auto a:angles) std::cout<<a<<" "; std::cout<<std::endl;
  batched.applyBatchParametrizedGate({2}, "rz", params);
  auto results = batched.getBatchStateVectors();
  std::cout << "\nVerifying normalization:" << std::endl;
  for (std::size_t b = 0; b < B; ++b) {
    double norm = 0.0;
    for (const auto &amp : results[b]) {
      norm += std::norm(amp);
    }
    std::cout << "  Sample " << b << ": norm = " << norm << std::endl;
    assert(std::abs(norm - 1.0) < 1e-10);
  }
  std::cout << "✅ Test 2 PASSED" << std::endl;
}

// Test 3: Numerical equivalence with for-loop
void test_numerical_equivalence() {
  std::cout << "\n=== Test 3: Numerical Equivalence (Batch vs For-Loop) ===" << std::endl;
  const std::size_t n = 4;
  const std::size_t B = 8;
  const std::size_t depth = 5;
  std::cout << "  Configuration: n=" << n << ", B=" << B << ", depth=" << depth << std::endl;
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 2 * M_PI);
  std::vector<std::vector<double>> batchParams(B);
  for (std::size_t b = 0; b < B; ++b) for (std::size_t d = 0; d < depth; ++d) batchParams[b].push_back(dist(rng));
  TestBackend env;
  std::cout << "  Running for-loop version..." << std::endl;
  std::vector<std::vector<std::complex<double>>> forloopResults;
  for (std::size_t b = 0; b < B; ++b)
    forloopResults.push_back(run_forloop_ry<double>(n, batchParams[b], env));
  std::cout << "  Running batch version..." << std::endl;
  FormoTensorState<double> batched(n, B, env.scratch, env.handle, env.rng);
  batched.setZeroState();
  for (std::size_t d = 0; d < depth; ++d) {
    std::vector<std::vector<double>> params(B);
    for (std::size_t b = 0; b < B; ++b) params[b] = {batchParams[b][d]};
    batched.applyBatchParametrizedGate({static_cast<int32_t>(d % n)}, "ry", params);
  }
  auto batchResults = batched.getBatchStateVectors();
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
  std::cout << "✅ Test 3 PASSED" << std::endl;
}

// Test 4: Error handling
void test_error_handling() {
  std::cout << "\n=== Test 4: Error Handling ===" << std::endl;
  const std::size_t n = 2;
  const std::size_t B = 4;
  TestBackend env;
  FormoTensorState<double> batched(n, B, env.scratch, env.handle, env.rng);
  batched.setZeroState();
  std::cout << "  Testing wrong angle count..." << std::endl;
  try {
    std::vector<double> wrongSize = {0.1, 0.2, 0.3}; // Only 3, but B=4
    std::vector<std::vector<double>> params(wrongSize.size());
    for (std::size_t i = 0; i < wrongSize.size(); ++i) params[i] = {wrongSize[i]};
    batched.applyBatchParametrizedGate({0}, "ry", params);
    std::cerr << "  ❌ Should have thrown exception for wrong angle count" << std::endl;
    assert(false);
  } catch (const std::runtime_error &e) {
    std::cout << "  ✓ Correctly caught error: " << e.what() << std::endl;
  }
  std::cout << "  Testing batch method in non-batch mode..." << std::endl;
  try {
    FormoTensorState<double> single(n, env.scratch, env.handle, env.rng);
    single.setZeroState();
    std::vector<std::vector<double>> params{{0.1}, {0.2}};
    single.applyBatchParametrizedGate({0}, "ry", params);
    std::cerr << "  ❌ Should have thrown exception in non-batch mode" << std::endl;
    assert(false);
  } catch (const std::runtime_error &e) {
    std::cout << "  ✓ Correctly caught error: " << e.what() << std::endl;
  }
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
  TestBackend env;
  FormoTensorState<double> batched(n, B, env.scratch, env.handle, env.rng);
  batched.setZeroState();
  std::mt19937 rng(123);
  std::uniform_real_distribution<double> dist(0.0, 2 * M_PI);
  std::cout << "  Applying feature encoding layer..." << std::endl;
  for (std::size_t q = 0; q < n; ++q) {
    std::vector<double> encodingAngles(B);
    for (std::size_t b = 0; b < B; ++b) encodingAngles[b] = dist(rng);
    std::vector<std::vector<double>> params(B);
    for (std::size_t b = 0; b < B; ++b) params[b] = {encodingAngles[b]};
    batched.applyBatchParametrizedGate({static_cast<int32_t>(q)}, "ry", params);
  }
  std::cout << "  Applying " << numLayers << " variational layers..." << std::endl;
  for (std::size_t layer = 0; layer < numLayers; ++layer) {
    for (std::size_t q = 0; q < n; ++q) {
      double sharedAngle = dist(rng);
      void *d_gate = allocateGateMatrix<double>(ry_gate<double>(sharedAngle));
      batched.applyGate(/*controls*/{}, {static_cast<int32_t>(q)}, d_gate, /*adjoint*/false);
      cudaFree(d_gate);
    }
    for (std::size_t q = 0; q < n - 1; ++q) {
      void *d_x = allocateGateMatrix<double>(x_gate<double>());
      batched.applyGate(/*controls*/{static_cast<int32_t>(q)}, {static_cast<int32_t>(q + 1)}, d_x, /*adjoint*/false);
      cudaFree(d_x);
    }
  }
  std::cout << "  Verifying normalization..." << std::endl;
  auto results = batched.getBatchStateVectors();
  for (std::size_t b = 0; b < B; ++b) {
    double norm = 0.0;
    for (const auto &amp : results[b]) norm += std::norm(amp);
    if (std::abs(norm - 1.0) > 1e-10) {
      std::cerr << "  ❌ Sample " << b << " not normalized: norm = " << norm << std::endl;
      assert(false);
    }
  }
  std::cout << "  ✓ QML circuit executed successfully" << std::endl;
  std::cout << "  ✓ All " << B << " batch samples are normalized" << std::endl;
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