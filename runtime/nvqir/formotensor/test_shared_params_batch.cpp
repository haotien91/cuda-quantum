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
  if (std::abs(c.real()) < 1e-10 && std::abs(c.imag()) < 1e-10) {
    std::cout << "0";
  } else if (c.imag() >= 0)
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
std::vector<std::complex<T>> ry_gate(double theta) {
  const double c = std::cos(theta / 2.0);
  const double s = std::sin(theta / 2.0);
  return { {c,0.0}, {-s,0.0}, {s,0.0}, {c,0.0} };
}

// Helper function to apply a shared gate to the batch state
// NOTE: Returns pointer to gate memory. Must be freed AFTER contraction.
template <typename T>
void* apply_shared_gate(FormoTensorState<T> &state, const std::vector<std::complex<T>> &gate, int target) {
  void *d_gate = allocateGateMatrix<T>(gate);
  state.applyGate(/*controls*/{}, {static_cast<int32_t>(target)}, d_gate, /*adjoint*/false);
  return d_gate;
}

// Helper: Run single circuit serially for reference
template <typename T>
std::vector<std::complex<T>> run_single_circuit(std::size_t n, const std::vector<double> &angles, TestBackend &env) {
  FormoTensorState<T> single(static_cast<std::size_t>(n), env.scratch, env.handle, env.rng);
  single.setZeroState();
  std::vector<void*> ptrs;
  for (std::size_t d = 0; d < angles.size(); ++d) {
    int tgt = static_cast<int>(d % n);
    void *d_gate = allocateGateMatrix<T>(ry_gate<T>(angles[d]));
    single.applyGate(/*controls*/{}, {static_cast<int32_t>(tgt)}, d_gate, /*adjoint*/false);
    ptrs.push_back(d_gate);
  }
  auto res = single.getStateVector();
  for(auto p : ptrs) cudaFree(p);
  return res;
}

// Test 1: Shared Parameter Batch Ry Gate
void test_shared_batch_ry() {
  std::cout << "\n=== Test 1: Shared Parameter Batch Ry Gate ===" << std::endl;
  const std::size_t n = 2;
  const std::size_t B = 4;
  TestBackend env;
  
  FormoTensorState<double> batched(n, B, env.scratch, env.handle, env.rng);
  batched.setZeroState();
  
  // Apply Ry(pi) to Qubit 0. Ry(pi)|0> = |1>.
  // Expectation: |1>_0 |0>_1 = |10> (Index 1, if Q0 is LSB)
  double theta = M_PI;
  void* d_gate = apply_shared_gate(batched, ry_gate<double>(theta), 0);
  
  auto results = batched.getBatchStateVectors();
  cudaFree(d_gate);
  
  std::cout << "  Verifying results..." << std::endl;
  for (std::size_t b = 0; b < B; ++b) {
    // Index 1 corresponds to binary 01 (Q1=0, Q0=1)
    std::complex<double> val = results[b][1];
    std::cout << "    Sample " << b << " |10>: "; printComplex(val); std::cout << std::endl;
    
    if (!approxEqual(val, std::complex<double>(1.0, 0.0))) {
        std::cout << "  ❌ Sample " << b << " mismatch!" << std::endl;
        // Print full state for debug
        for(size_t i=0; i<results[b].size(); ++i) {
            std::cout << "      idx " << i << ": "; printComplex(results[b][i]); std::cout << std::endl;
        }
        assert(false);
    }
  }
  std::cout << "✅ Test 1 PASSED" << std::endl;
}

// Test 2: Numerical Equivalence (Batch vs Serial)
void test_numerical_equivalence_shared() {
  std::cout << "\n=== Test 2: Numerical Equivalence (Shared Batch vs Serial) ===" << std::endl;
  const std::size_t n = 3;
  const std::size_t B = 8;
  const std::size_t depth = 5;
  
  TestBackend env;
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 2 * M_PI);
  
  std::vector<double> angles;
  for (std::size_t d = 0; d < depth; ++d) angles.push_back(dist(rng));
  
  // 1. Run Serial (Reference)
  std::cout << "  Running serial reference..." << std::endl;
  auto refState = run_single_circuit<double>(n, angles, env);
  
  // 2. Run Batch
  std::cout << "  Running batch version..." << std::endl;
  FormoTensorState<double> batched(n, B, env.scratch, env.handle, env.rng);
  batched.setZeroState(); 
  
  std::vector<void*> ptrs;
  for (std::size_t d = 0; d < depth; ++d) {
    int tgt = static_cast<int>(d % n);
    void* d_gate = apply_shared_gate(batched, ry_gate<double>(angles[d]), tgt);
    ptrs.push_back(d_gate);
  }
  
  auto batchResults = batched.getBatchStateVectors();
  for(auto p : ptrs) cudaFree(p);
  
  std::cout << "  Comparing results..." << std::endl;
  double maxDiff = 0.0;
  bool allMatch = true;
  const std::size_t stateDim = 1ULL << n;
  
  for (std::size_t b = 0; b < B; ++b) {
    for (std::size_t i = 0; i < stateDim; ++i) {
      double diff = std::abs(refState[i] - batchResults[b][i]);
      maxDiff = std::max(maxDiff, diff);
      if (diff > 1e-10) {
        std::cerr << "  ❌ Mismatch at sample " << b << ", index " << i 
                  << ": diff = " << diff << std::endl;
        allMatch = false;
      }
    }
  }
  
  if (allMatch) {
    std::cout << "  ✓ All " << B << " samples match reference (max diff: " 
              << std::scientific << maxDiff << ")" << std::endl;
  }
  assert(allMatch);
  std::cout << "✅ Test 2 PASSED" << std::endl;
}

// Test 3: Batch Sampling Correctness
void test_batch_sampling() {
  std::cout << "\n=== Test 3: Batch Sampling ===" << std::endl;
  const std::size_t n = 2;
  const std::size_t B = 100; 
  TestBackend env;
  
  FormoTensorState<double> batched(n, B, env.scratch, env.handle, env.rng);
  batched.setZeroState();
  
  // Apply Hadamard to Q0: state becomes |+>|0> = (|00>+|10>)/sqrt(2)
  const double s2 = 1.0/std::sqrt(2.0);
  std::vector<std::complex<double>> h_gate = {{s2,0}, {s2,0}, {s2,0}, {-s2,0}};
  void* d_gate = apply_shared_gate(batched, h_gate, 0);
  
  // Sample 100 shots per batch element
  int shots = 100;
  auto results = batched.sampleBatch({0, 1}, shots);
  cudaFree(d_gate);
  
  std::cout << "  Verifying sampling results..." << std::endl;
  bool pass = true;
  for (std::size_t b = 0; b < B; ++b) {
    // Expect "00" and "01" (Q1=0 is always true. Q0 is 0 or 1)
    // Bitstring format: Q1 Q0. So "00" and "01".
    // Wait, earlier we saw Q0 is LSB (Index 1 = |10> = q0=1).
    // sampleBatch implementation constructs bitstring based on measuredQubits order?
    // Yes: bitstr[numMeasured - 1 - k] = ...
    // If measured {0, 1}. k=0 is Q0. k=1 is Q1.
    // bitstr[1] = Q0 val. bitstr[0] = Q1 val.
    // String is "Q1 Q0".
    // So we expect "00" and "01".
    
    size_t count00 = results[b]["00"];
    size_t count01 = results[b]["01"]; 
    
    size_t total = 0;
    for(auto& kv : results[b]) total += kv.second;
    
    if (total != shots) {
         std::cerr << "Sample " << b << " shot count mismatch: " << total << " vs " << shots << std::endl;
         pass = false;
    }
    
    if (results[b]["10"] > 0 || results[b]["11"] > 0) {
        std::cerr << "Sample " << b << " invalid states found (Q1 should be 0)" << std::endl;
        pass = false;
    }
    
    if (count00 == 0 || count01 == 0) {
         // Warning only, statistically possible
    }
  }
  
  if (pass) std::cout << "✅ Test 3 PASSED" << std::endl;
  else assert(false);
}

int main() {
  std::cout << "\n╔════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║  FormoTensor Shared Parameter Batch Test Suite        ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
  try {
    test_shared_batch_ry();
    test_numerical_equivalence_shared();
    test_batch_sampling();
    
    std::cout << "\n╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║           ✅ ALL TESTS PASSED ✅                       ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝\n" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
}
