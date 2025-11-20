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

// --- Helper Functions ---

// Check if two complex numbers are approximately equal
template <typename T>
bool approxEqual(const std::complex<T> &a, const std::complex<T> &b, T tolerance = 1e-6) {
  return std::abs(a - b) < tolerance;
}

// Print complex number
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

// Backend Setup
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

// Gate Generators
template <typename T>
std::vector<std::complex<T>> h_gate() {
  const T s2 = 1.0 / std::sqrt(2.0);
  return {{s2, 0}, {s2, 0}, {s2, 0}, {-s2, 0}};
}

template <typename T>
std::vector<std::complex<T>> x_gate() {
  return {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
}

// Helper to apply a gate and return the pointer (caller must free)
template <typename T>
void* apply_gate_ptr(FormoTensorState<T> &state, const std::vector<std::complex<T>> &gate, 
                     const std::vector<int> &controls, const std::vector<int> &targets) {
  void *d_gate = allocateGateMatrix<T>(gate);
  
  // Cast to int32_t vectors
  std::vector<int32_t> c_ids(controls.begin(), controls.end());
  std::vector<int32_t> t_ids(targets.begin(), targets.end());
  
  state.applyGate(c_ids, t_ids, d_gate, /*adjoint*/false);
  return d_gate;
}

// --- GHZ Test ---

void test_ghz_batch() {
  std::cout << "\n=== Test: GHZ State (Entanglement) in Batch Mode ===" << std::endl;
  
  // Configuration
  const std::size_t n = 3;       // 3 qubits
  const std::size_t B = 4;       // 4 batch samples
  TestBackend env;
  
  std::cout << "  Configuration: n=" << n << ", B=" << B << std::endl;

  // 1. Initialize Batch State |000>
  FormoTensorState<double> batched(n, B, env.scratch, env.handle, env.rng);
  batched.setZeroState(); // All batches are |000>

  std::vector<void*> ptrs;

  // 2. Apply H on Qubit 0
  // State -> (|000> + |100>) / sqrt(2)
  // Note: Assuming Q0 is LSB for state vector index, |100> is index 1.
  std::cout << "  Applying H(0)..." << std::endl;
  ptrs.push_back(apply_gate_ptr(batched, h_gate<double>(), {}, {0}));

  // 3. Apply CNOT(0, 1)
  // Control: 0, Target: 1
  // State -> (|000> + |110>) / sqrt(2)
  // |110> (Q0=1, Q1=1, Q2=0) is index 3 (binary 011).
  std::cout << "  Applying CNOT(0, 1)..." << std::endl;
  ptrs.push_back(apply_gate_ptr(batched, x_gate<double>(), {0}, {1}));

  // 4. Apply CNOT(1, 2)
  // Control: 1, Target: 2
  // State -> (|000> + |111>) / sqrt(2)
  // |111> (Q0=1, Q1=1, Q2=1) is index 7 (binary 111).
  std::cout << "  Applying CNOT(1, 2)..." << std::endl;
  ptrs.push_back(apply_gate_ptr(batched, x_gate<double>(), {1}, {2}));

  // 5. Verify Results
  std::cout << "  Computing Batch State Vectors..." << std::endl;
  auto results = batched.getBatchStateVectors();

  // Clean up gate memory
  for (auto p : ptrs) cudaFree(p);

  std::cout << "  Verifying GHZ State..." << std::endl;
  
  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  
  for (std::size_t b = 0; b < B; ++b) {
    const auto& sv = results[b];
    
    // We expect only two non-zero amplitudes:
    // |000> (Index 0) -> 1/sqrt(2)
    // |111> (Index 7) -> 1/sqrt(2)
    
    std::complex<double> amp000 = sv[0];
    std::complex<double> amp111 = sv[7]; // 2^3 - 1
    
    // Check non-zero components
    bool ok0 = approxEqual(amp000, std::complex<double>(inv_sqrt2, 0.0));
    bool ok1 = approxEqual(amp111, std::complex<double>(inv_sqrt2, 0.0));
    
    if (!ok0 || !ok1) {
      std::cerr << "  ❌ Sample " << b << " Failed!" << std::endl;
      std::cout << "     Expected: " << inv_sqrt2 << " for |000> and |111>" << std::endl;
      std::cout << "     Got |000>: "; printComplex(amp000); std::cout << std::endl;
      std::cout << "     Got |111>: "; printComplex(amp111); std::cout << std::endl;
      
      // Dump full state
      std::cout << "     Full State:" << std::endl;
      for(size_t i=0; i<sv.size(); ++i) {
          if(std::abs(sv[i]) > 1e-6) {
            std::cout << "       Index " << i << ": "; printComplex(sv[i]); std::cout << std::endl;
          }
      }
      assert(false);
    }
    
    // Check other components are zero
    double totalProb = std::norm(amp000) + std::norm(amp111);
    if (std::abs(totalProb - 1.0) > 1e-5) {
        // This means there are other non-zero components
        std::cerr << "  ❌ Sample " << b << " has extra components (Prob sum = " << totalProb << ")" << std::endl;
        assert(false);
    }
  }
  
  std::cout << "✅ GHZ Test PASSED: Correctly generated (|000> + |111>)/sqrt(2) for all " << B << " batches." << std::endl;
}

int main() {
  try {
    test_ghz_batch();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
}

