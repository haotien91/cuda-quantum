// Include standard library headers BEFORE namespace-internal headers
#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <cutensornet.h>

#include "formotensor_state.h"
#include "formotensor_state.inc"
#include "formotensor_utils.h"

using namespace nvqir;

// Helper: Timer
class Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_time;
public:
    void start() { start_time = Clock::now(); }
    double stop() {
        auto end_time = Clock::now();
        return std::chrono::duration<double>(end_time - start_time).count();
    }
};

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
std::vector<std::complex<T>> random_single_qubit_gate(std::mt19937& rng) {
    std::uniform_real_distribution<T> dist(0.0, 2 * M_PI);
    T theta = dist(rng);
    // Use Ry for simplicity
    T c = std::cos(theta / 2.0);
    T s = std::sin(theta / 2.0);
    return {{c, 0}, {-s, 0}, {s, 0}, {c, 0}};
}

template <typename T>
std::vector<std::complex<T>> cnot_gate() {
    return {{1,0}, {0,0}, {0,0}, {0,0},
            {0,0}, {1,0}, {0,0}, {0,0},
            {0,0}, {0,0}, {0,0}, {1,0},
            {0,0}, {0,0}, {1,0}, {0,0}};
}

// Benchmark Configuration
struct BenchConfig {
    size_t numQubits;
    size_t depth;
    size_t batchSize;
};

// Run Serial Benchmark
// Returns average time per sample (total time / B) ? No, return Total Time.
double run_serial_benchmark(const BenchConfig& cfg, TestBackend& env) {
    std::cout << "  Running Serial Benchmark (B=" << cfg.batchSize << ")..." << std::flush;
    
    // Pre-generate gates so we measure execution time, not generation time
    // Although generation is fast.
    // For shared parameters, we use the SAME circuit B times.
    
    std::vector<std::vector<std::complex<double>>> layerGates(cfg.depth);
    std::vector<int> layerTargets(cfg.depth);
    
    for(size_t d=0; d<cfg.depth; ++d) {
        layerGates[d] = random_single_qubit_gate<double>(env.rng);
        layerTargets[d] = d % cfg.numQubits;
    }

    Timer timer;
    timer.start();
    
    for (size_t b = 0; b < cfg.batchSize; ++b) {
        FormoTensorState<double> state(cfg.numQubits, env.scratch, env.handle, env.rng);
        state.setZeroState();
        
        std::vector<void*> ptrs; // To manage memory
        
        for(size_t d=0; d<cfg.depth; ++d) {
             void* d_gate = allocateGateMatrix<double>(layerGates[d]);
             ptrs.push_back(d_gate);
             state.applyGate({}, {static_cast<int32_t>(layerTargets[d])}, d_gate, false);
        }
        
        // Force computation
        auto res = state.getStateVector();
        
        for(auto p : ptrs) cudaFree(p);
    }
    
    double elapsed = timer.stop();
    std::cout << " Done. Time: " << elapsed << "s" << std::endl;
    return elapsed;
}

// Run Batch Benchmark
double run_batch_benchmark(const BenchConfig& cfg, TestBackend& env) {
    std::cout << "  Running Shared Batch Benchmark (B=" << cfg.batchSize << ")..." << std::flush;
    
    std::vector<std::vector<std::complex<double>>> layerGates(cfg.depth);
    std::vector<int> layerTargets(cfg.depth);
    
    for(size_t d=0; d<cfg.depth; ++d) {
        layerGates[d] = random_single_qubit_gate<double>(env.rng);
        layerTargets[d] = d % cfg.numQubits;
    }

    Timer timer;
    timer.start();
    
    {
        FormoTensorState<double> state(cfg.numQubits, cfg.batchSize, env.scratch, env.handle, env.rng);
        state.setZeroState();
        
        std::vector<void*> ptrs; 
        
        for(size_t d=0; d<cfg.depth; ++d) {
             void* d_gate = allocateGateMatrix<double>(layerGates[d]);
             ptrs.push_back(d_gate);
             state.applyGate({}, {static_cast<int32_t>(layerTargets[d])}, d_gate, false);
        }
        
        // Force computation
        auto res = state.getBatchStateVectors();
        
        for(auto p : ptrs) cudaFree(p);
    } // Destructor cleans up
    
    double elapsed = timer.stop();
    std::cout << " Done. Time: " << elapsed << "s" << std::endl;
    return elapsed;
}

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << "   FormoTensor Shared Parameter Batch Benchmark" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    try {
        TestBackend env;
        
        // Warmup
        {
            FormoTensorState<double> dummy(2, env.scratch, env.handle, env.rng);
            dummy.setZeroState();
            dummy.getStateVector();
        }

        // Scenarios
        // Increase qubit count to make contraction dominant
        // Note: With n=20, state vector size is 2^20 * 16 bytes = 16MB.
        // Batch=100 -> 1.6GB. 
        size_t N = 20; 
        size_t D = 50;
        
        std::vector<size_t> batchSizes = {1, 10, 50, 100};
        
        std::cout << "\nCircuit: " << N << " Qubits, Depth " << D << " (Random Single Qubit Gates)" << std::endl;
        std::cout << std::setw(10) << "BatchSize" 
                  << std::setw(15) << "Serial(s)" 
                  << std::setw(15) << "Batch(s)" 
                  << std::setw(15) << "Speedup" << std::endl;
        std::cout << "---------------------------------------------------------" << std::endl;
        
        for (size_t B : batchSizes) {
            BenchConfig cfg{N, D, B};
            
            double t_serial = run_serial_benchmark(cfg, env);
            double t_batch = run_batch_benchmark(cfg, env);
            double speedup = t_serial / t_batch;
            
            std::cout << std::setw(10) << B 
                      << std::setw(15) << std::fixed << std::setprecision(4) << t_serial 
                      << std::setw(15) << std::fixed << std::setprecision(4) << t_batch 
                      << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

