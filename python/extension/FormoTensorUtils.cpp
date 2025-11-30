/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "nvqir/CircuitSimulator.h"
#include "nvqir/formotensor/FormoTensorCircuitSimulator.h"
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace nvqir {
// Forward declaration of the function to get the active simulator
CircuitSimulator *getCircuitSimulatorInternal();
} // namespace nvqir

namespace {

// Helper to get the active FormoTensor simulator
template <typename ScalarType>
nvqir::SimulatorFormoTensor<ScalarType> *getFormoTensorSimulator() {
  auto *sim = nvqir::getCircuitSimulatorInternal();
  if (!sim) {
    throw std::runtime_error("No active circuit simulator found.");
  }

  // Try to cast to FormoTensor simulator
  // Note: We might need to check the name first or use dynamic_cast if RTTI is
  // enabled and classes are polymorphic enough. CircuitSimulator has virtual
  // functions so dynamic_cast should work.
  auto *formoSim =
      dynamic_cast<nvqir::SimulatorFormoTensor<ScalarType> *>(sim);
  
  return formoSim;
}

// Helper to check if we are using float or double precision
bool isDoublePrecision() {
  auto *sim = nvqir::getCircuitSimulatorInternal();
  if (!sim) return true; // Default to double
  return sim->isDoublePrecision();
}

} // namespace

PYBIND11_MODULE(_formotensor_utils, m) {
  m.doc() = "FormoTensor utility functions for batch processing.";

  m.def(
      "load_batch_state",
      [](py::array_t<std::complex<double>> stateData, std::size_t batchSize) {
        py::buffer_info buf = stateData.request();
        
        // Check dimensions? 
        // We expect a flat buffer or [Batch, StateDim]
        // For simplicity, we just take the flat data and trust the user/wrapper.
        
        if (isDoublePrecision()) {
           auto *sim = getFormoTensorSimulator<double>();
           if (!sim) {
             throw std::runtime_error("Active simulator is not FormoTensor (double).");
           }
           
           auto *ptr = static_cast<std::complex<double> *>(buf.ptr);
           std::vector<std::complex<double>> dataVec(ptr, ptr + buf.size);
           sim->loadBatchStateData(dataVec, batchSize);
        } else {
           // If simulator is float, we need to convert double input (from python complex128) to float
           auto *sim = getFormoTensorSimulator<float>();
           if (!sim) {
             throw std::runtime_error("Active simulator is not FormoTensor (float).");
           }
           
           auto *ptr = static_cast<std::complex<double> *>(buf.ptr);
           std::vector<std::complex<float>> dataVec(buf.size);
           for(size_t i=0; i<buf.size; ++i) {
               dataVec[i] = static_cast<std::complex<float>>(ptr[i]);
           }
           sim->loadBatchStateData(dataVec, batchSize);
        }
      },
      "Load a batch of initial quantum states into the FormoTensor simulator.",
      py::arg("state_data"), py::arg("batch_size"));

  m.def(
      "set_batch_size",
      [](std::size_t batchSize) {
        if (isDoublePrecision()) {
           auto *sim = getFormoTensorSimulator<double>();
           if (sim) sim->setBatchSize(batchSize);
        } else {
           auto *sim = getFormoTensorSimulator<float>();
           if (sim) sim->setBatchSize(batchSize);
        }
      },
      "Set the batch size for the FormoTensor simulator.",
      py::arg("batch_size"));
}
