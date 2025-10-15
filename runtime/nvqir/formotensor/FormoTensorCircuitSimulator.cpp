/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "FormoTensorCircuitSimulator.h"
#include "formotensor_state.inc"

namespace nvqir {

// C API for registration
extern "C" nvqir::CircuitSimulator *getCircuitSimulator_formotensor() {
  return new SimulatorFormoTensor<double>();
}

} // namespace nvqir
