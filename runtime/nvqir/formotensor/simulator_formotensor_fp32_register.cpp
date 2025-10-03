/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "FormoTensorCircuitSimulator.h"
#include "../NVQIR.h"

namespace nvqir {
void registerFormoTensorSimulatorF32() {
  // Register the FormoTensor simulator backend with FP32 precision
  nvqir::registerCircuitSimulator("formotensor-fp32", getCircuitSimulator_formotensor);
}
} // namespace nvqir
