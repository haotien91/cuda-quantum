# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
from .mlir._mlir_libs import _formotensor_utils

def load_state(state_data: np.ndarray):
    """
    Load a batch of initial quantum states into the FormoTensor simulator.
    
    Args:
        state_data (np.ndarray): A numpy array of shape (batch_size, state_dim) 
                                 containing the initial state vectors.
                                 The state dimension must be a power of 2.
    """
    if not isinstance(state_data, np.ndarray):
        state_data = np.array(state_data)
        
    if state_data.ndim != 2:
        raise ValueError("state_data must be a 2D array of shape (batch_size, state_dim)")
        
    batch_size = state_data.shape[0]
    state_dim = state_data.shape[1]
    
    # Check if state_dim is power of 2
    if (state_dim & (state_dim - 1) != 0) or state_dim == 0:
        raise ValueError(f"State dimension {state_dim} must be a power of 2")
        
    # Flatten the array for C++ consumption
    # Ensure it's contiguous and complex128 (or compatible)
    flat_data = np.ascontiguousarray(state_data.flatten(), dtype=np.complex128)
    
    _formotensor_utils.load_batch_state(flat_data, batch_size)

def set_batch_size(batch_size: int):
    """
    Set the batch size for the FormoTensor simulator.
    
    Args:
        batch_size (int): The number of samples in the batch.
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0")
        
    _formotensor_utils.set_batch_size(batch_size)
