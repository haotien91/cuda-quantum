import cudaq
import numpy as np
import pytest

# Skip if formotensor target is not available
try:
    cudaq.set_target('formotensor')
except:
    pytest.skip("formotensor target not available", allow_module_level=True)

def test_batch_loading_and_execution():
    # 1. Define parameters
    num_qubits = 2
    batch_size = 5
    state_dim = 1 << num_qubits
    
    # 2. Create random batch state
    # Shape: (batch_size, state_dim)
    np.random.seed(42)
    data = np.random.rand(batch_size, state_dim) + 1j * np.random.rand(batch_size, state_dim)
    # Normalize each state
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    
    # 3. Load state into simulator
    # This should implicitly set batch size
    cudaq.formotensor.load_state(data)
    
    # 4. Define a simple kernel (Identity for verification)
    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(num_qubits)
        # No operations, just return state
        
    # 5. Get state
    # Should return flattened array of size batch_size * state_dim
    flat_state = cudaq.get_state(kernel)
    
    # 6. Verify
    assert len(flat_state) == batch_size * state_dim
    
    # Reshape and compare
    reshaped_state = np.array(flat_state).reshape(batch_size, state_dim)
    
    # Check fidelity/closeness
    # Note: Phase might differ globally, but here we loaded the state directly, 
    # so it should match exactly if no operations were performed.
    assert np.allclose(data, reshaped_state, atol=1e-6)
    
    print("Batch state loading and retrieval verified successfully!")

def test_set_batch_size_explicit():
    batch_size = 10
    cudaq.formotensor.set_batch_size(batch_size)
    # We can't easily verify internal state without running something, 
    # but this checks if the call succeeds.
    print("set_batch_size called successfully.")

if __name__ == "__main__":
    try:
        test_batch_loading_and_execution()
        test_set_batch_size_explicit()
    except Exception as e:
        print(f"Test failed: {e}")
        exit(1)
