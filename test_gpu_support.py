#!/usr/bin/env python3
"""
Test GPU support for CHAOS quantum simulator
"""

def test_gpu_availability():
    """Test if GPU support is properly detected and configured"""
    from quantum_circuit import QuantumCircuit, GPU_AVAILABLE, array_lib
    
    print("=== CHAOS GPU Support Test ===")
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"Array Library: {array_lib.__name__}")
    
    # Test basic circuit creation
    circuit = QuantumCircuit(3)
    print(f"State vector type: {type(circuit.state_vector)}")
    print(f"State vector device: ", end="")
    
    if hasattr(circuit.state_vector, 'device'):
        print(f"GPU device {circuit.state_vector.device}")
    else:
        print("CPU (NumPy)")
    
    # Test basic operations
    circuit.apply_gate("H", 0)
    circuit.apply_cnot(0, 1)  # Use the dedicated CNOT method
    circuit.run()
    
    print(f"Final state vector type: {type(circuit.state_vector)}")
    print("Circuit operations completed successfully!")
    
    # Test measurement (this should work on both GPU/CPU)
    result = circuit.measure()
    print(f"Measurement result: {result}")
    
    return GPU_AVAILABLE

if __name__ == "__main__":
    gpu_available = test_gpu_availability()
    
    if not gpu_available:
        print("\n=== To enable GPU support ===")
        print("Install CuPy with:")
        print("  pip install cupy-cuda12x  # For CUDA 12.x")
        print("  pip install cupy-cuda11x  # For CUDA 11.x")
        print("\nThen verify with:")
        print("  python -c \"import cupy; print(f'GPU: {cupy.cuda.Device().compute_capability}')\"")
