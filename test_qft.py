import numpy as np
from quantum_circuit import QuantumCircuit

def test_qft_iqft_identity():
    """
    Tests if applying QFT followed by IQFT returns the initial state.
    This verifies that IQFT is the correct inverse of QFT.
    """
    print("\n--- Testing QFT-IQFT Identity ---")
    num_qubits = 3
    
    # Create a circuit and prepare a non-trivial initial state, e.g., |101>
    circuit = QuantumCircuit(num_qubits)
    circuit.apply_gate('X', 0)
    circuit.apply_gate('X', 2)
    circuit.run() # Run to prepare the state
    
    initial_state = np.copy(circuit.state_vector)
    print(f"Initial state prepared (should be |101>)")
    print(circuit)
    
    # Clear operations to apply QFT and IQFT on the prepared state
    circuit.operations = []
    
    # Apply QFT, then IQFT
    print("Applying QFT...")
    circuit.apply_qft()
    print("Applying IQFT...")
    circuit.apply_iqft()
    
    # Run the QFT and IQFT operations
    circuit.run()
    
    final_state = circuit.state_vector
    
    print("Final state after QFT and IQFT:")
    print(circuit)
    
    # Check if the final state is close to the initial state
    assert np.allclose(initial_state, final_state), "Final state does not match initial state!"
    
    print("✅ Test Passed: QFT and IQFT are correct inverses.")

if __name__ == "__main__":
    test_qft_iqft_identity()
