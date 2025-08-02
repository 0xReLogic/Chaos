"""
Test script for validating the Phase 4 refactor of the QuantumCircuit.
This script creates a Bell state and prints the final state probabilities.
"""

from quantum_circuit import create_bell_state

def test_bell_state_creation():
    print("--- Testing Bell State Creation (Post-Phase 4) ---")
    
    # 1. Create the Bell state circuit
    bell_circuit = create_bell_state()
    print("Initial state:")
    print(bell_circuit)
    
    # 2. Run the circuit to apply the gates
    bell_circuit.run()
    
    # 3. Print the final state and probabilities
    print("\nFinal state after applying H(0) and CNOT(0, 1):")
    print(bell_circuit)
    
    print("Expected result: Probabilities for |00> and |11> should be ~0.5 each.")
    print("Probabilities for |01> and |10> should be 0.")

if __name__ == "__main__":
    test_bell_state_creation()
