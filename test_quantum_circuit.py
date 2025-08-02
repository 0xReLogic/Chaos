"""
CHAOS - Quantum Computing Simulator
Module: test_quantum_circuit.py
Description: Test script for the Quantum Circuit implementation
"""

import numpy as np
from qubit import Qubit
from quantum_gates import apply_gate
from quantum_circuit import QuantumCircuit, create_bell_state, create_deutsch_algorithm
from collections import Counter

def test_basic_circuit():
    """Test basic quantum circuit operations"""
    print("\n=== Testing Basic Quantum Circuit ===")
    
    # Create a circuit with 2 qubits
    circuit = QuantumCircuit(2)
    print("Initial circuit state:")
    print(circuit)
    
    # Apply X gate to the first qubit
    circuit.apply_gate("X", 0)
    print("\nAfter applying X gate to qubit 0:")
    print(circuit)
    
    # Apply Hadamard gate to the second qubit
    circuit.apply_gate("H", 1)
    print("\nAfter applying H gate to qubit 1:")
    print(circuit)
    
    # Run the circuit
    circuit.run()
    print("\nAfter running the circuit:")
    print(circuit)
    
    # Measure all qubits
    results = circuit.measure()
    print(f"\nMeasurement results: {results}")
    print("Circuit state after measurement:")
    print(circuit)

def test_cnot_gate():
    """Test CNOT gate in a quantum circuit"""
    print("\n=== Testing CNOT Gate ===")
    
    # Create a circuit with 2 qubits
    circuit = QuantumCircuit(2)
    
    # Apply X to the first qubit (control)
    circuit.apply_gate("X", 0)
    
    # Apply CNOT with first qubit as control, second as target
    circuit.apply_cnot(0, 1)
    
    print("Circuit before running:")
    print(circuit)
    
    # Run the circuit
    circuit.run()
    print("\nAfter running the circuit:")
    print(circuit)
    
    # Measure all qubits
    results = circuit.measure()
    print(f"\nMeasurement results: {results}")
    print("Expected: [1, 1] (both qubits in state |1⟩)")

def test_bell_state():
    """Test creation of a Bell state"""
    print("\n=== Testing Bell State Creation ===")
    
    # Create a Bell state circuit
    circuit = create_bell_state()
    
    print("Bell state circuit before running:")
    print(circuit)
    
    # Run the circuit
    circuit.run()
    print("\nAfter running the circuit:")
    print(circuit)
    
    # Test statistical behavior of Bell state
    print("\n=== Testing Statistical Behavior of Bell State ===")
    results = []
    num_trials = 1000
    
    for _ in range(num_trials):
        bell_circuit = create_bell_state()
        bell_circuit.run()
        results.append(tuple(bell_circuit.measure()))
    
    # Count occurrences
    counts = Counter(results)
    print(f"Results from {num_trials} measurements:")
    print(f"Measured [0, 0]: {counts[(0, 0)]} times ({counts[(0, 0)]/num_trials*100:.1f}%)")
    print(f"Measured [0, 1]: {counts[(0, 1)]} times ({counts[(0, 1)]/num_trials*100:.1f}%)")
    print(f"Measured [1, 0]: {counts[(1, 0)]} times ({counts[(1, 0)]/num_trials*100:.1f}%)")
    print(f"Measured [1, 1]: {counts[(1, 1)]} times ({counts[(1, 1)]/num_trials*100:.1f}%)")
    print("Expected: approximately 50% [0, 0] and 50% [1, 1], with [0, 1] and [1, 0] being rare or absent")

def test_deutsch_algorithm():
    """Test Deutsch's Algorithm"""
    print("\n=== Testing Deutsch's Algorithm ===")
    
    # Test with constant function f(x) = 0
    print("\nTesting with constant function f(x) = 0:")
    circuit_const_0 = create_deutsch_algorithm("constant_0")
    circuit_const_0.run()
    result = circuit_const_0.measure(0)  # Measure only the first qubit
    print(f"Measurement result: {result}")
    print(f"Function is {'constant' if result == 0 else 'balanced'}")
    print("Expected: 0 (constant)")
    
    # Test with constant function f(x) = 1
    print("\nTesting with constant function f(x) = 1:")
    circuit_const_1 = create_deutsch_algorithm("constant_1")
    circuit_const_1.run()
    result = circuit_const_1.measure(0)
    print(f"Measurement result: {result}")
    print(f"Function is {'constant' if result == 0 else 'balanced'}")
    print("Expected: 0 (constant)")
    
    # Test with balanced function f(x) = x
    print("\nTesting with balanced function f(x) = x:")
    circuit_identity = create_deutsch_algorithm("identity")
    circuit_identity.run()
    result = circuit_identity.measure(0)
    print(f"Measurement result: {result}")
    print(f"Function is {'constant' if result == 0 else 'balanced'}")
    print("Expected: 1 (balanced)")
    
    # Test with balanced function f(x) = NOT x
    print("\nTesting with balanced function f(x) = NOT x:")
    circuit_negation = create_deutsch_algorithm("negation")
    circuit_negation.run()
    result = circuit_negation.measure(0)
    print(f"Measurement result: {result}")
    print(f"Function is {'constant' if result == 0 else 'balanced'}")
    print("Expected: 1 (balanced)")

if __name__ == "__main__":
    print("CHAOS - Quantum Computing Simulator")
    print("Quantum Circuit Testing")
    print("=" * 50)
    
    test_basic_circuit()
    test_cnot_gate()
    test_bell_state()
    test_deutsch_algorithm()
    
    print("\nAll tests completed.")