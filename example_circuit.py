"""
CHAOS - Quantum Computing Simulator
Module: example_circuit.py
Description: Examples of using quantum circuits
"""

import numpy as np
from qubit import Qubit
from quantum_gates import apply_gate
from quantum_circuit import QuantumCircuit, create_bell_state, create_deutsch_algorithm
from collections import Counter

def basic_circuit_example():
    """Basic example of creating and using a quantum circuit"""
    print("=== Basic Quantum Circuit Example ===")
    
    # Create a circuit with 3 qubits
    circuit = QuantumCircuit(3)
    print("Initial circuit state:")
    print(circuit)
    
    # Apply some gates
    circuit.apply_gate("X", 0)  # NOT gate on qubit 0
    circuit.apply_gate("H", 1)  # Hadamard gate on qubit 1
    circuit.apply_cnot(0, 2)    # CNOT with qubit 0 as control, qubit 2 as target
    
    print("\nCircuit after adding gates (before running):")
    print(circuit)
    
    # Run the circuit
    circuit.run()
    print("\nCircuit after running:")
    print(circuit)
    
    # Measure all qubits
    results = circuit.measure()
    print(f"\nMeasurement results: {results}")
    print("Circuit state after measurement:")
    print(circuit)

def bell_state_example():
    """Example of creating and measuring a Bell state"""
    print("\n=== Bell State Example ===")
    
    # Create a Bell state circuit
    circuit = create_bell_state()
    print("Bell state circuit (before running):")
    print(circuit)
    
    # Run the circuit
    circuit.run()
    print("\nBell state circuit after running:")
    print(circuit)
    
    # Measure the Bell state multiple times
    print("\nMeasuring the Bell state multiple times:")
    results = []
    num_trials = 10
    
    for i in range(num_trials):
        # Create a fresh Bell state for each measurement
        bell_circuit = create_bell_state()
        bell_circuit.run()
        measurement = bell_circuit.measure()
        results.append(measurement)
        print(f"Trial {i+1}: {measurement}")
    
    # Count the occurrences of each result
    counts = Counter([tuple(result) for result in results])
    print("\nResults summary:")
    for result, count in counts.items():
        print(f"{result}: {count} times ({count/num_trials*100:.1f}%)")
    
    print("\nNote: Bell states should only give [0, 0] or [1, 1] results")
    print("due to quantum entanglement.")

def deutsch_algorithm_example():
    """Example of using Deutsch's Algorithm"""
    print("\n=== Deutsch's Algorithm Example ===")
    
    print("\nDeutsch's Algorithm determines whether a function f: {0,1} -> {0,1}")
    print("is constant or balanced with a single function evaluation.")
    print("- Constant: f(0) = f(1)")
    print("- Balanced: f(0) ≠ f(1)")
    
    # Test with all four possible functions
    function_types = ["constant_0", "constant_1", "identity", "negation"]
    function_descriptions = {
        "constant_0": "f(x) = 0 for all x (constant)",
        "constant_1": "f(x) = 1 for all x (constant)",
        "identity": "f(x) = x (balanced)",
        "negation": "f(x) = NOT x (balanced)"
    }
    
    for f_type in function_types:
        print(f"\nTesting with {function_descriptions[f_type]}:")
        
        # Create and run the circuit
        circuit = create_deutsch_algorithm(f_type)
        circuit.run()
        
        # Measure the first qubit
        result = circuit.measure(0)
        
        # Interpret the result
        if result == 0:
            conclusion = "constant"
        else:
            conclusion = "balanced"
        
        print(f"Measurement result: {result}")
        print(f"Conclusion: Function is {conclusion}")
        print(f"Actual: {function_descriptions[f_type].split('(')[1]}")

def quantum_teleportation_example():
    """Example of quantum teleportation protocol"""
    print("\n=== Quantum Teleportation Example ===")
    
    print("Quantum teleportation transfers the state of one qubit to another")
    print("using entanglement and classical communication.")
    
    # Create a circuit with 3 qubits
    circuit = QuantumCircuit(3)
    
    # Prepare the state to teleport (qubit 0)
    # Let's create a superposition state
    circuit.apply_gate("H", 0)
    circuit.apply_gate("T", 0)  # Add some phase to make it interesting
    
    # Print the state we want to teleport
    print("\nState to teleport (qubit 0):")
    q0 = circuit.get_qubit(0)
    prob_zero, prob_one = q0.get_probabilities()
    print(f"State: {q0.state}")
    print(f"Probabilities: |0⟩: {prob_zero:.4f}, |1⟩: {prob_one:.4f}")
    
    # Create entanglement between qubits 1 and 2
    circuit.apply_gate("H", 1)
    circuit.apply_cnot(1, 2)
    
    # Perform the teleportation protocol
    circuit.apply_cnot(0, 1)
    circuit.apply_gate("H", 0)
    
    # Measure qubits 0 and 1
    circuit.run()
    m0 = circuit.measure(0)
    m1 = circuit.measure(1)
    
    print(f"\nMeasurement results: m0={m0}, m1={m1}")
    
    # Apply corrections to qubit 2 based on measurements
    if m1 == 1:
        circuit.apply_gate("X", 2)
    if m0 == 1:
        circuit.apply_gate("Z", 2)
    
    circuit.run()
    
    # Check if teleportation was successful
    print("\nTeleported state (qubit 2):")
    q2 = circuit.get_qubit(2)
    prob_zero, prob_one = q2.get_probabilities()
    print(f"State: {q2.state}")
    print(f"Probabilities: |0⟩: {prob_zero:.4f}, |1⟩: {prob_one:.4f}")
    
    print("\nThe teleported state should match the original state (accounting for global phase).")

if __name__ == "__main__":
    print("CHAOS - Quantum Computing Simulator Examples")
    print("=" * 50)
    
    basic_circuit_example()
    bell_state_example()
    deutsch_algorithm_example()
    quantum_teleportation_example()
    
    print("\nExamples completed.")