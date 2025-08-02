"""
CHAOS - Quantum Computing Simulator
Module: example.py
Description: Simple examples of using the CHAOS library
"""

import numpy as np
from qubit import Qubit

def basic_qubit_example():
    """Basic example of creating and measuring qubits"""
    print("=== Basic Qubit Example ===")
    
    # Create a qubit in state |0⟩
    q0 = Qubit(0)
    print("Qubit in state |0⟩:")
    print(q0)
    
    # Create a qubit in state |1⟩
    q1 = Qubit(1)
    print("\nQubit in state |1⟩:")
    print(q1)
    
    # Create a qubit in superposition
    q_super = Qubit([1/np.sqrt(2), 1/np.sqrt(2)])
    print("\nQubit in superposition (50% |0⟩, 50% |1⟩):")
    print(q_super)
    
    # Measure the qubit in superposition
    result = q_super.measure()
    print(f"\nMeasurement result: {result}")
    print("Qubit state after measurement:")
    print(q_super)

def statistical_measurement_example():
    """Example demonstrating statistical nature of quantum measurements"""
    print("\n=== Statistical Measurement Example ===")
    
    # Number of trials
    num_trials = 1000
    
    # Create a qubit with 30% chance of |0⟩ and 70% chance of |1⟩
    alpha = np.sqrt(0.3)
    beta = np.sqrt(0.7)
    
    print(f"Creating {num_trials} qubits with state:")
    print(f"α|0⟩ + β|1⟩ where |α|² = 0.3 and |β|² = 0.7")
    
    # Count measurement results
    zeros = 0
    ones = 0
    
    for _ in range(num_trials):
        q = Qubit([alpha, beta])
        result = q.measure()
        if result == 0:
            zeros += 1
        else:
            ones += 1
    
    # Calculate percentages
    percent_zeros = zeros / num_trials * 100
    percent_ones = ones / num_trials * 100
    
    print(f"\nResults after {num_trials} measurements:")
    print(f"Measured |0⟩: {zeros} times ({percent_zeros:.1f}%)")
    print(f"Measured |1⟩: {ones} times ({percent_ones:.1f}%)")
    print(f"Expected: approximately 30% for |0⟩ and 70% for |1⟩")

if __name__ == "__main__":
    print("CHAOS - Quantum Computing Simulator Examples")
    print("=" * 50)
    
    basic_qubit_example()
    statistical_measurement_example()
    
    print("\nExamples completed.")