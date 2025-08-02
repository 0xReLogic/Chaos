"""
CHAOS - Quantum Computing Simulator
Module: test_qubit.py
Description: Test script for the Qubit implementation
"""

from qubit import Qubit
import numpy as np
from collections import Counter

def test_initialization():
    """Test qubit initialization"""
    print("\n=== Testing Qubit Initialization ===")
    
    # Initialize to |0⟩
    q0 = Qubit(0)
    print("Qubit initialized to |0⟩:")
    print(q0)
    
    # Initialize to |1⟩
    q1 = Qubit(1)
    print("\nQubit initialized to |1⟩:")
    print(q1)
    
    # Initialize to custom state (superposition)
    # 1/√2 |0⟩ + 1/√2 |1⟩
    q_super = Qubit([1/np.sqrt(2), 1/np.sqrt(2)])
    print("\nQubit initialized to superposition (1/√2 |0⟩ + 1/√2 |1⟩):")
    print(q_super)
    
    # Test initialize_to_zero and initialize_to_one methods
    q_custom = Qubit([0.6, 0.8])
    print("\nQubit initialized to custom state [0.6, 0.8] (will be normalized):")
    print(q_custom)
    
    q_custom.initialize_to_zero()
    print("\nAfter initialize_to_zero():")
    print(q_custom)
    
    q_custom.initialize_to_one()
    print("\nAfter initialize_to_one():")
    print(q_custom)

def test_measurement():
    """Test qubit measurement"""
    print("\n=== Testing Qubit Measurement ===")
    
    # Create a qubit in superposition
    q = Qubit([1/np.sqrt(2), 1/np.sqrt(2)])
    print("Qubit in superposition (50% |0⟩, 50% |1⟩):")
    print(q)
    
    # Measure once
    result = q.measure()
    print(f"\nMeasurement result: {result}")
    print("Qubit state after measurement:")
    print(q)
    
    # Test multiple measurements on identical qubits
    print("\n=== Testing Multiple Measurements ===")
    results = []
    num_trials = 1000
    
    for _ in range(num_trials):
        # Create a fresh qubit in superposition for each measurement
        q = Qubit([1/np.sqrt(2), 1/np.sqrt(2)])
        results.append(q.measure())
    
    # Count occurrences
    counts = Counter(results)
    print(f"Results from {num_trials} measurements:")
    print(f"Measured 0: {counts[0]} times ({counts[0]/num_trials*100:.1f}%)")
    print(f"Measured 1: {counts[1]} times ({counts[1]/num_trials*100:.1f}%)")
    print("Expected: approximately 50% for each")

def test_biased_qubit():
    """Test a qubit with biased probabilities"""
    print("\n=== Testing Biased Qubit ===")
    
    # Create a qubit with 70% chance of |0⟩ and 30% chance of |1⟩
    # sqrt(0.7) ≈ 0.837, sqrt(0.3) ≈ 0.548
    q = Qubit([np.sqrt(0.7), np.sqrt(0.3)])
    print("Qubit with 70% chance of |0⟩ and 30% chance of |1⟩:")
    print(q)
    
    # Test multiple measurements
    results = []
    num_trials = 1000
    
    for _ in range(num_trials):
        # Create a fresh qubit with the same bias for each measurement
        q = Qubit([np.sqrt(0.7), np.sqrt(0.3)])
        results.append(q.measure())
    
    # Count occurrences
    counts = Counter(results)
    print(f"Results from {num_trials} measurements:")
    print(f"Measured 0: {counts[0]} times ({counts[0]/num_trials*100:.1f}%)")
    print(f"Measured 1: {counts[1]} times ({counts[1]/num_trials*100:.1f}%)")
    print("Expected: approximately 70% for 0 and 30% for 1")

if __name__ == "__main__":
    print("CHAOS - Quantum Computing Simulator")
    print("Qubit Testing")
    print("=" * 50)
    
    test_initialization()
    test_measurement()
    test_biased_qubit()
    
    print("\nAll tests completed.")