"""
CHAOS - Quantum Computing Simulator
Module: test_quantum_gates.py
Description: Test script for the Quantum Gates implementation
"""

import numpy as np
from qubit import Qubit
from quantum_gates import (
    X_GATE, Y_GATE, Z_GATE, H_GATE, S_GATE, T_GATE, I_GATE,
    apply_gate, SINGLE_QUBIT_GATES
)
from collections import Counter

def test_x_gate():
    """Test the X (NOT) gate"""
    print("\n=== Testing X Gate (NOT Gate) ===")
    
    # Apply X gate to |0⟩
    q0 = Qubit(0)
    print("Initial state |0⟩:")
    print(q0)
    
    apply_gate(q0, "X")
    print("\nAfter applying X gate:")
    print(q0)
    
    # Apply X gate to |1⟩
    q1 = Qubit(1)
    print("\nInitial state |1⟩:")
    print(q1)
    
    apply_gate(q1, "X")
    print("\nAfter applying X gate:")
    print(q1)

def test_h_gate():
    """Test the Hadamard gate"""
    print("\n=== Testing H Gate (Hadamard Gate) ===")
    
    # Apply H gate to |0⟩
    q0 = Qubit(0)
    print("Initial state |0⟩:")
    print(q0)
    
    apply_gate(q0, "H")
    print("\nAfter applying H gate:")
    print(q0)
    
    # Apply H gate to |1⟩
    q1 = Qubit(1)
    print("\nInitial state |1⟩:")
    print(q1)
    
    apply_gate(q1, "H")
    print("\nAfter applying H gate:")
    print(q1)
    
    # Test statistical behavior of H gate
    print("\n=== Testing Statistical Behavior of H Gate ===")
    results = []
    num_trials = 1000
    
    for _ in range(num_trials):
        q = Qubit(0)
        apply_gate(q, "H")
        results.append(q.measure())
    
    # Count occurrences
    counts = Counter(results)
    print(f"Results from {num_trials} measurements after applying H to |0⟩:")
    print(f"Measured 0: {counts[0]} times ({counts[0]/num_trials*100:.1f}%)")
    print(f"Measured 1: {counts[1]} times ({counts[1]/num_trials*100:.1f}%)")
    print("Expected: approximately 50% for each")

def test_z_gate():
    """Test the Z gate"""
    print("\n=== Testing Z Gate (Phase Flip Gate) ===")
    
    # Apply Z gate to |0⟩
    q0 = Qubit(0)
    print("Initial state |0⟩:")
    print(q0)
    
    apply_gate(q0, "Z")
    print("\nAfter applying Z gate:")
    print(q0)
    
    # Apply Z gate to |1⟩
    q1 = Qubit(1)
    print("\nInitial state |1⟩:")
    print(q1)
    
    apply_gate(q1, "Z")
    print("\nAfter applying Z gate:")
    print(q1)
    
    # Apply Z gate to superposition
    q_super = Qubit([1/np.sqrt(2), 1/np.sqrt(2)])
    print("\nInitial superposition state (|0⟩ + |1⟩)/√2:")
    print(q_super)
    
    apply_gate(q_super, "Z")
    print("\nAfter applying Z gate:")
    print(q_super)

def test_gate_combinations():
    """Test combinations of gates"""
    print("\n=== Testing Gate Combinations ===")
    
    # Test H -> X -> H sequence on |0⟩
    q = Qubit(0)
    print("Initial state |0⟩:")
    print(q)
    
    apply_gate(q, "H")
    print("\nAfter applying H gate:")
    print(q)
    
    apply_gate(q, "X")
    print("\nAfter applying X gate:")
    print(q)
    
    apply_gate(q, "H")
    print("\nAfter applying H gate again:")
    print(q)
    
    # Test the "winning" condition from todo.txt:
    # Apply Hadamard to |0⟩ and measure multiple times to get 50% 0 and 50% 1
    print("\n=== Testing 'Winning' Condition ===")
    print("Applying Hadamard to |0⟩ and measuring multiple times")
    
    results = []
    num_trials = 1000
    
    for _ in range(num_trials):
        q = Qubit(0)
        apply_gate(q, "H")
        results.append(q.measure())
    
    # Count occurrences
    counts = Counter(results)
    print(f"Results from {num_trials} measurements:")
    print(f"Measured 0: {counts[0]} times ({counts[0]/num_trials*100:.1f}%)")
    print(f"Measured 1: {counts[1]} times ({counts[1]/num_trials*100:.1f}%)")
    print("Expected: approximately 50% for each")

if __name__ == "__main__":
    print("CHAOS - Quantum Computing Simulator")
    print("Quantum Gates Testing")
    print("=" * 50)
    
    test_x_gate()
    test_h_gate()
    test_z_gate()
    test_gate_combinations()
    
    print("\nAll tests completed.")