"""
CHAOS - Quantum Computing Simulator
Module: example_gates.py
Description: Examples of using quantum gates
"""

import numpy as np
from qubit import Qubit
from quantum_gates import (
    X_GATE, Y_GATE, Z_GATE, H_GATE, S_GATE, T_GATE, I_GATE,
    apply_gate, SINGLE_QUBIT_GATES
)

def basic_gate_example():
    """Basic example of applying quantum gates to qubits"""
    print("=== Basic Quantum Gate Example ===")
    
    # Create a qubit in state |0⟩
    q = Qubit(0)
    print("Initial qubit state |0⟩:")
    print(q)
    
    # Apply X gate (NOT gate)
    apply_gate(q, "X")
    print("\nAfter applying X gate:")
    print(q)
    
    # Apply H gate to create superposition
    q = Qubit(0)  # Reset to |0⟩
    apply_gate(q, "H")
    print("\nAfter applying H gate to |0⟩:")
    print(q)
    
    # Measure the qubit in superposition
    result = q.measure()
    print(f"\nMeasurement result: {result}")
    print("Qubit state after measurement:")
    print(q)

def quantum_coin_flip():
    """Example of a quantum coin flip using the Hadamard gate"""
    print("\n=== Quantum Coin Flip Example ===")
    
    print("A quantum coin flip uses a qubit in state |0⟩,")
    print("applies a Hadamard gate to create a superposition,")
    print("and then measures the qubit to get a random result.")
    
    # Create a qubit in state |0⟩
    q = Qubit(0)
    
    # Apply H gate to create superposition
    apply_gate(q, "H")
    print("\nQubit in superposition after applying H gate:")
    print(q)
    
    # Measure the qubit to get the coin flip result
    result = q.measure()
    print(f"\nCoin flip result: {'Heads' if result == 0 else 'Tails'}")
    
    # Perform multiple coin flips
    num_flips = 1000
    results = []
    
    for _ in range(num_flips):
        q = Qubit(0)
        apply_gate(q, "H")
        results.append(q.measure())
    
    heads = results.count(0)
    tails = results.count(1)
    
    print(f"\nResults from {num_flips} quantum coin flips:")
    print(f"Heads: {heads} ({heads/num_flips*100:.1f}%)")
    print(f"Tails: {tails} ({tails/num_flips*100:.1f}%)")

def phase_flip_example():
    """Example demonstrating phase flip with Z gate"""
    print("\n=== Phase Flip Example ===")
    
    # Create a qubit in superposition
    q = Qubit(0)
    apply_gate(q, "H")
    print("Qubit in superposition (|0⟩ + |1⟩)/√2:")
    print(q)
    
    # Apply Z gate to flip the phase of |1⟩
    apply_gate(q, "Z")
    print("\nAfter applying Z gate (phase flip):")
    print(q)
    
    # Apply H gate again
    apply_gate(q, "H")
    print("\nAfter applying H gate again:")
    print(q)
    
    # Measure the qubit
    result = q.measure()
    print(f"\nMeasurement result: {result}")
    print("Qubit state after measurement:")
    print(q)

if __name__ == "__main__":
    print("CHAOS - Quantum Computing Simulator Examples")
    print("=" * 50)
    
    basic_gate_example()
    quantum_coin_flip()
    phase_flip_example()
    
    print("\nExamples completed.")