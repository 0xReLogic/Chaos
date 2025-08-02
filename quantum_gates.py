"""
CHAOS - Quantum Computing Simulator
Module: quantum_gates.py
Description: Implementation of Quantum Gates for qubit manipulation
"""

import numpy as np
from typing import Union, List, Tuple, Dict, Optional
from qubit import Qubit

class QuantumGate:
    """
    Base class for quantum gates.
    
    A quantum gate is a unitary transformation that acts on qubits.
    It is represented by a matrix that, when applied to a qubit's state vector,
    transforms it to a new state.
    """
    
    def __init__(self, name: str, matrix: np.ndarray):
        """
        Initialize a quantum gate with a name and matrix representation.
        
        Args:
            name: The name of the gate (e.g., "X", "H", "CNOT")
            matrix: The matrix representation of the gate
        """
        self.name = name
        self.matrix = np.array(matrix, dtype=complex)
        
        # Verify that the matrix is unitary
        # For a unitary matrix U, U† * U = I (identity matrix)
        # where U† is the conjugate transpose of U
        n = self.matrix.shape[0]
        identity = np.eye(n)
        conjugate_transpose = self.matrix.conj().T
        product = np.dot(conjugate_transpose, self.matrix)
        
        # Check if the product is approximately the identity matrix
        if not np.allclose(product, identity, rtol=1e-5, atol=1e-8):
            raise ValueError(f"The matrix for gate {name} is not unitary")
    
    def apply(self, qubit: Qubit) -> None:
        """
        Apply the gate to a qubit, transforming its state.
        
        Args:
            qubit: The qubit to transform
        """
        qubit.state = np.dot(self.matrix, qubit.state)
    
    def __str__(self) -> str:
        """String representation of the gate."""
        return f"{self.name} Gate:\n{self.matrix}"


# Define common single-qubit gates

# Pauli-X Gate (NOT Gate)
# Flips |0⟩ to |1⟩ and |1⟩ to |0⟩
X_GATE = QuantumGate("X", [
    [0, 1],
    [1, 0]
])

# Pauli-Y Gate
# Rotates the qubit state around the Y-axis of the Bloch sphere
Y_GATE = QuantumGate("Y", [
    [0, -1j],
    [1j, 0]
])

# Pauli-Z Gate
# Flips the phase of |1⟩ state
Z_GATE = QuantumGate("Z", [
    [1, 0],
    [0, -1]
])

# Hadamard Gate
# Creates superposition: |0⟩ -> (|0⟩ + |1⟩)/√2, |1⟩ -> (|0⟩ - |1⟩)/√2
H_GATE = QuantumGate("H", [
    [1/np.sqrt(2), 1/np.sqrt(2)],
    [1/np.sqrt(2), -1/np.sqrt(2)]
])

# S Gate (Phase Gate)
# Rotates |1⟩ by 90 degrees
S_GATE = QuantumGate("S", [
    [1, 0],
    [0, 1j]
])

# T Gate
# Rotates |1⟩ by 45 degrees
T_GATE = QuantumGate("T", [
    [1, 0],
    [0, np.exp(1j * np.pi / 4)]
])

# Identity Gate
# Does nothing to the qubit
I_GATE = QuantumGate("I", [
    [1, 0],
    [0, 1]
])

# Dictionary of all single-qubit gates for easy access
SINGLE_QUBIT_GATES = {
    "X": X_GATE,
    "Y": Y_GATE,
    "Z": Z_GATE,
    "H": H_GATE,
    "S": S_GATE,
    "T": T_GATE,
    "I": I_GATE
}

# Function to apply a gate to a qubit
def apply_gate(qubit: Qubit, gate: Union[str, QuantumGate]) -> None:
    """
    Apply a quantum gate to a qubit.
    
    Args:
        qubit: The qubit to transform
        gate: Either a QuantumGate object or a string key from SINGLE_QUBIT_GATES
    
    Raises:
        ValueError: If the gate string is not recognized
        TypeError: If the gate is neither a string nor a QuantumGate
    """
    if isinstance(gate, str):
        if gate in SINGLE_QUBIT_GATES:
            SINGLE_QUBIT_GATES[gate].apply(qubit)
        else:
            raise ValueError(f"Unknown gate: {gate}")
    elif isinstance(gate, QuantumGate):
        gate.apply(qubit)
    else:
        raise TypeError("Gate must be a string or QuantumGate object")