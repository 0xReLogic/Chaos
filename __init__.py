"""
CHAOS - Quantum Computing Simulator
"""

from .qubit import Qubit
from .quantum_gates import (
    QuantumGate, 
    X_GATE, Y_GATE, Z_GATE, H_GATE, S_GATE, T_GATE, I_GATE,
    apply_gate, SINGLE_QUBIT_GATES
)
from .quantum_circuit import (
    QuantumCircuit,
    create_bell_state,
    create_deutsch_algorithm
)

__version__ = "0.3.0"