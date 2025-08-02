# CHAOS - A Physics-Accurate Quantum Computing Simulator

CHAOS is a multi-qubit quantum computing simulator built in Python. It is designed from the ground up to be physically accurate, modeling quantum phenomena like superposition and entanglement through a professional, state-vector-based architecture.

## Vision & Philosophy

In Greek mythology, **Chaos** is the primordial void from which the cosmos was born. This project embodies that spirit: it provides a foundational framework to simulate the probabilistic, indeterminate nature of quantum mechanics, from which definite, classical answers emerge upon measurement.

Unlike simpler simulators that manage qubits individually, CHAOS adopts the industry-standard approach used in professional and academic research, ensuring that its behavior correctly reflects the underlying mathematics of quantum mechanics.

## Core Architectural Pillars

The simulator's accuracy and power rest on three fundamental pillars:

1.  **Global State Vector**: The entire multi-qubit system is represented by a single, unified state vector of size 2^n (where n is the number of qubits). This is the only way to correctly capture system-wide correlations and entanglement.
2.  **Tensor Product Gate Application**: Quantum gates are not applied to qubits in isolation. Instead, they are expanded into full-system operators using the tensor product (Kronecker product). For example, applying a Hadamard gate to the first of three qubits involves creating an `H ⊗ I ⊗ I` operator, which then acts on the entire state vector. This is computationally intensive but physically correct.
3.  **Probabilistic Measurement & State Collapse**: Measurement is a probabilistic process based on the amplitudes of the state vector. When a qubit is measured, the system's state vector collapses into a new, valid state consistent with the measurement outcome, accurately modeling quantum mechanics.

## Key Features

-   **Stateful, Multi-Qubit Circuits**: Create circuits with any number of qubits.
-   **Rich State Visualization**: A human-readable `print()` output for any circuit, automatically calculating and displaying:
    -   Marginal probabilities for each qubit.
    -   An entanglement status (`Entangled` or `Separable`).
    -   Full system state probabilities.
-   **Accurate Partial Measurement**: Measure a single qubit and watch the entire system state collapse correctly.
-   **Iconic State Generators**: Built-in functions to instantly create famous entangled states like the Bell State and GHZ State.

## Installation

```bash
# It is recommended to use a virtual environment
python -m venv venv
# Windows: venv\Scripts\activate | MacOS/Linux: source venv/bin/activate

pip install -r requirements.txt
```

## Usage Guide

### Example 1: Creating a Bell State (2-Qubit Entanglement)

The Bell State is the simplest and most famous example of entanglement.

```python
from quantum_circuit import create_bell_state

# This helper function creates a 2-qubit circuit,
# applies H to the first qubit, then CNOT(0, 1).
bell_circuit = create_bell_state()
bell_circuit.run()

print(bell_circuit)
```

**Expected Output:**
```
Quantum Circuit (2 qubits, Entangled)
=====================================
Qubit 0: |0⟩=50.0%, |1⟩=50.0%
Qubit 1: |0⟩=50.0%, |1⟩=50.0%
-------------------------------------
System State Probabilities:
  |00⟩: 50.0%
  |11⟩: 50.0%
```
This output correctly shows that the system is entangled and will only ever be measured as `00` or `11`.

### Example 2: Creating a GHZ State (Multi-Qubit Entanglement)

The Greenberger–Horne–Zeilinger (GHZ) state extends entanglement to three or more qubits.

```python
from quantum_circuit import create_ghz_state

ghz_circuit = create_ghz_state(3)
ghz_circuit.run()

print(ghz_circuit)
```

**Expected Output:**
```
Quantum Circuit (3 qubits, Entangled)
=====================================
Qubit 0: |0⟩=50.0%, |1⟩=50.0%
Qubit 1: |0⟩=50.0%, |1⟩=50.0%
Qubit 2: |0⟩=50.0%, |1⟩=50.0%
-------------------------------------
System State Probabilities:
  |000⟩: 50.0%
  |111⟩: 50.0%
```
This shows that all three qubits are linked; they will all be `0` or all be `1` upon measurement.

## Project Roadmap

-   **Phase 1-3 (Complete):** Foundational implementation of qubits, gates, and basic circuits.
-   **Phase 4 (Complete):** The Great Refactor to a global state vector architecture.
-   **Phase 5 (Complete):** Implementation of partial measurement and rich state visualization.
-   **Phase 6 (Next):** Implementation of complex quantum algorithms like the Quantum Fourier Transform (QFT) and Grover's search algorithm.
