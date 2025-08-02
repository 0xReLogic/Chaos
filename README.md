# CHAOS - Quantum Computing Simulator

CHAOS is a quantum computing simulator implemented in Python. This project aims to simulate the behavior of quantum computers on classical computers.

## Concept

In Greek mythology, Chaos is the primordial void from which everything is born. Similarly, quantum computers operate in the realm of probability and superposition before "collapsing" into a definite answer.

## Features

This project builds a Python library that can:

1. Define Qubits, the basic units of quantum computation that can exist in state 0, 1, or both simultaneously (superposition).
2. Implement Quantum Gates, operations (such as rotations or flips) that manipulate the state of Qubits.
3. Simulate Quantum Entanglement, the "spooky" phenomenon where two Qubits become mysteriously connected.
4. Run Quantum Circuits (sequences of gates) and "measure" the results to get probabilistic answers.

## Installation

This project uses Python and NumPy. To install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate

# Install dependencies
pip install numpy
```

## Usage

### Qubit

```python
from qubit import Qubit

# Create a qubit in state |0⟩
q0 = Qubit(0)

# Create a qubit in state |1⟩
q1 = Qubit(1)

# Create a qubit in superposition
# 1/√2 |0⟩ + 1/√2 |1⟩ (50% probability of 0, 50% probability of 1)
import numpy as np
q_super = Qubit([1/np.sqrt(2), 1/np.sqrt(2)])

# Measure the qubit
result = q_super.measure()
print(f"Measurement result: {result}")  # 0 or 1 with 50% probability

# View probabilities
prob_zero, prob_one = q_super.get_probabilities()
print(f"Probability |0⟩: {prob_zero}, Probability |1⟩: {prob_one}")
```

### Quantum Gates

```python
from qubit import Qubit
from quantum_gates import apply_gate

# Create a qubit in state |0⟩
q = Qubit(0)

# Apply X gate (NOT gate) - flips |0⟩ to |1⟩
apply_gate(q, "X")
print(q)  # Should show state |1⟩

# Create a new qubit in state |0⟩
q = Qubit(0)

# Apply Hadamard gate to create superposition
apply_gate(q, "H")
print(q)  # Should show 50% |0⟩, 50% |1⟩

# Available gates:
# - "X": Pauli-X (NOT) gate
# - "Y": Pauli-Y gate
# - "Z": Pauli-Z (phase flip) gate
# - "H": Hadamard gate
# - "S": S gate (phase gate)
# - "T": T gate
# - "I": Identity gate

# Quantum coin flip example
q = Qubit(0)
apply_gate(q, "H")
result = q.measure()
print(f"Coin flip result: {'Heads' if result == 0 else 'Tails'}")
```

## Development

This project is developed in several phases:

### Phase 1: Quantum Particle (The Qubit)
- [x] Milestone 1.1: Create a Qubit class in Python. Internally, represent its state as a 2D vector with complex numbers (using NumPy).
- [x] Milestone 1.2: Implement functions to initialize a Qubit to state |0⟩ or |1⟩.
- [x] Milestone 1.3: Implement a measure() function that will "collapse" the Qubit's superposition to 0 or 1 based on its amplitude probabilities.

### Phase 2: Laws of Physics (The Quantum Gates)
- [x] Milestone 2.1: Represent each quantum gate (Pauli-X, Hadamard, CNOT) as 2x2 or 4x4 matrices.
- [x] Milestone 2.2: Create an apply_gate(qubit, gate) function that performs matrix multiplication to transform the Qubit state.

### Phase 3: Mini Universe (The Quantum Circuit)
- [ ] Milestone 3.1: Create a QuantumCircuit class that can manage multiple Qubits (a quantum register).
- [ ] Milestone 3.2: Implement methods to add a series of gates to the circuit.
- [ ] Milestone 3.3: Create a run() function that will execute all gates sequentially in the circuit and return the final measurement results.
