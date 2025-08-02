# CHAOS - Quantum Computing Simulator

CHAOS is a quantum computing simulator implemented in Python. This project aims to simulate the behavior of quantum computers on classical computers.

## Concept

In Greek mythology, Chaos is the primordial void from which everything is born. Similarly, quantum computers operate in the realm of probability and superposition before "collapsing" into a definite answer.

The core of CHAOS is built around a powerful, modern architecture:

-   **Global State Vector**: Instead of managing individual qubits, the `QuantumCircuit` now operates on a single, global state vector of size 2^n (where n is the number of qubits). This is the standard, professional approach for quantum simulation, allowing for the accurate representation of complex, system-wide states like entanglement.
-   **Tensor Product Operations**: All quantum gates, including single-qubit and controlled gates, are applied to the state vector using their full matrix representation expanded via tensor products. This ensures that the simulation is mathematically and physically accurate.
-   **True Entanglement**: The simulator now correctly models entanglement without shortcuts. Applying a CNOT gate to a superposition correctly generates a Bell state, where the fates of the qubits are intertwined, as demonstrated by the final state probabilities.

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

Here's how to use CHAOS to simulate a simple quantum circuit.

### 1. Installing Dependencies

Make sure you have the required packages installed:

```bash
pip install -r requirements.txt
```

### 2. Creating an Entangled Bell State

The new architecture makes creating and observing complex quantum states straightforward. Here's how to create a Bell State, a classic example of maximum entanglement:

```python
from quantum_circuit import create_bell_state

# create_bell_state is a helper function that builds the circuit
# It applies a Hadamard gate to qubit 0, and a CNOT with control=0, target=1
bell_circuit = create_bell_state()

# Run the simulation to apply the gates
bell_circuit.run()

# Print the final state vector and probabilities
# The output will show ~50% probability for |00> and ~50% for |11>
print(bell_circuit)

# Measure the system
# This will collapse the state to either [0, 0] or [1, 1]
results = bell_circuit.measure()
print(f"Measurement results: {results}")
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

### Quantum Circuits

```python
from quantum_circuit import QuantumCircuit, create_bell_state, create_deutsch_algorithm

# Create a circuit with 3 qubits
circuit = QuantumCircuit(3)

# Apply gates to specific qubits
circuit.apply_gate("X", 0)  # Apply X gate to qubit 0
circuit.apply_gate("H", 1)  # Apply H gate to qubit 1
circuit.apply_cnot(0, 2)    # Apply CNOT with qubit 0 as control, qubit 2 as target

# Run the circuit to execute all gates
circuit.run()

# Measure all qubits
results = circuit.measure()
print(f"Measurement results: {results}")

# Create a Bell state (entangled qubits)
bell_circuit = create_bell_state()
bell_circuit.run()
bell_results = bell_circuit.measure()
print(f"Bell state measurement: {bell_results}")  # Should be either [0,0] or [1,1]

# Run Deutsch's Algorithm to determine if a function is constant or balanced
deutsch_circuit = create_deutsch_algorithm("constant_0")  # f(x) = 0 for all x
deutsch_circuit.run()
result = deutsch_circuit.measure(0)  # Measure only the first qubit
print(f"Function is {'constant' if result == 0 else 'balanced'}")
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
- [x] Milestone 3.1: Create a QuantumCircuit class that can manage multiple Qubits (a quantum register).
- [x] Milestone 3.2: Implement methods to add a series of gates to the circuit.
- [x] Milestone 3.3: Create a run() function that will execute all gates sequentially in the circuit and return the final measurement results.
