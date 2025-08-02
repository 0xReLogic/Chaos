"""
CHAOS - Quantum Computing Simulator
Module: quantum_circuit.py
Description: Implementation of Quantum Circuits for managing multiple qubits and operations
"""

import random
import numpy as np
from typing import Union, List, Dict, Tuple, Optional, Any
from qubit import Qubit
from quantum_gates import QuantumGate, I_GATE, SINGLE_QUBIT_GATES, apply_gate

class QuantumCircuit:
    """
    Represents a quantum circuit with multiple qubits and gates.
    
    A quantum circuit manages a collection of qubits (quantum register) and
    allows applying sequences of quantum gates to them.
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize a quantum circuit with a specified number of qubits.
        
        Args:
            num_qubits: The number of qubits in the circuit
        """
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be positive")
        
        self.num_qubits = num_qubits
        # Initialize all qubits to |0⟩ state
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.state_vector[0] = 1
        # List to store operations (gates and their target qubits)
        self.operations = []
    

    
    def apply_gate(self, gate: Union[str, QuantumGate], qubit_index: int) -> None:
        """
        Add a single-qubit gate operation to the circuit.
        
        Args:
            gate: The gate to apply (either a string name or QuantumGate object)
            qubit_index: The index of the target qubit
            
        Raises:
            IndexError: If the qubit index is out of range
            ValueError: If the gate name is not recognized
        """
        if qubit_index < 0 or qubit_index >= self.num_qubits:
            raise IndexError(f"Qubit index {qubit_index} out of range (0 to {self.num_qubits-1})")
        
        # Store the operation for later execution
        self.operations.append(("single", gate, qubit_index))
    
    def apply_controlled_gate(self, gate: Union[str, QuantumGate], control_index: int, target_index: int) -> None:
        """
        Add a controlled gate operation to the circuit.
        
        A controlled gate applies the gate to the target qubit only if the control qubit is in state |1⟩.
        
        Args:
            gate: The gate to apply (either a string name or QuantumGate object)
            control_index: The index of the control qubit
            target_index: The index of the target qubit
            
        Raises:
            IndexError: If any qubit index is out of range
            ValueError: If the gate name is not recognized or if control and target are the same
        """
        if control_index < 0 or control_index >= self.num_qubits:
            raise IndexError(f"Control qubit index {control_index} out of range (0 to {self.num_qubits-1})")
        if target_index < 0 or target_index >= self.num_qubits:
            raise IndexError(f"Target qubit index {target_index} out of range (0 to {self.num_qubits-1})")
        if control_index == target_index:
            raise ValueError("Control and target qubits must be different")
        
        # Store the operation for later execution
        self.operations.append(("controlled", gate, control_index, target_index))
    
    def apply_cnot(self, control_index: int, target_index: int) -> None:
        """
        Add a CNOT (Controlled-X) gate to the circuit.
        
        This is a convenience method for the common CNOT operation.
        
        Args:
            control_index: The index of the control qubit
            target_index: The index of the target qubit
        """
        self.apply_controlled_gate("X", control_index, target_index)
    
    def apply_hadamard_to_all(self) -> None:
        """Apply the Hadamard gate to all qubits in the circuit."""
        for i in range(self.num_qubits):
            self.apply_gate("H", i)
    
    def reset(self) -> None:
        """Reset all qubits to |0⟩ state and clear all operations."""
        self.state_vector = np.zeros(2**self.num_qubits, dtype=complex)
        self.state_vector[0] = 1
        self.operations = []
    
    def _execute_single_gate(self, gate: Union[str, QuantumGate], qubit_index: int) -> None:
        """
        Execute a single-qubit gate operation.
        
        Args:
            gate: The gate to apply
            qubit_index: The index of the target qubit
        """
        # Ensure gate is a QuantumGate object
        if isinstance(gate, str):
            gate = SINGLE_QUBIT_GATES[gate]

        # Create the operator for the entire system using tensor products
        # For a gate G on qubit k in an n-qubit system, the operator is I ⊗ ... ⊗ G ⊗ ... ⊗ I
        
        # Start with a 1x1 identity matrix
        operator = np.array([[1]], dtype=complex)
        
        for i in range(self.num_qubits):
            if i == qubit_index:
                # Place the actual gate at the target qubit position
                current_gate_matrix = gate.matrix
            else:
                # Place an identity gate on all other qubits
                current_gate_matrix = I_GATE.matrix
            
            # Tensor product with the operator built so far
            operator = np.kron(operator, current_gate_matrix)

        # Apply the full operator to the state vector
        self.state_vector = operator @ self.state_vector
    
    def _execute_controlled_gate(self, gate: Union[str, QuantumGate], control_index: int, target_index: int) -> None:
        """
        Execute a controlled gate operation.
        
        Args:
            gate: The gate to apply
            control_index: The index of the control qubit
            target_index: The index of the target qubit
        """
        # Define projector matrices
        P0 = np.array([[1, 0], [0, 0]], dtype=complex)  # Projector for |0⟩
        P1 = np.array([[0, 0], [0, 1]], dtype=complex)  # Projector for |1⟩
        
        # Ensure gate is a QuantumGate object
        if isinstance(gate, str):
            gate = SINGLE_QUBIT_GATES[gate]

        # Build the two parts of the controlled operator
        # Part 1: Control is |0⟩, apply Identity to target
        # Part 2: Control is |1⟩, apply Gate to target
        term1_list = []
        term2_list = []

        for i in range(self.num_qubits):
            if i == control_index:
                term1_list.append(P0)
                term2_list.append(P1)
            elif i == target_index:
                term1_list.append(I_GATE.matrix)
                term2_list.append(gate.matrix)
            else:
                term1_list.append(I_GATE.matrix)
                term2_list.append(I_GATE.matrix)

        # Build the full operators using tensor products
        operator1 = term1_list[0]
        operator2 = term2_list[0]
        for i in range(1, self.num_qubits):
            operator1 = np.kron(operator1, term1_list[i])
            operator2 = np.kron(operator2, term2_list[i])
            
        # The final operator is the sum of the two parts
        final_operator = operator1 + operator2

        # Apply the full operator to the state vector
        self.state_vector = final_operator @ self.state_vector
    
    def run(self) -> None:
        """
        Execute all operations in the circuit in sequence.
        
        This method applies all the gates that have been added to the circuit
        in the order they were added.
        """
        for operation in self.operations:
            if operation[0] == "single":
                _, gate, qubit_index = operation
                self._execute_single_gate(gate, qubit_index)
            elif operation[0] == "controlled":
                _, gate, control_index, target_index = operation
                self._execute_controlled_gate(gate, control_index, target_index)
    
    def measure(self, qubit_index: Optional[int] = None) -> Union[int, List[int]]:
        """
        Measure one or all qubits in the circuit.
        
        Args:
            qubit_index: The index of the qubit to measure, or None to measure all qubits
            
        Returns:
            If qubit_index is specified, returns the measurement result (0 or 1) for that qubit.
            If qubit_index is None, returns a list of measurement results for all qubits.
            
        Raises:
            IndexError: If the qubit index is out of range
        """
        if qubit_index is not None:
            if not (0 <= qubit_index < self.num_qubits):
                raise IndexError(f"Qubit index {qubit_index} out of range for {self.num_qubits} qubits.")

            # 1. Calculate the probability of measuring 0 for the specified qubit
            prob_zero = 0
            # The bit to check corresponds to the qubit's position from the left in binary representation (e.g., q2, q1, q0)
            bit_position_mask = 1 << (self.num_qubits - 1 - qubit_index)

            for i, amp in enumerate(self.state_vector):
                # If the bit at the qubit's position is 0, add its probability to prob_zero
                if (i & bit_position_mask) == 0:
                    prob_zero += np.abs(amp)**2

            # 2. Choose the measurement outcome based on the calculated probability
            measured_value = 0 if random.random() < prob_zero else 1

            # 3. Collapse the state vector
            new_state_vector = np.zeros_like(self.state_vector)
            for i, amp in enumerate(self.state_vector):
                # Check if the bit at the qubit's position matches the measured value
                is_match = ((i & bit_position_mask) != 0) == measured_value
                if is_match:
                    new_state_vector[i] = amp
            
            # 4. Normalize the new state vector
            norm = np.linalg.norm(new_state_vector)
            if norm == 0:
                # This case should ideally not be reached in a valid quantum state
                raise ValueError("Cannot collapse to a zero-norm state.")
            self.state_vector = new_state_vector / norm

            return measured_value

        # Calculate probabilities for each state in the state vector
        probabilities = np.abs(self.state_vector)**2
        
        # Choose a state based on the probabilities
        possible_states = np.arange(len(self.state_vector))
        measured_state_index = np.random.choice(possible_states, p=probabilities)
        
        # Collapse the state vector to the measured state
        self.state_vector = np.zeros_like(self.state_vector)
        self.state_vector[measured_state_index] = 1
        
        # Convert the integer index to a list of classical bits
        # Example: index 5 (101) for 3 qubits -> [1, 0, 1]
        binary_representation = format(measured_state_index, f'0{self.num_qubits}b')
        return [int(bit) for bit in binary_representation]
    
    def run_and_measure(self, qubit_index: Optional[int] = None) -> Union[int, List[int]]:
        """
        Run the circuit and then measure one or all qubits.
        
        This is a convenience method that combines run() and measure().
        
        Args:
            qubit_index: The index of the qubit to measure, or None to measure all qubits
            
        Returns:
            If qubit_index is specified, returns the measurement result (0 or 1) for that qubit.
            If qubit_index is None, returns a list of measurement results for all qubits.
        """
        self.run()
        return self.measure(qubit_index)

    def __str__(self) -> str:
        """Provides a rich, intuitive string representation of the quantum circuit's state."""
        # 1. Calculate marginal probabilities for each qubit
        marginal_probs = []
        for i in range(self.num_qubits):
            prob_zero = 0
            mask = 1 << (self.num_qubits - 1 - i)
            for j, amp in enumerate(self.state_vector):
                if (j & mask) == 0:
                    prob_zero += np.abs(amp)**2
            marginal_probs.append(prob_zero)

        # 2. Check for entanglement
        # Heuristic: If the probability of any basis state is not equal to the product of its marginals, it's entangled.
        is_entangled = False
        system_probabilities = np.abs(self.state_vector)**2
        if self.num_qubits > 1:
            for i, prob in enumerate(system_probabilities):
                if prob > 1e-9:
                    product_of_marginals = 1.0
                    for q_idx in range(self.num_qubits):
                        # Get the bit value (0 or 1) for this qubit in this basis state
                        bit_val = (i >> (self.num_qubits - 1 - q_idx)) & 1
                        if bit_val == 0:
                            product_of_marginals *= marginal_probs[q_idx]
                        else:
                            product_of_marginals *= (1 - marginal_probs[q_idx])
                    
                    if not np.isclose(prob, product_of_marginals):
                        is_entangled = True
                        break

        # 3. Build the output string
        status = "Entangled" if is_entangled else "Separable"
        header = f"Quantum Circuit ({self.num_qubits} qubits, {status})"
        output = f"{header}\n{'=' * len(header)}\n"

        for i in range(self.num_qubits):
            prob0 = marginal_probs[i]
            prob1 = 1 - prob0
            output += f"Qubit {i}: |0⟩={prob0:.1%}, |1⟩={prob1:.1%}\n"
        
        output += "-" * len(header) + "\n"
        output += "System State Probabilities:\n"

        for i, prob in enumerate(system_probabilities):
            if prob > 1e-9:
                basis_state = format(i, f'0{self.num_qubits}b')
                output += f"  |{basis_state}⟩: {prob:.1%}\n"
        
        return output
    




def run_and_measure(self, qubit_index: Optional[int] = None) -> Union[int, List[int]]:
    """
    Run the circuit and then measure one or all qubits.
    
    This is a convenience method that combines run() and measure().
    
    Args:
        qubit_index: The index of the qubit to measure, or None to measure all qubits
        
    Returns:
        If qubit_index is specified, returns the measurement result (0 or 1) for that qubit.
        If qubit_index is None, returns a list of measurement results for all qubits.
    """
    self.run()
    return self.measure(qubit_index)

def __str__(self) -> str:
    """
    String representation of the quantum circuit.
    
    Provides a rich, intuitive string representation of the quantum circuit's state.
    """
    # 1. Calculate marginal probabilities for each qubit
    marginal_probs = []
    for i in range(self.num_qubits):
        prob_zero = 0
        mask = 1 << (self.num_qubits - 1 - i)
        for j, amp in enumerate(self.state_vector):
            if (j & mask) == 0:
                prob_zero += np.abs(amp)**2
        marginal_probs.append(prob_zero)

    # 2. Check for entanglement
    # Heuristic: If the probability of any basis state is not equal to the product of its marginals, it's entangled.
    is_entangled = False
    system_probabilities = np.abs(self.state_vector)**2
    if self.num_qubits > 1:
        for i, prob in enumerate(system_probabilities):
            if prob > 1e-9:
                product_of_marginals = 1.0
                for q_idx in range(self.num_qubits):
                    # Get the bit value (0 or 1) for this qubit in this basis state
                    bit_val = (i >> (self.num_qubits - 1 - q_idx)) & 1
# Predefined quantum algorithms

def create_bell_state() -> QuantumCircuit:
    """
    Create a Bell state (maximally entangled state) circuit.
    
    The Bell state is created by applying a Hadamard gate to the first qubit,
    followed by a CNOT gate with the first qubit as control and the second as target.
    
    Returns:
        A quantum circuit configured to create a Bell state
    """
    circuit = QuantumCircuit(2)
    circuit.apply_gate("H", 0)  # Apply Hadamard to first qubit
    circuit.apply_cnot(0, 1)    # Apply CNOT with first qubit as control, second as target
    return circuit

def create_ghz_state(num_qubits: int = 3) -> QuantumCircuit:
    """
    Create a GHZ (Greenberger-Horne-Zeilinger) state circuit for n qubits.

    The GHZ state is a maximally entangled state of three or more qubits.
    It's created by applying a Hadamard to the first qubit, then a series of
    CNOTs from the first qubit to all other qubits.

    Args:
        num_qubits: The number of qubits for the GHZ state (must be >= 2).

    Returns:
        A quantum circuit configured to create a GHZ state.
    """
    if num_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits.")

    circuit = QuantumCircuit(num_qubits)
    # Start with a Hadamard on the first qubit
    circuit.apply_gate('H', 0)

    # Cascade CNOTs from the first qubit to all others
    for i in range(1, num_qubits):
        circuit.apply_cnot(0, i)

    return circuit