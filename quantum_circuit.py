"""
CHAOS - Quantum Computing Simulator
Module: quantum_circuit.py
Description: Implementation of Quantum Circuits for managing multiple qubits and operations
"""

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
            raise NotImplementedError("Partial measurement is not yet implemented in Phase 4. Use measure() to measure all qubits.")

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
        """
        String representation of the quantum circuit.
        
        Returns:
            A string showing the state of each qubit in the circuit
        """
        result = f"Quantum Circuit with {self.num_qubits} qubits:\n"
        result += f"State Vector: {self.state_vector}\n\n"
        result += "State Probabilities:\n"
        
        probabilities = np.abs(self.state_vector)**2
        for i, prob in enumerate(probabilities):
            if prob > 1e-9: # Only show states with non-negligible probability
                # Format state as a binary string, e.g., |01⟩ for index 1 in a 2-qubit system
                state_str = format(i, f'0{self.num_qubits}b')
                result += f"|{state_str}⟩: {prob:.4f}\n"
        return result


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