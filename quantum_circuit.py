"""
CHAOS - Quantum Computing Simulator
Module: quantum_circuit.py
Description: Implementation of Quantum Circuits for managing multiple qubits and operations
"""

import numpy as np
from typing import Union, List, Dict, Tuple, Optional, Any
from qubit import Qubit
from quantum_gates import QuantumGate, SINGLE_QUBIT_GATES, apply_gate

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
        self.qubits = [Qubit(0) for _ in range(num_qubits)]
        # List to store operations (gates and their target qubits)
        self.operations = []
    
    def get_qubit(self, index: int) -> Qubit:
        """
        Get a specific qubit from the circuit.
        
        Args:
            index: The index of the qubit (0-based)
            
        Returns:
            The qubit at the specified index
            
        Raises:
            IndexError: If the index is out of range
        """
        if index < 0 or index >= self.num_qubits:
            raise IndexError(f"Qubit index {index} out of range (0 to {self.num_qubits-1})")
        return self.qubits[index]
    
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
        for qubit in self.qubits:
            qubit.initialize_to_zero()
        self.operations = []
    
    def _execute_single_gate(self, gate: Union[str, QuantumGate], qubit_index: int) -> None:
        """
        Execute a single-qubit gate operation.
        
        Args:
            gate: The gate to apply
            qubit_index: The index of the target qubit
        """
        qubit = self.qubits[qubit_index]
        apply_gate(qubit, gate)
    
    def _execute_controlled_gate(self, gate: Union[str, QuantumGate], control_index: int, target_index: int) -> None:
        """
        Execute a controlled gate operation.
        
        Args:
            gate: The gate to apply
            control_index: The index of the control qubit
            target_index: The index of the target qubit
        """
        control_qubit = self.qubits[control_index]
        target_qubit = self.qubits[target_index]
        
        # Get the probability of the control qubit being in state |1⟩
        prob_one = control_qubit.get_probabilities()[1]
        
        # If the control qubit is definitely in state |0⟩, do nothing
        if prob_one == 0:
            return
        
        # If the control qubit is definitely in state |1⟩, apply the gate to the target
        if prob_one == 1:
            apply_gate(target_qubit, gate)
            return
        
        # If the control qubit is in superposition, we need to handle entanglement
        # This is a simplified approach for educational purposes
        # In a real quantum computer, this would involve tensor products and more complex math
        
        # Measure the control qubit (this collapses the superposition)
        control_result = control_qubit.measure()
        
        # If the control qubit collapsed to |1⟩, apply the gate to the target
        if control_result == 1:
            apply_gate(target_qubit, gate)
    
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
            if qubit_index < 0 or qubit_index >= self.num_qubits:
                raise IndexError(f"Qubit index {qubit_index} out of range (0 to {self.num_qubits-1})")
            return self.qubits[qubit_index].measure()
        else:
            return [qubit.measure() for qubit in self.qubits]
    
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
        for i, qubit in enumerate(self.qubits):
            prob_zero, prob_one = qubit.get_probabilities()
            result += f"Qubit {i}: {qubit.state} (|0⟩: {prob_zero:.4f}, |1⟩: {prob_one:.4f})\n"
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

def create_deutsch_algorithm(f_type: str) -> QuantumCircuit:
    """
    Create a circuit for Deutsch's Algorithm.
    
    Deutsch's Algorithm determines whether a function f: {0,1} -> {0,1} is constant or balanced
    with a single function evaluation.
    
    Args:
        f_type: The type of function to implement:
            - "constant_0": f(x) = 0 for all x
            - "constant_1": f(x) = 1 for all x
            - "identity": f(x) = x (balanced)
            - "negation": f(x) = NOT x (balanced)
            
    Returns:
        A quantum circuit configured for Deutsch's Algorithm
        
    Raises:
        ValueError: If f_type is not recognized
    """
    if f_type not in ["constant_0", "constant_1", "identity", "negation"]:
        raise ValueError(f"Unknown function type: {f_type}")
    
    circuit = QuantumCircuit(2)
    
    # Initialize qubits: |0⟩ for first qubit, |1⟩ for second qubit
    circuit.qubits[1].initialize_to_one()
    
    # Apply Hadamard gates to both qubits
    circuit.apply_gate("H", 0)
    circuit.apply_gate("H", 1)
    
    # Apply the function (implemented as quantum gates)
    if f_type == "constant_1":
        # For f(x) = 1, apply X to the second qubit
        circuit.apply_gate("X", 1)
    elif f_type == "identity":
        # For f(x) = x, apply CNOT
        circuit.apply_cnot(0, 1)
    elif f_type == "negation":
        # For f(x) = NOT x, apply X to first qubit, then CNOT, then X to first qubit again
        circuit.apply_gate("X", 0)
        circuit.apply_cnot(0, 1)
        circuit.apply_gate("X", 0)
    
    # Apply final Hadamard to the first qubit
    circuit.apply_gate("H", 0)
    
    return circuit