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

# Optional GPU acceleration with CuPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CHAOS: GPU acceleration enabled with CuPy")
    array_lib = cp
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # Fallback to NumPy
    array_lib = np
    print("CHAOS: GPU not available, using CPU with NumPy")

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
        # Initialize state vector with GPU support if available
        self.state_vector = array_lib.zeros(2**num_qubits, dtype=complex)
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

    def apply_c_rot(self, k: int, control_index: int, target_index: int) -> None:
        """
        Add a controlled-R_k rotation gate to the circuit's operations queue.

        Args:
            k (int): The rotation parameter for R_k.
            control_index (int): The index of the control qubit.
            target_index (int): The index of the target qubit.
        """
        if control_index < 0 or control_index >= self.num_qubits:
            raise IndexError(f"Control qubit index {control_index} out of range")
        if target_index < 0 or target_index >= self.num_qubits:
            raise IndexError(f"Target qubit index {target_index} out of range")
        if control_index == target_index:
            raise ValueError("Control and target qubits must be different")
        
        self.operations.append(("c_rot", k, control_index, target_index))

    def apply_qft(self, qubits: Optional[List[int]] = None, swaps: bool = True) -> None:
        """
        Add the Quantum Fourier Transform (QFT) operations to the circuit.

        Args:
            qubits: A list of qubit indices to apply QFT on. Defaults to all qubits.
            swaps: Whether to add the final SWAP gates to reverse the qubit order.
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))
        
        n = len(qubits)

        for i in range(n):
            self.apply_gate('H', qubits[i])
            for j in range(i + 1, n):
                k = j - i + 1
                self.apply_c_rot(k, qubits[j], qubits[i])

        if swaps:
            for i in range(n // 2):
                self.apply_cnot(qubits[i], qubits[n - 1 - i])
                self.apply_cnot(qubits[n - 1 - i], qubits[i])
                self.apply_cnot(qubits[i], qubits[n - 1 - i])

    def apply_iqft(self, qubits: Optional[List[int]] = None, swaps: bool = True) -> None:
        """
        Add the Inverse Quantum Fourier Transform (IQFT) operations to the circuit.
        This is the precise mathematical inverse of the QFT operation.

        Args:
            qubits: A list of qubit indices to apply IQFT on. Defaults to all qubits.
            swaps: Whether to add the initial SWAP gates to reverse the qubit order.
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))
        
        n = len(qubits)

        if swaps:
            for i in range(n // 2):
                self.apply_cnot(qubits[i], qubits[n - 1 - i])
                self.apply_cnot(qubits[n - 1 - i], qubits[i])
                self.apply_cnot(qubits[i], qubits[n - 1 - i])

        # The QFT algorithm applies H and C-ROTs in a nested loop.
        # To get the inverse, we must apply the inverse of those gates in the exact reverse order.
        # The last gate applied in QFT for n=3 was H on qubit 2.
        # The first gate in IQFT must be H on qubit 2.
        for i in reversed(range(n)):
            # For a given qubit i, the QFT first applied H, then a series of C-ROTs.
            # The inverse must first undo the C-ROTs, then undo the H.
            
            # Undo the C-ROT gates.
            # The QFT loop was `for j in range(i + 1, n)`. The reverse is needed.
            for j in reversed(range(i + 1, n)):
                k = j - i + 1
                self.apply_c_rot(-k, qubits[j], qubits[i])
            
            # Undo the Hadamard gate.
            self.apply_gate('H', qubits[i])
    
    def apply_oracle(self, marked_state_str: str) -> None:
        """
        Adds a phase-flip oracle for a specific computational basis state to the queue.

        Args:
            marked_state_str: A string representing the basis state to mark, e.g., '101'.
        """
        if len(marked_state_str) != self.num_qubits:
            raise ValueError("Length of marked_state_str must equal the number of qubits.")
        if not all(c in '01' for c in marked_state_str):
            raise ValueError("marked_state_str must be a binary string.")

        marked_state_int = int(marked_state_str, 2)
        self.operations.append(("oracle", marked_state_int))

    def apply_grover_iteration(self, marked_state_str: str) -> None:
        """
        Applies one full iteration of Grover's algorithm (Oracle + Amplifier).

        Args:
            marked_state_str: The computational basis state to search for, e.g., '101'.
        """
        # 1. Apply the oracle to mark the state
        self.apply_oracle(marked_state_str)

        # 2. Apply the diffusion operator to amplify the marked state's amplitude
        self.apply_grover_amplifier()

    def apply_modular_exponentiation(self, a: int, N: int, control_qubits: list[int], ancilla_qubits: list[int]):
        """
        Adds the full modular exponentiation circuit U_a: |x>|y> -> |x>|y * a^x mod N>.

        This is constructed by applying controlled modular multipliers for each qubit
        in the control register.

        Args:
            a: The base of the exponentiation.
            N: The modulus.
            control_qubits: The list of qubits representing the exponent x.
            ancilla_qubits: The list of qubits representing the register y.
        """
        # The control qubits are processed from most significant to least significant
        # to correspond to the powers of 2 in the exponent.
        for i, control_qubit in enumerate(reversed(control_qubits)):
            # The power of 'a' for this control qubit is 2^i
            a_power_2_i = pow(a, 2**i, N)
            self.apply_c_modular_multiplier(a_power_2_i, N, control_qubit, ancilla_qubits)

    def apply_c_modular_multiplier(self, a: int, N: int, control_qubit: int, ancilla_qubits: list[int]):
        """
        Adds a controlled modular multiplier operation to the queue.

        Args:
            a: The base for the multiplication.
            N: The modulus.
            control_qubit: The index of the control qubit.
            ancilla_qubits: A list of indices for the target ancilla qubits.
        """
        self.operations.append(("c_mod_mul", a, N, control_qubit, ancilla_qubits))

    def apply_grover_amplifier(self) -> None:
        """
        Applies the Grover diffusion operator (amplifier).

        This operator reflects the state vector about the mean amplitude,
        amplifying the amplitude of the marked state.
        It is constructed as H^(⊗n) @ (2|0><0| - I) @ H^(⊗n).
        """
        # Apply H to all qubits
        self.apply_hadamard_to_all()

        # Apply oracle for the |0...0> state
        zero_state_str = '0' * self.num_qubits
        self.apply_oracle(zero_state_str)

        # Apply H to all qubits again
        self.apply_hadamard_to_all()

    def apply_hadamard_to_all(self) -> None:
        """Apply the Hadamard gate to all qubits in the circuit."""
        for i in range(self.num_qubits):
            self.apply_gate("H", i)
    
    def reset(self) -> None:
        """Reset all qubits to |0⟩ state and clear all operations."""
        # Reset state vector with GPU support if available
        self.state_vector = array_lib.zeros(2**self.num_qubits, dtype=complex)
        self.state_vector[0] = 1
        self.operations = []
    
    def _to_cpu(self, array):
        """Convert GPU array to CPU array if needed."""
        if GPU_AVAILABLE and hasattr(array, 'get'):
            return array.get()  # CuPy to NumPy
        return array
    
    def _to_gpu(self, array):
        """Convert CPU array to GPU array if available."""
        if GPU_AVAILABLE and array_lib == cp:
            return cp.asarray(array)
        return array
    
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
        
        # Apply the gate operator to the state vector
        self.state_vector = operator @ self.state_vector

    def _apply_controlled_operator(self, gate_matrix: np.ndarray, control_qubit: int, target_qubits: list[int]):
        """
        Applies a generic controlled operator to the state vector by building the full matrix.

        Args:
            gate_matrix: The unitary matrix to be applied to the target qubits.
            control_qubit: The index of the control qubit.
            target_qubits: A list of indices for the target qubits.
        """
        num_total_qubits = self.num_qubits
        total_dim = 2**num_total_qubits
        final_op = np.zeros((total_dim, total_dim), dtype=complex)

        control_mask = 1 << (num_total_qubits - 1 - control_qubit)
        target_masks = [1 << (num_total_qubits - 1 - i) for i in target_qubits]
        num_target_qubits = len(target_qubits)

        for i in range(total_dim):
            if (i & control_mask) == 0:
                final_op[i, i] = 1
            else:
                target_state_val = 0
                for k, mask in enumerate(target_masks):
                    if (i & mask) != 0:
                        target_state_val |= (1 << (num_target_qubits - 1 - k))
                
                result_vector = gate_matrix[:, target_state_val]

                for new_target_val, amplitude in enumerate(result_vector):
                    if not np.isclose(amplitude, 0):
                        new_full_state_idx = i
                        for mask in target_masks:
                            new_full_state_idx &= ~mask
                        
                        for k, mask in enumerate(target_masks):
                            if (new_target_val & (1 << (num_target_qubits - 1 - k))) != 0:
                                new_full_state_idx |= mask

                        final_op[new_full_state_idx, i] = amplitude
        
        self.state_vector = final_op @ self.state_vector

    def _execute_controlled_gate(self, gate: Union[str, QuantumGate], control_index: int, target_index: int) -> None:
        """
        Execute a controlled gate operation.
        
        Args:
            gate: The gate to apply
            control_index: The index of the control qubit
            target_index: The index of the target qubit
        """
        # Ensure gate is a QuantumGate object
        if isinstance(gate, str):
            gate = SINGLE_QUBIT_GATES[gate]

        self._apply_controlled_operator(gate.matrix, control_index, [target_index])

    def _execute_c_rot(self, k: int, control_index: int, target_index: int) -> None:
        """
        Execute a controlled-R_k rotation gate.
        """
        if k == 0:
            return

        angle_sign = -1 if k < 0 else 1
        k_abs = abs(k)
        theta = angle_sign * (2 * np.pi / (2**k_abs))
        
        r_k_matrix = np.array([
            [1, 0],
            [0, np.exp(1j * theta)]
        ], dtype=complex)

        self._apply_controlled_operator(r_k_matrix, control_index, [target_index])

    def _execute_oracle(self, marked_state_int: int) -> None:
        """
        Executes the phase-flip oracle for a given marked state.

        Args:
            marked_state_int: The integer representation of the state to be marked.
        """
        oracle_matrix = np.identity(2**self.num_qubits, dtype=complex)
        oracle_matrix[marked_state_int, marked_state_int] = -1
        self.state_vector = oracle_matrix @ self.state_vector
    
    def _execute_c_modular_multiplier(self, a: int, N: int, control_qubit: int, ancilla_qubits: list[int]):
        """
        Executes a controlled modular multiplication: |y> -> |(a*y) mod N>.
        """
        num_ancilla = len(ancilla_qubits)
        ancilla_size = 2**num_ancilla

        # Create the permutation matrix for the modular multiplication on the ancilla
        perm_matrix = np.zeros((ancilla_size, ancilla_size), dtype=complex)
        for y in range(ancilla_size):
            if y >= N:
                perm_matrix[y, y] = 1
            else:
                new_y = (a * y) % N
                perm_matrix[new_y, y] = 1

        # Build and apply the full controlled gate operator
        self._apply_controlled_operator(perm_matrix, control_qubit, ancilla_qubits)
    
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
            elif operation[0] == "c_rot":
                _, k, control_index, target_index = operation
                self._execute_c_rot(k, control_index, target_index)
            elif operation[0] == "oracle":
                _, marked_state_int = operation
                self._execute_oracle(marked_state_int)
            elif operation[0] == "c_mod_mul":
                _, a, N, control_qubit, ancilla_qubits = operation
                self._execute_c_modular_multiplier(a, N, control_qubit, ancilla_qubits)
    
    def measure(self, qubit_index: Optional[Union[int, List[int]]] = None) -> Union[int, List[int]]:
        """
        Measure one, multiple, or all qubits in the circuit.

        This method simulates measurement without collapsing the state vector for simplicity,
        as the primary use case in this simulator is reading the final state.

        Args:
            qubit_index: An int for a single qubit, a list of ints for multiple
                         qubits, or None to measure all qubits.

        Returns:
            An int for a single qubit measurement, or a list of ints for multiple.
        """
        # Perform a full system measurement once - convert to CPU for NumPy operations
        state_cpu = self._to_cpu(self.state_vector)
        probabilities = np.abs(state_cpu)**2
        # Ensure probabilities sum to 1 to avoid numpy errors with floating point inaccuracies
        probabilities /= np.sum(probabilities)
        basis_states = np.arange(2**self.num_qubits)
        measured_state_int = random.choices(basis_states, weights=probabilities, k=1)[0]
        full_measurement_str = format(measured_state_int, f'0{self.num_qubits}b')
        full_results = [int(bit) for bit in full_measurement_str]

        if qubit_index is None:
            # Case 1: Measure all qubits
            return full_results

        if isinstance(qubit_index, list):
            # Case 2: Measure a specific list of qubits
            return [full_results[i] for i in qubit_index]

        if isinstance(qubit_index, int):
            # Case 3: Measure a single qubit
            if not (0 <= qubit_index < self.num_qubits):
                raise IndexError(f"Qubit index {qubit_index} out of range.")
            return full_results[qubit_index]

        raise TypeError(f"Invalid type for qubit_index: {type(qubit_index)}")

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
    
    def is_entangled(self) -> bool:
        """
        Check if the quantum system is in an entangled state.
        
        Returns:
            bool: True if the system is entangled, False otherwise.
        """
        if self.num_qubits <= 1:
            return False
        
        # Calculate marginal probabilities for each qubit
        marginal_probs = []
        for i in range(self.num_qubits):
            prob_zero = 0
            mask = 1 << (self.num_qubits - 1 - i)
            for j, amp in enumerate(self.state_vector):
                if (j & mask) == 0:
                    prob_zero += np.abs(amp)**2
            marginal_probs.append(prob_zero)

        # Check for entanglement
        system_probabilities = np.abs(self.state_vector)**2
        for i, prob in enumerate(system_probabilities):
            if prob > 1e-9:
                product_of_marginals = 1.0
                for q_idx in range(self.num_qubits):
                    bit_val = (i >> (self.num_qubits - 1 - q_idx)) & 1
                    if bit_val == 0:
                        product_of_marginals *= marginal_probs[q_idx]
                    else:
                        product_of_marginals *= (1 - marginal_probs[q_idx])
                
                if not np.isclose(prob, product_of_marginals):
                    return True
        return False


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

def create_deutsch_algorithm(f_type: str) -> QuantumCircuit:
    """
    Create a Deutsch algorithm circuit for a specified function type.
    
    The Deutsch algorithm determines whether a given function f is constant or balanced
    using only one function evaluation, demonstrating quantum speedup.
    
    Args:
        f_type: The type of function to test. Options are:
            - "constant_0": f(x) = 0 for all x
            - "constant_1": f(x) = 1 for all x  
            - "identity": f(x) = x
            - "negation": f(x) = NOT x
    
    Returns:
        A quantum circuit configured to run Deutsch's algorithm
        
    Raises:
        ValueError: If f_type is not one of the supported function types
    """
    if f_type not in ["constant_0", "constant_1", "identity", "negation"]:
        raise ValueError(f"Unsupported function type: {f_type}")
    
    # Deutsch algorithm requires 2 qubits: input qubit and ancilla qubit
    circuit = QuantumCircuit(2)
    
    # Step 1: Initialize ancilla qubit to |1⟩ 
    circuit.apply_gate("X", 1)
    
    # Step 2: Apply Hadamard to both qubits
    circuit.apply_gate("H", 0)  # Input qubit
    circuit.apply_gate("H", 1)  # Ancilla qubit
    
    # Step 3: Apply the oracle for function f
    if f_type == "constant_0":
        # f(x) = 0: Do nothing (identity operation)
        pass
    elif f_type == "constant_1":
        # f(x) = 1: Flip the ancilla qubit
        circuit.apply_gate("X", 1)
    elif f_type == "identity":
        # f(x) = x: CNOT with input as control, ancilla as target
        circuit.apply_cnot(0, 1)
    elif f_type == "negation":
        # f(x) = NOT x: X gate on input, then CNOT
        circuit.apply_gate("X", 0)
        circuit.apply_cnot(0, 1)
        circuit.apply_gate("X", 0)  # Undo the X gate on input
    
    # Step 4: Apply Hadamard to input qubit
    circuit.apply_gate("H", 0)
    
    return circuit