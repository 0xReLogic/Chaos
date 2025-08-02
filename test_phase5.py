import numpy as np
from quantum_circuit import QuantumCircuit

def create_bell_state():
    """Creates a 2-qubit circuit in the Bell state |Φ+⟩."""
    circuit = QuantumCircuit(2)
    circuit.apply_gate('H', 0)
    circuit.apply_cnot(0, 1)
    circuit.run()
    return circuit


def test_partial_measurement():
    """Tests the partial measurement functionality on a Bell state."""
    print("--- Testing Partial Measurement (Phase 5) ---")

    # 1. Create a Bell state
    bell_circuit = create_bell_state()
    print("\nInitial Bell State:")
    print(bell_circuit)

    # 2. Measure only the first qubit (index 0)
    print("\nMeasuring qubit 0...")
    measured_value = bell_circuit.measure(qubit_index=0)
    print(f"Result of measuring qubit 0: |{measured_value}⟩")

    # 3. Check the collapsed state
    print("\nState after measurement:")
    print(bell_circuit)

    print("--- Verification ---")
    if measured_value == 0:
        # Expect state to be |00⟩ = [1, 0, 0, 0]
        expected_state = np.array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
        print("Expected state to collapse to |00⟩.")
    else: # measured_value == 1
        # Expect state to be |11⟩ = [0, 0, 0, 1]
        expected_state = np.array([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j])
        print("Expected state to collapse to |11⟩.")

    # np.allclose checks if two arrays are element-wise equal within a tolerance
    is_correct = np.allclose(bell_circuit.state_vector, expected_state)
    if is_correct:
        print("\nSUCCESS: The state collapsed as expected.")
    else:
        print("\nFAILURE: The state did not collapse correctly.")
        print(f"  - Actual state:   {bell_circuit.state_vector}")
        print(f"  - Expected state: {expected_state}")


if __name__ == "__main__":
    test_partial_measurement()
