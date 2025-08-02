import numpy as np
import math
from quantum_circuit import QuantumCircuit

def test_grover_search():
    """
    Tests Grover's search algorithm for a 3-qubit system.
    """
    num_qubits = 3
    # The state we want to find
    marked_state_str = '110'
    marked_state_int = int(marked_state_str, 2)

    print(f"\n--- Testing Grover's Search for |{marked_state_str}> ---")

    # 1. Initialize circuit and create uniform superposition
    qc = QuantumCircuit(num_qubits)
    qc.apply_hadamard_to_all()

    # 2. Determine the optimal number of iterations
    N = 2**num_qubits
    optimal_iterations = math.floor(math.pi / 4 * math.sqrt(N))
    print(f"Search space size N = {N}. Optimal iterations ≈ {optimal_iterations}.")

    # 3. Apply Grover iterations
    for i in range(optimal_iterations):
        print(f"Applying Grover iteration {i + 1}...")
        qc.apply_grover_iteration(marked_state_str)

    # 4. Run the circuit
    qc.run()

    # 5. Verify the result
    print("\nFinal state probabilities:")
    print(qc)

    # Find the probability of the marked state
    probabilities = np.abs(qc.state_vector)**2
    marked_state_prob = probabilities[marked_state_int]
    print(f"Probability of finding marked state |{marked_state_str}>: {marked_state_prob:.2%}")

    # Assert that the marked state has the highest probability
    most_likely_state = np.argmax(probabilities)
    assert most_likely_state == marked_state_int, f"Search failed! Expected {marked_state_str}, but found {most_likely_state:0{num_qubits}b}."

    print(f"\n✅ Test Passed: Grover's algorithm successfully found the marked state |{marked_state_str}>.")

if __name__ == "__main__":
    test_grover_search()
