import numpy as np
from quantum_circuit import QuantumCircuit
from fractions import Fraction
import math

def run_shor_period_finding(a: int, N: int):
    """
    Runs the quantum part of Shor's algorithm to find the period 'r' of a^x mod N.
    """
    # 1. Determine the number of qubits required.
    # Control register needs to be large enough to store the period with high probability.
    # Ancilla register needs to be large enough to store N.
    n = math.ceil(math.log2(N))
    num_control_qubits = 2 * n
    num_ancilla_qubits = n
    total_qubits = num_control_qubits + num_ancilla_qubits

    control_qubits = list(range(num_control_qubits))
    ancilla_qubits = list(range(num_control_qubits, total_qubits))

    print(f"--- Running Shor's Period Finding for a={a}, N={N} ---")
    print(f"Control Qubits: {num_control_qubits}, Ancilla Qubits: {num_ancilla_qubits}")

    # 2. Create the quantum circuit.
    qc = QuantumCircuit(total_qubits)

    # 3. Initialize the state.
    # Apply Hadamard to control qubits to create superposition.
    for i in control_qubits:
        qc.apply_gate('H', i)

    # Set ancilla register to |1> (which is |0...01>).
    qc.apply_gate('X', ancilla_qubits[-1])

    # 4. Apply the modular exponentiation.
    qc.apply_modular_exponentiation(a, N, control_qubits, ancilla_qubits)

    # 5. Apply the Inverse QFT on the control register.
    qc.apply_iqft(control_qubits, swaps=True)

    # 6. Run the simulation.
    print("Running circuit...")
    qc.run()
    print("Circuit run complete.")

    # 7. Measure the control qubits.
    measurement_results = qc.measure(control_qubits)
    measurement_int = int("".join(map(str, measurement_results)), 2)
    
    print(f"Measurement result (integer): {measurement_int}")

    # 8. Classical post-processing to find the period 'r'.
    if measurement_int == 0:
        print("Measurement is 0, cannot determine period. Please run again.")
        return

    phase = measurement_int / (2**num_control_qubits)
    print(f"Phase = {phase:.4f}")

    # Use continued fractions to find the period r.
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator
    print(f"Continued fraction approximation: {frac.numerator}/{frac.denominator}")
    print(f"Deduced period r = {r}")

    # 9. Validate the period.
    if pow(a, r, N) == 1:
        print(f"SUCCESS: {a}^{r} mod {N} = 1. Period found is correct.")
    else:
        print(f"FAILURE: {a}^{r} mod {N} != 1. Period found is incorrect. The algorithm may fail, try another run.")

if __name__ == "__main__":
    N = 15
    # We choose a=7, which is coprime to 15.
    # The period of 7^x mod 15 is 4. (7^1=7, 7^2=4, 7^3=13, 7^4=1)
    a = 7 
    run_shor_period_finding(a, N)
