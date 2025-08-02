from quantum_circuit import create_ghz_state

print("--- Testing GHZ State Creation ---")

# 1. Create a 3-qubit GHZ state circuit
ghz_circuit = create_ghz_state(3)

# 2. Run the circuit to generate the state
ghz_circuit.run()

# 3. Print the final state using the rich __str__ method
print("Final state of the 3-qubit GHZ circuit:")
print(ghz_circuit)

print("\n--- Verification ---")
print("A 3-qubit GHZ state should be an equal superposition of |000> and |111>.")
print("The output should show ~50% probability for these two states and 0% for all others.")
print("The circuit should also be marked as 'Entangled'.")
