"""
CHAOS - Quantum Computing Simulator
Module: qubit.py
Description: Implementation of Qubit, the fundamental unit of quantum computation
"""

import numpy as np
from typing import Union, List, Tuple
import random

class Qubit:
    """
    Represents a quantum bit (qubit), the fundamental unit of quantum information.
    
    A qubit can exist in state |0⟩, state |1⟩, or any superposition of these states.
    Internally, a qubit is represented as a 2D complex vector [alpha, beta] where:
    - |alpha|^2 is the probability of measuring |0⟩
    - |beta|^2 is the probability of measuring |1⟩
    - |alpha|^2 + |beta|^2 = 1 (normalization condition)
    """
    
    def __init__(self, state: Union[int, List[complex], np.ndarray] = 0):
        """
        Initialize a qubit in a specific state.
        
        Args:
            state: Can be:
                - 0: Initialize to |0⟩ state [1, 0]
                - 1: Initialize to |1⟩ state [0, 1]
                - List or numpy array: Custom state vector [alpha, beta]
        """
        if isinstance(state, int):
            if state == 0:
                # |0⟩ state
                self.state = np.array([1+0j, 0+0j], dtype=complex)
            elif state == 1:
                # |1⟩ state
                self.state = np.array([0+0j, 1+0j], dtype=complex)
            else:
                raise ValueError("Integer state must be 0 or 1")
        elif isinstance(state, (list, np.ndarray)):
            # Custom state vector
            self.state = np.array(state, dtype=complex)
            
            # Ensure it's a 2D vector
            if len(self.state) != 2:
                raise ValueError("Qubit state vector must have exactly 2 elements")
                
            # Normalize the state vector
            norm = np.linalg.norm(self.state)
            if norm == 0:
                raise ValueError("State vector cannot be zero")
            self.state = self.state / norm
        else:
            raise TypeError("State must be an integer (0 or 1) or a list/array of 2 complex numbers")
    
    def initialize_to_zero(self):
        """Initialize the qubit to the |0⟩ state."""
        self.state = np.array([1+0j, 0+0j], dtype=complex)
        
    def initialize_to_one(self):
        """Initialize the qubit to the |1⟩ state."""
        self.state = np.array([0+0j, 1+0j], dtype=complex)
    
    def measure(self) -> int:
        """
        Measure the qubit, collapsing its state to either |0⟩ or |1⟩.
        
        Returns:
            int: 0 or 1, the result of the measurement
        """
        # Calculate probabilities
        prob_zero = np.abs(self.state[0])**2
        
        # Generate a random number between 0 and 1
        random_value = random.random()
        
        # Collapse the state based on probability
        if random_value <= prob_zero:
            self.state = np.array([1+0j, 0+0j], dtype=complex)
            return 0
        else:
            self.state = np.array([0+0j, 1+0j], dtype=complex)
            return 1
    
    def get_probabilities(self) -> Tuple[float, float]:
        """
        Get the probabilities of measuring |0⟩ and |1⟩.
        
        Returns:
            Tuple[float, float]: (probability of |0⟩, probability of |1⟩)
        """
        prob_zero = np.abs(self.state[0])**2
        prob_one = np.abs(self.state[1])**2
        return (prob_zero, prob_one)
    
    def __str__(self) -> str:
        """
        String representation of the qubit.
        
        Returns:
            str: String representation showing the state vector and probabilities
        """
        prob_zero, prob_one = self.get_probabilities()
        return (f"Qubit State: [{self.state[0]:.4f}, {self.state[1]:.4f}]\n"
                f"Probability |0⟩: {prob_zero:.4f} ({prob_zero*100:.1f}%)\n"
                f"Probability |1⟩: {prob_one:.4f} ({prob_one*100:.1f}%)")