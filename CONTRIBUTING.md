# Contributing to CHAOS

First off, thank you for considering contributing to CHAOS! It's people like you that make open source projects thrive.

CHAOS has achieved solid performance in quantum simulation - **20+ qubit simulation with GPU acceleration and significant memory efficiency improvements**. We're focused on expanding quantum computing capabilities and making quantum algorithms more accessible to developers and researchers.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/0xReLogic/Chaos/issues). If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.

### Suggesting Enhancements

If you have an idea for an enhancement, please open an issue with the label `enhancement`. Provide a clear description of the enhancement and why it would be beneficial.

### Pull Requests

1.  Fork the repo and create your branch from `main`.
2.  If you've added code that should be tested, add tests.
3.  Ensure the test suite passes.
4.  Make sure your code lints.
5.  Issue that pull request!

## Development Environment Setup

### Prerequisites
- Python 3.8 or higher
- Git for version control
- **Recommended**: NVIDIA GPU with CUDA for large-scale quantum simulation testing

### Quick Setup
```bash
# Fork and clone the repository
git clone https://github.com/0xReLogic/Chaos.git
cd Chaos

# Create development environment
python -m venv chaos-dev
# Windows
chaos-dev\Scripts\activate
# macOS/Linux  
source chaos-dev/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install GPU acceleration for 15+ qubit testing
pip install cupy-cuda11x  # or cupy-cuda12x for CUDA 12.x

# Verify installation
python test_ultimate_scaling.py
```

### Testing Your Changes

CHAOS uses `test_ultimate_scaling.py` as the comprehensive test suite that includes all quantum algorithms with unlimited scaling capability.

```bash
# Main Test - Run this for all contributions
python test_ultimate_scaling.py  # Interactive test - choose algorithm and qubit range

# Quick Verification Tests
python test_quantum_circuit.py   # Core circuit functionality (~30 seconds)
python test_quantum_gates.py     # Individual gate operations (~10 seconds)  
python test_qubit.py             # Basic qubit operations (~5 seconds)

# Performance & GPU Tests (optional but recommended)
python test_memory_claims.py     # Memory efficiency verification
python test_gpu_support.py       # GPU acceleration tests (requires CUDA)

# Stress Test Examples (for performance testing)
# Small test (safe for any hardware)
(echo 5 && echo 2 && echo 5) | python test_ultimate_scaling.py  # Bell states, 2-5 qubits

# Medium test (requires decent hardware)  
(echo 1 && echo 1 && echo 15) | python test_ultimate_scaling.py # QFT, 1-15 qubits

# Extreme test (only for powerful systems)
(echo 1 && echo 1 && echo unlimited) | python test_ultimate_scaling.py # QFT unlimited mode
```

**Testing Checklist for Contributors:**
- [ ] `test_ultimate_scaling.py` runs without errors
- [ ] Your changes don't break existing algorithms
- [ ] Memory usage stays reasonable (under 100MB for 15 qubits)
- [ ] Code works with both NumPy (CPU) and CuPy (GPU) if applicable

## Styleguides

### Git Commit Messages

-   Use the present tense ("Add feature" not "Added feature").
-   Use the imperative mood ("Move cursor to..." not "Moves cursor to...").
-   Limit the first line to 72 characters or less.
-   Reference issues and pull requests liberally after the first line.

### Python Styleguide

All Python code must adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/). We use `black` for auto-formatting.

**CHAOS-Specific Guidelines:**
- **Performance-Critical Code**: Use NumPy/CuPy best practices for large array operations
- **Memory Efficiency**: Avoid matrix construction for quantum gates; prefer direct state manipulation
- **Documentation**: Include complexity analysis for algorithms (e.g., O(2^n) memory, O(n²) time)
- **GPU Compatibility**: Ensure new features work with both NumPy (CPU) and CuPy (GPU) backends

```python
# Good: Direct state manipulation (memory efficient)
from math import sqrt

def apply_hadamard_direct(state_vector, qubit_index, total_qubits):
    """Apply Hadamard gate directly to state vector (O(2^n) memory, O(2^n) time)"""
    step = 2 ** qubit_index
    for i in range(0, len(state_vector), 2 * step):
        for j in range(step):
            a = state_vector[i + j]
            b = state_vector[i + j + step]
            state_vector[i + j] = (a + b) / sqrt(2)
            state_vector[i + j + step] = (a - b) / sqrt(2)

# Avoid: Matrix construction (memory explosion - O(4^n) memory!)
# full_matrix = np.kron(I, np.kron(H, I))  # Don't do this for large circuits
# This would need 64GB+ for just 15 qubits!
```

We appreciate your contributions!

## Priority Areas for Contribution

CHAOS has achieved breakthrough performance with **25+ qubit simulation** capabilities. Current focus areas:

### High-Priority: Performance & Scalability
- **Memory Optimization**: Push beyond 25 qubits (current limit ~30 qubits)
  - State vector compression techniques
  - Sparse matrix representations for specific algorithms
  - Memory-mapped state vectors for ultra-large simulations

- **Algorithm Optimization**: Improve existing implementations
  - QFT: Already optimized, but room for GPU-specific improvements
  - Grover: Adaptive iteration counting optimization
  - Shor: Parallel modular exponentiation
  
- **Multi-GPU Support**: Distribute computation across multiple GPUs
- **Benchmarking**: Comprehensive performance comparisons vs Qiskit, Cirq

### Medium-Priority: Advanced Quantum Features  
- **Noise Models**: NISQ device simulation
  - Depolarizing, amplitude damping, phase damping channels
  - Gate error models, measurement errors
  
- **Error Correction**: Quantum error correction codes
  - Shor [[9,1,3]] code implementation
  - Surface code simulation
  
- **Variational Algorithms**: 
  - QAOA for optimization problems
  - VQE for quantum chemistry
  - Quantum Neural Networks

### Medium-Priority: Usability & Integration
- **Circuit Visualization**: Interactive quantum circuit diagrams
- **Higher-Level API**: Simplified circuit construction
  ```python
  # Current: qc.apply_gate('H', 0); qc.apply_cnot(0, 1)
  # Desired: qc.add(H(0), CNOT(0,1))
  ```
- **Ecosystem Integration**: Import/export for Qiskit, Cirq circuits
- **Educational Tools**: Interactive Jupyter notebooks, tutorials

### Lower-Priority: Documentation & Community
- **API Documentation**: Comprehensive documentation with examples
- **Tutorial Content**: Step-by-step guides for quantum algorithms
- **Community Examples**: Real-world use cases and research applications
- **Performance Guides**: Best practices for large-scale quantum simulation

## Contribution Process

### For Bug Fixes & Small Features
1. Fork the repository
2. Create a feature branch: `git checkout -b fix/issue-description`
3. Make your changes with appropriate tests
4. **Run the test suite**: Follow the [Testing Checklist](#testing-your-changes) above
5. Submit a pull request with clear description

### For Major Features & Algorithms
1. **Open an Issue First**: Discuss your proposal with the community
2. **Design Review**: Get feedback on your approach before coding
3. **Implementation**: Follow the coding guidelines and include comprehensive tests
4. **Documentation**: Update relevant docs and examples
5. **Performance Testing**: Verify your changes work well at scale

### Algorithm Contribution Guidelines

When adding new quantum algorithms to CHAOS:

```python
# Example: Adding a new algorithm to test_ultimate_scaling.py
elif selected_algorithm == 'YourNewAlgorithm':
    # Complexity: O(?) gates, O(?) memory
    # Description: What does this algorithm do?
    
    if current_qubits >= minimum_qubits_required:
        # Your algorithm implementation
        qc.your_algorithm_setup()
        qc.your_algorithm_execution()
    else:
        # Fallback for small systems
        qc.basic_setup()
```

**Algorithm Requirements:**
- [ ] Complexity analysis in comments
- [ ] Works with 1+ qubits (with graceful degradation)
- [ ] Memory efficient (direct state manipulation preferred)
- [ ] GPU compatible (NumPy/CuPy agnostic)

### Performance Testing Guidelines

For changes affecting simulation performance, run these benchmarks:

```bash
# Memory Efficiency Test (should stay under 100MB for 20 qubits)
python -c "
import tracemalloc
tracemalloc.start()
from quantum_circuit import QuantumCircuit
qc = QuantumCircuit(20)
qc.apply_hadamard_to_all()  # Worst case for memory
qc.run()
current, peak = tracemalloc.get_traced_memory()
print(f'Peak memory: {peak / 1024**2:.1f} MB (should be < 100MB)')
tracemalloc.stop()
"

# GPU Compatibility Test
python -c "
try:
    import cupy as cp
    from quantum_circuit import QuantumCircuit
    qc = QuantumCircuit(15)  # Test reasonable size
    qc.apply_hadamard_to_all()
    qc.run()
    print('GPU acceleration working')
except ImportError:
    print('GPU acceleration not available (optional)')
except Exception as e:
    print(f'GPU test failed: {e}')
"

# Performance Regression Test
python -c "
import time
from quantum_circuit import QuantumCircuit
start = time.time()
qc = QuantumCircuit(18)
qc.apply_qft()  # Standard benchmark
qc.run()
elapsed = time.time() - start
print(f'18-qubit QFT: {elapsed:.2f}s (baseline: ~30s on CPU, ~5s on GPU)')
"
```

**Performance Standards:**
- 15 qubits: < 10 seconds
- 20 qubits: < 100 seconds  
- 25 qubits: < 1000 seconds (GPU required)
- Memory: < 5MB per qubit for QFT

## Recognition

We appreciate all contributions to CHAOS! Contributors will be:
- Added to the README contributors section
- Mentioned in release notes for significant features
- Given credit in code comments and documentation

## Questions?

- **General Questions**: Open a GitHub Discussion
- **Bug Reports**: Create a GitHub Issue with the `bug` label
- **Feature Requests**: Create a GitHub Issue with the `enhancement` label
- **Performance Issues**: Include system specs and circuit details

Thank you for helping make quantum computing more accessible! 
