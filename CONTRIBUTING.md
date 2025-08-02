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
python test_quantum_circuit.py
```

### Testing Your Changes
```bash
# Run the full test suite
python -m pytest

# Test specific algorithms
python test_qft.py        # Quantum Fourier Transform
python test_grover.py     # Grover's Search Algorithm  
python test_shor.py       # Shor's Period-Finding
python test_ghz.py        # GHZ State Generation

# Performance testing (requires GPU)
python -c "from quantum_circuit import QuantumCircuit; qc = QuantumCircuit(18); print('18-qubit test ready')"
```

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
def apply_hadamard_direct(state_vector, qubit_index, total_qubits):
    step = 2 ** qubit_index
    for i in range(0, len(state_vector), 2 * step):
        for j in range(step):
            a = state_vector[i + j]
            b = state_vector[i + j + step]
            state_vector[i + j] = (a + b) / sqrt(2)
            state_vector[i + j + step] = (a - b) / sqrt(2)

# Avoid: Matrix construction (memory explosion)
# full_matrix = np.kron(I, np.kron(H, I))  # Don't do this for large circuits
```

We appreciate your contributions!

## Priority Areas for Contribution

CHAOS has achieved breakthrough performance with 20+ qubit simulation capabilities. We're now focusing on expanding the quantum computing ecosystem:

### High-Priority: Performance & Scalability
- **Memory Optimization**: Further improve memory efficiency for 25+ qubit simulation
- **Algorithm Optimization**: Enhance existing quantum algorithms for better performance
- **Distributed Computing**: Implement multi-GPU or cluster-based quantum simulation
- **Benchmarking**: Create comprehensive performance benchmarks across different hardware

### Medium-Priority: Advanced Quantum Features  
- **Noise Models**: Implement realistic noise channels for NISQ device simulation
- **Error Correction**: Add quantum error correction codes (Shor, Steane, Surface codes)
- **Variational Algorithms**: QAOA, VQE, and other variational quantum algorithms
- **Advanced Gates**: Custom gates, parametric gates, and composite gate operations

### Medium-Priority: Usability & Integration
- **Circuit Visualization**: Interactive circuit diagrams and state visualization
- **Abstract API**: Higher-level circuit building interface (`qc.add(H(0), CNOT(0,1))`)
- **Ecosystem Bridges**: Import/export compatibility with Qiskit, Cirq, PennyLane
- **Educational Tools**: Tutorials, interactive examples, and learning resources

### Lower-Priority: Documentation & Community
- **API Documentation**: Comprehensive documentation with examples
- **Tutorial Content**: Step-by-step guides for quantum algorithms
- **Community Examples**: Real-world use cases and research applications
- **Performance Guides**: Best practices for large-scale quantum simulation

## Contribution Process

### For Bug Fixes & Small Features
1. Fork the repository
2. Create a feature branch: `git checkout -b fix/issue-description`
3. Make your changes with tests
4. Ensure all tests pass: `python -m pytest`
5. Submit a pull request with clear description

### For Major Features & Algorithms
1. **Open an Issue First**: Discuss your proposal with the community
2. **Design Review**: Get feedback on your approach before coding
3. **Implementation**: Follow the coding guidelines and include comprehensive tests
4. **Documentation**: Update relevant docs and examples
5. **Performance Testing**: Verify your changes don't break large-scale simulation

### Performance Testing Guidelines
For changes affecting simulation performance:
```bash
# Test memory efficiency (should stay under 100MB for 20 qubits)
python -c "
import numpy as np
from quantum_circuit import QuantumCircuit
qc = QuantumCircuit(20)
qc.apply_hadamard_to_all()
qc.run()
print(f'Memory usage OK: {qc.state_vector.nbytes / 1024**2:.1f} MB')
"

# Test GPU compatibility (if available)
python -c "
try:
    import cupy as cp
    print('GPU acceleration available')
except ImportError:
    print('GPU acceleration not available (optional)')
"
```

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
