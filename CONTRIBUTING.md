# Contributing to CHAOS

First off, thank you for considering contributing to CHAOS! It's people like you that make open source such a great community.

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

## Styleguides

### Git Commit Messages

-   Use the present tense ("Add feature" not "Added feature").
-   Use the imperative mood ("Move cursor to..." not "Moves cursor to...").
-   Limit the first line to 72 characters or less.
-   Reference issues and pull requests liberally after the first line.

### Python Styleguide

All Python code must adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/). We use `black` for auto-formatting.

We appreciate your contributions!

## Areas for Contribution

We have a long-term vision for CHAOS and welcome contributions in the following areas:

### Performance & Advanced Simulation
- **Profiling and Optimization**: Identify and accelerate bottlenecks in the simulation code, especially matrix expansion and application.
- **GPU Acceleration**: Help integrate libraries like `cupy` to offload heavy computations to the GPU.
- **Noise Models**: Implement basic noise channels (e.g., depolarizing, bit-flip, phase-flip) to allow for more realistic simulations of noisy intermediate-scale quantum (NISQ) hardware.
- **Parametric Circuits**: Add support for circuits with variable parameters.

### Usability & Integration
- **Abstract API**: Design and implement a more fluent, higher-level API for building circuits (e.g., `qc.add(H(0), CNOT(0,1))`).
- **Interactive Visualizations**: Improve the existing circuit diagrams or create new interactive visualization tools.
- **Ecosystem Integration**: Create converters to import/export circuits from/to industry-standard formats like Qiskit's `QuantumCircuit` or Cirq's `Circuit`.
