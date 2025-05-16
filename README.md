# Q-Classifier

Q-Classifier is a PyTorch-based framework implementing the quantum classification architecture proposed in "Data re-uploading for a universal quantum classifier", where classical input data is re-encoded at each layer into single- or multi-qubit systems processed by parameterized unitaries and optionally subject to noise, enabling modular experimentation with quantum-inspired models and their robustness.

## Features

- **Quantum Classifiers**: Implements single-qubit, chunked, and multi-qubit quantum classifiers.
- **Noise Channels**: Supports depolarizing and thermal relaxation noise via Kraus operators.
- **Dataset Generation**: Includes toy datasets (circles, diagonal band, logo), and MNIST variants for classification experiments.
- **Visualization Tools**: Bloch sphere plotting, classification heatmaps, dataset previews.
- **Training Tools**: Early stopping, multiple initialization selection, and batch-based optimization.
- **Jupyter Notebook Example**: Demonstrates the full training and evaluation pipeline.

## Getting Started

### Installation

Install dependencies via:

```bash
pip install torch numpy matplotlib qutip
```

### Structure

```
qclassifier/
├── datasets.py             # Synthetic datasets (circles, band, MNIST, image-based)
├── layers.py               # Parametrized quantum layers (unitary transformations)
├── models.py               # Concrete quantum classifiers (1 qubit, chunked, multi-qubit)
├── models_base.py          # Abstract base classes for all classifiers
├── noise.py                # Noise channels using Kraus operators
├── training_tools.py       # Training loops, early stopping, initialization search
├── utils.py                # Bloch sphere conversion and orthogonal state generation
├── visualization.py        # Dataset and Bloch sphere visualizations
examples/
└── example_circles.ipynb   # Jupyter example with full training and evaluation
```

## Example Usage

Run the notebook in `examples/example_circles.ipynb` to:

- Generate the "3 circles" dataset
- Train various classifier architectures (single-qubit, multi-qubit)
- Visualize classification boundaries and Bloch encodings
- Compare model performance as a function of network depth

## Notes

- The implementation is simulation-based and does not run on actual quantum hardware.
- Density matrices are evolved manually using batched matrix multiplications and noise operators.
- Maximally orthogonal states are used to represent output classes.
