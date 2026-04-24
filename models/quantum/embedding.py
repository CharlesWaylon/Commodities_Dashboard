"""
Quantum feature map for commodity data (PennyLane).

Adapted from QuantumComputing/quantum_embedding.py but extended to:
  - support a configurable number of qubits (set via config.N_QUBITS)
  - use a layered entanglement structure (two rounds) for richer correlations
  - expose both the raw circuit and a convenience state-vector function

How it works (for non-physicists):
  Each qubit encodes one classical feature via two rotation gates:
    RY(x_i)    — rotates around Y-axis by the feature value in radians
    RZ(x_i²)  — rotates around Z-axis by x_i squared, adding non-linearity

  CNOT gates then "entangle" adjacent qubits, which means the quantum state
  can represent correlations between features that no single qubit carries
  alone — analogous to interaction terms in a regression model, but in a
  high-dimensional Hilbert space.

  Two entanglement layers instead of one increase the expressibility of the
  circuit, allowing it to represent more complex correlation structures.
"""

import pennylane as qml
import numpy as np

from models.config import N_QUBITS

dev = qml.device("default.qubit", wires=N_QUBITS)


def _feature_map(x: np.ndarray) -> None:
    """
    Encode a 1D feature vector x (length N_QUBITS) into quantum state.

    Circuit structure (2-layer entanglement):
      Layer 1: RY + RZ on each qubit → CNOT chain (i → i+1)
      Layer 2: RY on each qubit → CNOT chain (reversed: i+1 → i)

    The reversed CNOT chain in layer 2 closes the entanglement loop and
    captures correlations in the opposite direction.
    """
    if len(x) != N_QUBITS:
        raise ValueError(
            f"Feature vector length {len(x)} does not match N_QUBITS={N_QUBITS}. "
            "Update config.N_QUBITS or re-run build_quantum_features()."
        )

    # Layer 1: angle encoding
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)
        qml.RZ(x[i] ** 2, wires=i)

    # Layer 1 entanglement: forward CNOT chain
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])

    # Layer 2: refine encoding with second rotation
    for i in range(N_QUBITS):
        qml.RY(x[i] * np.pi / 2, wires=i)   # half-angle to avoid redundancy

    # Layer 2 entanglement: reverse CNOT chain (closes the correlation loop)
    for i in range(N_QUBITS - 1, 0, -1):
        qml.CNOT(wires=[i, i - 1])


@qml.qnode(dev)
def quantum_state_circuit(x: np.ndarray) -> np.ndarray:
    """
    Apply feature map and return the full state vector.

    The state vector has 2^N_QUBITS complex amplitudes. This is used by
    the quantum kernel to compute inner products between encoded data points.
    """
    _feature_map(x)
    return qml.state()
