"""
Quantum kernel (fidelity kernel) for commodity feature vectors.

Adapted from QuantumComputing/quantum_kernal_model.py.

The quantum kernel measures the "similarity" between two data points
by computing how much their quantum states overlap:

    K(x1, x2) = |⟨ψ(x1)|ψ(x2)⟩|²

Where |ψ(x)⟩ is the quantum state produced by embedding x into the circuit.

Why does this matter?
  In classical kernel methods (like SVMs), the kernel replaces the explicit
  computation of dot products in a high-dimensional feature space. A quantum
  kernel does the same, but the feature space is the quantum Hilbert space —
  which grows exponentially with the number of qubits (2^N dimensions).

  The bet is that the entanglement structure in the circuit captures non-linear
  dependencies between commodities that are hard to express with classical
  kernels like RBF or polynomial.

Performance note:
  Each kernel evaluation runs a full quantum circuit simulation, so the Gram
  matrix (all pairs in the training set) is O(n²) circuit calls. Keep training
  sets to a few hundred samples for reasonable run times on a laptop simulator.
"""

import numpy as np
from models.quantum.embedding import quantum_state_circuit


def quantum_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Fidelity (squared inner product) between two quantum-embedded feature vectors.

    Returns a value in [0, 1]:
      1.0 → identical points (perfect overlap)
      0.0 → orthogonal states (maximally different in Hilbert space)
    """
    state1 = quantum_state_circuit(x1)
    state2 = quantum_state_circuit(x2)
    return float(np.abs(np.vdot(state1, state2)) ** 2)


def kernel_matrix(X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
    """
    Compute the quantum kernel (Gram) matrix.

    Parameters
    ----------
    X1 : np.ndarray, shape (n, N_QUBITS)
        First set of feature vectors.
    X2 : np.ndarray, shape (m, N_QUBITS), optional
        Second set. If None, computes the square Gram matrix K[i,j] = K(X1_i, X1_j).
        If provided, computes the cross-kernel K[i,j] = K(X1_i, X2_j) — used at
        prediction time where X1=X_test and X2=X_train.

    Returns
    -------
    np.ndarray, shape (n, n) or (n, m)
    """
    if X2 is None:
        X2 = X1
        symmetric = True
    else:
        symmetric = False

    n, m = len(X1), len(X2)
    K = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            if symmetric and j < i:
                K[i, j] = K[j, i]   # exploit symmetry to halve circuit calls
            else:
                K[i, j] = quantum_kernel(X1[i], X2[j])

    return K
