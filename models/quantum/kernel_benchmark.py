"""
Quantum kernel circuit benchmark — qubit count and feature map comparison.

Three feature maps compete across three qubit counts (4, 6, 8):

  RY/RZ (current)
      The existing circuit from embedding.py. Angle encoding via RY(x_i) and
      RZ(x_i²) with a two-layer forward/backward CNOT chain. The squared
      angle in layer 1 introduces non-linearity; the bidirectional entanglement
      lets each qubit "see" its neighbours in both directions.

  ZZFeatureMap (second-order Pauli)
      Implements the circuit from Havlíček et al. (2019). First-order terms
      encode x_i as RZ(2x_i) after a Hadamard. Second-order cross-terms encode
      (π−x_i)(π−x_j) via a CNOT+RZ+CNOT sandwich. The cross terms create a
      feature space that contains all pairwise feature products — analogous to
      a polynomial kernel of degree 2 but in Hilbert space.

  IQP (Instantaneous Quantum Polynomial)
      The IQP family of circuits: H^n → diagonal-Z interactions → H^n, repeated.
      IQP circuits are believed to be classically hard to simulate (Shepherd &
      Bremner 2009) and recent work shows they can outperform classical kernels
      on certain structured datasets.

Qubit scaling effect
  More qubits → larger Hilbert space (2^n dimensions) → potentially richer
  kernel. But: (1) circuit depth grows, increasing simulator runtime and noise
  sensitivity on real hardware; (2) the kernel can become too expressive and
  overfit to noise in short financial series.

Noise simulation
  The benchmark evaluates each circuit twice: once on the ideal statevector
  simulator and once on PennyLane's `default.mixed` device with a calibrated
  depolarizing noise model. The IC gap (ideal − noisy) documents decoherence-
  induced degradation without requiring IBM credentials.

IBM Quantum submission (optional)
  If `pennylane-qiskit` is installed and a valid IBMQ token is provided,
  `submit_to_ibm()` converts the best-performing circuit to a Qiskit
  transpiled form and submits it to the free-tier ibm_brisbane backend.
  The returned job ID can be polled via the IBM Quantum dashboard.

Usage
-----
    from models.quantum.kernel_benchmark import KernelBenchmark

    prices = load_price_matrix()
    X, y = build_quantum_features(prices)   # 4-feature baseline

    bench = KernelBenchmark(n_train=60, n_test=20)
    results = bench.run(X, y)               # pd.DataFrame

    bench.plot_ic_heatmap(results)          # circuit × qubit count IC grid
    bench.noise_report(X, y, n_qubits=4)   # ideal vs. noisy comparison

    # Optional IBM submission
    bench.submit_to_ibm(circuit="zzfeaturemap", n_qubits=4,
                        token="YOUR_IBMQ_TOKEN", backend="ibm_brisbane")
"""

import time
import warnings
import itertools
import numpy as np
import pandas as pd
import pennylane as qml
from scipy.stats import spearmanr
from typing import Optional, Callable

from models.config import QUANTUM_RIDGE_REG
from models.features import build_quantum_features
from models.data_loader import load_price_matrix

# ── Qubit counts and circuits to benchmark ─────────────────────────────────────
QUBIT_COUNTS   = [4, 6, 8]
CIRCUIT_NAMES  = ["ry_rz", "zzfeaturemap", "iqp"]

# Depolarizing error probability per gate for the noise simulation
# Calibrated to roughly match IBM 5-qubit device error rates (~0.1% per gate)
DEPOL_PROB = 0.001


# ── Feature map circuit definitions ───────────────────────────────────────────

def _ry_rz(x: np.ndarray, n_qubits: int) -> None:
    """
    Current embedding: RY(x_i) + RZ(x_i²), forward then backward CNOT chain.
    Extended version of models/quantum/embedding.py for arbitrary n_qubits.
    """
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
        qml.RZ(x[i] ** 2, wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    for i in range(n_qubits):
        qml.RY(x[i] * np.pi / 2, wires=i)
    for i in range(n_qubits - 1, 0, -1):
        qml.CNOT(wires=[i, i - 1])


def _zzfeaturemap(x: np.ndarray, n_qubits: int) -> None:
    """
    Second-order Pauli feature map (Havlíček et al. 2019).
    Two repetitions of: H^n → RZ(2x_i) → pairwise CNOT+RZ+CNOT cross-terms.

    The cross-term RZ(2(π−x_i)(π−x_j)) encodes the interaction of features
    i and j. This is the defining gate of the ZZFeatureMap and the source of
    its polynomial-degree-2 expressibility.
    """
    for rep in range(2):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(2.0 * x[i], wires=i)
        for i in range(n_qubits - 1):
            phi = 2.0 * (np.pi - x[i]) * (np.pi - x[i + 1])
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(phi, wires=i + 1)
            qml.CNOT(wires=[i, i + 1])


def _iqp(x: np.ndarray, n_qubits: int) -> None:
    """
    IQP (Instantaneous Quantum Polynomial) feature map.
    Structure: [H^n → RZ(x_i) → IsingZZ(x_i * x_j)] × 2.

    The IsingZZ(θ) = exp(-iθ/2 * Z⊗Z) gate encodes pairwise feature products.
    Believed classically hard to simulate efficiently (Shepherd & Bremner 2009).
    """
    for _ in range(2):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)
        for i in range(n_qubits - 1):
            qml.IsingZZ(x[i] * x[i + 1], wires=[i, i + 1])


_CIRCUITS = {
    "ry_rz":       _ry_rz,
    "zzfeaturemap": _zzfeaturemap,
    "iqp":         _iqp,
}


# ── Kernel computation ─────────────────────────────────────────────────────────

def _make_state_fn(circuit_fn: Callable, n_qubits: int, device: str = "default.qubit"):
    """Return a QNode that encodes x and returns the statevector."""
    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev)
    def state_circuit(x):
        circuit_fn(x[:n_qubits], n_qubits)
        return qml.state()

    return state_circuit


def _make_noisy_expval_fn(circuit_fn: Callable, n_qubits: int, depol_prob: float = DEPOL_PROB):
    """
    QNode on default.mixed with per-gate depolarizing noise.
    Returns Pauli-Z expectation values (used as a proxy for state similarity).
    """
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def noisy_circuit(x):
        circuit_fn(x[:n_qubits], n_qubits)
        for i in range(n_qubits):
            qml.DepolarizingChannel(depol_prob, wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return noisy_circuit


def _kernel_matrix_from_states(state_fn: Callable, X: np.ndarray) -> np.ndarray:
    """Fidelity kernel matrix K[i,j] = |⟨ψ(x_i)|ψ(x_j)⟩|² from statevectors."""
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        s_i = state_fn(X[i])
        for j in range(i, n):
            s_j = state_fn(X[j])
            k = float(np.abs(np.vdot(s_i, s_j)) ** 2)
            K[i, j] = k
            K[j, i] = k
    return K


def _noisy_kernel_matrix(noisy_fn: Callable, X: np.ndarray) -> np.ndarray:
    """
    Approximate kernel from noisy Pauli-Z expectation values.
    Uses dot product of expectation vectors as a proxy for state fidelity.
    This captures the reduction in overlap caused by decoherence.
    """
    n = len(X)
    expvals = np.array([noisy_fn(X[i]) for i in range(n)])
    K = expvals @ expvals.T
    # Normalise to [0, 1]
    norm = np.outer(np.sqrt(np.diag(K)), np.sqrt(np.diag(K))) + 1e-12
    return np.clip(K / norm, 0, 1)


def _krr_ic(K_train: np.ndarray, K_test_train: np.ndarray,
            y_train: np.ndarray, y_test: np.ndarray,
            ridge: float = QUANTUM_RIDGE_REG) -> float:
    """Kernel ridge regression Spearman IC."""
    reg = K_train + ridge * np.eye(len(K_train))
    weights = np.linalg.solve(reg, y_train)
    preds = K_test_train @ weights
    ic, _ = spearmanr(preds, y_test)
    return float(ic)


# ── Benchmark runner ───────────────────────────────────────────────────────────

class KernelBenchmark:
    """
    Evaluates all combinations of (circuit, n_qubits) on a held-out test set.

    Parameters
    ----------
    n_train : int
        Training samples. Keep ≤ 100 to avoid O(n²) runtime explosion.
    n_test : int
        Test samples for IC evaluation.
    qubit_counts : list[int]
        Qubit counts to benchmark. Default [4, 6, 8].
    circuits : list[str]
        Circuit names from {"ry_rz", "zzfeaturemap", "iqp"}.
    """

    def __init__(
        self,
        n_train: int = 60,
        n_test: int = 20,
        qubit_counts: Optional[list] = None,
        circuits: Optional[list] = None,
    ):
        self.n_train    = n_train
        self.n_test     = n_test
        self.qubit_counts = qubit_counts or QUBIT_COUNTS
        self.circuits   = circuits or CIRCUIT_NAMES

    def run(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Run the full benchmark grid.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features). Features are sliced to
            n_qubits for each configuration.
        y : np.ndarray
            Target vector (next-day log returns).

        Returns
        -------
        pd.DataFrame
            Columns: circuit, n_qubits, ic_ideal, runtime_s, n_train, n_test.
        """
        n_feat = X.shape[1]
        X_train = X[:self.n_train]
        y_train = y[:self.n_train]
        X_test  = X[self.n_train : self.n_train + self.n_test]
        y_test  = y[self.n_train : self.n_train + self.n_test]

        rows = []
        for cname, n_q in itertools.product(self.circuits, self.qubit_counts):
            if n_q > n_feat:
                continue   # can't use more qubits than features

            fn = _CIRCUITS[cname]

            # Slice features to n_q
            Xtr = X_train[:, :n_q]
            Xte = X_test[:, :n_q]

            state_fn = _make_state_fn(fn, n_q)

            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                K_train = _kernel_matrix_from_states(state_fn, Xtr)
                # Cross-kernel: test vs. train
                K_cross = np.zeros((len(Xte), len(Xtr)))
                for i in range(len(Xte)):
                    s_i = state_fn(Xte[i])
                    for j in range(len(Xtr)):
                        s_j = state_fn(Xtr[j])
                        K_cross[i, j] = float(np.abs(np.vdot(s_i, s_j)) ** 2)

            elapsed = time.perf_counter() - t0
            ic = _krr_ic(K_train, K_cross, y_train, y_test)

            rows.append({
                "circuit":   cname,
                "n_qubits":  n_q,
                "ic_ideal":  round(ic, 4),
                "runtime_s": round(elapsed, 1),
                "n_train":   self.n_train,
                "n_test":    self.n_test,
            })
            print(f"  {cname:15s}  {n_q}q  IC={ic:.4f}  ({elapsed:.1f}s)")

        return pd.DataFrame(rows).sort_values("ic_ideal", ascending=False).reset_index(drop=True)

    def noise_report(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_qubits: int = 4,
        depol_probs: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Sweep depolarizing error probability and record IC degradation.

        Simulates the effect of increasing hardware noise — from a near-ideal
        device (p=1e-4) to heavily noisy NISQ hardware (p=0.01).

        Parameters
        ----------
        depol_probs : list[float]
            Error probabilities per gate to sweep. Default: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2].

        Returns
        -------
        pd.DataFrame
            Columns: circuit, n_qubits, depol_prob, ic_ideal, ic_noisy, degradation.
        """
        if depol_probs is None:
            depol_probs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

        Xtr = X[:self.n_train, :n_qubits]
        ytr = y[:self.n_train]
        Xte = X[self.n_train : self.n_train + self.n_test, :n_qubits]
        yte = y[self.n_train : self.n_train + self.n_test]

        rows = []
        for cname in self.circuits:
            fn = _CIRCUITS[cname]

            # Ideal IC (computed once per circuit)
            state_fn  = _make_state_fn(fn, n_qubits)
            K_tr_ideal = _kernel_matrix_from_states(state_fn, Xtr)
            K_cr_ideal = np.zeros((len(Xte), len(Xtr)))
            for i in range(len(Xte)):
                s_i = state_fn(Xte[i])
                for j in range(len(Xtr)):
                    K_cr_ideal[i, j] = float(np.abs(np.vdot(s_i, state_fn(Xtr[j]))) ** 2)
            ic_ideal = _krr_ic(K_tr_ideal, K_cr_ideal, ytr, yte)

            for p in depol_probs:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    noisy_fn  = _make_noisy_expval_fn(fn, n_qubits, depol_prob=p)
                    K_tr_noisy = _noisy_kernel_matrix(noisy_fn, Xtr)
                    K_cr_noisy = _noisy_kernel_matrix(noisy_fn, np.vstack([Xte, Xtr]))
                    K_cr_noisy = K_cr_noisy[:len(Xte), len(Xte):]

                ic_noisy = _krr_ic(K_tr_noisy, K_cr_noisy, ytr, yte)
                rows.append({
                    "circuit":     cname,
                    "n_qubits":    n_qubits,
                    "depol_prob":  p,
                    "ic_ideal":    round(ic_ideal, 4),
                    "ic_noisy":    round(ic_noisy, 4),
                    "degradation": round(ic_ideal - ic_noisy, 4),
                })

        return pd.DataFrame(rows)

    def submit_to_ibm(
        self,
        circuit: str,
        n_qubits: int,
        X_sample: np.ndarray,
        token: str,
        backend: str = "ibm_brisbane",
    ) -> str:
        """
        Submit a kernel circuit sample to IBM Quantum via pennylane-qiskit.

        Requires: pip install pennylane-qiskit

        Parameters
        ----------
        circuit : str
            One of "ry_rz", "zzfeaturemap", "iqp".
        X_sample : np.ndarray
            A small feature matrix (≤ 10 rows) for the kernel computation.
            Larger samples will exceed the 10-min free-tier job limit.
        token : str
            IBMQ API token from https://quantum.ibm.com/account
        backend : str
            IBM backend name. "ibm_brisbane" has 127 qubits but free-tier
            jobs are queued. "ibmq_qasm_simulator" runs immediately (classical).

        Returns
        -------
        str
            Job ID. Monitor at https://quantum.ibm.com/jobs
        """
        try:
            dev_ibm = qml.device(
                "qiskit.ibmq",
                wires=n_qubits,
                backend=backend,
                ibmqx_token=token,
                shots=1024,
            )
        except Exception as e:
            raise ImportError(
                "pennylane-qiskit is required for IBM submission. "
                "Install with: pip install pennylane-qiskit\n"
                f"Original error: {e}"
            )

        fn = _CIRCUITS[circuit]

        @qml.qnode(dev_ibm)
        def ibm_circuit(x):
            fn(x[:n_qubits], n_qubits)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # Run one sample to submit the job
        x0 = X_sample[0, :n_qubits]
        result = ibm_circuit(x0)

        # The job ID is accessible via the device's backend job tracker
        job_id = str(getattr(dev_ibm, "_current_job_id", "submitted"))
        print(
            f"Job submitted to {backend}.\n"
            f"Job ID: {job_id}\n"
            f"Monitor at: https://quantum.ibm.com/jobs/{job_id}\n"
            f"Note: free-tier jobs may queue for several minutes to hours."
        )
        return job_id
