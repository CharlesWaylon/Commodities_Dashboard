"""
QNN Hybrid — parameterized quantum circuit as a differentiable PyTorch layer.

Architecture
------------
    Input (n_features)
        ↓
    Classical pre-layer: Linear(n_features → n_qubits) + Tanh × π
        (scales to [−π, π] for angle encoding)
        ↓
    Quantum layer: PQC with L=2 variational layers
        Each layer: RY(x_i) + RZ(θ_{l,i,0}) + RX(θ_{l,i,1}) → CNOT chain
        Measurement: ⟨Z_i⟩ for each qubit
        (PennyLane TorchLayer — differentiable via parameter shift)
        ↓
    Classical post-layer: Linear(n_qubits → 1)
        ↓
    Scalar next-day return forecast

The parameterized quantum circuit (PQC) acts as a non-linear hidden layer.
Feature encoding is done via RY rotations. Trainable RZ and RX gates
constitute the variational parameters θ. Entanglement via CNOT chain allows
the quantum layer to represent multi-qubit correlations.

Research question
-----------------
Does the quantum layer capture non-linear feature interactions not captured
by an equivalent classical layer?

Experimental design:
  QNNForecaster    — uses the quantum PQC as the hidden layer
  ClassicalControl — replaces the quantum layer with Linear(n_q, n_q) + Tanh

Both models are trained identically (same data split, optimizer, epochs).
The `comparison_study()` function runs both and returns:
  - IC on test set for each model
  - Training loss curves (to check if quantum converges differently)
  - Parameter count comparison
  - Expressibility estimate: variance of output distribution over random inputs
    (higher variance → model can express more diverse functions)

Training note
-------------
Gradient computation through the PQC uses the parameter shift rule, which
requires 2 circuit evaluations per parameter per sample. With n_qubits=4,
n_layers=2 variational layers, and batch_size=16, expect ~5-30 minutes per
epoch on a laptop simulator. Use small training sets (n_train=100-200) and
few epochs (20-50) for initial experiments.

Usage
-----
    from models.quantum.qnn_hybrid import QNNForecaster, comparison_study

    prices = load_price_matrix()
    X, y = build_quantum_features(prices)   # (n_samples, 4) scaled features

    qnn = QNNForecaster(n_qubits=4, n_layers=2)
    history = qnn.fit(X, y, max_epochs=30, verbose=True)

    ic      = qnn.ic_score(X, y)
    express = qnn.expressibility(n_samples=200)

    results = comparison_study(X, y, max_epochs=30)
    # returns {"quantum": {...}, "classical": {...}, "comparison": pd.DataFrame}
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import spearmanr

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import pennylane as qml
    _PENNYLANE_AVAILABLE = True
except ImportError:
    _PENNYLANE_AVAILABLE = False

from models.config import QUANTUM_RIDGE_REG, TEST_FRACTION, RANDOM_SEED, N_QUBITS
from models.features import build_quantum_features

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_VARIATIONAL_LAYERS = 2
BATCH_SIZE           = 16    # small batches — each forward pass runs n_q circuits
LR_QNN               = 5e-3
WEIGHT_DECAY         = 1e-4
EARLY_STOP_PATIENCE  = 10
GRAD_CLIP            = 1.0


def _require_deps():
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required. pip install torch")
    if not _PENNYLANE_AVAILABLE:
        raise ImportError("pennylane is required. pip install pennylane")


# ── Quantum circuit definition ─────────────────────────────────────────────────

def _make_quantum_layer(n_qubits: int, n_layers: int):
    """
    Build a PennyLane TorchLayer for use as a PyTorch nn.Module.

    Circuit per variational layer:
      1. RY(x_i)         — encode the (pre-processed) input feature i
      2. RZ(θ_{l,i,0})   — trainable Z-rotation (phase)
      3. RX(θ_{l,i,1})   — trainable X-rotation (amplitude)
      4. CNOT chain       — entangle adjacent qubits

    Measurement: ⟨Z_i⟩ for each qubit → n_qubits output values in [−1, +1].

    Why RZ then RX?
      RZ controls phase differences between |0⟩ and |1⟩. RX rotates between
      them. Together they span the full single-qubit Bloch sphere, giving the
      PQC maximum expressibility per qubit per layer.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def quantum_circuit(inputs, weights):
        # inputs:  (n_qubits,) — pre-scaled to [−π, π] by the pre-layer
        # weights: (n_layers, n_qubits, 2) — trainable parameters

        for l in range(n_layers):
            # Encode input via RY rotations
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # Trainable rotations
            for i in range(n_qubits):
                qml.RZ(weights[l, i, 0], wires=i)
                qml.RX(weights[l, i, 1], wires=i)
            # Entanglement: forward CNOT chain
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Close loop: last qubit back to first (ring topology)
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    return qml.qnn.TorchLayer(quantum_circuit, weight_shapes)


# ── Full model architectures ───────────────────────────────────────────────────

class _QNNModel(nn.Module if _TORCH_AVAILABLE else object):
    """
    Pre-layer → Quantum PQC → post-layer.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input feature vector.
    n_qubits : int
        Number of qubits (= width of the quantum layer).
    n_layers : int
        Number of variational layers in the PQC.
    """

    def __init__(self, n_features: int, n_qubits: int, n_layers: int):
        super().__init__()
        self.pre   = nn.Linear(n_features, n_qubits)
        self.qlayer = _make_quantum_layer(n_qubits, n_layers)
        self.post  = nn.Linear(n_qubits, 1)
        self.n_qubits = n_qubits

    def forward(self, x):
        # Scale pre-layer output to [−π, π] for angle encoding
        h = torch.tanh(self.pre(x)) * np.pi
        q = self.qlayer(h)                   # (batch, n_qubits) ∈ [−1, +1]
        return self.post(q).squeeze(-1)      # (batch,)


class _ClassicalModel(nn.Module if _TORCH_AVAILABLE else object):
    """
    Structurally equivalent classical control: same pre/post layers,
    but the quantum layer is replaced by Linear(n_q, n_q) + Tanh.
    """

    def __init__(self, n_features: int, n_qubits: int):
        super().__init__()
        self.pre    = nn.Linear(n_features, n_qubits)
        self.hidden = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.Tanh(),
        )
        self.post   = nn.Linear(n_qubits, 1)

    def forward(self, x):
        h = torch.tanh(self.pre(x)) * np.pi
        h = self.hidden(h)
        return self.post(h).squeeze(-1)


# ── Training utilities ─────────────────────────────────────────────────────────

def _train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    max_epochs: int,
    verbose: bool,
    label: str = "",
) -> list:
    """Shared training loop for both quantum and classical models."""
    import torch

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_v  = torch.tensor(X_val,   dtype=torch.float32)
    y_v  = torch.tensor(y_val,   dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=False
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR_QNN, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-5
    )
    loss_fn = nn.MSELoss()

    best_val  = float("inf")
    best_sd   = None
    patience  = 0
    history   = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        t_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            t_loss += loss.item() * len(Xb)
        t_loss /= max(len(X_tr), 1)

        model.eval()
        with torch.no_grad():
            v_loss = float(loss_fn(model(X_v), y_v))

        scheduler.step(v_loss)
        history.append({"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss})

        if v_loss < best_val:
            best_val = v_loss
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"  [{label}] Early stop at epoch {epoch}")
                break

        if verbose and epoch % 5 == 0:
            print(f"  [{label}] Epoch {epoch:3d}  train={t_loss:.6f}  val={v_loss:.6f}")

    if best_sd:
        model.load_state_dict(best_sd)

    return history


# ── Public interface ──────────────────────────────────────────────────────────

class QNNForecaster:
    """
    Training and inference wrapper for the QNN hybrid model.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the PQC. Default from config.N_QUBITS (4).
    n_layers : int
        Variational layers in the PQC.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = N_VARIATIONAL_LAYERS):
        _require_deps()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self._model: Optional[_QNNModel] = None
        self.train_history: list = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_epochs: int = 30,
        verbose: bool = False,
    ) -> list:
        """
        Train the QNN hybrid.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features). Typically from
            build_quantum_features(), already scaled to [−π, π].
        y : np.ndarray
            Target next-day returns.
        """
        import torch
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        n = len(X)
        split    = int(n * (1 - TEST_FRACTION))
        val_cut  = int(split * 0.85)

        X_train, y_train = X[:val_cut],    y[:val_cut]
        X_val,   y_val   = X[val_cut:split], y[val_cut:split]

        self._model = _QNNModel(
            n_features=X.shape[1],
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
        )

        history = _train_model(
            self._model, X_train, y_train, X_val, y_val,
            max_epochs=max_epochs, verbose=verbose, label="QNN"
        )
        self.train_history = history
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        import torch
        self._model.eval()
        with torch.no_grad():
            return self._model(torch.tensor(X, dtype=torch.float32)).numpy()

    def ic_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Spearman IC on the held-out test set."""
        split = int(len(X) * (1 - TEST_FRACTION))
        preds = self.predict(X[split:])
        ic, _ = spearmanr(preds, y[split:])
        return round(float(ic), 4)

    def parameter_count(self) -> int:
        """Total number of trainable parameters (classical + quantum)."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def expressibility(self, n_samples: int = 200) -> float:
        """
        Estimate expressibility as the variance of the model output over random
        inputs drawn from N(0,1). Higher variance → more expressive function.

        A purely classical linear model has variance ∝ ||W||². A quantum model
        with sufficient entanglement can exceed this — measuring whether it does
        is the core research question.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        import torch
        X_rand = torch.randn(n_samples, self._model.pre.in_features) * np.pi
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_rand).numpy()
        return round(float(np.var(outputs)), 6)

    def circuit_diagram(self, sample_input: Optional[np.ndarray] = None) -> str:
        """
        Return a PennyLane text diagram of the quantum circuit.

        Useful for confirming the circuit depth and gate count before
        running on real hardware.
        """
        if sample_input is None:
            sample_input = np.zeros(self.n_qubits)

        dev = qml.device("default.qubit", wires=self.n_qubits)
        n_l = self.n_layers

        @qml.qnode(dev)
        def draw_circuit(x):
            dummy_weights = np.zeros((n_l, self.n_qubits, 2))
            for l in range(n_l):
                for i in range(self.n_qubits):
                    qml.RY(x[i], wires=i)
                for i in range(self.n_qubits):
                    qml.RZ(dummy_weights[l, i, 0], wires=i)
                    qml.RX(dummy_weights[l, i, 1], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return qml.draw(draw_circuit)(sample_input[:self.n_qubits])


# ── Comparison study ──────────────────────────────────────────────────────────

def comparison_study(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int = N_QUBITS,
    n_layers: int = N_VARIATIONAL_LAYERS,
    max_epochs: int = 30,
    verbose: bool = False,
) -> dict:
    """
    Train both the QNN hybrid and the equivalent classical model on the same
    data and return a side-by-side comparison.

    This is the canonical experiment for answering the Tier-4 research question:
    does the quantum layer provide a measurable advantage over a classical
    layer of the same width?

    Parameters
    ----------
    X, y : np.ndarray
        Features and targets from build_quantum_features().

    Returns
    -------
    dict with keys:
        quantum   — QNNForecaster instance
        classical — _ClassicalModel instance
        summary   — pd.DataFrame with IC, param_count, expressibility
        qnn_history, classical_history — training loss curves
    """
    _require_deps()
    import torch
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    n      = len(X)
    split  = int(n * (1 - TEST_FRACTION))
    val_cut = int(split * 0.85)

    X_train, y_train = X[:val_cut],      y[:val_cut]
    X_val,   y_val   = X[val_cut:split], y[val_cut:split]
    X_test,  y_test  = X[split:],        y[split:]

    # ── Quantum model ──
    qnn = QNNForecaster(n_qubits=n_qubits, n_layers=n_layers)
    qnn_hist = qnn.fit(X, y, max_epochs=max_epochs, verbose=verbose)
    qnn_ic   = qnn.ic_score(X, y)
    qnn_expr = qnn.expressibility()
    qnn_params = qnn.parameter_count()

    # ── Classical control ──
    cls_model = _ClassicalModel(n_features=X.shape[1], n_qubits=n_qubits)
    cls_hist  = _train_model(
        cls_model, X_train, y_train, X_val, y_val,
        max_epochs=max_epochs, verbose=verbose, label="Classical"
    )

    cls_model.eval()
    import torch
    with torch.no_grad():
        cls_preds = cls_model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    cls_ic, _ = spearmanr(cls_preds, y_test)
    cls_ic    = round(float(cls_ic), 4)

    X_rand = torch.randn(200, X.shape[1]) * np.pi
    with torch.no_grad():
        cls_expr = round(float(cls_model(X_rand).numpy().var()), 6)

    cls_params = sum(p.numel() for p in cls_model.parameters() if p.requires_grad)

    summary = pd.DataFrame([
        {
            "model":           "QNN Hybrid",
            "ic_test":         qnn_ic,
            "param_count":     qnn_params,
            "expressibility":  qnn_expr,
            "n_qubits":        n_qubits,
            "n_layers":        n_layers,
        },
        {
            "model":           "Classical Control",
            "ic_test":         cls_ic,
            "param_count":     cls_params,
            "expressibility":  cls_expr,
            "n_qubits":        n_qubits,
            "n_layers":        "—",
        },
    ])

    return {
        "quantum":            qnn,
        "classical":          cls_model,
        "summary":            summary,
        "qnn_history":        pd.DataFrame(qnn_hist),
        "classical_history":  pd.DataFrame(cls_hist),
    }
