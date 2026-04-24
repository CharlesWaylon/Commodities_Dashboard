"""
Hybrid quantum-classical regression model.

Adapted from QuantumComputing/hybrid_model.py with improvements:
  - Fits a kernel ridge regression using the quantum Gram matrix
  - Exposes a scikit-learn-compatible interface (fit / predict / score)
  - Stores training kernel so it doesn't need to be recomputed on predict
  - Adds R² score method for quick evaluation

Architecture overview:
  Classical input → quantum feature map → Hilbert space inner product (kernel)
  → kernel ridge regression (classical linear algebra) → prediction

The model is essentially kernel ridge regression (KRR), but the kernel
function is evaluated on a quantum simulator instead of with a classical
formula like RBF. KRR solves:

    weights = (K + λI)⁻¹ y
    ŷ = K_test @ weights

Where K is the N×N training Gram matrix, λ is regularisation (QUANTUM_RIDGE_REG),
and K_test is the (M×N) cross-kernel between test and training points.

Practical expectations:
  With ~3-5 qubits on synthetic or real commodity data, this model is
  a research prototype, not a production forecaster. The value is in
  exploring whether quantum kernels capture correlations that classical
  kernels miss — not in beating XGBoost on a benchmark.
"""

import numpy as np
from sklearn.metrics import r2_score

from models.quantum.kernel import kernel_matrix
from models.config import QUANTUM_RIDGE_REG


class QuantumHybridRegressor:
    """
    Kernel ridge regressor with a quantum fidelity kernel.

    Parameters
    ----------
    ridge_reg : float
        Tikhonov regularisation added to the diagonal of the Gram matrix
        for numerical stability. Default from config.QUANTUM_RIDGE_REG.
    """

    def __init__(self, ridge_reg: float = QUANTUM_RIDGE_REG):
        self.ridge_reg = ridge_reg
        self.X_train_  = None
        self.K_train_  = None
        self.weights_  = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumHybridRegressor":
        """
        Compute the quantum Gram matrix and solve for kernel ridge weights.

        This is the slow step — O(n²) quantum circuit simulations.
        For n=100 training samples with 4 qubits, expect ~30–120 seconds
        on a laptop simulator.
        """
        self.X_train_ = X
        self.K_train_ = kernel_matrix(X)         # square Gram matrix (n × n)
        n = len(X)
        reg_matrix = self.K_train_ + self.ridge_reg * np.eye(n)
        self.weights_ = np.linalg.solve(reg_matrix, y)  # more stable than pinv
        self._is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict returns for new samples.

        Computes the cross-kernel between X_test and X_train,
        then applies the weights learned during fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        K_test = kernel_matrix(X_test, self.X_train_)   # (m × n) cross-kernel
        return K_test @ self.weights_

    def score(self, X_test: np.ndarray, y_true: np.ndarray) -> float:
        """R² coefficient of determination on test data."""
        y_pred = self.predict(X_test)
        return float(r2_score(y_true, y_pred))
