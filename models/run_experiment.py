"""
End-to-end experiment runner — connects data → features → baselines → quantum model.

Run from the Commodities_Dashboard directory:
    python -m models.run_experiment

What it does:
  1. Downloads 2 years of daily prices for the modeling commodity set
  2. Builds the feature matrix and scales to quantum angle range [-π, π]
  3. Splits into train / test sets (chronologically — no shuffling)
  4. Fits and evaluates two classical baselines
  5. Fits the quantum hybrid model on a small subset (to keep runtime reasonable)
  6. Prints an R² comparison table

Expected runtime:
  Data download:    ~5s
  Feature building: <1s
  Baselines:        <1s
  Quantum (n=50):   ~2–10 min on a laptop (O(n²) circuit simulations)
  Set QUANTUM_TRAIN_N lower if it's too slow.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from models.data_loader import load_price_matrix, train_test_split_by_date
from models.features import build_quantum_features
from models.classical.baseline import PersistenceModel, RollingMeanModel
from models.quantum.hybrid import QuantumHybridRegressor
from models.config import DEFAULT_TARGET, TEST_FRACTION

# How many training samples to use for the quantum model.
# Kernel matrix is O(n²) circuit calls — keep small for first runs.
QUANTUM_TRAIN_N = 50
QUANTUM_TEST_N  = 20


def main():
    print("=" * 60)
    print("Commodities Predictive Model — Experiment Runner")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print(f"\n[1/5] Downloading price history...")
    prices = load_price_matrix()
    print(f"      Loaded {len(prices)} trading days × {prices.shape[1]} commodities")

    # ── 2. Build features ─────────────────────────────────────────
    print(f"\n[2/5] Building feature matrix (target: {DEFAULT_TARGET})...")
    X, y = build_quantum_features(prices, target_name=DEFAULT_TARGET)
    print(f"      Feature matrix: {X.shape}  |  Target: {y.shape}")

    # ── 3. Train / test split ─────────────────────────────────────
    split = int(len(X) * (1 - TEST_FRACTION))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"\n[3/5] Split: {len(X_train)} train / {len(X_test)} test samples")

    # ── 4. Classical baselines ────────────────────────────────────
    print("\n[4/5] Fitting classical baselines...")

    persistence = PersistenceModel().fit(X_train, y_train)
    persistence_r2 = persistence.score(X_test, y_test)

    rolling_mean = RollingMeanModel().fit(X_train, y_train)
    rolling_r2 = rolling_mean.score(X_test, y_test)

    print(f"      Persistence R²:   {persistence_r2:.4f}")
    print(f"      Rolling Mean R²:  {rolling_r2:.4f}")

    # ── 5. Quantum model (on a small subset) ──────────────────────
    print(f"\n[5/5] Fitting quantum hybrid model...")
    print(f"      (using first {QUANTUM_TRAIN_N} train samples, "
          f"{QUANTUM_TEST_N} test samples to limit simulation time)")
    print("      This may take several minutes...\n")

    Xq_train = X_train[:QUANTUM_TRAIN_N]
    yq_train = y_train[:QUANTUM_TRAIN_N]
    Xq_test  = X_test[:QUANTUM_TEST_N]
    yq_test  = y_test[:QUANTUM_TEST_N]

    qmodel = QuantumHybridRegressor()
    qmodel.fit(Xq_train, yq_train)
    quantum_r2 = qmodel.score(Xq_test, yq_test)

    # ── Results table ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Results — forecasting next-day return of {DEFAULT_TARGET}")
    print("=" * 60)
    print(f"  {'Model':<30} {'R²':>8}")
    print(f"  {'-'*38}")
    print(f"  {'Persistence (lag-1 return)':<30} {persistence_r2:>8.4f}")
    print(f"  {'Rolling Mean (5-day)':<30} {rolling_r2:>8.4f}")
    print(f"  {'Quantum Hybrid (KRR, 4-qubit)':<30} {quantum_r2:>8.4f}")
    print("=" * 60)
    print("\nNote: R² near 0 is expected for next-day return forecasting.")
    print("The goal is to beat the baselines — not to achieve high R².\n")


if __name__ == "__main__":
    main()
