"""
QAOA-based portfolio optimizer for a 10–15 asset subset of the commodity universe.

Maps the mean-variance portfolio selection problem to a QUBO (Quadratic
Unconstrained Binary Optimization), converts it to an Ising Hamiltonian,
and solves approximately with QAOA on a PennyLane simulator.

Binary decision variable: x_i ∈ {0, 1}  → include asset i in portfolio.

The QUBO objective (lower = better):

    QUBO(x) = x^T Σ x  −  λ · μ^T x  +  P · (Σᵢ xᵢ − k)²

Where:
  Σ  = sample covariance matrix of daily log-returns  (risk)
  μ  = mean daily log-return vector                   (expected return)
  λ  = risk–return tradeoff (higher → more return-seeking)
  P  = penalty coefficient enforcing Σ xᵢ = k
  k  = target number of assets to hold

QAOA approximates the ground state of the cost Hamiltonian derived from
this QUBO by alternating p cost + mixer unitary layers on a uniform
superposition, then optimising the 2p variational angles (γ, β) with a
classical solver (COBYLA — gradient-free, robust for noisy landscapes).

Practical scale: 10–15 qubits runs comfortably on a laptop simulator in
a few minutes at p=2.  Increase p or n_assets at the cost of runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pennylane as qml
import pennylane.numpy as pnp
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Optional

from models.config import MODELING_COMMODITIES
from models.data_loader import load_price_matrix_from_db


# ── Module-level defaults ──────────────────────────────────────────────────────

QAOA_N_ASSETS      = 12    # asset universe size (qubits = this value)
QAOA_K_SELECTED    = 5     # assets to hold in the final portfolio
QAOA_P_LAYERS      = 2     # QAOA circuit depth (cost+mixer pairs)
QAOA_LAMBDA        = 2.0   # risk-return tradeoff weight
QAOA_PENALTY       = 5.0   # Lagrange multiplier for the k-asset budget constraint
QAOA_LOOKBACK_DAYS = 252   # trading days of history for covariance estimation
QAOA_OPT_STEPS     = 120   # COBYLA iteration budget


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class QAOAResult:
    """Output of QAOAPortfolioOptimizer.optimize()."""
    selected_assets:  list[str]
    weights:          dict[str, float]   # equal-weight within selected assets
    expected_return:  float              # annualised (252 × daily mean)
    portfolio_vol:    float              # annualised (√252 × daily std)
    sharpe:           float              # expected_return / portfolio_vol (rf = 0)
    qubo_matrix:      np.ndarray         # (n_assets × n_assets) QUBO Q
    optimal_params:   np.ndarray         # final [γ₁,…,γₚ, β₁,…,βₚ]
    cost_history:     list[float] = field(default_factory=list)


# ── Step 1 — QUBO builder ─────────────────────────────────────────────────────

def build_qubo(
    mu:      np.ndarray,
    cov:     np.ndarray,
    k:       int,
    lam:     float = QAOA_LAMBDA,
    penalty: float = QAOA_PENALTY,
) -> np.ndarray:
    """
    Construct symmetric QUBO matrix Q such that min_x x^T Q x encodes
    the mean-variance selection problem.

    For binary x, x_i² = x_i, so the budget quadratic
    P·(Σ xᵢ − k)² = P·Σ xᵢ + 2P·Σᵢ<ⱼ xᵢxⱼ − 2Pk·Σ xᵢ + Pk²
    collapses (dropping the constant Pk²) to:
      diagonal additions:    P·(1 − 2k)  per asset
      off-diagonal additions: 2P          per pair

    Returns
    -------
    np.ndarray, shape (n, n), symmetric.
    """
    n = len(mu)
    Q = cov.copy().astype(float)

    # Off-diagonal budget penalty (symmetric)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2.0 * penalty
            Q[j, i]  = Q[i, j]

    # Diagonal: covariance (already present) + return reward + budget linear term
    for i in range(n):
        Q[i, i] += -lam * mu[i] + penalty * (1.0 - 2.0 * k)

    return Q


# ── Step 2 — QUBO → Ising conversion ─────────────────────────────────────────

def qubo_to_ising(
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Map QUBO {0,1} variables to Ising {−1,+1} spin variables.

    Substitution: x_i = (1 − z_i) / 2
    Then x_iˣx_j = (1 − z_i)(1 − z_j) / 4.

    The Ising Hamiltonian is H = Σᵢ hᵢ Zᵢ + Σᵢ<ⱼ Jᵢⱼ Zᵢ Zⱼ + const.

    Returns
    -------
    h      : (n,) bias  coefficients on Z_i
    J      : (n,n) coupling coefficients on Z_i Z_j  (upper-triangular used)
    offset : constant energy shift (irrelevant for optimisation)
    """
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0

    for i in range(n):
        h[i]    += Q[i, i] / 2.0
        offset  += Q[i, i] / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            Qij      = Q[i, j]
            J[i, j] += Qij / 4.0
            h[i]    -= Qij / 4.0
            h[j]    -= Qij / 4.0
            offset  += Qij / 4.0

    return h, J, offset


# ── Step 3 — PennyLane cost + mixer Hamiltonians ──────────────────────────────

def _cost_hamiltonian(h: np.ndarray, J: np.ndarray) -> qml.Hamiltonian:
    """
    H_C = Σᵢ hᵢ Zᵢ + Σᵢ<ⱼ Jᵢⱼ Zᵢ Zⱼ
    """
    n = len(h)
    coeffs: list[float] = []
    ops: list = []

    for i in range(n):
        if abs(h[i]) > 1e-12:
            coeffs.append(float(h[i]))
            ops.append(qml.PauliZ(i))

    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                coeffs.append(float(J[i, j]))
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    if not coeffs:
        return qml.Hamiltonian([0.0], [qml.Identity(0)])
    return qml.Hamiltonian(coeffs, ops)


def _mixer_hamiltonian(n: int) -> qml.Hamiltonian:
    """Standard transverse-field mixer H_M = −Σᵢ Xᵢ."""
    return qml.Hamiltonian([-1.0] * n, [qml.PauliX(i) for i in range(n)])


# ── Step 4 — QAOA QNode ───────────────────────────────────────────────────────

def _make_qaoa_qnode(
    cost_h:  qml.Hamiltonian,
    mixer_h: qml.Hamiltonian,
    n:       int,
    p:       int,
) -> qml.QNode:
    """
    Build an analytic (shot-free) QNode that returns ⟨H_C⟩.
    params layout: [γ₁, …, γₚ, β₁, …, βₚ]
    """
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="autograd")
    def circuit(params):
        gammas = params[:p]
        betas  = params[p:]
        for wire in range(n):
            qml.Hadamard(wires=wire)
        for layer in range(p):
            qml.qaoa.cost_layer(gammas[layer], cost_h)
            qml.qaoa.mixer_layer(betas[layer], mixer_h)
        return qml.expval(cost_h)

    return circuit


def _make_sampler_qnode(
    cost_h:   qml.Hamiltonian,
    mixer_h:  qml.Hamiltonian,
    n:        int,
    p:        int,
    n_shots:  int = 512,
) -> qml.QNode:
    """Shot-based QNode that returns bitstring samples from the optimised state."""
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, shots=n_shots)
    def sampler(params):
        gammas = params[:p]
        betas  = params[p:]
        for wire in range(n):
            qml.Hadamard(wires=wire)
        for layer in range(p):
            qml.qaoa.cost_layer(gammas[layer], cost_h)
            qml.qaoa.mixer_layer(betas[layer], mixer_h)
        return qml.sample()

    return sampler


# ── Step 5 — Bitstring decoder ────────────────────────────────────────────────

def _decode_bitstrings(
    samples:  np.ndarray,
    h:        np.ndarray,
    J:        np.ndarray,
    k:        int,
) -> np.ndarray:
    """
    Return the bitstring from `samples` (shape shots × n) with the lowest
    Ising energy that also satisfies exactly k ones.  Falls back to globally
    best energy if no k-asset solution appears in the sample set.
    """
    def ising_energy(bits: np.ndarray) -> float:
        z = 1.0 - 2.0 * bits.astype(float)
        return float(h @ z + z @ J @ z)

    best_constrained:   tuple[float, Optional[np.ndarray]] = (np.inf, None)
    best_unconstrained: tuple[float, Optional[np.ndarray]] = (np.inf, None)

    for row in samples:
        e = ising_energy(row)
        if e < best_unconstrained[0]:
            best_unconstrained = (e, row.copy())
        if int(row.sum()) == k and e < best_constrained[0]:
            best_constrained = (e, row.copy())

    bits = best_constrained[1]
    if bits is None:
        bits = best_unconstrained[1]

    return bits  # shape (n,) of 0/1


# ── Main class ────────────────────────────────────────────────────────────────

class QAOAPortfolioOptimizer:
    """
    Mean-variance portfolio selection via QUBO + PennyLane QAOA.

    Typical usage
    -------------
    prices = load_price_matrix_from_db()
    opt    = QAOAPortfolioOptimizer(n_assets=12, k=5, p=2)
    result = opt.fit(prices).optimize()
    print(result.selected_assets, result.sharpe)

    Parameters
    ----------
    n_assets : int
        Universe size — how many assets to pass as qubits (10–15 recommended).
        The top Sharpe-ranked assets in the DB are chosen automatically.
    k : int
        Assets to hold (budget constraint: exactly k ones in the bitstring).
    p : int
        QAOA depth.  p=1 is fast; p=2–3 gives meaningfully better solutions.
    lam : float
        Risk–return weight.  Higher → optimiser chases return over risk reduction.
    penalty : float
        Lagrange multiplier for Σ xᵢ = k.  Should be larger than the largest
        diagonal QUBO entry to guarantee constraint satisfaction.
    lookback : int
        History window (trading days) for covariance estimation.
    opt_steps : int
        COBYLA iteration budget.  120 is sufficient for p ≤ 3.
    """

    def __init__(
        self,
        n_assets:  int   = QAOA_N_ASSETS,
        k:         int   = QAOA_K_SELECTED,
        p:         int   = QAOA_P_LAYERS,
        lam:       float = QAOA_LAMBDA,
        penalty:   float = QAOA_PENALTY,
        lookback:  int   = QAOA_LOOKBACK_DAYS,
        opt_steps: int   = QAOA_OPT_STEPS,
    ):
        self.n_assets  = n_assets
        self.k         = k
        self.p         = p
        self.lam       = lam
        self.penalty   = penalty
        self.lookback  = lookback
        self.opt_steps = opt_steps

        self._asset_names: list[str]        = []
        self._mu:          np.ndarray | None = None
        self._cov:         np.ndarray | None = None
        self._is_fit:      bool              = False

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, prices: pd.DataFrame) -> "QAOAPortfolioOptimizer":
        """
        Compute log-returns statistics from a price matrix.

        The top `n_assets` assets are selected by annualised Sharpe ratio so
        the qubit count stays within the simulator's comfort zone.

        Parameters
        ----------
        prices : pd.DataFrame
            Date-indexed close prices, one column per asset.
            Typically the output of load_price_matrix_from_db().
        """
        log_ret = np.log(prices / prices.shift(1)).dropna()
        if len(log_ret) > self.lookback:
            log_ret = log_ret.iloc[-self.lookback :]

        mu_all    = log_ret.mean()
        std_all   = log_ret.std().replace(0, np.nan)
        sharpe_sr = (mu_all / std_all).fillna(0.0)

        top_n          = min(self.n_assets, len(prices.columns))
        selected_cols  = sharpe_sr.nlargest(top_n).index.tolist()

        sub               = log_ret[selected_cols]
        self._asset_names = selected_cols
        self._mu          = sub.mean().values
        self._cov         = sub.cov().values
        self._is_fit      = True
        return self

    # ── optimize ─────────────────────────────────────────────────────────────

    def optimize(self, n_shots: int = 512) -> QAOAResult:
        """
        Build the QUBO, run QAOA, decode the optimal bitstring.

        Parameters
        ----------
        n_shots : int
            Bitstring samples drawn from the final QAOA state for decoding.

        Returns
        -------
        QAOAResult
        """
        if not self._is_fit:
            raise RuntimeError("Call fit() before optimize().")

        k = min(self.k, len(self._asset_names))
        n = len(self._asset_names)

        # ── Build QUBO and Ising model ────────────────────────────────────────
        Q         = build_qubo(self._mu, self._cov, k, self.lam, self.penalty)
        h, J, _   = qubo_to_ising(Q)

        cost_h  = _cost_hamiltonian(h, J)
        mixer_h = _mixer_hamiltonian(n)

        circuit = _make_qaoa_qnode(cost_h, mixer_h, n, self.p)
        sampler = _make_sampler_qnode(cost_h, mixer_h, n, self.p, n_shots)

        # ── Initialise variational parameters ─────────────────────────────────
        # γ ∈ (0, π): cost rotation angles — small positive values to start
        # β ∈ (0, π/2): mixer angles — near π/4 maximises exploration
        rng    = np.random.default_rng(42)
        p0     = np.concatenate([
            rng.uniform(0.01, 0.5, self.p),
            rng.uniform(0.2,  0.5, self.p),
        ])

        cost_history: list[float] = []

        def objective(params_flat: np.ndarray) -> float:
            val = float(circuit(pnp.array(params_flat, requires_grad=True)))
            cost_history.append(val)
            return val

        result = minimize(
            objective,
            x0=p0,
            method="COBYLA",
            options={"maxiter": self.opt_steps, "rhobeg": 0.3},
        )
        optimal_params = result.x

        # ── Sample and decode ─────────────────────────────────────────────────
        raw_samples = sampler(pnp.array(optimal_params, requires_grad=False))
        bits        = _decode_bitstrings(np.array(raw_samples), h, J, k)

        selected = [self._asset_names[i] for i, b in enumerate(bits) if b == 1]
        if not selected:
            selected = self._asset_names[:k]

        # ── Compute portfolio metrics (equal-weight, annualised) ──────────────
        idx    = [self._asset_names.index(a) for a in selected]
        w      = np.ones(len(idx)) / len(idx)
        mu_p   = float(np.array([self._mu[i]  for i in idx]) @ w) * 252
        cov_sub = self._cov[np.ix_(idx, idx)]
        vol_p  = float(np.sqrt(w @ cov_sub @ w) * np.sqrt(252))
        sharpe = mu_p / vol_p if vol_p > 1e-9 else 0.0

        return QAOAResult(
            selected_assets = selected,
            weights         = {a: round(1.0 / len(selected), 4) for a in selected},
            expected_return = round(mu_p, 4),
            portfolio_vol   = round(vol_p, 4),
            sharpe          = round(sharpe, 4),
            qubo_matrix     = Q,
            optimal_params  = optimal_params,
            cost_history    = cost_history,
        )


# ── Convenience entry-point ───────────────────────────────────────────────────

def run_qaoa_portfolio(
    lookback_days: int            = QAOA_LOOKBACK_DAYS,
    n_assets:      int            = QAOA_N_ASSETS,
    k:             int            = QAOA_K_SELECTED,
    p:             int            = QAOA_P_LAYERS,
    lam:           float          = QAOA_LAMBDA,
    penalty:       float          = QAOA_PENALTY,
    opt_steps:     int            = QAOA_OPT_STEPS,
    prices:        Optional[pd.DataFrame] = None,
) -> QAOAResult:
    """
    One-call entry point: load prices from the DB (or accept a pre-built
    DataFrame), fit the optimizer, run QAOA, and return the result.

    Safe to call from a Streamlit page, a notebook, or a script.

    Example
    -------
    from models.quantum.qaoa_portfolio import run_qaoa_portfolio
    result = run_qaoa_portfolio(n_assets=12, k=5, p=2)
    print(result.selected_assets)
    print(f"Sharpe: {result.sharpe:.2f}")
    """
    if prices is None:
        prices = load_price_matrix_from_db()

    opt = QAOAPortfolioOptimizer(
        n_assets  = n_assets,
        k         = k,
        p         = p,
        lam       = lam,
        penalty   = penalty,
        lookback  = lookback_days,
        opt_steps = opt_steps,
    )
    opt.fit(prices)
    return opt.optimize()
