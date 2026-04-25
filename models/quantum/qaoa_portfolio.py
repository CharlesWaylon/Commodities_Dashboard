"""
QAOA Portfolio Optimizer — commodity basket selection as a QUBO problem.

Problem formulation
-------------------
Given N commodities with return signals μ_i and pairwise correlations ρ_{ij},
select K commodities to maximise risk-adjusted expected return:

    maximise   Σ_i  μ_i x_i  −  λ Σ_{i<j} ρ_{ij} x_i x_j
    subject to Σ_i x_i = K,    x_i ∈ {0, 1}

  μ_i   — momentum/signal for commodity i (e.g. 10-day cumulative return)
  ρ_{ij} — 21-day rolling correlation (penalises correlated picks)
  λ     — diversification weight. Higher λ → more correlation-averse basket
  K     — target portfolio size (default 3 of 9 commodities)

The constraint Σ x_i = K is absorbed via a quadratic penalty P·(Σ x_i − K)².
This converts the constrained problem into an unconstrained QUBO:

    Q_{ii} = −μ_i + P(1 − 2K)
    Q_{ij} = 2λ ρ_{ij} + 2P          (i ≠ j)

QUBO to Ising mapping: x_i = (1 − z_i)/2, z_i ∈ {−1, +1}
The Ising Hamiltonian H_C = Σ h_i Z_i + Σ_{i<j} J_{ij} Z_i Z_j
is fed into QAOA's cost layer.

QAOA circuit (p layers)
-----------------------
    |+⟩^n → [exp(−iγ₁H_C) exp(−iβ₁H_B)] × p layers → measure

  H_B = Σ X_i  (transverse-field mixer, drives tunnelling between bit strings)
  Variational parameters: γ = [γ₁…γ_p], β = [β₁…β ₚ]
  Optimised by classical gradient descent via PennyLane's autograd.

Why QAOA for commodity selection?
  With 9–10 commodities, the state space is 2⁹ = 512 bit strings.
  Classical brute force is trivial. But QAOA is the research vehicle:
  (1) at ~50 qubits (all 28 tracked futures) the problem becomes
      classically hard; (2) the framework generalises to constraint-rich
      versions (sector limits, leverage limits) that are hard to express
      in continuous optimisers.

Usage
-----
    from models.quantum.qaoa_portfolio import QAOAPortfolioOptimizer

    prices = load_price_matrix()
    optimizer = QAOAPortfolioOptimizer(n_select=3, n_layers=2, lambda_=0.5)
    optimizer.fit(prices)

    basket  = optimizer.optimal_basket()          # list of commodity names
    probs   = optimizer.measurement_probs()        # dict of bitstring → probability
    energy  = optimizer.cost_landscape()           # pd.DataFrame of γ,β grid sweep
    report  = optimizer.portfolio_report(prices)   # return/corr stats of chosen basket
"""

import warnings
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from typing import Optional

from models.config import MODELING_COMMODITIES, ROLLING_VOL_WINDOW
from models.features import log_returns

# ── Default hyperparameters ───────────────────────────────────────────────────
N_SELECT        = 3       # K: commodities to select
N_LAYERS        = 2       # p: QAOA circuit depth
LAMBDA_CORR     = 0.5     # λ: correlation penalty weight
PENALTY         = 5.0     # P: equality constraint penalty
MOMENTUM_WINDOW = 10      # days for signal computation (μ_i)
CORR_WINDOW     = 21      # days for ρ_{ij} computation
N_OPTIM_STEPS   = 100     # gradient descent steps for QAOA params
LR_QAOA         = 0.05    # learning rate for QAOA optimiser
N_SHOTS         = 1024    # measurement shots for probability estimation


class QAOAPortfolioOptimizer:
    """
    QAOA-based combinatorial portfolio optimiser.

    Parameters
    ----------
    n_select : int
        K: number of commodities to include in the optimal basket.
    n_layers : int
        p: QAOA circuit depth. Higher p → better approximation ratio but
        slower training. p=2 is sufficient for small N; p=4+ for harder instances.
    lambda_ : float
        Correlation penalty weight. 0 → pure return maximisation (no
        diversification); 1 → equal weight on return and correlation.
    penalty : float
        Quadratic penalty for violating the cardinality constraint Σx_i = K.
        Should be > max(|μ_i|) to ensure feasibility.
    commodities : dict, optional
        {display_name: ticker}. Defaults to MODELING_COMMODITIES.
    """

    def __init__(
        self,
        n_select:    int   = N_SELECT,
        n_layers:    int   = N_LAYERS,
        lambda_:     float = LAMBDA_CORR,
        penalty:     float = PENALTY,
        commodities: Optional[dict] = None,
    ):
        self.n_select    = n_select
        self.n_layers    = n_layers
        self.lambda_     = lambda_
        self.penalty     = penalty
        self.commodities = commodities or MODELING_COMMODITIES

        self._commodity_names: Optional[list]  = None
        self._n_qubits: Optional[int]          = None
        self._Q: Optional[np.ndarray]          = None
        self._cost_h                           = None
        self._mixer_h                          = None
        self._optimal_params: Optional[tuple]  = None
        self._probs: Optional[dict]            = None

    # ── QUBO and Hamiltonian construction ─────────────────────────────────────

    def _compute_signals(self, prices: pd.DataFrame) -> np.ndarray:
        """10-day cumulative log-return normalised to [−1, +1]."""
        ret = log_returns(prices)
        signals = ret.iloc[-MOMENTUM_WINDOW:].sum()
        names = [n for n in self._commodity_names if n in signals.index]
        raw = signals[names].values.astype(float)
        scale = np.abs(raw).max() + 1e-9
        return raw / scale

    def _compute_corr(self, prices: pd.DataFrame) -> np.ndarray:
        """21-day rolling correlation matrix (last window)."""
        ret = log_returns(prices)
        available = [n for n in self._commodity_names if n in ret.columns]
        corr = ret[available].iloc[-CORR_WINDOW:].corr().values
        return corr

    def _build_qubo(self, mu: np.ndarray, corr: np.ndarray) -> np.ndarray:
        """
        Construct the QUBO matrix Q such that minimising x^T Q x (QUBO convention)
        is equivalent to maximising the portfolio objective.

        Note: QUBO minimises, so we negate the return signal.
        """
        n = len(mu)
        Q = np.zeros((n, n))
        P = self.penalty
        K = self.n_select

        for i in range(n):
            # Diagonal: −μ_i (negated for minimisation) + P*(1 − 2K)
            Q[i, i] = -mu[i] + P * (1 - 2 * K)

        for i in range(n):
            for j in range(i + 1, n):
                # Off-diagonal: correlation penalty + constraint term (symmetric)
                Q[i, j] = self.lambda_ * corr[i, j] + 2 * P
                Q[j, i] = Q[i, j]

        return Q

    def _qubo_to_ising(self, Q: np.ndarray) -> tuple:
        """
        Map QUBO (x ∈ {0,1}) to Ising (z ∈ {−1,+1}) via x_i = (1 − z_i)/2.

        Returns h (linear coefficients) and J (quadratic coefficients) for:
            H_C = Σ_i h_i Z_i + Σ_{i<j} J_{ij} Z_i Z_j
        """
        n = Q.shape[0]
        h = np.zeros(n)
        J = {}

        for i in range(n):
            h[i] = -0.5 * (Q[i, i] + sum(Q[i, j] for j in range(n) if j != i))

        for i in range(n):
            for j in range(i + 1, n):
                if Q[i, j] != 0:
                    J[(i, j)] = Q[i, j] / 4.0

        return h, J

    def _build_hamiltonians(self, h: np.ndarray, J: dict):
        """Construct PennyLane Hamiltonian objects for cost and mixer."""
        n = len(h)

        cost_coeffs = []
        cost_obs    = []

        # Linear terms: h_i Z_i
        for i, hi in enumerate(h):
            if abs(hi) > 1e-10:
                cost_coeffs.append(float(hi))
                cost_obs.append(qml.PauliZ(i))

        # Quadratic terms: J_{ij} Z_i Z_j
        for (i, j), Jij in J.items():
            if abs(Jij) > 1e-10:
                cost_coeffs.append(float(Jij))
                cost_obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

        if not cost_coeffs:
            cost_coeffs = [0.0]
            cost_obs    = [qml.Identity(0)]

        cost_h  = qml.Hamiltonian(cost_coeffs, cost_obs)

        # Mixer Hamiltonian: H_B = Σ X_i
        mixer_h = qml.Hamiltonian(
            [1.0] * n,
            [qml.PauliX(i) for i in range(n)],
        )

        return cost_h, mixer_h

    # ── QAOA circuit ──────────────────────────────────────────────────────────

    def _make_qaoa_circuit(self, n_qubits: int, cost_h, mixer_h):
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="autograd")
        def circuit(params):
            gammas = params[: self.n_layers]
            betas  = params[self.n_layers :]

            # Equal superposition
            for i in range(n_qubits):
                qml.Hadamard(wires=i)

            # QAOA layers
            for l in range(self.n_layers):
                qml.CommutingEvolution(cost_h, gammas[l])
                for i in range(n_qubits):
                    qml.RX(-2 * betas[l], wires=i)

            return qml.expval(cost_h)

        @qml.qnode(dev, wires=n_qubits)
        def sample_circuit(params):
            gammas = params[: self.n_layers]
            betas  = params[self.n_layers :]

            for i in range(n_qubits):
                qml.Hadamard(wires=i)

            for l in range(self.n_layers):
                qml.CommutingEvolution(cost_h, gammas[l])
                for i in range(n_qubits):
                    qml.RX(-2 * betas[l], wires=i)

            return qml.probs(wires=range(n_qubits))

        return circuit, sample_circuit

    # ── public interface ──────────────────────────────────────────────────────

    def fit(self, prices: pd.DataFrame) -> "QAOAPortfolioOptimizer":
        """
        Build the QUBO, construct the QAOA Hamiltonian, and optimise parameters.

        Parameters
        ----------
        prices : pd.DataFrame
            Closing prices from load_price_matrix(). Must include all
            commodities in self.commodities.
        """
        self._commodity_names = [
            n for n in self.commodities if n in prices.columns
        ]
        self._n_qubits = len(self._commodity_names)

        if self._n_qubits < 2:
            raise ValueError("Need at least 2 commodities.")
        if self.n_select >= self._n_qubits:
            raise ValueError("n_select must be < number of commodities.")

        mu   = self._compute_signals(prices)
        corr = self._compute_corr(prices)
        self._Q = self._build_qubo(mu, corr)

        h, J = self._qubo_to_ising(self._Q)
        self._cost_h, self._mixer_h = self._build_hamiltonians(h, J)

        circuit, sample_circuit = self._make_qaoa_circuit(
            self._n_qubits, self._cost_h, self._mixer_h
        )

        # Initialise parameters: gammas near 0, betas near π/4
        np.random.seed(42)
        init_params = pnp.array(
            np.concatenate([
                np.random.uniform(0.0, 0.5, self.n_layers),
                np.random.uniform(0.2, 0.6, self.n_layers),
            ]),
            requires_grad=True,
        )

        optimiser = qml.AdamOptimizer(stepsize=LR_QAOA)
        params = init_params

        best_energy = float("inf")
        best_params = params.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for step in range(N_OPTIM_STEPS):
                params, energy = optimiser.step_and_cost(circuit, params)
                if float(energy) < best_energy:
                    best_energy = float(energy)
                    best_params = params.copy()

        self._optimal_params = best_params

        # Sample probability distribution from optimal parameters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probs_array = sample_circuit(best_params)

        n = self._n_qubits
        self._probs = {
            format(i, f"0{n}b"): float(probs_array[i])
            for i in range(2 ** n)
        }

        return self

    def optimal_basket(self) -> list:
        """
        Return the K commodity names with highest selection probability,
        enforcing the cardinality constraint Σ x_i = K exactly.

        Strategy: rank all bitstrings with exactly K ones by probability,
        return the highest-probability feasible solution.
        """
        if self._probs is None:
            raise RuntimeError("Call fit() first.")

        K = self.n_select
        feasible = {
            bs: p for bs, p in self._probs.items()
            if bs.count("1") == K
        }
        if not feasible:
            raise RuntimeError("No feasible solution found — increase n_layers or reduce n_select.")

        best_bs = max(feasible, key=feasible.get)
        return [
            self._commodity_names[i]
            for i, bit in enumerate(best_bs)
            if bit == "1"
        ]

    def measurement_probs(self, top_n: int = 20) -> pd.DataFrame:
        """
        Top-N measurement probabilities sorted descending.

        Returns
        -------
        pd.DataFrame
            Columns: bitstring, probability, n_ones, feasible (bool), commodities.
        """
        if self._probs is None:
            raise RuntimeError("Call fit() first.")

        K = self.n_select
        rows = []
        for bs, p in sorted(self._probs.items(), key=lambda t: -t[1])[:top_n]:
            selected = [self._commodity_names[i] for i, b in enumerate(bs) if b == "1"]
            rows.append({
                "bitstring":   bs,
                "probability": round(p, 6),
                "n_ones":      bs.count("1"),
                "feasible":    bs.count("1") == K,
                "commodities": ", ".join(selected),
            })
        return pd.DataFrame(rows)

    def cost_landscape(
        self,
        gamma_range: Optional[np.ndarray] = None,
        beta_range:  Optional[np.ndarray] = None,
        n_points:    int = 20,
    ) -> pd.DataFrame:
        """
        Sweep the 2D cost landscape over (γ₁, β₁) for p=1 QAOA layer.
        Useful for visualising the loss surface and checking for barren plateaus.

        Returns
        -------
        pd.DataFrame
            Columns: gamma, beta, cost.
        """
        if self._cost_h is None:
            raise RuntimeError("Call fit() first.")

        dev_sweep = qml.device("default.qubit", wires=self._n_qubits)

        @qml.qnode(dev_sweep)
        def sweep_circuit(g, b):
            for i in range(self._n_qubits):
                qml.Hadamard(wires=i)
            qml.CommutingEvolution(self._cost_h, g)
            for i in range(self._n_qubits):
                qml.RX(-2 * b, wires=i)
            return qml.expval(self._cost_h)

        g_vals = gamma_range if gamma_range is not None else np.linspace(0, np.pi, n_points)
        b_vals = beta_range  if beta_range  is not None else np.linspace(0, np.pi / 2, n_points)

        rows = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for g in g_vals:
                for b in b_vals:
                    cost = float(sweep_circuit(float(g), float(b)))
                    rows.append({"gamma": round(float(g), 4), "beta": round(float(b), 4), "cost": round(cost, 6)})

        return pd.DataFrame(rows)

    def portfolio_report(self, prices: pd.DataFrame) -> dict:
        """
        Descriptive statistics for the selected basket vs. equal-weight benchmark.

        Returns
        -------
        dict with keys:
            basket           — list of selected commodities
            basket_return_ann — annualised return of equal-weight basket
            basket_vol_ann    — annualised vol of equal-weight basket
            basket_sharpe     — Sharpe ratio
            mean_pairwise_corr — mean correlation within basket
            vs_equal_weight    — comparison dict
        """
        basket = self.optimal_basket()
        all_names = self._commodity_names

        ret = log_returns(prices).dropna()
        basket_ret   = ret[basket].mean(axis=1)
        eq_weight    = ret[all_names].mean(axis=1)

        def stats(r):
            ann_ret = float(r.mean() * 252)
            ann_vol = float(r.std() * np.sqrt(252))
            return {
                "ann_return": round(ann_ret, 4),
                "ann_vol":    round(ann_vol, 4),
                "sharpe":     round(ann_ret / ann_vol if ann_vol > 0 else np.nan, 3),
            }

        corr_within = ret[basket].corr()
        off_diag = corr_within.values[np.triu_indices(len(basket), k=1)]
        mean_corr = float(off_diag.mean()) if len(off_diag) > 0 else np.nan

        return {
            "basket":              basket,
            "basket_stats":        stats(basket_ret),
            "equal_weight_stats":  stats(eq_weight),
            "mean_pairwise_corr":  round(mean_corr, 4),
            "n_commodities_total": len(all_names),
            "n_selected":          self.n_select,
        }
