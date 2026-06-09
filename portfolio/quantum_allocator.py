"""
portfolio/quantum_allocator.py — QAOA portfolio optimisation as ONE allocator option.

The legacy ``models/portfolio_optimizer.py`` (cascade-informed QAOA) stays live for
the existing pages. This module re-casts the SAME quantum optimisation as a
first-class ``Allocator`` in the new Layer-3 framework, so it competes head-to-head
against the classical mean-variance / risk-parity allocators on the SAME
net-of-cost backtest. It ships into production only where it WINS — gated by
``QAOA_ALLOCATOR_ENABLED`` (default off) and a measured win over the classical
baseline (see ``portfolio.compete``).

It reuses the QUBO/QAOA building blocks from ``models.quantum.qaoa_portfolio`` but
takes its μ and Σ from the signal forecast and the Layer-3 risk model (not from a
private price re-load), so it solves exactly the same selection problem as the
classical rival. If PennyLane is unavailable or the solve errors, it falls back to
the classical exact optimum — quantum is never a single point of failure.
"""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np

from portfolio.allocators import MeanVarianceSelectAllocator, _SelectAllocator

logger = logging.getLogger(__name__)

# QAOA defaults tuned for backtest feasibility (kept cheap: shallow circuit, few
# optimiser steps). Quality at n≈12 is bounded above by the classical exact optimum.
QAOA_P_LAYERS = 1
QAOA_OPT_STEPS = 40
QAOA_N_SHOTS = 256
QAOA_PENALTY = 5.0


def qaoa_allocator_enabled() -> bool:
    """Feature flag governing PRODUCTION eligibility (not research comparison)."""
    return os.getenv("QAOA_ALLOCATOR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


class QAOAAllocator(_SelectAllocator):
    """Quantum (QAOA) cardinality-constrained mean-variance selection allocator."""

    def __init__(self, *args, p: int = QAOA_P_LAYERS, opt_steps: int = QAOA_OPT_STEPS,
                 n_shots: int = QAOA_N_SHOTS, penalty: float = QAOA_PENALTY, seed: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = int(p)
        self.opt_steps = int(opt_steps)
        self.n_shots = int(n_shots)
        self.penalty = float(penalty)
        self.seed = int(seed)
        self._fell_back = False

    def _select(self, mu: np.ndarray, cov: np.ndarray) -> List[int]:
        try:
            return self._select_qaoa(mu, cov)
        except Exception as e:  # pragma: no cover - environment-dependent
            if not self._fell_back:
                logger.warning("QAOAAllocator: falling back to classical exact optimum (%s)", e)
                self._fell_back = True
            return MeanVarianceSelectAllocator(
                k=self.k, lam=self.lam, n_universe=self.n_universe
            )._select(mu, cov)

    def _select_qaoa(self, mu: np.ndarray, cov: np.ndarray) -> List[int]:
        import pennylane.numpy as pnp
        from scipy.optimize import minimize

        from models.quantum.qaoa_portfolio import (
            _cost_hamiltonian,
            _decode_bitstrings,
            _make_qaoa_qnode,
            _make_sampler_qnode,
            _mixer_hamiltonian,
            build_qubo,
            qubo_to_ising,
        )

        n = len(mu)
        Q = build_qubo(mu, cov, self.k, self.lam, self.penalty)
        h, J, _ = qubo_to_ising(Q)
        cost_h = _cost_hamiltonian(h, J)
        mixer_h = _mixer_hamiltonian(n)
        circuit = _make_qaoa_qnode(cost_h, mixer_h, n, self.p)
        sampler = _make_sampler_qnode(cost_h, mixer_h, n, self.p, self.n_shots)

        rng = np.random.default_rng(self.seed)
        p0 = np.concatenate([rng.uniform(0.01, 0.5, self.p), rng.uniform(0.2, 0.5, self.p)])

        def objective(params_flat: np.ndarray) -> float:
            return float(circuit(pnp.array(params_flat, requires_grad=True)))

        res = minimize(objective, x0=p0, method="COBYLA",
                       options={"maxiter": self.opt_steps, "rhobeg": 0.3})
        samples = sampler(pnp.array(res.x, requires_grad=False))
        bits = _decode_bitstrings(np.array(samples), h, J, self.k)
        idx = [i for i, b in enumerate(bits) if b == 1]
        return idx or list(range(min(self.k, n)))
