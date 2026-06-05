"""
portfolio/compete.py — allocator bake-off on the same net-of-cost backtest.

Runs the candidate allocators — classical mean-variance selection, risk-parity
(inverse-vol), and the quantum QAOA selection — through the IDENTICAL walk-forward
portfolio backtest and ranks them by realized net-of-cost Sharpe. This is how the
quantum optimiser earns (or fails to earn) its place: "ships only where it wins".

``production_allocator(...)`` encodes that policy: QAOA is chosen for production
only if its feature flag is on AND it beat the best classical baseline here;
otherwise the classical winner is returned. So quantum stays first-class and
flagged, but never ships unless it is measurably better.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from portfolio.allocators import (
    AllocatorConfig,
    MeanVarianceSelectAllocator,
    RiskScaledAllocator,
    SingleHorizonFrameAllocator,
)
from portfolio.backtest import BacktestConfig, BacktestResult, run_backtest
from portfolio.quantum_allocator import QAOAAllocator, qaoa_allocator_enabled

CLASSICAL = ("classical_mv", "risk_parity")
QUANTUM = "qaoa"


@dataclass
class Bakeoff:
    results: Dict[str, BacktestResult]
    table: pd.DataFrame
    winner: str
    quantum_wins: bool

    def verdict(self) -> str:
        q = self.results.get(QUANTUM)
        best_classical = max(
            (n for n in self.results if n in CLASSICAL),
            key=lambda n: _safe(self.results[n].sharpe),
            default=None,
        )
        if q is None or best_classical is None:
            return f"winner: {self.winner}"
        verb = "BEATS" if self.quantum_wins else "does NOT beat"
        return (
            f"QAOA {verb} the best classical baseline "
            f"({QUANTUM} Sharpe {q.sharpe:+.2f} vs {best_classical} "
            f"{self.results[best_classical].sharpe:+.2f}) → "
            f"{'QAOA eligible to ship (flag-gated)' if self.quantum_wins else 'QAOA stays gated off'}."
        )


def _safe(x) -> float:
    return float(x) if x is not None and x == x else float("-inf")


def _build_allocators(horizon: int, k: int, n_universe: int, target_vol: float) -> Dict[str, object]:
    return {
        "classical_mv": SingleHorizonFrameAllocator(
            MeanVarianceSelectAllocator(k=k, n_universe=n_universe, target_vol=target_vol), horizon
        ),
        "risk_parity": SingleHorizonFrameAllocator(
            RiskScaledAllocator(AllocatorConfig(target_vol=target_vol)), horizon
        ),
        QUANTUM: SingleHorizonFrameAllocator(
            QAOAAllocator(k=k, n_universe=n_universe, target_vol=target_vol), horizon
        ),
    }


def run_bakeoff(
    signal,
    panel: pd.DataFrame,
    bt_config: Optional[BacktestConfig] = None,
    horizon: int = 21,
    k: int = 5,
    n_universe: int = 12,
    include: Optional[List[str]] = None,
) -> Bakeoff:
    """Backtest each allocator on the same panel/config; rank by net Sharpe."""
    bt = bt_config or BacktestConfig()
    allocators = _build_allocators(horizon, k, n_universe, bt.allocator.target_vol)
    if include:
        allocators = {n: a for n, a in allocators.items() if n in include}

    results: Dict[str, BacktestResult] = {}
    for name, alloc in allocators.items():
        results[name] = run_backtest(signal, panel, bt, allocator=alloc)

    rows = [{
        "allocator": n, "net_sharpe": r.sharpe, "ann_return": r.ann_return,
        "ann_vol": r.ann_vol, "max_drawdown": r.max_drawdown,
        "avg_turnover": r.avg_turnover, "n_rebalances": r.n_rebalances,
    } for n, r in results.items()]
    table = pd.DataFrame(rows).sort_values("net_sharpe", ascending=False).reset_index(drop=True)
    winner = table.iloc[0]["allocator"] if not table.empty else "none"

    best_classical = max((n for n in results if n in CLASSICAL),
                         key=lambda n: _safe(results[n].sharpe), default=None)
    quantum_wins = (
        QUANTUM in results and best_classical is not None
        and _safe(results[QUANTUM].sharpe) > _safe(results[best_classical].sharpe)
    )
    return Bakeoff(results=results, table=table, winner=winner, quantum_wins=quantum_wins)


def production_allocator(bakeoff: Bakeoff, horizon: int, k: int, n_universe: int, target_vol: float):
    """
    The 'ships only where it wins' policy: return the QAOA allocator ONLY if its flag
    is on AND it beat the best classical baseline; otherwise the classical winner.
    """
    allocators = _build_allocators(horizon, k, n_universe, target_vol)
    if qaoa_allocator_enabled() and bakeoff.quantum_wins:
        return QUANTUM, allocators[QUANTUM]
    classical_winner = max(
        (n for n in bakeoff.results if n in CLASSICAL),
        key=lambda n: _safe(bakeoff.results[n].sharpe), default="risk_parity",
    )
    return classical_winner, allocators[classical_winner]


def main(argv: Optional[List[str]] = None) -> int:
    from signals.base import get_signal
    from portfolio.backtest import _load_panel

    ap = argparse.ArgumentParser(description="Allocator bake-off (classical vs QAOA) on the same backtest.")
    ap.add_argument("--signal", required=True)
    ap.add_argument("--panel", default="long_core", choices=["aligned", "long_core"])
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--rebalance-days", type=int, default=126, help="coarser cadence keeps QAOA feasible")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--target-vol", type=float, default=0.10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-universe", type=int, default=12)
    args = ap.parse_args(argv)

    panel = _load_panel(args.panel)
    bt = BacktestConfig(rebalance_days=args.rebalance_days, cost_bps=args.cost_bps,
                        allocator=AllocatorConfig(target_vol=args.target_vol))
    bake = run_bakeoff(get_signal(args.signal), panel, bt, horizon=args.horizon,
                       k=args.k, n_universe=args.n_universe)
    print("=" * 78)
    print(f"ALLOCATOR BAKE-OFF · {args.signal} · panel={args.panel} · "
          f"rebal={args.rebalance_days}d · H{args.horizon} · k={args.k}/{args.n_universe}")
    print("-" * 78)
    with pd.option_context("display.float_format", lambda v: f"{v:+.3f}"):
        print(bake.table.to_string(index=False))
    print("-" * 78)
    print(bake.verdict())
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
