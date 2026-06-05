"""
portfolio/backtest.py — the Layer-3 portfolio backtest (step 3.3).

Walks the allocator forward through history and produces the REALIZED, net-of-cost
equity curve — the Layer-3 analogue of the signal gate. Where the gate scores raw
predictive power (IC), this scores the tradeable result: a risk-managed, costed P&L
with Sharpe, drawdown, turnover and exposure.

Each rebalance (every ``rebalance_days``):
  1. estimate the risk model as-of the decision date (PIT, data ≤ d),
  2. compute the signal's forecasts as-of d (PIT),
  3. map them to target weights via the SleeveAllocator,
  4. charge transaction cost on the turnover from the drifted book to the target,
  5. hold and let weights DRIFT with prices until the next rebalance.

Costs are folded into the equity curve, so every reported metric is net of cost.
Same TransactionCostModel as the gate, so paper and book agree.

LAYER DISCIPLINE: may import signals/data; never streamlit/pages/app.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from evaluation.costs import TransactionCostModel
from portfolio.allocators import AllocatorConfig, SleeveAllocator
from portfolio.risk import estimate_risk_model


@dataclass
class BacktestConfig:
    rebalance_days: int = 21          # holding period between rebalances (≈ monthly)
    risk_lookback: int = 252          # covariance/vol window
    warmup: int = 252                 # min rows before the first rebalance
    cost_bps: float = 10.0            # per-side transaction cost
    periods_per_year: float = 252.0
    allocator: AllocatorConfig = field(default_factory=AllocatorConfig)


@dataclass
class BacktestResult:
    equity: pd.Series                 # net-of-cost equity curve (starts at 1.0)
    daily_returns: pd.Series          # net daily portfolio returns
    turnover: pd.Series               # per-rebalance traded notional
    gross_leverage: pd.Series         # per-rebalance gross
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    cagr: float
    avg_turnover: float
    avg_gross: float
    realized_vol: float
    n_rebalances: int
    signal_name: str

    def summary(self) -> str:
        return (
            f"net Sharpe {self.sharpe:+.2f} | ann.ret {self.ann_return:+.1%} | "
            f"ann.vol {self.ann_vol:.1%} | maxDD {self.max_drawdown:.1%} | "
            f"avg turnover {self.avg_turnover:.2f}/reb | avg gross {self.avg_gross:.2f} | "
            f"{self.n_rebalances} rebalances"
        )


def run_backtest(
    signal,
    panel: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    allocator=None,
) -> BacktestResult:
    """
    Walk ``signal`` through ``panel`` and return realized net-of-cost P&L.

    ``allocator`` is any object exposing ``allocate(forecast_frame, risk_model) ->
    AllocationResult`` (the default SleeveAllocator, or a SingleHorizonFrameAllocator
    wrapping a selection allocator such as MV-select or QAOA). This is what lets the
    classical and quantum allocators compete on the SAME backtest engine.
    """
    cfg = config or BacktestConfig()
    panel = panel.sort_index()
    dates = panel.index
    simple_ret = panel.pct_change()
    cost_model = TransactionCostModel(cost_bps=cfg.cost_bps)
    allocator = allocator if allocator is not None else SleeveAllocator(cfg.allocator)

    start_i = max(cfg.warmup, cfg.risk_lookback)
    equity = 1.0
    w_cur = pd.Series(dtype=float)
    eq_dates: List[pd.Timestamp] = []
    eq_vals: List[float] = []
    turnovers: List[float] = []
    grosses: List[float] = []
    reb_dates: List[pd.Timestamp] = []

    for i in range(start_i, len(dates) - 1):
        d = dates[i]
        # ── rebalance? ────────────────────────────────────────────────────────
        if (i - start_i) % cfg.rebalance_days == 0:
            rm = estimate_risk_model(panel, d, lookback=cfg.risk_lookback)
            if rm is not None:
                fc = signal.compute(d, panel)
                if fc is not None and not fc.dropna(how="all").empty:
                    res = allocator.allocate(fc, rm)
                    w_tgt = res.weights
                    if not w_tgt.empty:
                        turnovers.append(cost_model.turnover(w_cur, w_tgt))
                        equity *= (1.0 - cost_model.cost(w_cur, w_tgt))
                        grosses.append(res.gross_leverage)
                        reb_dates.append(d)
                        w_cur = w_tgt

        # ── realize next-day return on the held book ──────────────────────────
        r_next = simple_ret.iloc[i + 1]
        if not w_cur.empty:
            r = r_next.reindex(w_cur.index).fillna(0.0)
            port_ret = float((w_cur * r).sum())
            equity *= (1.0 + port_ret)
            denom = 1.0 + port_ret
            if denom > 0:
                w_cur = w_cur * (1.0 + r) / denom   # weights drift with prices
        eq_dates.append(dates[i + 1])
        eq_vals.append(equity)

    equity_curve = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_dates), name="equity")
    daily = equity_curve.pct_change().dropna()
    ppy = cfg.periods_per_year

    if len(daily) > 1 and daily.std() > 0:
        ann_ret = float(daily.mean() * ppy)
        ann_vol = float(daily.std() * np.sqrt(ppy))
        sharpe = ann_ret / ann_vol
    else:
        ann_ret = ann_vol = sharpe = float("nan")
    n_years = len(equity_curve) / ppy if len(equity_curve) else float("nan")
    cagr = float(equity_curve.iloc[-1] ** (1 / n_years) - 1) if len(equity_curve) and n_years > 0 else float("nan")
    dd = equity_curve / equity_curve.cummax() - 1.0
    max_dd = float(dd.min()) if len(dd) else float("nan")

    return BacktestResult(
        equity=equity_curve,
        daily_returns=daily,
        turnover=pd.Series(turnovers, index=pd.DatetimeIndex(reb_dates)),
        gross_leverage=pd.Series(grosses, index=pd.DatetimeIndex(reb_dates)),
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        cagr=cagr,
        avg_turnover=float(np.mean(turnovers)) if turnovers else float("nan"),
        avg_gross=float(np.mean(grosses)) if grosses else float("nan"),
        realized_vol=ann_vol,
        n_rebalances=len(reb_dates),
        signal_name=getattr(signal, "name", "signal"),
    )


# ── CLI ──────────────────────────────────────────────────────────────────────
def _load_panel(source: str):
    from models.data_loader import load_long_history_core_panel, load_price_matrix_from_db

    if source == "long_core":
        p = load_long_history_core_panel()
        if p is None or p.empty:
            raise RuntimeError("long_core panel empty — run `python -m services.deep_history_ingest`.")
        return p
    return load_price_matrix_from_db()


def main(argv: Optional[List[str]] = None) -> int:
    from signals.base import get_signal

    ap = argparse.ArgumentParser(description="Walk-forward portfolio backtest of a signal.")
    ap.add_argument("--signal", required=True)
    ap.add_argument("--panel", default="aligned", choices=["aligned", "long_core"])
    ap.add_argument("--rebalance-days", type=int, default=21)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--target-vol", type=float, default=0.10)
    args = ap.parse_args(argv)

    panel = _load_panel(args.panel)
    cfg = BacktestConfig(
        rebalance_days=args.rebalance_days,
        cost_bps=args.cost_bps,
        allocator=AllocatorConfig(target_vol=args.target_vol),
    )
    res = run_backtest(get_signal(args.signal), panel, cfg)
    print("=" * 78)
    print(f"PORTFOLIO BACKTEST · {res.signal_name} · panel={args.panel} · "
          f"rebal={args.rebalance_days}d · cost={args.cost_bps}bps")
    print("-" * 78)
    print(res.summary())
    print(f"CAGR {res.cagr:+.1%} · realized vol {res.realized_vol:.1%} (target {args.target_vol:.0%})")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
