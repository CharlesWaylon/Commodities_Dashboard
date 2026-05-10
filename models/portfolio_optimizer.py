"""
models/portfolio_optimizer.py

Cascade-informed QAOA portfolio optimizer.

Wraps QAOAPortfolioOptimizer to substitute cascade-derived expected returns
(from macro-adjusted forecasts) for the historical log-return mean vector,
keeping the sample covariance matrix for risk estimation.

Entry points
────────────
run_cascade_portfolio(cascade_forecasts, confidences, prices, macro_row, ...)
    → CascadePortfolioResult   (QAOA allocation with cascade mu)

backtest_cascade_vs_baseline(prices, macro_df, ...)
    → BacktestSummary          (walk-forward: cascade vs momentum vs equal-weight)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models.quantum.qaoa_portfolio import (
    QAOAPortfolioOptimizer,
    QAOAResult,
    QAOA_N_ASSETS,
    QAOA_K_SELECTED,
    QAOA_P_LAYERS,
    QAOA_LAMBDA,
    QAOA_PENALTY,
    QAOA_LOOKBACK_DAYS,
    QAOA_OPT_STEPS,
)


# ── Cascade coefficient tables ─────────────────────────────────────────────────

COMM_EFFECTS: Dict[str, Dict[str, float]] = {
    "WTI Crude Oil":           {"real_yields": -0.20, "usd": -0.50, "risk_off": -0.40},
    "Brent Crude Oil":         {"real_yields": -0.20, "usd": -0.50, "risk_off": -0.40},
    "Natural Gas (Henry Hub)": {"real_yields":  0.00, "usd": -0.20, "risk_off": -0.20},
    "Gasoline (RBOB)":         {"real_yields": -0.15, "usd": -0.45, "risk_off": -0.35},
    "Heating Oil":             {"real_yields": -0.15, "usd": -0.45, "risk_off": -0.35},
    "Gold (COMEX)":            {"real_yields": -0.60, "usd":  0.30, "risk_off":  0.50},
    "Silver (COMEX)":          {"real_yields": -0.40, "usd": -0.10, "risk_off":  0.25},
    "Copper (COMEX)":          {"real_yields":  0.10, "usd": -0.30, "risk_off": -0.50},
    "Corn (CBOT)":             {"real_yields":  0.00, "usd": -0.30, "risk_off": -0.25},
    "Wheat (CBOT SRW)":        {"real_yields":  0.00, "usd": -0.25, "risk_off": -0.20},
    "Soybeans (CBOT)":         {"real_yields":  0.00, "usd": -0.30, "risk_off": -0.25},
    "Feeder Cattle":           {"real_yields":  0.00, "usd": -0.10, "risk_off": -0.30},
    "Lean Hogs":               {"real_yields":  0.00, "usd": -0.10, "risk_off": -0.30},
}

SECTOR_MAP: Dict[str, str] = {
    "WTI Crude Oil":           "Energy",
    "Brent Crude Oil":         "Energy",
    "Natural Gas (Henry Hub)": "Energy",
    "Gasoline (RBOB)":         "Energy",
    "Heating Oil":             "Energy",
    "Gold (COMEX)":            "Metals",
    "Silver (COMEX)":          "Metals",
    "Copper (COMEX)":          "Metals",
    "Corn (CBOT)":             "Grains",
    "Wheat (CBOT SRW)":        "Grains",
    "Soybeans (CBOT)":         "Grains",
    "Feeder Cattle":           "Livestock",
    "Lean Hogs":               "Livestock",
}


# ── Result containers ──────────────────────────────────────────────────────────

@dataclass
class CascadePortfolioResult:
    """Output of CascadePortfolioOptimizer.optimize() or run_cascade_portfolio()."""
    qaoa_result:           QAOAResult
    cascade_forecasts:     Dict[str, float]             # commodity → final_pct (weekly %)
    confidences:           Dict[str, float]              # commodity → 0..1
    channel_contributions: Dict[str, Dict[str, float]]   # commodity → {channel: pp}
    macro_state:           Dict[str, float]
    baseline_result:       Optional[QAOAResult] = None   # historical-mu QAOA for comparison
    causal_narrative:      str = ""

    @property
    def selected_assets(self) -> List[str]:
        return self.qaoa_result.selected_assets

    @property
    def weights(self) -> Dict[str, float]:
        return self.qaoa_result.weights

    @property
    def expected_return(self) -> float:
        return self.qaoa_result.expected_return

    @property
    def portfolio_vol(self) -> float:
        return self.qaoa_result.portfolio_vol

    @property
    def sharpe(self) -> float:
        return self.qaoa_result.sharpe


@dataclass
class BacktestSummary:
    """Result of backtest_cascade_vs_baseline()."""
    strategy_returns:   Dict[str, pd.Series]     # periodic log-returns per strategy
    cumulative_returns: Dict[str, pd.Series]     # compounded wealth index (starts at 1.0)
    sharpe_ratios:      Dict[str, float]
    max_drawdowns:      Dict[str, float]
    annual_returns:     Dict[str, float]
    n_periods:          int
    rebalance_dates:    List[pd.Timestamp]
    k:                  int
    rebalance_days:     int

    def summary_df(self) -> pd.DataFrame:
        rows = []
        for s in self.sharpe_ratios:
            rows.append({
                "Strategy":     s,
                "Ann. Return":  f"{self.annual_returns.get(s, 0.0):.1%}",
                "Sharpe":       f"{self.sharpe_ratios[s]:.2f}",
                "Max Drawdown": f"{self.max_drawdowns.get(s, 0.0):.1%}",
            })
        return pd.DataFrame(rows)


# ── Cascade signal helpers ─────────────────────────────────────────────────────

def _macro_signals(macro_row: Dict) -> Tuple[float, float, float]:
    """Extract (real_yields_sig, usd_sig, risk_off_sig) from a macro row dict."""
    dxy_z = float(macro_row.get("dxy_zscore63", macro_row.get("dxy_zscore", 0.0)) or 0.0)
    vix   = float(macro_row.get("vix", 18.0) or 18.0)
    tlt_m = float(macro_row.get("tlt_mom21", 0.0) or 0.0)
    return (-tlt_m / 0.05), (dxy_z / 2.0), ((vix - 15.0) / 20.0)


def compute_cascade_forecast(
    commodity:  str,
    macro_row:  Dict,
    prices_df:  pd.DataFrame,
    lookback:   int = 22,
) -> Tuple[float, float, float, float, float, Dict[str, float]]:
    """
    Returns (base_pct, macro_adj, final_pct, band_90, confidence, channel_breakdown).
    Compatible with _compute_cascade_for() in pages/4_Models.py.
    """
    effects = COMM_EFFECTS.get(commodity, {})
    ry, usd, ro = _macro_signals(macro_row)

    base_pct = 0.0
    try:
        col = next((c for c in prices_df.columns if commodity.lower() in c.lower()), None)
        if col is not None:
            s = prices_df[col].dropna()
            if len(s) >= lookback + 1:
                base_pct = float((s.iloc[-1] / s.iloc[-lookback] - 1) * 100)
    except Exception:
        pass

    ch: Dict[str, float] = {
        "Real Yields": ry  * effects.get("real_yields", 0.0),
        "USD":         usd * effects.get("usd",         0.0),
        "Risk-Off":    ro  * effects.get("risk_off",    0.0),
    }
    macro_adj = sum(ch.values())
    final_pct = base_pct + macro_adj

    signs = [np.sign(v) for v in ch.values() if v != 0.0]
    conflicting = len(set(signs)) > 1 if signs else False
    band_90 = abs(macro_adj) * (1.5 if conflicting else 0.8) + abs(base_pct) * 0.3 + 0.5
    confidence = max(
        0.05, min(0.95, min(abs(macro_adj) / 2.0, 1.0) - (0.2 if conflicting else 0.0) + 0.35)
    )
    return base_pct, macro_adj, final_pct, band_90, confidence, ch


def cascade_mu_from_forecasts(
    forecasts:   Dict[str, float],    # commodity → final_pct (weekly %)
    asset_names: List[str],
    hist_mu:     np.ndarray,          # daily log-return means from historical data
    confidences: Dict[str, float],    # commodity → 0..1
) -> np.ndarray:
    """
    Blend cascade weekly-% forecasts with historical daily log-return mu.
    Blend weight = confidence (high confidence → trust cascade more).

    Unit conversion: weekly % → daily log-return = log(1 + pct/100) / 5
    """
    mu = hist_mu.copy()
    for i, name in enumerate(asset_names):
        if name in forecasts:
            conf = confidences.get(name, 0.5)
            cascade_daily = np.log(max(1.0 + forecasts[name] / 100.0, 1e-6)) / 5.0
            mu[i] = conf * cascade_daily + (1.0 - conf) * hist_mu[i]
    return mu


def _build_causal_narrative(
    macro_state:       Dict,
    cascade_forecasts: Dict[str, float],
    selected:          List[str],
) -> str:
    """Auto-generate a one-paragraph narrative describing the cascade allocation."""
    dxy_z = float(macro_state.get("dxy_zscore63", macro_state.get("dxy_zscore", 0.0)) or 0.0)
    vix   = float(macro_state.get("vix", 18.0) or 18.0)
    tlt_m = float(macro_state.get("tlt_mom21", 0.0) or 0.0)

    lines: List[str] = []

    if abs(tlt_m) > 2:
        direction = "rising" if tlt_m < 0 else "falling"
        lines.append(
            f"Real yields are {direction} (TLT {tlt_m:+.1f}%), "
            f"{'pressuring' if tlt_m < 0 else 'supporting'} rate-sensitive metals."
        )

    if abs(dxy_z) > 1.0:
        lines.append(
            f"USD is {'strong' if dxy_z > 0 else 'weak'} (z-score {dxy_z:+.1f}), "
            f"{'creating headwinds' if dxy_z > 0 else 'providing tailwinds'} for dollar-denominated commodities."
        )

    if vix > 20:
        lines.append(f"Elevated volatility (VIX {vix:.0f}) drives risk-off flows, supporting gold and safe havens.")
    elif vix < 15:
        lines.append(f"Low volatility (VIX {vix:.0f}) supports risk-on commodities.")

    top3 = sorted(cascade_forecasts.items(), key=lambda x: x[1], reverse=True)[:3]
    bot2 = sorted(cascade_forecasts.items(), key=lambda x: x[1])[:2]

    if top3:
        lines.append(f"Cascade favours {', '.join(f'{c.split(chr(40))[0].strip()} ({v:+.1f}%)' for c, v in top3)}.")
    if bot2 and bot2[0][1] < 0:
        lines.append(f"Underweights {', '.join(f'{c.split(chr(40))[0].strip()} ({v:+.1f}%)' for c, v in bot2)}.")

    if selected:
        sel_str = ", ".join(s.split("(")[0].strip() for s in selected)
        lines.append(f"QAOA selects: {sel_str}.")

    return " ".join(lines) or "Cascade allocation computed from current macro state."


# ── Main optimizer class ───────────────────────────────────────────────────────

class CascadePortfolioOptimizer:
    """
    Wraps QAOAPortfolioOptimizer, substituting cascade-derived expected returns
    for the historical log-return mean (mu) while keeping the covariance matrix.

    Usage
    -----
    opt = CascadePortfolioOptimizer(k=5, p=2)
    opt.fit(prices, cascade_forecasts, confidences, channel_contribs, macro_row)
    result = opt.optimize()        # → CascadePortfolioResult
    """

    def __init__(
        self,
        n_assets:     int   = QAOA_N_ASSETS,
        k:            int   = QAOA_K_SELECTED,
        p:            int   = QAOA_P_LAYERS,
        lam:          float = QAOA_LAMBDA,
        penalty:      float = QAOA_PENALTY,
        lookback:     int   = QAOA_LOOKBACK_DAYS,
        opt_steps:    int   = QAOA_OPT_STEPS,
        run_baseline: bool  = False,    # set True to also run unmodified QAOA for comparison
    ):
        self.n_assets     = n_assets
        self.k            = k
        self.p            = p
        self.lam          = lam
        self.penalty      = penalty
        self.lookback     = lookback
        self.opt_steps    = opt_steps
        self.run_baseline = run_baseline

        self._qaoa: QAOAPortfolioOptimizer = QAOAPortfolioOptimizer(
            n_assets, k, p, lam, penalty, lookback, opt_steps
        )
        self._cascade_fcast: Dict[str, float]       = {}
        self._confidences:   Dict[str, float]       = {}
        self._channels:      Dict[str, Dict]        = {}
        self._macro_state:   Dict                   = {}
        self._hist_mu:       Optional[np.ndarray]   = None
        self._prices_ref:    Optional[pd.DataFrame] = None
        self._is_fit         = False

    def fit(
        self,
        prices:            pd.DataFrame,
        cascade_forecasts: Dict[str, float],
        confidences:       Dict[str, float],
        channel_contribs:  Optional[Dict[str, Dict]] = None,
        macro_row:         Optional[Dict]             = None,
    ) -> "CascadePortfolioOptimizer":
        """
        1. Fit QAOA on historical prices (computes asset universe, cov, hist mu).
        2. Override mu with confidence-blended cascade forecasts.
        """
        self._prices_ref = prices
        self._qaoa.fit(prices)

        # Save historical mu before overriding
        self._hist_mu = self._qaoa._mu.copy()

        # Blend cascade mu into the QAOA's internal mu vector
        self._qaoa._mu = cascade_mu_from_forecasts(
            cascade_forecasts,
            self._qaoa._asset_names,
            self._hist_mu,
            confidences,
        )

        self._cascade_fcast = cascade_forecasts
        self._confidences   = confidences
        self._channels      = channel_contribs or {}
        self._macro_state   = macro_row or {}
        self._is_fit        = True
        return self

    def optimize(self, n_shots: int = 512) -> CascadePortfolioResult:
        if not self._is_fit:
            raise RuntimeError("Call fit() before optimize().")

        cascade_res = self._qaoa.optimize(n_shots=n_shots)

        baseline_res: Optional[QAOAResult] = None
        if self.run_baseline and self._prices_ref is not None:
            try:
                base = QAOAPortfolioOptimizer(
                    self.n_assets, self.k, self.p, self.lam,
                    self.penalty, self.lookback, self.opt_steps,
                )
                base.fit(self._prices_ref)
                baseline_res = base.optimize(n_shots=n_shots)
            except Exception:
                pass

        return CascadePortfolioResult(
            qaoa_result           = cascade_res,
            cascade_forecasts     = self._cascade_fcast,
            confidences           = self._confidences,
            channel_contributions = self._channels,
            macro_state           = self._macro_state,
            baseline_result       = baseline_res,
            causal_narrative      = _build_causal_narrative(
                self._macro_state,
                self._cascade_fcast,
                cascade_res.selected_assets,
            ),
        )


# ── One-call entry point ───────────────────────────────────────────────────────

def run_cascade_portfolio(
    cascade_forecasts: Dict[str, float],
    confidences:       Dict[str, float],
    prices:            pd.DataFrame,
    macro_row:         Optional[Dict]  = None,
    channel_contribs:  Optional[Dict]  = None,
    n_assets:          int   = QAOA_N_ASSETS,
    k:                 int   = QAOA_K_SELECTED,
    p:                 int   = QAOA_P_LAYERS,
    lam:               float = QAOA_LAMBDA,
    penalty:           float = QAOA_PENALTY,
    lookback:          int   = QAOA_LOOKBACK_DAYS,
    opt_steps:         int   = QAOA_OPT_STEPS,
    run_baseline:      bool  = False,
    n_shots:           int   = 512,
) -> CascadePortfolioResult:
    """
    One-call entry: fit cascade-informed optimizer and return allocation.

    Parameters
    ----------
    cascade_forecasts : dict  commodity → weekly expected return (%)
    confidences       : dict  commodity → confidence in cascade forecast (0–1)
    prices            : DataFrame  date-indexed close prices
    macro_row         : dict  latest macro state for narrative generation
    channel_contribs  : dict  commodity → {channel: contribution_pp}
    run_baseline      : bool  if True, also runs unmodified QAOA for comparison (2× runtime)
    """
    opt = CascadePortfolioOptimizer(
        n_assets=n_assets, k=k, p=p, lam=lam, penalty=penalty,
        lookback=lookback, opt_steps=opt_steps, run_baseline=run_baseline,
    )
    opt.fit(prices, cascade_forecasts, confidences, channel_contribs, macro_row)
    return opt.optimize(n_shots=n_shots)


# ── Walk-forward backtest ──────────────────────────────────────────────────────

def _sharpe(returns: pd.Series, periods_per_year: float = 12.0) -> float:
    if returns.std() < 1e-9:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def _max_drawdown(cum: pd.Series) -> float:
    peak = cum.cummax()
    dd   = (cum - peak) / peak
    return float(dd.min())


def _annual_return(cum: pd.Series, periods_per_year: float = 12.0) -> float:
    if len(cum) < 2 or cum.iloc[0] <= 0:
        return 0.0
    n_years = (len(cum) - 1) / periods_per_year
    if n_years <= 0:
        return 0.0
    return float((cum.iloc[-1] / cum.iloc[0]) ** (1.0 / n_years) - 1.0)


def backtest_cascade_vs_baseline(
    prices:          pd.DataFrame,
    macro_df:        pd.DataFrame,
    k:               int = QAOA_K_SELECTED,
    rebalance_days:  int = 21,
    lookback_days:   int = 252,
    min_history:     int = 126,
    commodity_names: Optional[List[str]] = None,
) -> BacktestSummary:
    """
    Walk-forward backtest comparing three selection strategies.

    Strategies (all equal-weight within selected assets):
      equal_weight    : all commodities always held, 1/N
      momentum_top_k  : top-k by 21-day momentum, rebalanced monthly
      cascade_top_k   : top-k by (cascade_final × confidence), rebalanced monthly

    The cascade uses COMM_EFFECTS with macro signals from macro_df (aligned by
    date). When macro_df is empty, cascade falls back to momentum-only.

    Parameters
    ----------
    prices          : date-indexed price matrix, one column per commodity
    macro_df        : date-indexed macro signals (dxy_zscore63, vix, tlt_mom21)
    k               : assets to hold per rebalance period
    rebalance_days  : trading days between rebalances
    lookback_days   : momentum lookback window (days)
    min_history     : minimum history before first backtest period starts
    commodity_names : restrict universe; defaults to all COMM_EFFECTS keys present in prices
    """
    # ── Align commodity universe ──────────────────────────────────────────────
    if commodity_names is None:
        commodity_names = [
            c for c in COMM_EFFECTS if any(c.lower() in col.lower() for col in prices.columns)
        ]
    # map display name → price column name
    col_map: Dict[str, str] = {}
    for comm in commodity_names:
        match = next((c for c in prices.columns if comm.lower() in c.lower()), None)
        if match:
            col_map[comm] = match
    commodity_names = list(col_map.keys())

    if not commodity_names or len(commodity_names) < k:
        return BacktestSummary(
            strategy_returns={}, cumulative_returns={}, sharpe_ratios={},
            max_drawdowns={}, annual_returns={}, n_periods=0,
            rebalance_dates=[], k=k, rebalance_days=rebalance_days,
        )

    sub_prices = prices[[col_map[c] for c in commodity_names]].copy()
    sub_prices.columns = commodity_names
    sub_prices = sub_prices.dropna(how="all")

    # ── Build rebalance date index ────────────────────────────────────────────
    dates      = sub_prices.index
    start_idx  = min_history
    rebal_idxs = list(range(start_idx, len(dates) - rebalance_days, rebalance_days))

    if len(rebal_idxs) < 3:
        return BacktestSummary(
            strategy_returns={}, cumulative_returns={}, sharpe_ratios={},
            max_drawdowns={}, annual_returns={}, n_periods=0,
            rebalance_dates=[], k=k, rebalance_days=rebalance_days,
        )

    # ── Pre-compute log-returns ───────────────────────────────────────────────
    log_ret = np.log(sub_prices / sub_prices.shift(1))

    # ── Align macro_df to price index ────────────────────────────────────────
    macro_aligned: Optional[pd.DataFrame] = None
    if not macro_df.empty:
        try:
            macro_aligned = macro_df.reindex(dates, method="ffill")
        except Exception:
            macro_aligned = None

    # ── Period-return accumulators ────────────────────────────────────────────
    period_rets: Dict[str, List[float]] = {
        "Equal Weight":    [],
        "Momentum Top-k":  [],
        "Cascade Top-k":   [],
    }
    rebalance_dates: List[pd.Timestamp] = []

    for idx in rebal_idxs:
        t_start = idx
        t_end   = min(idx + rebalance_days, len(dates) - 1)
        if t_end <= t_start:
            continue
        rebalance_dates.append(dates[t_start])

        history = sub_prices.iloc[max(0, t_start - lookback_days): t_start]
        fwd_prices = sub_prices.iloc[t_start: t_end + 1]
        if len(history) < 22 or len(fwd_prices) < 2:
            for s in period_rets:
                period_rets[s].append(0.0)
            continue

        # Forward period returns per commodity
        fwd_ret: Dict[str, float] = {}
        for c in commodity_names:
            p0 = fwd_prices[c].iloc[0]
            p1 = fwd_prices[c].iloc[-1]
            if p0 > 0:
                fwd_ret[c] = float(np.log(p1 / p0))

        # ── Equal-weight ──────────────────────────────────────────────────────
        ew_comms = [c for c in commodity_names if c in fwd_ret]
        ew_ret   = float(np.mean([fwd_ret[c] for c in ew_comms])) if ew_comms else 0.0
        period_rets["Equal Weight"].append(ew_ret)

        # ── Momentum top-k ────────────────────────────────────────────────────
        mom_signal: Dict[str, float] = {}
        for c in commodity_names:
            s = history[c].dropna()
            if len(s) >= 22:
                mom_signal[c] = float(np.log(s.iloc[-1] / s.iloc[-22]))
        top_mom = sorted(mom_signal, key=lambda x: mom_signal[x], reverse=True)[:k]
        mom_ret = (
            float(np.mean([fwd_ret[c] for c in top_mom if c in fwd_ret]))
            if top_mom else 0.0
        )
        period_rets["Momentum Top-k"].append(mom_ret)

        # ── Cascade top-k ─────────────────────────────────────────────────────
        macro_row: Dict = {}
        if macro_aligned is not None:
            try:
                row = macro_aligned.iloc[t_start]
                macro_row = row.to_dict()
            except Exception:
                pass

        cascade_signal: Dict[str, float] = {}
        for c in commodity_names:
            try:
                _, _, final_pct, _, conf, _ = compute_cascade_forecast(
                    c, macro_row, history, lookback=22
                )
                cascade_signal[c] = final_pct * conf
            except Exception:
                # fall back to momentum if cascade fails
                cascade_signal[c] = mom_signal.get(c, 0.0) * 100

        top_cascade = sorted(cascade_signal, key=lambda x: cascade_signal[x], reverse=True)[:k]
        casc_ret = (
            float(np.mean([fwd_ret[c] for c in top_cascade if c in fwd_ret]))
            if top_cascade else 0.0
        )
        period_rets["Cascade Top-k"].append(casc_ret)

    # ── Aggregate statistics ──────────────────────────────────────────────────
    periods_per_year = 252.0 / rebalance_days

    sharpe_ratios:  Dict[str, float] = {}
    max_drawdowns:  Dict[str, float] = {}
    annual_returns: Dict[str, float] = {}
    cumulative:     Dict[str, pd.Series] = {}
    strategy_rets:  Dict[str, pd.Series] = {}

    for s, rets in period_rets.items():
        if not rets:
            continue
        ser = pd.Series(rets, index=rebalance_dates[:len(rets)])
        cum  = (1.0 + ser).cumprod()
        strategy_rets[s]  = ser
        cumulative[s]      = cum
        sharpe_ratios[s]   = _sharpe(ser, periods_per_year)
        max_drawdowns[s]   = _max_drawdown(cum)
        annual_returns[s]  = _annual_return(cum, periods_per_year)

    return BacktestSummary(
        strategy_returns   = strategy_rets,
        cumulative_returns = cumulative,
        sharpe_ratios      = sharpe_ratios,
        max_drawdowns      = max_drawdowns,
        annual_returns     = annual_returns,
        n_periods          = len(rebalance_dates),
        rebalance_dates    = rebalance_dates,
        k                  = k,
        rebalance_days     = rebalance_days,
    )
