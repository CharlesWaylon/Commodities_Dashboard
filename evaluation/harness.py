"""
evaluation/harness.py — THE GATE.

This is the single, headless, look-ahead-safe evaluator referenced by the
CLAUDE.md rules. Nothing in the dashboard becomes "live" until it passes here.
The gate, not any model, is the product's spine (North Star principle #1).

What it does, for one ``Signal`` at one or more horizons:

  1. Generates point-in-time forecasts at every evaluable date (the signal only
     ever sees data <= the decision date — see signals/base.py).
  2. Scores each horizon SEPARATELY against realised H-day forward returns:
       • out-of-sample Spearman IC (cross-sectional, per date)
       • IC information ratio + t-stat, computed on a DE-OVERLAPPED subsample
         (dates spaced >= H apart) so autocorrelated overlapping targets do not
         inflate significance — this is the purge/embargo discipline applied to
         the IC series.
       • directional hit rate
       • NET-OF-COST long-short PnL (dollar-neutral book rebalanced every ~H days,
         held to the next rebalance, charged the transaction-cost model).
  3. Splits the de-overlapped IC series into walk-forward folds and checks the IC
     SIGN is STABLE across folds (not driven by one lucky window).
  4. Emits a machine-readable scorecard (-> signal_scorecard table) and a
     human-readable diff vs the previous run, plus a PROMOTE / REJECT verdict.

Definition of done for Phase 0:
    python -m evaluation.harness --signal momentum_xs --horizons 5,10,21
produces a costed, walk-forward, look-ahead-safe scorecard with a pass/fail
verdict, logged.

This module never imports streamlit / pages / app (enforced by .importlinter).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluation.costs import TransactionCostModel
from signals.base import FORECAST_FIELD, Signal, get_signal


# ── Configuration ────────────────────────────────────────────────────────────
@dataclass
class PassFailConfig:
    """The promotability contract. A horizon is PROMOTABLE iff all hold."""

    min_ic_mean: float = 0.0          # OOS IC must be positive
    min_ic_ir: float = 0.30           # IC information ratio (mean/std of de-overlapped IC)
    min_fold_sign_frac: float = 0.60  # fraction of folds whose IC shares the overall sign
    min_ls_sharpe_net: float = 0.0    # net-of-cost long-short Sharpe must be positive


@dataclass
class HarnessConfig:
    n_splits: int = 5                 # walk-forward folds for sign-stability
    min_history: int = 260            # rows of price history before first eval date
    min_cross_section: int = 5        # min instruments with a view to score a date
    cost_bps: float = 10.0            # per-side transaction cost (basis points)
    long_short_quantile: float = 0.0  # 0 => rank-weighted book; >0 => top/bottom q only
    passfail: PassFailConfig = field(default_factory=PassFailConfig)


# ── Scorecards ────────────────────────────────────────────────────────────────
@dataclass
class HorizonScorecard:
    signal_name: str
    horizon: int
    n_obs: int                # number of daily cross-sectional IC observations
    n_obs_deoverlapped: int   # non-overlapping subsample size (drives the t-stat)
    ic_mean: float
    ic_ir: float
    ic_tstat: float
    hit_rate: float
    ls_sharpe_net: float
    ls_return_net_ann: float
    avg_turnover: float
    cost_bps: float
    n_folds: int
    fold_ic_means: List[float]
    fold_sign_frac: float
    verdict: str              # "promote" | "reject"
    reasons: List[str]        # why it failed (empty if promoted)


@dataclass
class SignalScorecard:
    signal_name: str
    economic_rationale: str
    run_at: str               # UTC ISO
    horizons: List[HorizonScorecard]
    config_json: str

    def overall_verdict(self) -> str:
        return "promote" if any(h.verdict == "promote" for h in self.horizons) else "reject"


# ── Metric helpers ─────────────────────────────────────────────────────────────
def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation via pandas (no scipy dependency)."""
    if len(a) < 3:
        return np.nan
    return float(a.rank().corr(b.rank()))


def _forward_returns(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """H-day forward cumulative log return: fwd.loc[t] = Σ ret over (t, t+H]."""
    log_ret = np.log(panel).diff()
    cum = log_ret.rolling(horizon).sum()
    return cum.shift(-horizon)


def _deoverlap(dates: List[pd.Timestamp], positions: Dict[pd.Timestamp, int], horizon: int) -> List[pd.Timestamp]:
    """Greedily pick dates spaced >= ``horizon`` trading rows apart (non-overlapping)."""
    picked: List[pd.Timestamp] = []
    last_pos = -10**9
    for d in dates:
        p = positions[d]
        if p - last_pos >= horizon:
            picked.append(d)
            last_pos = p
    return picked


def _long_short_weights(score: pd.Series, quantile: float) -> pd.Series:
    """Dollar-neutral weights with gross leverage 1 from a cross-sectional score."""
    s = score.dropna()
    if quantile and quantile > 0:
        lo, hi = s.quantile(quantile), s.quantile(1 - quantile)
        longs = s[s >= hi].index
        shorts = s[s <= lo].index
        w = pd.Series(0.0, index=s.index)
        if len(longs):
            w[longs] = 1.0 / len(longs)
        if len(shorts):
            w[shorts] = -1.0 / len(shorts)
    else:
        w = s - s.mean()  # demeaned ranks-ish; dollar neutral
    gross = w.abs().sum()
    if gross == 0:
        return w
    return w / gross


# ── Core evaluation ─────────────────────────────────────────────────────────────
def run_signal(
    signal: Signal,
    panel: pd.DataFrame,
    horizons: Optional[Tuple[int, ...]] = None,
    config: Optional[HarnessConfig] = None,
) -> SignalScorecard:
    """Evaluate ``signal`` on ``panel`` and return a full scorecard."""
    config = config or HarnessConfig()
    horizons = tuple(horizons or signal.horizons)
    panel = panel.sort_index()
    cost_model = TransactionCostModel(cost_bps=config.cost_bps)

    index = panel.index
    positions = {d: i for i, d in enumerate(index)}
    max_h = max(horizons)

    # Evaluable dates: enough history behind, and >= max_h rows ahead for a target.
    eval_dates = list(index[config.min_history : len(index) - max_h])

    # Generate point-in-time forecasts once per date (all horizons at once).
    forecasts: Dict[pd.Timestamp, pd.DataFrame] = {}
    for t in eval_dates:
        out = signal.compute(t, panel)
        if out is None or out.dropna(how="all").empty:
            continue
        forecasts[t] = out
    fc_dates = sorted(forecasts)

    horizon_cards: List[HorizonScorecard] = []
    for h in horizons:
        fwd = _forward_returns(panel, h)

        # ── per-date cross-sectional IC + directional hits ────────────────────
        ic_by_date: Dict[pd.Timestamp, float] = {}
        hit_num = hit_den = 0
        for t in fc_dates:
            if (h, FORECAST_FIELD) not in forecasts[t].columns:
                continue
            f = forecasts[t][(h, FORECAST_FIELD)].dropna()
            if f.empty:
                continue
            r = fwd.loc[t].reindex(f.index)
            valid = f.notna() & r.notna()
            if valid.sum() < config.min_cross_section:
                continue
            ic = _spearman(f[valid], r[valid])
            if np.isnan(ic):
                continue
            ic_by_date[t] = ic
            hit_num += int((np.sign(f[valid]) == np.sign(r[valid])).sum())
            hit_den += int(valid.sum())

        ic_dates = sorted(ic_by_date)
        ic_series = pd.Series([ic_by_date[d] for d in ic_dates], index=pd.DatetimeIndex(ic_dates))

        # ── de-overlap for an honest IR / t-stat ──────────────────────────────
        deov_dates = _deoverlap(ic_dates, positions, h)
        deov_ic = ic_series.reindex(pd.DatetimeIndex(deov_dates)).dropna()

        ic_mean = float(deov_ic.mean()) if len(deov_ic) else float("nan")
        ic_std = float(deov_ic.std(ddof=1)) if len(deov_ic) > 1 else float("nan")
        ic_ir = ic_mean / ic_std if ic_std and np.isfinite(ic_std) and ic_std > 0 else float("nan")
        ic_tstat = ic_ir * np.sqrt(len(deov_ic)) if np.isfinite(ic_ir) else float("nan")
        hit_rate = hit_num / hit_den if hit_den else float("nan")

        # ── walk-forward folds on the de-overlapped IC series ─────────────────
        fold_means: List[float] = []
        if len(deov_ic) >= config.n_splits:
            chunks = np.array_split(deov_ic.to_numpy(), config.n_splits)
            fold_means = [float(np.mean(c)) for c in chunks if len(c)]
        overall_sign = np.sign(ic_mean) if np.isfinite(ic_mean) else 0.0
        if fold_means and overall_sign != 0:
            fold_sign_frac = float(np.mean([np.sign(m) == overall_sign for m in fold_means]))
        else:
            fold_sign_frac = float("nan")

        # ── net-of-cost long-short PnL on non-overlapping holding periods ──────
        ls_returns: List[float] = []
        turnovers: List[float] = []
        prev_w = pd.Series(dtype=float)
        for i, t in enumerate(deov_dates):
            if (h, FORECAST_FIELD) not in forecasts[t].columns:
                continue
            score = forecasts[t][(h, FORECAST_FIELD)].dropna()
            if len(score) < config.min_cross_section:
                continue
            w = _long_short_weights(score, config.long_short_quantile)
            # holding period: t -> next picked date (≈ h trading days)
            t_next = deov_dates[i + 1] if i + 1 < len(deov_dates) else None
            if t_next is None:
                p0, p1 = positions[t], min(positions[t] + h, len(index) - 1)
            else:
                p0, p1 = positions[t], positions[t_next]
            cum_log = np.log(panel).diff().iloc[p0 + 1 : p1 + 1].sum()
            simple_ret = np.expm1(cum_log).reindex(w.index).fillna(0.0)
            gross = float((w * simple_ret).sum())
            cost = cost_model.cost(prev_w, w)
            ls_returns.append(gross - cost)
            turnovers.append(cost_model.turnover(prev_w, w))
            prev_w = w

        ls = pd.Series(ls_returns, dtype=float)
        periods_per_year = 252.0 / h
        if len(ls) > 1 and ls.std(ddof=1) > 0:
            ls_sharpe = float(ls.mean() / ls.std(ddof=1) * np.sqrt(periods_per_year))
        else:
            ls_sharpe = float("nan")
        ls_ret_ann = float(ls.mean() * periods_per_year) if len(ls) else float("nan")
        avg_turnover = float(np.mean(turnovers)) if turnovers else float("nan")

        # ── verdict ────────────────────────────────────────────────────────────
        pf = config.passfail
        reasons: List[str] = []
        if not (np.isfinite(ic_mean) and ic_mean > pf.min_ic_mean):
            reasons.append(f"IC mean {ic_mean:.4f} ≤ {pf.min_ic_mean}")
        if not (np.isfinite(ic_ir) and ic_ir >= pf.min_ic_ir):
            reasons.append(f"IC IR {ic_ir:.3f} < {pf.min_ic_ir}")
        if not (np.isfinite(fold_sign_frac) and fold_sign_frac >= pf.min_fold_sign_frac):
            reasons.append(f"fold sign-stability {fold_sign_frac:.2f} < {pf.min_fold_sign_frac}")
        if not (np.isfinite(ls_sharpe) and ls_sharpe >= pf.min_ls_sharpe_net):
            reasons.append(f"net LS Sharpe {ls_sharpe:.3f} < {pf.min_ls_sharpe_net}")
        verdict = "promote" if not reasons else "reject"

        horizon_cards.append(
            HorizonScorecard(
                signal_name=signal.name,
                horizon=h,
                n_obs=len(ic_series),
                n_obs_deoverlapped=len(deov_ic),
                ic_mean=ic_mean,
                ic_ir=ic_ir,
                ic_tstat=ic_tstat,
                hit_rate=hit_rate,
                ls_sharpe_net=ls_sharpe,
                ls_return_net_ann=ls_ret_ann,
                avg_turnover=avg_turnover,
                cost_bps=config.cost_bps,
                n_folds=len(fold_means),
                fold_ic_means=fold_means,
                fold_sign_frac=fold_sign_frac,
                verdict=verdict,
                reasons=reasons,
            )
        )

    return SignalScorecard(
        signal_name=signal.name,
        economic_rationale=signal.economic_rationale,
        run_at=datetime.now(timezone.utc).isoformat(),
        horizons=horizon_cards,
        config_json=json.dumps(asdict(config), default=str),
    )


# ── Persistence (the experiment ledger, step 0.4) ────────────────────────────────
def persist_scorecard(card: SignalScorecard) -> int:
    """Write one row per horizon to the ``signal_scorecard`` table. Returns rows written."""
    from database.db import get_db, init_db
    from database.models import SignalScorecardRow

    init_db()  # idempotent CREATE TABLE IF NOT EXISTS
    n = 0
    now = datetime.now(timezone.utc).isoformat()

    def _f(x):
        # Coerce numpy scalars / NaN to DB-safe Python floats (None for NaN).
        if x is None:
            return None
        x = float(x)
        return None if not np.isfinite(x) else x

    with get_db() as db:
        for h in card.horizons:
            db.add(
                SignalScorecardRow(
                    run_at=card.run_at,
                    signal_name=card.signal_name,
                    horizon=int(h.horizon),
                    n_obs=int(h.n_obs),
                    ic_mean=_f(h.ic_mean),
                    ic_ir=_f(h.ic_ir),
                    ic_tstat=_f(h.ic_tstat),
                    hit_rate=_f(h.hit_rate),
                    ls_sharpe_net=_f(h.ls_sharpe_net),
                    ls_return_net_ann=_f(h.ls_return_net_ann),
                    avg_turnover=_f(h.avg_turnover),
                    cost_bps=_f(h.cost_bps),
                    verdict=h.verdict,
                    detail_json=json.dumps(asdict(h), default=_json_default),
                    config_json=card.config_json,
                    inserted_at=now,
                )
            )
            n += 1
    return n


def load_previous(signal_name: str, before_run_at: str) -> Dict[int, dict]:
    """Most-recent prior run's per-horizon metrics, for the human-readable diff."""
    from database.db import get_db
    from database.models import SignalScorecardRow

    try:
        with get_db() as db:
            prev_run = (
                db.query(SignalScorecardRow.run_at)
                .filter(SignalScorecardRow.signal_name == signal_name)
                .filter(SignalScorecardRow.run_at < before_run_at)
                .order_by(SignalScorecardRow.run_at.desc())
                .first()
            )
            if not prev_run:
                return {}
            rows = (
                db.query(SignalScorecardRow)
                .filter(SignalScorecardRow.signal_name == signal_name)
                .filter(SignalScorecardRow.run_at == prev_run[0])
                .all()
            )
            return {
                r.horizon: {"ic_mean": r.ic_mean, "ic_ir": r.ic_ir, "ls_sharpe_net": r.ls_sharpe_net, "verdict": r.verdict}
                for r in rows
            }
    except Exception:
        return {}


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


# ── Human-readable rendering ─────────────────────────────────────────────────────
def _fmt(x: float, nd: int = 4) -> str:
    return "  n/a " if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(card: SignalScorecard, previous: Optional[Dict[int, dict]] = None) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"SIGNAL SCORECARD  ·  {card.signal_name}  ·  {card.run_at}")
    lines.append("-" * 78)
    lines.append(f"rationale: {card.economic_rationale}")
    lines.append("-" * 78)
    header = f"{'H':>4} {'IC':>9} {'IC_IR':>8} {'t-stat':>8} {'hit':>7} {'LS_Shrp':>9} {'LS_ret':>9} {'turn':>7}  verdict"
    lines.append(header)
    for h in card.horizons:
        lines.append(
            f"{h.horizon:>4} {_fmt(h.ic_mean):>9} {_fmt(h.ic_ir,3):>8} {_fmt(h.ic_tstat,2):>8} "
            f"{_fmt(h.hit_rate,3):>7} {_fmt(h.ls_sharpe_net,3):>9} {_fmt(h.ls_return_net_ann,3):>9} "
            f"{_fmt(h.avg_turnover,2):>7}  {h.verdict.upper()}"
        )
        if h.reasons:
            lines.append(f"       ↳ reject: {'; '.join(h.reasons)}")
    lines.append("-" * 78)

    if previous:
        lines.append("Δ vs previous run:")
        for h in card.horizons:
            p = previous.get(h.horizon)
            if not p:
                lines.append(f"  H={h.horizon}: (no prior run)")
                continue
            d_ic = h.ic_mean - p["ic_mean"] if p["ic_mean"] is not None else float("nan")
            d_sh = h.ls_sharpe_net - p["ls_sharpe_net"] if p["ls_sharpe_net"] is not None else float("nan")
            lines.append(
                f"  H={h.horizon}: IC {_fmt(p['ic_mean'])}→{_fmt(h.ic_mean)} (Δ{_fmt(d_ic)}), "
                f"LS_Sharpe {_fmt(p['ls_sharpe_net'],3)}→{_fmt(h.ls_sharpe_net,3)} (Δ{_fmt(d_sh,3)}), "
                f"{p['verdict']}→{h.verdict}"
            )
        lines.append("-" * 78)

    lines.append(f"OVERALL VERDICT: {card.overall_verdict().upper()}")
    lines.append("=" * 78)
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────────
def _load_panel(source: str = "aligned"):
    """
    Point-in-time price panel from the data layer.

    source="aligned"   -> production ~5y common panel (load_price_matrix_from_db).
    source="long_core" -> deep ~24y core-futures panel (research; multi-regime).
    """
    from models.data_loader import load_long_history_core_panel, load_price_matrix_from_db

    if source == "long_core":
        panel = load_long_history_core_panel()
        if panel is None or panel.empty:
            raise RuntimeError(
                "long_core panel is empty — run `python -m services.deep_history_ingest` first."
            )
        return panel
    return load_price_matrix_from_db()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the out-of-sample evaluation gate on a signal.")
    ap.add_argument("--signal", required=True, help="registered signal name, e.g. momentum_xs")
    ap.add_argument("--horizons", default="5,10,21", help="comma-separated trading-day horizons")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="per-side transaction cost (bps)")
    ap.add_argument("--n-splits", type=int, default=5, help="walk-forward folds for sign-stability")
    ap.add_argument(
        "--min-cross-section",
        type=int,
        default=HarnessConfig.min_cross_section,
        help="min instruments with a view to score a date (lower for sub-universe "
        "signals, e.g. 4 for the energy-only inventory feed). Default 5.",
    )
    ap.add_argument("--no-db", action="store_true", help="do not persist to signal_scorecard")
    ap.add_argument(
        "--panel", default="aligned", choices=["aligned", "long_core"],
        help="price panel: 'aligned' (production ~5y) or 'long_core' (deep ~24y "
        "core futures, research/multi-regime). Default aligned.",
    )
    args = ap.parse_args(argv)

    horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    signal = get_signal(args.signal)
    panel = _load_panel(args.panel)
    config = HarnessConfig(
        n_splits=args.n_splits,
        cost_bps=args.cost_bps,
        min_cross_section=args.min_cross_section,
    )

    card = run_signal(signal, panel, horizons=horizons, config=config)
    previous = load_previous(card.signal_name, card.run_at) if not args.no_db else None
    print(render(card, previous))

    if not args.no_db:
        n = persist_scorecard(card)
        print(f"[harness] persisted {n} scorecard rows to signal_scorecard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
