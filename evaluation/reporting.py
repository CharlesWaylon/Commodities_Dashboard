"""
Read-only reporting helpers for the signal-research surface.

Presentation-agnostic: returns plain DataFrames/Series so the Streamlit page stays
thin (computation lives here and in the signal layer, per the layered-architecture
rule). Everything is defensive — on any failure it returns an empty result rather
than raising, so the research page degrades gracefully.

Nothing here is "promoted": these are research diagnostics for signals that have
NOT passed the gate. The page renders them clearly labelled as such.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

# The research ensemble and its components / sibling signals (all gate-REJECTED as
# of 2026-06-04; shown as research, not promoted).
ENSEMBLE_NAME = "ensemble_v1"
ENSEMBLE_COMPONENTS = ("momentum_xs", "cot_risk_premium", "reversal_st")


def latest_scorecard(signal_names: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Latest gate run per signal as a tidy DataFrame.

    Columns: signal_name, horizon, ic_mean, ic_ir, ic_tstat, hit_rate,
    ls_sharpe_net, avg_turnover, verdict, run_at. Empty frame on any failure.
    """
    names = list(signal_names) if signal_names else [ENSEMBLE_NAME, *ENSEMBLE_COMPONENTS]
    cols = ["signal_name", "horizon", "ic_mean", "ic_ir", "ic_tstat", "hit_rate",
            "ls_sharpe_net", "avg_turnover", "verdict", "run_at"]
    try:
        from database.db import get_db
        from database.models import SignalScorecardRow as R

        rows: List[dict] = []
        with get_db() as db:
            for name in names:
                latest = (
                    db.query(R.run_at)
                    .filter(R.signal_name == name)
                    .order_by(R.run_at.desc())
                    .first()
                )
                if not latest:
                    continue
                recs = (
                    db.query(R)
                    .filter(R.signal_name == name, R.run_at == latest[0])
                    .order_by(R.horizon)
                    .all()
                )
                for r in recs:
                    rows.append({
                        "signal_name": r.signal_name, "horizon": r.horizon,
                        "ic_mean": r.ic_mean, "ic_ir": r.ic_ir, "ic_tstat": r.ic_tstat,
                        "hit_rate": r.hit_rate, "ls_sharpe_net": r.ls_sharpe_net,
                        "avg_turnover": r.avg_turnover, "verdict": r.verdict,
                        "run_at": r.run_at,
                    })
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame(columns=cols)


def ensemble_tilts(horizon: int = 10, top_n: int = 8) -> pd.DataFrame:
    """
    Current cross-sectional ensemble score per instrument at the latest date.

    Returns columns [instrument, score, side] sorted by score (longs first).
    `top_n` keeps the strongest |score| longs and shorts (0 => all). Empty on
    failure. This calls the registered ensemble signal — computation stays in the
    signal layer.
    """
    try:
        from models.data_loader import load_price_matrix_from_db
        from signals.base import FORECAST_FIELD, get_signal

        panel = load_price_matrix_from_db()
        if panel is None or panel.empty:
            return pd.DataFrame(columns=["instrument", "score", "side"])
        sig = get_signal(ENSEMBLE_NAME)
        asof = panel.index[-1]
        out = sig.compute(asof, panel)
        if out is None or (horizon, FORECAST_FIELD) not in out.columns:
            return pd.DataFrame(columns=["instrument", "score", "side"])
        s = out[(horizon, FORECAST_FIELD)].dropna().sort_values(ascending=False)
        if s.empty:
            return pd.DataFrame(columns=["instrument", "score", "side"])
        df = s.reset_index()
        df.columns = ["instrument", "score"]
        df["side"] = df["score"].apply(lambda x: "LONG" if x > 0 else "SHORT")
        if top_n and len(df) > 2 * top_n:
            df = pd.concat([df.head(top_n), df.tail(top_n)], ignore_index=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["instrument", "score", "side"])


def asof_date() -> Optional[pd.Timestamp]:
    """Latest price date the snapshot is computed as of (None on failure)."""
    try:
        from models.data_loader import load_price_matrix_from_db

        panel = load_price_matrix_from_db()
        return None if panel is None or panel.empty else panel.index[-1]
    except Exception:
        return None
