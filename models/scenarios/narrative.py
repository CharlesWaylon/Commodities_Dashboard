"""
Auto-generated investor-facing narrative for a consensus ScenarioBand.

Produces a single, plain-English sentence (occasionally two) summarising:
  1. Direction and magnitude of the consensus mean return.
  2. Downside risk: the bear-end price level and percent move.
  3. (Optional) Coupling of the strongest peers in the same VAR complex.

The intent is the "first paragraph" of the dashboard — what a portfolio
manager reads before deciding whether to dive deeper into the charts.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from models.scenarios.band import ScenarioBand


def _strength_word(daily_pct: float) -> str:
    """Map an average daily-return percentage to a directional adjective."""
    if abs(daily_pct) < 0.05:
        return "essentially flat"
    if abs(daily_pct) < 0.20:
        return "mildly bullish" if daily_pct > 0 else "mildly bearish"
    if abs(daily_pct) < 0.50:
        return "moderately bullish" if daily_pct > 0 else "moderately bearish"
    return "strongly bullish" if daily_pct > 0 else "strongly bearish"


def build_narrative(
    band: ScenarioBand,
    spot: float,
    ripple_engine=None,
    peer_top_n: int = 3,
) -> str:
    """
    Render an investor-facing narrative for a consensus band.

    Parameters
    ----------
    band : ScenarioBand
        The consensus output of ScenarioAggregator.aggregate().
    spot : float
        Latest observed price (anchor for the price-level statements).
    ripple_engine : CausalRipple, optional
        If supplied, append a sentence on how the strongest peers move with
        the source commodity at horizon = band.horizon.
    peer_top_n : int
        Number of peers to mention when ripple_engine is provided.

    Returns
    -------
    str
        Markdown-formatted narrative — one or two short sentences.
    """
    horizon = band.horizon
    bear_q_pct = int(round(band.bear_q * 100))
    bull_q_pct = int(round(band.bull_q * 100))

    # Cumulative log-return paths → price endpoints
    cum_mean = float(np.sum(band.mean))
    cum_bear = float(np.sum(band.bear))
    cum_bull = float(np.sum(band.bull))

    mean_daily_pct = (cum_mean / horizon) * 100.0
    bear_end = spot * float(np.exp(cum_bear))
    bull_end = spot * float(np.exp(cum_bull))
    mean_end = spot * float(np.exp(cum_mean))

    bear_move_pct = (bear_end / spot - 1.0) * 100.0
    bull_move_pct = (bull_end / spot - 1.0) * 100.0
    mean_move_pct = (mean_end / spot - 1.0) * 100.0

    direction = _strength_word(mean_daily_pct)

    sentence = (
        f"**{band.commodity} consensus is {direction}** at "
        f"**{mean_daily_pct:+.2f}%/day** "
        f"(→ {mean_move_pct:+.1f}% to **${mean_end:,.2f}** at {horizon} days), "
        f"with P{bear_q_pct} downside to **${bear_end:,.2f}** ({bear_move_pct:+.1f}%) "
        f"and P{bull_q_pct} upside to **${bull_end:,.2f}** ({bull_move_pct:+.1f}%)."
    )

    if ripple_engine is not None:
        try:
            members = [m for m in ripple_engine.members() if m != band.commodity]
            couplings = []
            for peer in members:
                coup = ripple_engine.coupling(band.commodity, peer, horizon)
                # Use the contemporaneous (h=1) coupling — by far the dominant
                # term in a VAR(1) energy/grains system; the user can drill into
                # the matrix if they want lag detail.
                couplings.append((peer, float(coup[0])))
            couplings.sort(key=lambda kv: abs(kv[1]), reverse=True)
            top = couplings[:peer_top_n]

            if top:
                parts = [
                    f"{abs(c)*100:.0f}% {'with' if c >= 0 else 'against'} **{peer}**"
                    for peer, c in top
                ]
                ripple_sentence = (
                    " On impact the "
                    + ripple_engine.group.replace("_", " ")
                    + " complex moves "
                    + ", ".join(parts[:-1])
                    + (", and " if len(parts) > 1 else "")
                    + (parts[-1] if parts else "")
                    + "."
                )
                sentence += ripple_sentence
        except Exception:
            pass

    return sentence
