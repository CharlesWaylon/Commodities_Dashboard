"""
Shared narrative engines for the Accendio Commodities Dashboard.

Provides two public functions:

  build_cascade_narrative(macro_state, forecasts) -> str
      Plain-English macro brief for pages/7_Macro_Market_Cascade.py and
      the home-page Intelligence Brief (app.py).  Re-exported from
      utils.macro_narrative so the coefficient tables stay co-located with
      compute_forecasts().

  build_chain_narrative(result) -> str
      2-3 sentence walk-through of a CausalChainResult trace.  Extracted
      from models.causal_chain so future pages (home brief, portfolio
      summaries) can call the same engine without importing the full chain
      model.

Circular-import note
--------------------
models.causal_chain imports this module (to get build_chain_narrative).
This module must NOT import models.causal_chain at module level.
The CausalChainResult type is referenced only under TYPE_CHECKING, and
get_trigger_family() is imported lazily inside build_chain_narrative().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import pandas as pd

if TYPE_CHECKING:
    from models.causal_chain import CausalChainResult

# ── Cascade narrative ─────────────────────────────────────────────────────────
# The function body lives in utils.macro_narrative alongside SECTORS and the
# coefficient tables it references.  We expose it under the canonical new name
# while keeping the old build_narrative name available in that module.

from utils.macro_narrative import build_narrative as build_cascade_narrative  # noqa: E402


# ── Chain narrative ───────────────────────────────────────────────────────────

def build_chain_narrative(result: "CausalChainResult") -> str:
    """
    Compose a 2-3 sentence plain-English explanation of a causal chain trace.

    Parameters
    ----------
    result : CausalChainResult
        Completed trace returned by CausalChain.trace().

    Returns
    -------
    str
        Markdown-formatted narrative (sentences joined with spaces).
    """
    from models.triggers import get_trigger_family

    parts: List[str] = []

    # Sentence 1: trigger event
    fam      = get_trigger_family(result.trigger_family)
    fam_name = fam.description if fam else result.trigger_family.replace("_", " ").title()
    tier = (
        "strong"   if result.trigger_strength >= 0.7 else
        "moderate" if result.trigger_strength >= 0.3 else
        "weak"
    )
    parts.append(
        f"On {result.trigger_date}, a {tier} **{fam_name}** signal fired "
        f"(strength {result.trigger_strength:.0%})."
    )

    # Sentence 2: statistical layer + regime
    stat   = next((n for n in result.nodes if n.layer == "statistical"), None)
    regime = next((n for n in result.nodes if n.layer == "regime"),      None)
    mid: List[str] = []

    if stat and stat.before_value and stat.after_value:
        chg = stat.change_pct or 0.0
        mid.append(
            f"Realized volatility in {result.commodity.split('(')[0].strip()} "
            f"{'rose' if chg > 0 else 'fell'} "
            f"from {stat.before_value:.1f}% to {stat.after_value:.1f}% "
            f"({'+'  if chg >= 0 else ''}{chg:.0%})"
        )

    if regime:
        r_label = regime.metadata.get("regime", "Neutral")
        mid.append(f"the macro regime was **{r_label}**")

    if mid:
        parts.append(". ".join(mid).capitalize() + ".")

    # Sentence 3: ensemble + portfolio + cascade overlay
    ens  = next((n for n in result.nodes if n.layer == "ensemble"),  None)
    port = next((n for n in result.nodes if n.layer == "portfolio"), None)

    ens_str = ""
    if ens:
        trusted = ens.metadata.get("trusted_tier", "")
        if ens.metadata.get("is_trained"):
            ens_str = f"The meta-predictor trusted the **{trusted}** tier. "
        else:
            ens_str = "The meta-predictor has not yet been trained (equal weights). "

    port_str = (
        f"Portfolio recommendation: {result.portfolio_recommendation}" if port else ""
    )

    cascade_str = ""
    if port and port.metadata.get("cascade_available"):
        c_action = port.metadata.get("cascade_action", "none")
        c_src    = port.metadata.get("cascade_source", "")
        c_sec    = port.metadata.get("cascade_sector", "")
        c_pct    = port.metadata.get("cascade_final", 0.0) * 100
        c_date   = port.metadata.get("cascade_date", "")
        if c_action == "confirm":
            cascade_str = (
                f" The cascade sector model ({c_sec}, {c_src}) independently "
                f"forecast {c_pct:+.2f}% — confirming the rule-based recommendation "
                f"and lifting confidence."
            )
        elif c_action == "override":
            cascade_str = (
                f" **Cascade override**: the {c_sec} sector model ({c_src}, "
                f"forecast date {c_date}) disagreed with the rule-based signal "
                f"and produced a {c_pct:+.2f}% forecast — the cascade direction "
                f"was adopted as the final recommendation."
            )

    if ens_str or port_str or cascade_str:
        parts.append((ens_str + port_str + cascade_str).strip())

    return " ".join(parts)
