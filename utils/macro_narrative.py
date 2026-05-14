"""
Shared macro-narrative utilities used by both app.py (Intelligence Brief)
and pages/7_Macro_Market_Cascade.py.

Extracted here so neither file needs to import the other, avoiding the
module-level side-effects that run when page files are imported directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
from typing import Dict, List


# ── Sector / commodity universe ───────────────────────────────────────────────

SECTORS: Dict[str, List[str]] = {
    "Energy":    ["WTI Crude Oil", "Brent Crude Oil", "Natural Gas (Henry Hub)"],
    "Metals":    ["Gold (COMEX)", "Silver (COMEX)", "Copper (COMEX)"],
    "Grains":    ["Corn (CBOT)", "Wheat (CBOT SRW)", "Soybeans (CBOT)"],
    "Livestock": ["Feeder Cattle", "Lean Hogs"],
}

# ── Regime-conditional effect matrices ────────────────────────────────────────
#
# Two tables replace the former single flat MACRO_EFFECTS dict.
# compute_forecasts() selects the active table via macro_state["risk_off"]
# (True when VIX ≥ 20; False otherwise).
#
# Coefficient interpretation
# --------------------------
#   macro_adj = real_yields_coeff × ry_signal
#             + usd_coeff         × usd_signal
#             + risk_off_coeff    × risk_off_signal
#
#   Signals are normalised to [-1, 1] (risk_off_signal to [0, 1]).
#   Coefficients are in percentage points per unit of normalised signal.
#
# Sources for regime-conditional values
# --------------------------------------
#   Safe-haven gold demand vs VIX:
#     Baur & Lucey (2010), "Is Gold a Hedge or Safe Haven?", Journal of Banking & Finance.
#     Hood & Malik (2013), "Is Gold the Best Hedge?", Quarterly Review of Economics & Finance.
#     → Gold risk_off jumps from +0.30 (risk-on) to +0.80 (risk-off).
#     → Gold's real-yield sensitivity attenuates in stress because safe-haven
#       buying overwhelms the carry-cost channel (Erb & Harvey 2013, FAJ).
#     → In simultaneous dollar/gold safe-haven rallies, Gold's usd coefficient
#       turns positive: Reboredo (2013), "Is Gold a Safe Haven or Hedge?", Resources Policy.
#
#   Copper demand-destruction asymmetry:
#     IMF World Economic Outlook (Apr 2012), "Commodity Market Review".
#     Frankel & Rose (2010), "Determinants of Agricultural and Mineral Commodity Prices",
#     Kennedy School Faculty Research Working Papers.
#     → Copper risk_off deepens from -0.40 (risk-on) to -0.70 (risk-off):
#       recession fears hit industrial demand expectations far harder when VIX ≥ 20.
#
#   Energy demand-destruction in risk-off:
#     Hamilton (2009), "Causes and Consequences of the Oil Shock of 2007-08", Brookings.
#     → WTI/Brent risk_off -0.30 → -0.60; demand-outlook repricing dominates supply signals.
#
#   Agricultural & Livestock safe-haven unwind:
#     Irwin & Good (2009), "Market Instability in a New Era of Corn, Soybean, and Wheat Prices",
#     Choices Magazine.
#     → Grains and Livestock face modest risk_off amplification (-0.20 → -0.35) from
#       speculative long liquidation in risk-off regimes; physical demand is stickier.
#
#   Silver dual-nature (industrial + safe-haven):
#     Hillier, Draper & Faff (2006), "Do Precious Metals Shine?", Financial Analysts Journal.
#     → In risk-off, silver's safe-haven component partially offsets its industrial decline:
#       risk_off +0.20 (risk-on) → +0.50 (risk-off), but well below gold's +0.80.

MACRO_EFFECTS_RISK_ON: Dict[str, Dict[str, float]] = {
    # VIX < 20 — risk-on environment
    # Safe-haven demand is suppressed; gold trades more like an inflation / real-yield
    # instrument; industrial metals track the economic cycle; energy follows demand growth.
    #                                   real_yields    usd     risk_off
    "WTI Crude Oil":           {"real_yields": -0.20, "usd": -0.50, "risk_off": -0.30},
    "Brent Crude Oil":         {"real_yields": -0.20, "usd": -0.50, "risk_off": -0.30},
    "Natural Gas (Henry Hub)": {"real_yields":  0.00, "usd": -0.20, "risk_off": -0.10},
    "Gold (COMEX)":            {"real_yields": -0.60, "usd": -0.20, "risk_off": +0.30},
    "Silver (COMEX)":          {"real_yields": -0.40, "usd": -0.15, "risk_off": +0.20},
    "Copper (COMEX)":          {"real_yields":  0.10, "usd": -0.30, "risk_off": -0.40},
    "Corn (CBOT)":             {"real_yields":  0.00, "usd": -0.30, "risk_off": -0.20},
    "Wheat (CBOT SRW)":        {"real_yields":  0.00, "usd": -0.25, "risk_off": -0.15},
    "Soybeans (CBOT)":         {"real_yields":  0.00, "usd": -0.30, "risk_off": -0.20},
    "Feeder Cattle":           {"real_yields":  0.00, "usd": -0.10, "risk_off": -0.20},
    "Lean Hogs":               {"real_yields":  0.00, "usd": -0.10, "risk_off": -0.20},
}

MACRO_EFFECTS_RISK_OFF: Dict[str, Dict[str, float]] = {
    # VIX ≥ 20 — risk-off environment
    # Safe-haven demand amplifies gold and silver positively. Industrial commodities
    # (copper, energy) face sharper demand-destruction discounting. The gold/USD inverse
    # relationship partially breaks down as both benefit from flight-to-safety.
    #                                   real_yields    usd     risk_off
    "WTI Crude Oil":           {"real_yields": -0.25, "usd": -0.55, "risk_off": -0.60},
    "Brent Crude Oil":         {"real_yields": -0.25, "usd": -0.55, "risk_off": -0.60},
    "Natural Gas (Henry Hub)": {"real_yields":  0.00, "usd": -0.20, "risk_off": -0.15},
    "Gold (COMEX)":            {"real_yields": -0.40, "usd": +0.15, "risk_off": +0.80},
    "Silver (COMEX)":          {"real_yields": -0.35, "usd": -0.10, "risk_off": +0.50},
    "Copper (COMEX)":          {"real_yields":  0.05, "usd": -0.35, "risk_off": -0.70},
    "Corn (CBOT)":             {"real_yields":  0.00, "usd": -0.30, "risk_off": -0.35},
    "Wheat (CBOT SRW)":        {"real_yields":  0.00, "usd": -0.25, "risk_off": -0.30},
    "Soybeans (CBOT)":         {"real_yields":  0.00, "usd": -0.30, "risk_off": -0.35},
    "Feeder Cattle":           {"real_yields":  0.00, "usd": -0.10, "risk_off": -0.40},
    "Lean Hogs":               {"real_yields":  0.00, "usd": -0.10, "risk_off": -0.40},
}

# Backwards-compatible alias — callers that import MACRO_EFFECTS directly get the
# risk-on table. compute_forecasts() ignores this alias and picks the regime table itself.
MACRO_EFFECTS: Dict[str, Dict[str, float]] = MACRO_EFFECTS_RISK_ON


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_macro(period: str = "2y") -> pd.DataFrame:
    from features.macro_overlays import build_macro_overlay_features
    try:
        return build_macro_overlay_features(period=period)
    except Exception:
        return pd.DataFrame()


# ── Macro state ───────────────────────────────────────────────────────────────

def get_macro_state(macro_df: pd.DataFrame) -> Dict:
    """Extract the latest macro state, returning a normalised signal dict."""
    if macro_df is None or macro_df.empty:
        return {
            "dxy_zscore": 0.0, "vix": 18.0, "tlt_mom21": 0.0,
            "real_yields_signal": 0.0, "usd_signal": 0.0, "risk_off_signal": 0.0,
            "risk_off": False, "crisis": False,
            "date": str(date.today()),
        }
    row   = macro_df.iloc[-1]
    dxy_z = float(row.get("dxy_zscore63", 0.0) or 0.0)
    vix   = float(row.get("vix", 18.0) or 18.0)
    tlt21 = float(row.get("tlt_mom21", 0.0) or 0.0)

    real_yields_signal = max(-1.0, min(1.0, -tlt21 / 0.05))
    usd_signal         = max(-1.0, min(1.0,  dxy_z / 2.0))
    risk_off_signal    = max(0.0,  min(1.0,  (vix - 15.0) / 20.0))

    return {
        "dxy_zscore":          round(dxy_z, 3),
        "vix":                 round(vix, 1),
        "tlt_mom21":           round(tlt21, 4),
        "real_yields_signal":  round(real_yields_signal, 3),
        "usd_signal":          round(usd_signal, 3),
        "risk_off_signal":     round(risk_off_signal, 3),
        "risk_off":            vix >= 20.0,
        "crisis":              vix >= 30.0,
        "date": str(macro_df.index[-1].date()),
    }


# ── Forecast computation ──────────────────────────────────────────────────────

def compute_forecasts(prices: pd.DataFrame, macro_state: Dict) -> pd.DataFrame:
    """
    One row per commodity: commodity, sector, base_pct, macro_adj_pct,
    final_pct, uncertainty_band, regime. All values are 1-week forecasts in pp.

    The active MACRO_EFFECTS table is chosen by macro_state["risk_off"]:
      True  (VIX ≥ 20) → MACRO_EFFECTS_RISK_OFF
      False (VIX <  20) → MACRO_EFFECTS_RISK_ON
    """
    ry  = macro_state["real_yields_signal"]
    usd = macro_state["usd_signal"]
    ro  = macro_state["risk_off_signal"]

    efx_table = (
        MACRO_EFFECTS_RISK_OFF if macro_state.get("risk_off", False)
        else MACRO_EFFECTS_RISK_ON
    )
    regime_label = "risk_off" if macro_state.get("risk_off", False) else "risk_on"

    rows = []
    for sector, comms in SECTORS.items():
        for comm in comms:
            if comm in prices.columns and len(prices[comm].dropna()) >= 22:
                log_ret  = np.log(prices[comm] / prices[comm].shift(1)).dropna()
                tail21   = log_ret.tail(21)
                base_pct = float(tail21.mean() * 5 * 100)
                std21    = float(tail21.std() * np.sqrt(5) * 100)
            else:
                base_pct = 0.0
                std21    = 1.0

            efx = efx_table.get(comm, {})
            macro_adj = (
                efx.get("real_yields", 0.0) * ry +
                efx.get("usd",         0.0) * usd +
                efx.get("risk_off",    0.0) * ro
            )
            final = base_pct + macro_adj
            rows.append({
                "commodity":        comm,
                "sector":           sector,
                "base_pct":         round(base_pct, 3),
                "macro_adj_pct":    round(macro_adj, 3),
                "final_pct":        round(final, 3),
                "uncertainty_band": round(1.65 * std21, 3),
                "regime":           regime_label,
            })
    return pd.DataFrame(rows)


# ── Narrative engine ──────────────────────────────────────────────────────────

def build_narrative(macro_state: Dict, forecasts: pd.DataFrame) -> str:
    """
    Generate a plain-English macro brief from the current macro state and
    sector forecasts. Returns a markdown string (paragraphs separated by \\n\\n).
    """
    ry   = macro_state["real_yields_signal"]
    usd  = macro_state["usd_signal"]
    ro   = macro_state["risk_off_signal"]
    vix  = macro_state["vix"]
    dxyz = macro_state["dxy_zscore"]
    tlt  = macro_state["tlt_mom21"]
    crisis   = macro_state["crisis"]
    risk_off = macro_state["risk_off"]

    paras = []

    # Macro regime headline
    if crisis:
        regime_lbl = "crisis conditions (VIX ≥ 30)"
    elif risk_off:
        regime_lbl = "risk-off conditions (VIX ≥ 20)"
    else:
        regime_lbl = "risk-on conditions"
    paras.append(
        f"**Macro environment ({macro_state['date']}):** Markets are in {regime_lbl} "
        f"with VIX at {vix:.1f}. "
        f"The DXY z-score stands at {dxyz:+.2f} ({'above' if dxyz > 0 else 'below'} "
        f"its 63-day average) and TLT 21-day momentum is "
        f"{'positive' if tlt > 0 else 'negative'} at {tlt*100:+.1f}%."
    )

    # Real yields channel
    if abs(ry) > 0.15:
        dir_str = "rising" if ry > 0 else "falling"
        gold_row = forecasts[forecasts["commodity"] == "Gold (COMEX)"]
        gold_adj = f"{gold_row.iloc[0]['macro_adj_pct']:+.2f}%" if not gold_row.empty else "N/A"
        cu_row   = forecasts[forecasts["commodity"] == "Copper (COMEX)"]
        cu_adj   = f"{cu_row.iloc[0]['macro_adj_pct']:+.2f}%" if not cu_row.empty else "N/A"
        paras.append(
            f"**Real yields channel:** Real yields are {dir_str} (TLT 21d momentum "
            f"{tlt*100:+.1f}%), "
            + (f"creating a headwind for Gold (macro adjustment: {gold_adj}) "
               f"as the opportunity cost of holding bullion rises. "
               f"Industrial Copper is less sensitive to rate changes "
               f"(macro adjustment: {cu_adj})."
               if ry > 0.15 else
               f"acting as a tailwind for Gold (macro adjustment: {gold_adj}) "
               f"and other precious metals.")
        )

    # USD channel
    if abs(usd) > 0.15:
        dir_str  = "strong" if usd > 0 else "weak"
        wti_row  = forecasts[forecasts["commodity"] == "WTI Crude Oil"]
        wti_adj  = f"{wti_row.iloc[0]['macro_adj_pct']:+.2f}%" if not wti_row.empty else "N/A"
        paras.append(
            f"**USD channel:** The dollar is {dir_str} (DXY z-score {dxyz:+.2f}). "
            + (f"As most commodity futures are USD-denominated, "
               f"non-US buyers face higher costs — WTI receives a macro adjustment of {wti_adj} "
               f"and most Agricultural prices face similar pressure."
               if usd > 0.15 else
               f"A weak dollar provides broad support for USD-denominated commodity prices; "
               f"WTI macro adjustment: {wti_adj}.")
        )

    # Risk sentiment channel
    if ro > 0.15:
        cu_row  = forecasts[forecasts["commodity"] == "Copper (COMEX)"]
        cu_adj  = f"{cu_row.iloc[0]['macro_adj_pct']:+.2f}%" if not cu_row.empty else "N/A"
        gld_row = forecasts[forecasts["commodity"] == "Gold (COMEX)"]
        gld_adj = f"{gld_row.iloc[0]['macro_adj_pct']:+.2f}%" if not gld_row.empty else "N/A"
        regime_note = (
            "The **risk-off coefficient table** is active (VIX ≥ 20): "
            "safe-haven amplification raises Gold's risk_off sensitivity to +0.80 "
            "(vs +0.30 in risk-on) and deepens Copper's to −0.70 (vs −0.40)."
            if risk_off else
            "The **risk-on coefficient table** is active (VIX < 20): "
            "safe-haven demand for Gold is muted (risk_off coefficient +0.30)."
        )
        paras.append(
            f"**Risk sentiment:** VIX at {vix:.1f} signals risk-off positioning. "
            f"Industrial commodities face demand destruction headwinds — "
            f"Copper macro adjustment: {cu_adj}. "
            f"Gold benefits from safe-haven flows (macro adjustment: {gld_adj}). "
            f"{regime_note}"
        )

    # Sector-level summaries
    sector_summaries = []
    for sector in SECTORS:
        df_s = forecasts[forecasts["sector"] == sector]
        if df_s.empty:
            continue
        avg_final = df_s["final_pct"].mean()
        avg_base  = df_s["base_pct"].mean()
        adj       = avg_final - avg_base
        dir_word  = "adds" if adj > 0 else "subtracts"
        sector_summaries.append(
            f"{sector} ({dir_word} {abs(adj):.2f}pp → {avg_final:+.2f}%)"
        )
    if sector_summaries:
        paras.append(
            "**Sector-level macro overlay:** " + "; ".join(sector_summaries) + "."
        )

    # Overall conclusion
    sector_avgs = forecasts.groupby("sector")["final_pct"].mean()
    if not sector_avgs.empty:
        best  = sector_avgs.idxmax()
        worst = sector_avgs.idxmin()
        paras.append(
            f"**Overall:** Under current macro conditions, **{best}** is the "
            f"most favoured sector (avg 1-wk forecast: {sector_avgs[best]:+.2f}%). "
            f"**{worst}** faces the greatest headwinds "
            f"(avg forecast: {sector_avgs[worst]:+.2f}%). "
            f"These are model-generated outlooks; macro signal quality depends on "
            f"the freshness of DXY, VIX, and TLT data."
        )

    return "\n\n".join(paras)
