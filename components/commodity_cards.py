"""
Commodity Cards with Trigger Ripple Effect  (Mission 8)
=========================================================
Renders commodity price cards that visually react to active triggers from
the lifecycle manager.

States
------
  normal       — no active triggers elevating this commodity
  elevated     — commodity is in the primary scope of one or more active triggers;
                 card shows an exposure badge and a pulsing border ripple
  recalibrating — trigger was activated < RECAL_WINDOW_SECONDS ago;
                 additionally shows a recalibration spinner while model
                 handlers are expected to still be in their adjustment window

Public API
----------
  commodity_card_html(name, price, pct_change, sector, triggers)  → str
  commodity_cards_grid(df, lifecycle, max_elevated, key)          → None (renders)
  recal_css()                                                      → str  (shared CSS)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import streamlit as st

from services.trigger_lifecycle import LIFECYCLE, ActiveTrigger

# How long after activation the card shows the "recalibrating" spinner
RECAL_WINDOW_SECONDS = 45

# Severity → visual properties (border colour, ripple colour, badge colours)
_SEV_PROPS = {
    "LOW": {
        "border":     "rgba(238,242,255,0.22)",
        "ripple":     "rgba(238,242,255,0.18)",
        "badge_bg":   "rgba(238,242,255,0.06)",
        "badge_text": "rgba(238,242,255,0.45)",
        "glow":       "none",
    },
    "MEDIUM": {
        "border":     "#F59E0B",
        "ripple":     "rgba(245,158,11,0.45)",
        "badge_bg":   "rgba(245,158,11,0.12)",
        "badge_text": "#F59E0B",
        "glow":       "0 0 8px rgba(245,158,11,0.18)",
    },
    "HIGH": {
        "border":     "#e07020",
        "ripple":     "rgba(224,112,32,0.45)",
        "badge_bg":   "rgba(224,112,32,0.12)",
        "badge_text": "#e07020",
        "glow":       "0 0 10px rgba(224,112,32,0.22)",
    },
    "CRITICAL": {
        "border":     "#D94F4F",
        "ripple":     "rgba(217,79,79,0.50)",
        "badge_bg":   "rgba(217,79,79,0.15)",
        "badge_text": "#D94F4F",
        "glow":       "0 0 14px rgba(217,79,79,0.30)",
    },
}

_SECTOR_ICONS = {
    "energy":      "⚡",
    "metals":      "⚙",
    "agriculture": "🌾",
    "livestock":   "🐄",
    "digital":     "₿",
}


def recal_css() -> str:
    """Return the shared CSS block for card animations. Inject once per page."""
    return """
<style>
/* ── Ripple pulse animation ─────────────────────────────────────────────── */
@keyframes ac-ripple-low {
  0%   { box-shadow: 0 0 0 0   rgba(238,242,255,0.25); }
  70%  { box-shadow: 0 0 0 10px rgba(238,242,255,0); }
  100% { box-shadow: 0 0 0 0   rgba(238,242,255,0); }
}
@keyframes ac-ripple-medium {
  0%   { box-shadow: 0 0 0 0   rgba(245,158,11,0.50); }
  70%  { box-shadow: 0 0 0 12px rgba(245,158,11,0); }
  100% { box-shadow: 0 0 0 0   rgba(245,158,11,0); }
}
@keyframes ac-ripple-high {
  0%   { box-shadow: 0 0 0 0   rgba(224,112,32,0.50); }
  70%  { box-shadow: 0 0 0 12px rgba(224,112,32,0); }
  100% { box-shadow: 0 0 0 0   rgba(224,112,32,0); }
}
@keyframes ac-ripple-critical {
  0%   { box-shadow: 0 0 0 0   rgba(217,79,79,0.60); }
  70%  { box-shadow: 0 0 0 14px rgba(217,79,79,0); }
  100% { box-shadow: 0 0 0 0   rgba(217,79,79,0); }
}

/* ── Recalibration indicator dots ───────────────────────────────────────── */
@keyframes ac-recal-dot {
  0%, 80%, 100% { transform: scale(0.7); opacity: 0.3; }
  40%           { transform: scale(1.0); opacity: 1.0; }
}
.ac-recal-dot {
  display: inline-block;
  width: 4px; height: 4px;
  border-radius: 50%;
  background: #7B9CFF;
  animation: ac-recal-dot 1.2s infinite ease-in-out;
  margin-right: 2px;
  vertical-align: middle;
}
.ac-recal-dot:nth-child(2) { animation-delay: 0.20s; }
.ac-recal-dot:nth-child(3) { animation-delay: 0.40s; }

/* ── Card base styles ───────────────────────────────────────────────────── */
.ac-card {
  background: #0C1228;
  border: 0.5px solid rgba(123,156,255,0.12);
  border-radius: 8px;
  padding: 12px 14px 10px;
  font-family: Arial, 'Helvetica Neue', sans-serif;
  transition: border-color 0.3s, box-shadow 0.3s;
  position: relative;
  overflow: hidden;
}
.ac-card-normal { border-left: 2.5px solid rgba(123,156,255,0.18); }
.ac-card-low      { border-left: 2.5px solid rgba(238,242,255,0.22);
                    animation: ac-ripple-low      2.0s ease-out infinite; }
.ac-card-medium   { border-left: 2.5px solid #F59E0B;
                    animation: ac-ripple-medium   1.8s ease-out infinite; }
.ac-card-high     { border-left: 2.5px solid #e07020;
                    animation: ac-ripple-high     1.6s ease-out infinite;
                    box-shadow: 0 0 10px rgba(224,112,32,0.22); }
.ac-card-critical { border-left: 2.5px solid #D94F4F;
                    animation: ac-ripple-critical 1.4s ease-out infinite;
                    box-shadow: 0 0 14px rgba(217,79,79,0.30); }

/* Card content */
.ac-card-sector {
  font-size: 9px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: rgba(238,242,255,0.28);
  margin-bottom: 2px;
}
.ac-card-name {
  font-size: 12px;
  font-weight: 500;
  color: #EEF2FF;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 6px;
}
.ac-card-price {
  font-size: 16px;
  font-weight: 300;
  color: #EEF2FF;
  letter-spacing: -0.01em;
  line-height: 1;
}
.ac-card-change {
  font-size: 11px;
  margin-top: 2px;
}

/* Exposure badge */
.ac-exposure-badge {
  display: flex;
  align-items: center;
  gap: 5px;
  margin-top: 8px;
  padding: 4px 7px;
  border-radius: 4px;
  font-size: 9px;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}
.ac-exposure-trigger {
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-weight: 500;
}
.ac-exposure-sev {
  font-weight: 700;
  flex-shrink: 0;
}

/* Recalibration row */
.ac-recal-row {
  display: flex;
  align-items: center;
  gap: 5px;
  margin-top: 5px;
  font-size: 9px;
  color: rgba(123,156,255,0.7);
  letter-spacing: 0.06em;
}

/* Magnitude mini-bar under exposure badge */
.ac-mag-track {
  height: 2px;
  background: rgba(238,242,255,0.06);
  border-radius: 1px;
  overflow: hidden;
  margin-top: 4px;
}
.ac-mag-fill {
  height: 100%;
  border-radius: 1px;
}

/* Multi-trigger stack indicator */
.ac-multi-trigger {
  position: absolute;
  top: 8px;
  right: 8px;
  font-size: 8px;
  background: rgba(123,156,255,0.12);
  color: rgba(123,156,255,0.7);
  border-radius: 3px;
  padding: 1px 5px;
  letter-spacing: 0.06em;
}
</style>
"""


def _highest_severity_index(triggers: List[ActiveTrigger]) -> int:
    _rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    return max((_rank.get(e.signal.severity.label(), 0) for e in triggers), default=0)


def _highest_severity(triggers: List[ActiveTrigger]) -> str:
    _rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    _rev  = {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}
    idx = _highest_severity_index(triggers)
    return _rev.get(idx, "LOW")


def _is_recalibrating(triggers: List[ActiveTrigger]) -> bool:
    now = datetime.now(timezone.utc)
    return any(
        (now - e.activated_at).total_seconds() < RECAL_WINDOW_SECONDS
        for e in triggers
    )


def commodity_card_html(
    name: str,
    price: float,
    pct_change: float,
    sector: str,
    triggers: Optional[List[ActiveTrigger]] = None,
) -> str:
    """
    Build the HTML string for a single commodity card.

    ``triggers`` should be the list returned by ``LIFECYCLE.elevated_by(name)``.
    Pass ``None`` or ``[]`` for normal (non-elevated) rendering.
    """
    triggers = triggers or []
    is_elevated = len(triggers) > 0
    recalibrating = is_elevated and _is_recalibrating(triggers)

    change_color = "#3DB87A" if pct_change >= 0 else "#D94F4F"
    change_sign  = "+" if pct_change >= 0 else ""
    sector_lower = sector.lower()
    sector_icon  = _SECTOR_ICONS.get(sector_lower, "◆")

    # Card animation class
    if not is_elevated:
        card_class = "ac-card ac-card-normal"
    else:
        sev = _highest_severity(triggers).lower()
        card_class = f"ac-card ac-card-{sev}"

    # Multi-trigger badge
    multi_html = ""
    if len(triggers) > 1:
        multi_html = f'<div class="ac-multi-trigger">+{len(triggers)} triggers</div>'

    # Exposure badge (show the most severe trigger)
    exposure_html = ""
    if is_elevated:
        top = max(triggers, key=lambda e: e.signal.severity.value)
        sev_label  = top.signal.severity.label()
        props      = _SEV_PROPS.get(sev_label, _SEV_PROPS["LOW"])
        short_name = top.signal.config.display_name.split("(")[0].strip()
        if len(short_name) > 22:
            short_name = short_name[:20] + "…"
        mag_pct = int(top.signal.magnitude_score * 100)

        exposure_html = f"""
<div class="ac-exposure-badge"
     style="background:{props['badge_bg']};border:0.5px solid {props['border']}">
  <span class="ac-exposure-trigger" style="color:{props['badge_text']}">{short_name}</span>
  <span class="ac-exposure-sev"     style="color:{props['badge_text']}">{sev_label}</span>
</div>
<div class="ac-mag-track">
  <div class="ac-mag-fill" style="width:{mag_pct}%;background:{props['border']}"></div>
</div>"""

    # Recalibration indicator
    recal_html = ""
    if recalibrating:
        recal_html = """
<div class="ac-recal-row">
  <span class="ac-recal-dot"></span>
  <span class="ac-recal-dot"></span>
  <span class="ac-recal-dot"></span>
  <span>recalibrating</span>
</div>"""

    return f"""
<div class="{card_class}">
  {multi_html}
  <div class="ac-card-sector">{sector_icon} {sector.upper()}</div>
  <div class="ac-card-name" title="{name}">{name}</div>
  <div class="ac-card-price">{price:,.2f}</div>
  <div class="ac-card-change" style="color:{change_color}">
    {change_sign}{pct_change:.2f}%
  </div>
  {exposure_html}
  {recal_html}
</div>"""


# ── Grid renderer ──────────────────────────────────────────────────────────────

_GRID_CSS = """
<style>
.ac-cards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
  gap: 10px;
  margin-top: 8px;
}
.ac-grid-section-label {
  font-size: 9px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: rgba(238,242,255,0.25);
  margin: 14px 0 4px;
  font-family: Arial, sans-serif;
}
</style>
"""


def commodity_cards_grid(
    df,                          # DataFrame with columns: Name, Sector, Price, Pct_Change
    lifecycle=None,
    max_elevated: int = 20,
    key: str = "cards",
) -> None:
    """
    Render a responsive grid of commodity cards.

    Elevated cards are shown first in a highlighted section.
    Remaining cards follow in a dimmer section.

    Parameters
    ----------
    df            : DataFrame with at minimum Name, Sector, Price, Pct_Change columns.
    lifecycle     : TriggerLifecycleManager instance (defaults to module-level LIFECYCLE).
    max_elevated  : Maximum elevated cards to render (prevents overwhelming the UI).
    key           : Unique string used to namespace session state.
    """
    if lifecycle is None:
        lifecycle = LIFECYCLE

    # Inject shared CSS once
    st.markdown(recal_css() + _GRID_CSS, unsafe_allow_html=True)

    snap = lifecycle.state_snapshot()
    elevated_set = set(snap["elevated_commodities"])

    elevated_rows = []
    normal_rows   = []

    for _, row in df.iterrows():
        name = row.get("Name", "")
        if name in elevated_set:
            elevated_rows.append(row)
        else:
            normal_rows.append(row)

    # ── Elevated section ──────────────────────────────────────────────────────
    if elevated_rows:
        n = len(elevated_rows)
        label = f"Elevated · {n} commodity{'s' if n != 1 else ''} active"
        st.markdown(f'<div class="ac-grid-section-label">{label}</div>', unsafe_allow_html=True)

        cards_html = '<div class="ac-cards-grid">'
        for row in elevated_rows[:max_elevated]:
            name   = row.get("Name", "")
            trig   = lifecycle.elevated_by(name)
            cards_html += commodity_card_html(
                name       = name,
                price      = float(row.get("Price", 0) or 0),
                pct_change = float(row.get("Pct_Change", 0) or 0),
                sector     = str(row.get("Sector", "")).lower(),
                triggers   = trig,
            )
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

    else:
        # Show trigger-free status hint only if triggers are actually active
        if snap["active_count"] > 0:
            st.markdown(
                '<div class="ac-grid-section-label" style="color:rgba(238,242,255,0.15)">'
                'No primary-scope elevations from current triggers</div>',
                unsafe_allow_html=True,
            )

    # ── Standard section ──────────────────────────────────────────────────────
    if normal_rows:
        if elevated_rows:
            st.markdown(
                '<div class="ac-grid-section-label">All commodities</div>',
                unsafe_allow_html=True,
            )

        cards_html = '<div class="ac-cards-grid">'
        for row in normal_rows:
            name = row.get("Name", "")
            cards_html += commodity_card_html(
                name       = name,
                price      = float(row.get("Price", 0) or 0),
                pct_change = float(row.get("Pct_Change", 0) or 0),
                sector     = str(row.get("Sector", "")).lower(),
                triggers   = [],
            )
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)
