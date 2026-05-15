"""
Macro Exposure Heatmap  (Mission 8)
=====================================
Two complementary views of the current macro trigger surface:

  1. Cluster Exposure Strip  — one tile per sector showing aggregate elevation
  2. Trigger Impact Matrix   — rows=triggers × cols=clusters, cells coloured
                               by severity (primary scope) or dimly (secondary)

Both are rendered as pure HTML via ``st.markdown(unsafe_allow_html=True)``
so they precisely match the Accendio brand theme without Plotly overhead.

Public API
----------
  cluster_exposure_strip(lifecycle)  → None  (renders 5-cluster summary row)
  trigger_impact_matrix(lifecycle)   → None  (renders 2-D trigger/cluster grid)
  macro_exposure_heatmap(lifecycle)  → None  (renders both in sequence)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

from models.config import COMMODITY_SECTORS, MODELING_COMMODITIES
from services.trigger_lifecycle import LIFECYCLE, _PRIMARY_CLUSTERS

# ── Constants ──────────────────────────────────────────────────────────────────

_CLUSTER_ORDER   = ["energy", "metals", "agriculture", "livestock", "digital"]
_CLUSTER_DISPLAY = {
    "energy":      "Energy",
    "metals":      "Metals",
    "agriculture": "Agriculture",
    "livestock":   "Livestock",
    "digital":     "Digital",
}
_CLUSTER_ICON = {
    "energy":      "⚡",
    "metals":      "⚙",
    "agriculture": "🌾",
    "livestock":   "🐄",
    "digital":     "₿",
}

_SEV_RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
_SEV_REV  = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}

# Severity → colour tokens (Accendio palette)
_SEV_COLOURS = {
    "NONE":     {
        "bg":      "rgba(238,242,255,0.03)",
        "border":  "rgba(123,156,255,0.10)",
        "text":    "rgba(238,242,255,0.18)",
        "accent":  "rgba(238,242,255,0.12)",
        "bar":     "rgba(238,242,255,0.08)",
    },
    "LOW":      {
        "bg":      "rgba(238,242,255,0.05)",
        "border":  "rgba(238,242,255,0.22)",
        "text":    "rgba(238,242,255,0.55)",
        "accent":  "rgba(238,242,255,0.30)",
        "bar":     "rgba(238,242,255,0.20)",
    },
    "MEDIUM":   {
        "bg":      "rgba(245,158,11,0.08)",
        "border":  "rgba(245,158,11,0.40)",
        "text":    "#F59E0B",
        "accent":  "#F59E0B",
        "bar":     "#F59E0B",
    },
    "HIGH":     {
        "bg":      "rgba(224,112,32,0.10)",
        "border":  "rgba(224,112,32,0.50)",
        "text":    "#e07020",
        "accent":  "#e07020",
        "bar":     "#e07020",
    },
    "CRITICAL": {
        "bg":      "rgba(217,79,79,0.12)",
        "border":  "rgba(217,79,79,0.55)",
        "text":    "#D94F4F",
        "accent":  "#D94F4F",
        "bar":     "#D94F4F",
        "glow":    "0 0 12px rgba(217,79,79,0.28)",
    },
}

# Build cluster → list of commodity names (intersection with MODELING_COMMODITIES)
_MODELED = set(MODELING_COMMODITIES.keys())
_CLUSTER_COMMODITIES: Dict[str, List[str]] = {}
for _name, _sector in COMMODITY_SECTORS.items():
    if _name in _MODELED:
        _CLUSTER_COMMODITIES.setdefault(_sector, []).append(_name)


# ── Data computation ───────────────────────────────────────────────────────────

def _compute_cluster_exposure(snap: dict) -> List[dict]:
    """
    For each cluster return an exposure record:
      cluster, display, icon, highest_severity, aggregate_magnitude,
      trigger_count, elevated_count, total_count, triggers
    """
    elevated_set = set(snap["elevated_commodities"])
    active_triggers = snap["triggers"]

    results = []
    for cluster in _CLUSTER_ORDER:
        elevating = [t for t in active_triggers if cluster in t["primary_clusters"]]
        all_comms = _CLUSTER_COMMODITIES.get(cluster, [])
        elev_comms = [c for c in all_comms if c in elevated_set]

        if elevating:
            highest_sev = max(
                (t["severity"] for t in elevating),
                key=lambda s: _SEV_RANK.get(s, 0),
            )
            agg_magnitude = max(t["magnitude_score"] for t in elevating)
        else:
            highest_sev   = "NONE"
            agg_magnitude = 0.0

        results.append({
            "cluster":           cluster,
            "display":           _CLUSTER_DISPLAY[cluster],
            "icon":              _CLUSTER_ICON[cluster],
            "highest_severity":  highest_sev,
            "aggregate_magnitude": agg_magnitude,
            "trigger_count":     len(elevating),
            "elevated_count":    len(elev_comms),
            "total_count":       len(all_comms),
            "triggers":          elevating,
        })
    return results


# ── Component 1: Cluster Exposure Strip ────────────────────────────────────────

_STRIP_CSS = """
<style>
.ac-strip {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 8px;
  margin-top: 6px;
  font-family: Arial, 'Helvetica Neue', sans-serif;
}
.ac-strip-tile {
  border-radius: 8px;
  padding: 12px 14px 10px;
  border: 0.5px solid;
  position: relative;
  transition: box-shadow 0.3s;
}
.ac-tile-icon {
  font-size: 18px;
  line-height: 1;
  margin-bottom: 4px;
}
.ac-tile-label {
  font-size: 9px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.ac-tile-sev {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.ac-tile-stats {
  font-size: 9px;
  letter-spacing: 0.04em;
  line-height: 1.6;
}
.ac-tile-stat-val {
  font-size: 14px;
  font-weight: 300;
  line-height: 1;
  letter-spacing: -0.01em;
  margin-bottom: 2px;
}
.ac-tile-bar-track {
  height: 3px;
  background: rgba(238,242,255,0.06);
  border-radius: 2px;
  overflow: hidden;
  margin-top: 8px;
}
.ac-tile-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.5s ease;
}
.ac-tile-trigger-list {
  margin-top: 6px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.ac-tile-trigger-item {
  font-size: 8px;
  letter-spacing: 0.04em;
  padding: 1px 5px;
  border-radius: 3px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
"""


def cluster_exposure_strip(lifecycle=None) -> None:
    """Render the 5-cluster exposure summary strip."""
    if lifecycle is None:
        lifecycle = LIFECYCLE

    snap = lifecycle.state_snapshot()
    clusters = _compute_cluster_exposure(snap)

    st.markdown(_STRIP_CSS, unsafe_allow_html=True)

    tiles_html = '<div class="ac-strip">'
    for c in clusters:
        sev    = c["highest_severity"]
        cols   = _SEV_COLOURS[sev]
        mag    = int(c["aggregate_magnitude"] * 100)
        glow   = cols.get("glow", "none")
        n_trig = c["trigger_count"]
        n_elev = c["elevated_count"]
        n_tot  = c["total_count"]

        # Short trigger name pills
        trigger_pills = ""
        for t in c["triggers"][:3]:
            short = t["display_name"].split("/")[0].split("(")[0].strip()
            if len(short) > 20:
                short = short[:18] + "…"
            trigger_pills += (
                f'<div class="ac-tile-trigger-item" '
                f'style="background:{cols["bg"]};color:{cols["accent"]}">'
                f'{short}</div>'
            )

        sev_display = sev if sev != "NONE" else "—"
        sev_color   = cols["text"]

        tiles_html += f"""
<div class="ac-strip-tile"
     style="background:{cols['bg']};border-color:{cols['border']};box-shadow:{glow}">
  <div class="ac-tile-icon">{c['icon']}</div>
  <div class="ac-tile-label" style="color:{cols['text']}">{c['display']}</div>
  <div class="ac-tile-sev"   style="color:{sev_color}">{sev_display}</div>
  <div class="ac-tile-stat-val" style="color:{cols['accent']}">{n_elev}<span style="font-size:10px;color:{cols['text']}"> / {n_tot}</span></div>
  <div class="ac-tile-stats" style="color:{cols['text']}">commodities elevated</div>
  <div class="ac-tile-stats" style="color:rgba(238,242,255,0.25);margin-top:2px">{n_trig} trigger{'s' if n_trig != 1 else ''} active</div>
  <div class="ac-tile-bar-track">
    <div class="ac-tile-bar-fill" style="width:{mag}%;background:{cols['bar']}"></div>
  </div>
  {'<div class="ac-tile-trigger-list">' + trigger_pills + '</div>' if trigger_pills else ''}
</div>"""

    tiles_html += "</div>"
    st.markdown(tiles_html, unsafe_allow_html=True)


# ── Component 2: Trigger Impact Matrix ────────────────────────────────────────

# All 14 registered trigger types and their display names (fallback if registry unavailable)
_ALL_TRIGGERS = [
    ("FOMC_RATE_DECISION",         "FOMC Rate Decision"),
    ("CPI_RELEASE",                "CPI Release"),
    ("NONFARM_PAYROLLS",           "Non-Farm Payrolls"),
    ("OPEC_PRODUCTION_DECISION",   "OPEC+ Production"),
    ("EIA_CRUDE_INVENTORY",        "EIA Crude Inventory"),
    ("EIA_GAS_STORAGE",            "EIA Gas Storage"),
    ("USDA_WASDE_REPORT",          "USDA WASDE"),
    ("FED_CHAIR_SPEECH",           "Fed Chair Speech"),
    ("DOLLAR_SHOCK",               "USD Dollar Shock"),
    ("GEOPOLITICAL_SHOCK",         "Geopolitical Shock"),
    ("RECESSION_FLAG",             "Recession Flag"),
    ("WEATHER_SHOCK",              "Weather Shock"),
    ("ENERGY_TRANSITION_SIGNAL",   "Energy Transition"),
    ("PPI_RELEASE",                "PPI Release"),
]

_MATRIX_CSS = """
<style>
.ac-matrix-wrap {
  width: 100%;
  overflow-x: auto;
  font-family: Arial, 'Helvetica Neue', sans-serif;
  margin-top: 6px;
}
.ac-matrix {
  width: 100%;
  border-collapse: separate;
  border-spacing: 3px;
  min-width: 500px;
}
.ac-matrix th {
  font-size: 8.5px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: rgba(238,242,255,0.32);
  padding: 4px 8px;
  text-align: center;
  font-weight: 400;
  background: transparent;
  white-space: nowrap;
}
.ac-matrix th.row-header {
  text-align: right;
  padding-right: 10px;
  min-width: 130px;
}
.ac-matrix td {
  padding: 0;
  border-radius: 4px;
  height: 28px;
  min-width: 68px;
  text-align: center;
  vertical-align: middle;
  position: relative;
}
.ac-matrix-cell {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  border-radius: 4px;
  font-size: 8px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 600;
  padding: 0 6px;
  white-space: nowrap;
}
.ac-matrix-row-label {
  font-size: 10px;
  color: rgba(238,242,255,0.55);
  text-align: right;
  padding: 4px 10px 4px 4px;
  white-space: nowrap;
  border-radius: 4px;
}
.ac-matrix-row-label.active {
  color: #EEF2FF;
  font-weight: 500;
}
.ac-matrix-active-dot {
  display: inline-block;
  width: 5px; height: 5px;
  border-radius: 50%;
  margin-right: 5px;
  vertical-align: middle;
  flex-shrink: 0;
}
.ac-matrix-priority {
  font-size: 8px;
  color: rgba(123,156,255,0.5);
  margin-left: 5px;
}
.ac-matrix-row-label-inner {
  display: flex;
  align-items: center;
  justify-content: flex-end;
}
</style>
"""


def trigger_impact_matrix(lifecycle=None) -> None:
    """
    Render the 14×5 trigger/cluster impact matrix.

    Active triggers are highlighted in full colour.
    Inactive triggers are shown as ghosted outlines.
    Within each cell: primary scope = solid fill, secondary = hairline border only.
    """
    if lifecycle is None:
        lifecycle = LIFECYCLE

    snap = lifecycle.state_snapshot()
    active_map: Dict[str, dict] = {t["trigger_type"]: t for t in snap["triggers"]}

    # Try to load display names from registry
    try:
        from services.trigger_config import REGISTRY
        registry_names = {c.trigger_type: c.display_name for c in REGISTRY.all()}
    except Exception:
        registry_names = {}

    st.markdown(_MATRIX_CSS, unsafe_allow_html=True)

    # Table header
    col_icons = [_CLUSTER_ICON[c] for c in _CLUSTER_ORDER]
    col_names = [_CLUSTER_DISPLAY[c] for c in _CLUSTER_ORDER]

    header = "<tr><th class='row-header'>Trigger</th>"
    for icon, name in zip(col_icons, col_names):
        header += f"<th>{icon} {name}</th>"
    header += "</tr>"

    rows_html = ""
    for ttype, fallback_name in _ALL_TRIGGERS:
        is_active = ttype in active_map
        act_data  = active_map.get(ttype, {})
        display_name = registry_names.get(ttype, fallback_name)
        # Truncate long names
        if len(display_name) > 22:
            display_name = display_name[:20] + "…"

        primary_clusters = set(_PRIMARY_CLUSTERS.get(ttype, set()))
        # All 5 clusters are in secondary scope (all triggers hit all clusters)
        secondary_clusters = set(_CLUSTER_ORDER) - primary_clusters

        if is_active:
            sev      = act_data.get("severity", "LOW")
            mag      = act_data.get("magnitude_score", 0.0)
            priority = act_data.get("cascade_priority", 2)
            row_color = _SEV_COLOURS[sev]["text"]
            dot_color = _SEV_COLOURS[sev]["accent"]
            label_class = "ac-matrix-row-label active"
            dot_html = f'<span class="ac-matrix-active-dot" style="background:{dot_color}"></span>'
            pri_html = f'<span class="ac-matrix-priority">P{priority}</span>'
        else:
            sev = "NONE"
            mag = 0.0
            row_color = "rgba(238,242,255,0.25)"
            label_class = "ac-matrix-row-label"
            dot_html = ""
            pri_html = ""

        row = f"""
<tr>
  <td>
    <div class="{label_class}">
      <div class="ac-matrix-row-label-inner">
        {dot_html}
        <span style="color:{row_color}">{display_name}</span>
        {pri_html}
      </div>
    </div>
  </td>"""

        for cluster in _CLUSTER_ORDER:
            is_primary   = cluster in primary_clusters
            is_secondary = cluster in secondary_clusters

            if is_active and is_primary:
                cols  = _SEV_COLOURS[sev]
                glow  = cols.get("glow", "none")
                cell  = (
                    f'<td><div class="ac-matrix-cell" '
                    f'style="background:{cols["bg"]};color:{cols["text"]};'
                    f'border:0.5px solid {cols["border"]};box-shadow:{glow}">'
                    f'{sev}</div></td>'
                )
            elif is_active and is_secondary:
                # Secondary scope: show a dim hairline border, no fill
                cell = (
                    f'<td><div class="ac-matrix-cell" '
                    f'style="background:transparent;color:rgba(238,242,255,0.18);'
                    f'border:0.5px solid rgba(238,242,255,0.10)">'
                    f'2°</div></td>'
                )
            else:
                # Inactive: ghost cell
                cell = (
                    '<td><div class="ac-matrix-cell" '
                    'style="background:rgba(238,242,255,0.02);color:rgba(238,242,255,0.08);'
                    'border:0.5px solid rgba(238,242,255,0.06)">—</div></td>'
                )
            row += cell

        row += "</tr>"
        rows_html += row

    matrix_html = f"""
<div class="ac-matrix-wrap">
  <table class="ac-matrix">
    <thead>{header}</thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""

    # Legend
    legend_parts = []
    for label, bg, border, text in [
        ("Primary (active)",   "rgba(245,158,11,0.08)",  "rgba(245,158,11,0.4)",  "#F59E0B"),
        ("Secondary (active)", "transparent",             "rgba(238,242,255,0.10)","rgba(238,242,255,0.3)"),
        ("Inactive",           "rgba(238,242,255,0.02)", "rgba(238,242,255,0.06)","rgba(238,242,255,0.18)"),
    ]:
        legend_parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:16px;'
            f'font-size:9px;color:{text};letter-spacing:.06em">'
            f'<span style="display:inline-block;width:14px;height:10px;border-radius:2px;'
            f'background:{bg};border:0.5px solid {border}"></span>'
            f'{label}</span>'
        )

    legend_html = (
        f'<div style="margin-top:10px;display:flex;flex-wrap:wrap;align-items:center;gap:4px">'
        + "".join(legend_parts) + "</div>"
    )

    st.markdown(matrix_html + legend_html, unsafe_allow_html=True)


# ── Combined renderer ──────────────────────────────────────────────────────────

def macro_exposure_heatmap(lifecycle=None) -> None:
    """
    Render the full macro exposure heatmap:
      1. Cluster Exposure Strip  (5 sector tiles)
      2. Trigger Impact Matrix   (14×5 grid)

    Both components use the same lifecycle snapshot for consistency.
    """
    if lifecycle is None:
        lifecycle = LIFECYCLE

    cluster_exposure_strip(lifecycle)
    st.markdown(
        '<div style="margin-top:18px;font-size:9px;letter-spacing:.16em;'
        'text-transform:uppercase;color:rgba(238,242,255,0.22);margin-bottom:2px">'
        'Trigger impact matrix · primary scope vs secondary transmission</div>',
        unsafe_allow_html=True,
    )
    trigger_impact_matrix(lifecycle)
