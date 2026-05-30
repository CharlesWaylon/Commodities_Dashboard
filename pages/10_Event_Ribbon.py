"""
Macro Event Ribbon — Page 10
==============================
Real-time feed of incoming TriggerSignal events, rendered as an animated
horizontal ribbon.  Also shows the Trigger Lifecycle Manager's current state
(which triggers are active and what commodities they are elevating).

Architecture
------------
  TriggerBus  ──►  TriggerBroadcaster (WS server, bg thread)
                          │  ws://localhost:8765
                      browser JS  ──►  event_ribbon() component

The broadcaster is started via @st.cache_resource so it survives page
navigation and reruns.

Demo panel
----------
If no real trigger signals are arriving (e.g. during development), use the
"Inject test event" panel to push a synthetic payload directly to the
broadcaster.
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone, timedelta

import streamlit as st

from components.event_ribbon import event_ribbon
from services.trigger_lifecycle import LIFECYCLE
from services.ws_broadcast import BROADCASTER
from utils.theme import (
    apply_theme, render_topbar, render_sidebar_nav, panel_header,
    SIGNAL, ASCEND, DESCEND, AMBER, VOID, DEPTH, ABYSS, ICE, ICE_MID, BORDER,
)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Accendio · Event Ribbon",
    page_icon="assets/accendio_icon_transparent_32.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
render_topbar()
render_sidebar_nav()


# ── Start broadcaster once ────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _start_broadcaster():
    BROADCASTER.start()
    return BROADCASTER


_broadcaster = _start_broadcaster()


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    f'<h1 style="margin-bottom:4px">Macro Event Ribbon</h1>'
    f'<p style="font-size:12px;color:rgba(238,242,255,0.38);margin-bottom:0">'
    f'Real-time TriggerSignal feed · auto-scrolling</p>',
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)


# ── Ribbon controls + broadcaster status (moved out of sidebar) ───────────────

_cc1, _cc2, _cc3 = st.columns([1, 1, 2])
with _cc1:
    ws_port = st.number_input(
        "WS port", min_value=1024, max_value=65535, value=8765, step=1,
        help="Port the WebSocket broadcaster is listening on.",
    )
    ws_url = f"ws://localhost:{ws_port}"
with _cc2:
    max_events = st.slider("Max cards in ribbon", min_value=3, max_value=20, value=10)
with _cc3:
    running = _broadcaster.is_running
    clients = _broadcaster.client_count
    status_color = ASCEND if running else DESCEND
    st.markdown(
        f'<div style="padding:10px 14px;background:{DEPTH};border:0.5px solid {BORDER};'
        f'border-radius:6px">'
        f'<div style="font-size:10px;color:rgba(238,242,255,0.35);letter-spacing:.1em;'
        f'text-transform:uppercase;margin-bottom:4px">Broadcaster</div>'
        f'<div style="font-size:12px;color:{status_color}">'
        f'{"● Running" if running else "○ Offline"} · '
        f'{clients} client{"s" if clients != 1 else ""} connected</div>'
        f'<div style="font-size:10px;color:rgba(238,242,255,0.28);margin-top:2px">'
        f'Signals route: TriggerBus → Broadcaster → WebSocket → Ribbon</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)


# ── Live ribbon ───────────────────────────────────────────────────────────────

panel_header("LIVE EVENT FEED", badge=f"ws://{ws_url.replace('ws://', '')}")
event_ribbon(ws_url=ws_url, height=200, max_events=max_events)


# ── Lifecycle state ───────────────────────────────────────────────────────────

st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
panel_header("ACTIVE TRIGGERS", badge="Lifecycle Manager")

snap = LIFECYCLE.state_snapshot()

col_a, col_b, col_c = st.columns(3)
col_a.metric("Active Triggers",      snap["active_count"])
col_b.metric("Elevated Commodities", len(snap["elevated_commodities"]))
col_c.metric("WS Clients",           _broadcaster.client_count)

if snap["active_count"] == 0:
    st.markdown(
        f'<div style="color:rgba(238,242,255,0.22);font-size:12px;'
        f'text-align:center;padding:24px 0;letter-spacing:.06em">'
        f'No triggers currently active</div>',
        unsafe_allow_html=True,
    )
else:
    import pandas as pd

    rows = []
    for t in snap["triggers"]:
        ttl_min = t["remaining_ttl_seconds"] / 60
        rows.append({
            "Trigger":        t["display_name"],
            "Severity":       t["severity"],
            "Magnitude":      f"{t['magnitude_score']:.0%}",
            "Direction":      t["direction"].replace("_", " "),
            "Primary Clusters": ", ".join(t["primary_clusters"]),
            "Commodities ↑":  t["primary_commodity_count"],
            "TTL (min)":      f"{ttl_min:.1f}",
            "Priority":       f"P{t['cascade_priority']}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Magnitude":      st.column_config.TextColumn(width="small"),
            "TTL (min)":      st.column_config.TextColumn(width="small"),
            "Priority":       st.column_config.TextColumn(width="small"),
            "Commodities ↑":  st.column_config.NumberColumn(width="small"),
        },
    )

    if snap["elevated_commodities"]:
        st.markdown(
            f'<p style="font-size:10px;color:rgba(238,242,255,0.28);'
            f'letter-spacing:.08em;text-transform:uppercase;margin-top:12px;margin-bottom:4px">'
            f'Elevated commodities</p>',
            unsafe_allow_html=True,
        )
        pills = " ".join(
            f'<span style="display:inline-block;font-size:10px;padding:2px 8px;'
            f'margin:2px;border-radius:4px;background:rgba(123,156,255,0.07);'
            f'color:rgba(123,156,255,0.6);border:0.5px solid rgba(123,156,255,0.18)">'
            f'{c}</span>'
            for c in snap["elevated_commodities"]
        )
        st.markdown(pills, unsafe_allow_html=True)


# ── Synthetic trigger: data & helpers ─────────────────────────────────────────

_SYNTHETIC_TTL_MINUTES = 20

_TRIGGER_OPTIONS = [
    "FOMC Rate Decision",
    "CPI Inflation Release",
    "OPEC+ Production Decision",
    "Geopolitical Supply Shock",
    "USDA WASDE Crop Report",
    "Non-Farm Payrolls Report",
    "EIA Weekly Crude Inventory Report",
    "USD Dollar Index Shock (DXY)",
    "NBER Recession Indicator (USREC)",
    "Weather / ENSO Climate Shock",
    "Energy Transition / Battery Metals Signal",
    "PPI Producer Price Index Release",
    "EIA Weekly Natural Gas Storage Report",
    "Fed Chair Speech / Press Conference",
]

_TRIGGER_TYPES = [
    "FOMC_RATE_DECISION", "CPI_RELEASE", "OPEC_PRODUCTION_DECISION",
    "GEOPOLITICAL_SHOCK", "USDA_WASDE_REPORT", "NONFARM_PAYROLLS",
    "EIA_CRUDE_INVENTORY", "DOLLAR_SHOCK", "RECESSION_FLAG",
    "WEATHER_SHOCK", "ENERGY_TRANSITION_SIGNAL", "PPI_RELEASE",
    "EIA_GAS_STORAGE", "FED_CHAIR_SPEECH",
]

_TYPE_TO_FAMILY = {
    "FOMC_RATE_DECISION": "fomc_rate_decision", "CPI_RELEASE": "cpi_release",
    "OPEC_PRODUCTION_DECISION": "opec_action", "GEOPOLITICAL_SHOCK": "geopolitical_shock",
    "USDA_WASDE_REPORT": "usda_wasde_report", "NONFARM_PAYROLLS": "nonfarm_payrolls",
    "EIA_CRUDE_INVENTORY": "eia_crude_inventory", "DOLLAR_SHOCK": "fed_tightening",
    "RECESSION_FLAG": "recession_flag", "WEATHER_SHOCK": "weather_shock",
    "ENERGY_TRANSITION_SIGNAL": "energy_transition", "PPI_RELEASE": "ppi_release",
    "EIA_GAS_STORAGE": "eia_gas_storage", "FED_CHAIR_SPEECH": "fed_chair_speech",
}

_CLUSTER_MAP = {
    "FOMC_RATE_DECISION":       ["energy", "metals", "agriculture", "livestock", "digital"],
    "CPI_RELEASE":              ["energy", "metals", "agriculture", "livestock", "digital"],
    "OPEC_PRODUCTION_DECISION": ["energy", "metals", "agriculture", "livestock", "digital"],
    "GEOPOLITICAL_SHOCK":       ["energy", "metals", "agriculture", "livestock", "digital"],
    "USDA_WASDE_REPORT":        ["agriculture", "energy", "livestock", "metals", "digital"],
    "NONFARM_PAYROLLS":         ["metals", "energy", "agriculture", "livestock", "digital"],
    "EIA_CRUDE_INVENTORY":      ["energy", "metals", "agriculture", "livestock", "digital"],
    "DOLLAR_SHOCK":             ["metals", "energy", "agriculture", "livestock", "digital"],
    "RECESSION_FLAG":           ["energy", "metals", "agriculture", "livestock", "digital"],
    "WEATHER_SHOCK":            ["agriculture", "energy", "metals", "livestock", "digital"],
    "ENERGY_TRANSITION_SIGNAL": ["metals", "energy", "agriculture", "livestock", "digital"],
    "PPI_RELEASE":              ["energy", "metals", "agriculture", "livestock", "digital"],
    "EIA_GAS_STORAGE":          ["energy", "metals", "agriculture", "livestock", "digital"],
    "FED_CHAIR_SPEECH":         ["metals", "energy", "agriculture", "livestock", "digital"],
}

_PRIORITY_MAP = {
    "FOMC_RATE_DECISION": 1, "CPI_RELEASE": 1, "OPEC_PRODUCTION_DECISION": 1,
    "GEOPOLITICAL_SHOCK": 1, "RECESSION_FLAG": 1, "NONFARM_PAYROLLS": 2,
    "USDA_WASDE_REPORT": 2, "DOLLAR_SHOCK": 2, "WEATHER_SHOCK": 2,
    "FED_CHAIR_SPEECH": 2, "EIA_CRUDE_INVENTORY": 2, "EIA_GAS_STORAGE": 3,
    "ENERGY_TRANSITION_SIGNAL": 3, "PPI_RELEASE": 3,
}

# Maps families to the macro-snapshot features the cascade amplifies.
_FAMILY_TO_AMPLIFIED_FEATURES = {
    "fed_tightening":     ["DXY return", "TLT return", "TLT yield proxy"],
    "fomc_rate_decision": ["DXY return", "TLT return", "TLT yield proxy"],
    "fed_chair_speech":   ["DXY return", "TLT return"],
    "cpi_release":        ["DXY return", "TLT yield proxy"],
    "ppi_release":        ["DXY return", "TLT yield proxy"],
    "nonfarm_payrolls":   ["VIX 5d return", "TLT return"],
    "recession_flag":     ["VIX 5d return", "TLT return", "TLT yield proxy"],
    "geopolitical_shock": ["VIX 5d return"],
}

# Maps families to the upstream sector path dampened in the sector model.
_FAMILY_TO_UPSTREAM = {
    "opec_action":         "Energy",
    "eia_crude_inventory": "Energy",
    "eia_gas_storage":     "Energy",
    "energy_transition":   "Energy",
    "geopolitical_shock":  "Energy",
    "weather_shock":       "Agriculture",
    "usda_wasde_report":   "Agriculture",
}

# Risk gates from models/config.py — copied here for display only.
_RISK_GATES = {
    "fed_tightening": "Flatten toward equal weight (20% blend) — diversify under rate shock",
    "weather_shock":  "Cap Agriculture allocation at 1.5x equal weight",
    "opec_action":    "Cap Energy allocation at 1.5x equal weight",
}
_ANY_STRONG_GATE = "Damp 30% toward yesterday's portfolio (any trigger at strength >= 0.9)"


from utils.synthetic_triggers import (
    cleanup_expired_synthetics, get_active_synthetics,
    clear_all_synthetics, render_synthetic_banner,
    SYNTHETIC_TTL_MINUTES,
)


def _inject_synthetic_trigger(trigger_type: str, family: str, strength: float,
                              display_name: str, direction: str,
                              clusters: list[str], severity: str) -> bool:
    """Write a synthetic trigger to the DB and push through LIFECYCLE + AlertEngine."""
    from database.db import get_db
    from database.models import TriggerEvent

    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=SYNTHETIC_TTL_MINUTES)
    today = now.strftime("%Y-%m-%d")
    meta = json.dumps({
        "synthetic": True,
        "synthetic_expires_at": expires.isoformat(),
        "direction": direction,
        "affected_clusters": clusters,
        "display_name": display_name,
        "trigger_type": trigger_type,
    })

    try:
        with get_db() as db:
            existing = (
                db.query(TriggerEvent)
                .filter(TriggerEvent.family == family,
                        TriggerEvent.trigger_date == today,
                        TriggerEvent.rationale.like("[SYNTHETIC]%"))
                .first()
            )
            if existing:
                existing.strength = strength
                existing.detected_at = now.isoformat()
                existing.inserted_at = now.isoformat()
                existing.rationale = f"[SYNTHETIC] {display_name}"
                existing.trigger_metadata = meta
            else:
                db.add(TriggerEvent(
                    detected_at=now.isoformat(),
                    trigger_date=today,
                    family=family,
                    strength=strength,
                    rationale=f"[SYNTHETIC] {display_name}",
                    affected_commodities=json.dumps([]),
                    trigger_metadata=meta,
                    inserted_at=now.isoformat(),
                ))
            db.commit()
    except Exception as exc:
        st.error(f"DB write failed: {exc}")
        return False

    # Push through LIFECYCLE so Macro Exposure page reacts
    try:
        from services.trigger_classifier import TriggerSignal, SeverityTier
        from services.trigger_config import REGISTRY

        cfg = REGISTRY.get(trigger_type)
        if cfg is None:
            return True

        sev_map = {"LOW": SeverityTier.LOW, "MEDIUM": SeverityTier.MEDIUM,
                   "HIGH": SeverityTier.HIGH, "CRITICAL": SeverityTier.CRITICAL}

        from models.config import MODELING_COMMODITIES
        all_comms = tuple(MODELING_COMMODITIES.keys())

        sig = TriggerSignal(
            signal_id=str(uuid.uuid4()),
            trigger_type=trigger_type,
            family_name=family,
            source_event_id=f"synthetic_{uuid.uuid4().hex[:8]}",
            magnitude_score=strength,
            severity=sev_map.get(severity, SeverityTier.HIGH),
            direction=direction,
            release_timestamp=now.isoformat(),
            classified_at=now.isoformat(),
            decay_expires_at=expires.isoformat(),
            affected_commodities=all_comms,
            affected_clusters=tuple(clusters),
            model_targets=cfg.model_targets,
            cascade_priority=cfg.cascade_priority,
            classification_method="synthetic",
            raw_deviation_score=strength * 4.0,
            event_type=trigger_type,
            actual_value=strength,
            expected_value=0.0,
            is_repeat=False,
            config=cfg,
        )

        LIFECYCLE.activate(sig)

        # Push through AlertEngine so Alerts page reacts
        try:
            from services.alert_engine import ENGINE
            from components.notification_panel import drain_pending
            ENGINE.evaluate(sig)
            drain_pending(ENGINE)
        except Exception:
            pass

    except Exception:
        pass

    return True


# Clean up expired synthetics on every page load
cleanup_expired_synthetics()


# ── Synthetic trigger injector ───────────────────────────────────────────────

st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
st.divider()

st.markdown(
    f'<h2 style="margin-bottom:2px">What-If Scenario Trigger</h2>'
    f'<p style="font-size:12px;color:rgba(238,242,255,0.35);margin-bottom:0">'
    f'Inject a synthetic macro event to see how the dashboard responds. '
    f'Auto-expires after {SYNTHETIC_TTL_MINUTES} minutes.</p>',
    unsafe_allow_html=True,
)
st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    sel_idx = st.selectbox(
        "Event type", range(len(_TRIGGER_OPTIONS)),
        format_func=lambda i: _TRIGGER_OPTIONS[i],
        key="demo_trigger_select",
    )
    trigger_type = _TRIGGER_TYPES[sel_idx]
    display_name = _TRIGGER_OPTIONS[sel_idx]
    family_name  = _TYPE_TO_FAMILY[trigger_type]
with c2:
    severity = st.selectbox(
        "Severity", ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        index=2, key="demo_severity",
    )
with c3:
    direction = st.selectbox(
        "Direction",
        ["downside_surprise", "upside_surprise", "neutral"],
        key="demo_direction",
    )

magnitude = st.slider(
    "Magnitude score", 0.0, 1.0,
    value={"LOW": 0.25, "MEDIUM": 0.52, "HIGH": 0.74, "CRITICAL": 0.91}[severity],
    step=0.01, key="demo_magnitude",
)

clusters = _CLUSTER_MAP.get(trigger_type, ["energy", "metals"])

if st.button("Inject scenario trigger", type="primary", key="demo_fire"):
    ok = _inject_synthetic_trigger(
        trigger_type, family_name, magnitude, display_name,
        direction, clusters, severity,
    )
    if ok:
        # Also broadcast to ribbon for visual feedback
        now = datetime.now(timezone.utc)
        payload = json.dumps({
            "signal_id":                str(uuid.uuid4()),
            "trigger_type":             trigger_type,
            "display_name":             display_name,
            "severity":                 severity,
            "magnitude_score":          round(magnitude, 4),
            "direction":                direction,
            "affected_clusters":        clusters,
            "affected_commodity_count": 48,
            "cascade_priority":         _PRIORITY_MAP.get(trigger_type, 2),
            "classified_at":            now.isoformat(),
            "decay_expires_at":         (now + timedelta(minutes=SYNTHETIC_TTL_MINUTES)).isoformat(),
            "is_repeat":                False,
            "classification_method":    "synthetic",
            "source":                   "synthetic",
            "short_description":        f"Scenario: {display_name}",
        })
        if _broadcaster.is_running:
            _broadcaster.broadcast_raw(payload)
        st.rerun()


# ── Impact preview panel ─────────────────────────────────────────────────────

active_synthetics = get_active_synthetics()

if active_synthetics:
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    panel_header("SCENARIO IMPACT PREVIEW", badge=f"{len(active_synthetics)} active")

    for synth in active_synthetics:
        fam  = synth["family"]
        mag  = synth["strength"]
        name = synth["display_name"]
        remaining = synth["remaining_min"]

        from features.macro_features import family_to_regime
        regime = family_to_regime(fam)
        regime_color = {
            "rate_shock":      "#F59E0B",
            "commodity_shock": "#e07020",
            "growth_shock":    "#D94F4F",
            "neutral":         "rgba(238,242,255,0.4)",
        }.get(regime, "rgba(238,242,255,0.4)")

        # Build impact items
        impact_rows = ""

        # Regime
        impact_rows += (
            f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
            f'border-bottom:0.5px solid rgba(123,156,255,0.08)">'
            f'<span style="color:rgba(238,242,255,0.5);font-size:11px">Regime override</span>'
            f'<span style="color:{regime_color};font-size:11px;font-weight:500">'
            f'{regime.replace("_", " ").title()}</span>'
            f'</div>'
        )

        # Affected clusters
        cluster_pills = " ".join(
            f'<span style="display:inline-block;font-size:10px;padding:1px 6px;'
            f'margin:1px;border-radius:3px;background:rgba(123,156,255,0.07);'
            f'color:rgba(123,156,255,0.65);border:0.5px solid rgba(123,156,255,0.15)">'
            f'{c.title()}</span>'
            for c in synth["clusters"]
        )
        impact_rows += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 0;border-bottom:0.5px solid rgba(123,156,255,0.08)">'
            f'<span style="color:rgba(238,242,255,0.5);font-size:11px">Affected clusters</span>'
            f'<span>{cluster_pills}</span>'
            f'</div>'
        )

        # Cascade amplification
        amp_features = _FAMILY_TO_AMPLIFIED_FEATURES.get(fam)
        if amp_features:
            amp_str = ", ".join(amp_features)
            impact_rows += (
                f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
                f'border-bottom:0.5px solid rgba(123,156,255,0.08)">'
                f'<span style="color:rgba(238,242,255,0.5);font-size:11px">Cascade amplification</span>'
                f'<span style="color:rgba(238,242,255,0.65);font-size:11px">{amp_str}</span>'
                f'</div>'
            )

        # Upstream sector damping
        upstream = _FAMILY_TO_UPSTREAM.get(fam)
        if upstream:
            impact_rows += (
                f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
                f'border-bottom:0.5px solid rgba(123,156,255,0.08)">'
                f'<span style="color:rgba(238,242,255,0.5);font-size:11px">Upstream sector damping</span>'
                f'<span style="color:{AMBER};font-size:11px">{upstream} paths intensified</span>'
                f'</div>'
            )

        # Portfolio risk gate
        gate = _RISK_GATES.get(fam)
        if gate:
            impact_rows += (
                f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
                f'border-bottom:0.5px solid rgba(123,156,255,0.08)">'
                f'<span style="color:rgba(238,242,255,0.5);font-size:11px">Portfolio risk gate</span>'
                f'<span style="color:{AMBER};font-size:11px">{gate}</span>'
                f'</div>'
            )
        if mag >= 0.9:
            impact_rows += (
                f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
                f'border-bottom:0.5px solid rgba(123,156,255,0.08)">'
                f'<span style="color:rgba(238,242,255,0.5);font-size:11px">Turnover damper</span>'
                f'<span style="color:{DESCEND};font-size:11px">{_ANY_STRONG_GATE}</span>'
                f'</div>'
            )

        # Models affected
        models_hit = ["Cascade Orchestrator", "Macro Router"]
        if upstream:
            models_hit.append("Sector Model")
        models_hit.append("Meta-Predictor")
        if gate or mag >= 0.9:
            models_hit.append("Portfolio Optimizer")
        model_str = " → ".join(models_hit)
        impact_rows += (
            f'<div style="display:flex;justify-content:space-between;padding:6px 0">'
            f'<span style="color:rgba(238,242,255,0.5);font-size:11px">Model pipeline</span>'
            f'<span style="color:rgba(238,242,255,0.55);font-size:11px">{model_str}</span>'
            f'</div>'
        )

        st.markdown(
            f'<div style="background:{DEPTH};border:0.5px solid {BORDER};'
            f'border-radius:8px;padding:14px 18px;margin-bottom:12px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-bottom:10px">'
            f'<div>'
            f'<span style="font-size:13px;color:{ICE};font-weight:400">{name}</span>'
            f'<span style="font-size:10px;color:rgba(238,242,255,0.3);margin-left:10px">'
            f'strength {mag:.0%} · {synth["direction"].replace("_", " ")}</span>'
            f'</div>'
            f'<span style="font-size:10px;color:rgba(238,242,255,0.25)">'
            f'{remaining:.0f} min remaining</span>'
            f'</div>'
            f'{impact_rows}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Drill-deeper links
    st.markdown(
        f'<div style="margin-top:8px;font-size:11px;color:rgba(238,242,255,0.35)">'
        f'Explore deeper: ',
        unsafe_allow_html=True,
    )
    _link_cols = st.columns(4)
    with _link_cols[0]:
        st.page_link("pages/7_Macro_Market_Cascade.py", label="Macro-Market Cascade")
    with _link_cols[1]:
        st.page_link("pages/8_Portfolio.py", label="Portfolio")
    with _link_cols[2]:
        st.page_link("pages/11_Macro_Exposure.py", label="Macro Exposure")
    with _link_cols[3]:
        st.page_link("pages/12_Alerts.py", label="Alerts")

    # Manual clear button
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    if st.button("Clear all synthetic triggers", key="clear_synthetics"):
        clear_all_synthetics()
        st.rerun()
else:
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    st.caption(
        f"No active scenario triggers. Inject one above to preview its impact across the "
        f"dashboard. Synthetic events auto-expire after {SYNTHETIC_TTL_MINUTES} minutes."
    )
