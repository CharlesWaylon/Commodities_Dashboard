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
    apply_theme, render_topbar, panel_header,
    SIGNAL, ASCEND, DESCEND, AMBER, VOID, DEPTH, ABYSS, ICE, ICE_MID, BORDER,
)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Accendio · Event Ribbon",
    page_icon="assets/accendio_icon_transparent_32.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_theme()
render_topbar()


# ── Start broadcaster once ────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _start_broadcaster():
    BROADCASTER.start()
    return BROADCASTER


_broadcaster = _start_broadcaster()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<p style="font-size:10px;letter-spacing:.14em;color:{ICE_MID};'
        f'text-transform:uppercase;margin-bottom:8px">Event Ribbon</p>',
        unsafe_allow_html=True,
    )

    ws_port = st.number_input(
        "WS port", min_value=1024, max_value=65535, value=8765, step=1,
        help="Port the WebSocket broadcaster is listening on.",
    )
    ws_url = f"ws://localhost:{ws_port}"

    max_events = st.slider("Max cards in ribbon", min_value=3, max_value=20, value=10)

    st.markdown("---")

    # Broadcaster status
    running = _broadcaster.is_running
    clients = _broadcaster.client_count
    status_color = ASCEND if running else DESCEND
    st.markdown(
        f'<div style="font-size:10px;color:rgba(238,242,255,0.35);letter-spacing:.1em;'
        f'text-transform:uppercase;margin-bottom:4px">Broadcaster</div>'
        f'<div style="font-size:11px;color:{status_color}">'
        f'{"● Running" if running else "○ Offline"}'
        f'</div>'
        f'<div style="font-size:10px;color:rgba(238,242,255,0.3);margin-top:2px">'
        f'{clients} client{"s" if clients != 1 else ""} connected'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Signals route: TriggerBus → Broadcaster → WebSocket → Ribbon")


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    f'<h1 style="margin-bottom:4px">Macro Event Ribbon</h1>'
    f'<p style="font-size:12px;color:rgba(238,242,255,0.38);margin-bottom:0">'
    f'Real-time TriggerSignal feed · last {max_events} events · auto-scrolling</p>',
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)


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


# ── Demo event injector ───────────────────────────────────────────────────────

st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

with st.expander("Inject test event  (development)", expanded=False):
    st.caption(
        "Pushes a synthetic TriggerSignal payload directly to the broadcaster. "
        "Use this when no real macro data is flowing to verify the ribbon renders correctly."
    )

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

    c1, c2, c3 = st.columns(3)
    with c1:
        sel_idx = st.selectbox(
            "Event type", range(len(_TRIGGER_OPTIONS)),
            format_func=lambda i: _TRIGGER_OPTIONS[i],
            key="demo_trigger_select",
        )
        trigger_type = _TRIGGER_TYPES[sel_idx]
        display_name = _TRIGGER_OPTIONS[sel_idx]
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

    if st.button("Broadcast test event", type="primary", key="demo_fire"):
        now = datetime.now(timezone.utc)
        payload = json.dumps({
            "signal_id":                str(uuid.uuid4()),
            "trigger_type":             trigger_type,
            "display_name":             display_name,
            "severity":                 severity,
            "magnitude_score":          round(magnitude, 4),
            "direction":                direction,
            "affected_clusters":        _CLUSTER_MAP.get(trigger_type, ["energy", "metals"]),
            "affected_commodity_count": 48,
            "cascade_priority":         _PRIORITY_MAP.get(trigger_type, 2),
            "classified_at":            now.isoformat(),
            "decay_expires_at":         (now + timedelta(hours=24)).isoformat(),
            "is_repeat":                False,
            "classification_method":    "z_score",
            "source":                   "demo",
            "short_description":        f"Test event: {display_name}",
        })
        if _broadcaster.is_running:
            _broadcaster.broadcast_raw(payload)
            st.success(
                f"Broadcast: {display_name} ({severity}) → "
                f"{_broadcaster.client_count} client(s) connected",
            )
        else:
            st.error(
                "Broadcaster is not running. Check that `websockets` is installed "
                "(`pip install 'websockets>=12.0'`) and refresh the page."
            )

    with st.expander("Stress test — broadcast N random events"):
        n_events = st.slider("Number of events", 1, 20, 5, key="stress_n")
        if st.button("Fire random events", key="stress_fire"):
            if not _broadcaster.is_running:
                st.error("Broadcaster not running.")
            else:
                for _ in range(n_events):
                    now = datetime.now(timezone.utc)
                    tidx = random.randint(0, len(_TRIGGER_TYPES) - 1)
                    ttype = _TRIGGER_TYPES[tidx]
                    sev   = random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
                    mag   = round(random.uniform(0.15, 0.98), 4)
                    pld   = json.dumps({
                        "signal_id":                str(uuid.uuid4()),
                        "trigger_type":             ttype,
                        "display_name":             _TRIGGER_OPTIONS[tidx],
                        "severity":                 sev,
                        "magnitude_score":          mag,
                        "direction":                random.choice(["upside_surprise", "downside_surprise", "neutral"]),
                        "affected_clusters":        _CLUSTER_MAP.get(ttype, ["energy"]),
                        "affected_commodity_count": random.randint(8, 48),
                        "cascade_priority":         _PRIORITY_MAP.get(ttype, 2),
                        "classified_at":            now.isoformat(),
                        "decay_expires_at":         (now + timedelta(hours=12)).isoformat(),
                        "is_repeat":                random.random() < 0.15,
                        "classification_method":    "z_score",
                        "source":                   "demo",
                        "short_description":        "",
                    })
                    _broadcaster.broadcast_raw(pld)
                st.success(f"Fired {n_events} random events.")


# ── Integration guide ─────────────────────────────────────────────────────────

with st.expander("Wire into live signal pipeline", expanded=False):
    st.markdown("""
**Step 1 — Start the broadcaster when your app loads:**
```python
from services.ws_broadcast import BROADCASTER
BROADCASTER.start()
```

**Step 2 — Register it as a reset handler on the lifecycle manager:**
```python
from services.trigger_lifecycle import LIFECYCLE
LIFECYCLE.register_reset_handler(
    lambda reset: None   # lifecycle resets don't go to ribbon
)
```

**Step 3 — Hook the cascade dispatcher to broadcast each classified signal:**
```python
from services.cascade_handlers import DISPATCHER
from services.ws_broadcast import broadcast_signal

# After classify_event(event) returns a TriggerSignal:
signal = classify_event(event)
if signal:
    broadcast_signal(signal)          # → ribbon
    LIFECYCLE.activate(signal)        # → lifecycle manager
    DISPATCHER.dispatch(signal)       # → model handlers
```

**Step 4 — Add the ribbon component to any page:**
```python
from components.event_ribbon import event_ribbon
event_ribbon(ws_url="ws://localhost:8765")
```
    """)
