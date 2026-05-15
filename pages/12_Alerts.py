"""
Alerts — Page 12
=================
Dashboard page for the Mission 9 Alert & Notification Layer.

  Panel A  Live Notification Panel  — drains ENGINE.pending_queue, shows cards
  Panel B  Rule Management          — toggle / inspect DEFAULT_RULES, add custom
  Panel C  Alert History            — full session history with filter controls
  Sidebar  Test Event Injector      — fire synthetic HIGH/CRITICAL signals

The page wires the AlertEngine to the TriggerBus so that real signals from
the classifier are automatically evaluated.  Demo events can also be injected
via the sidebar without a running bus.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import streamlit as st

from components.notification_panel import drain_pending, notification_panel, unread_count
from services.alert_engine import (
    Alert, AlertRule, DEFAULT_RULES, ENGINE, SeverityTier,
)
from utils.theme import (
    apply_theme, render_topbar, panel_header,
    SIGNAL, ASCEND, DESCEND, AMBER, ICE, ICE_MID,
    VOID, DEPTH, ABYSS, BORDER,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Accendio · Alerts",
    page_icon="assets/accendio_icon_transparent_32.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
render_topbar()


# ── Drain pending alerts from engine queue (runs on every render) ─────────────

new_count = drain_pending(ENGINE)


# ── Sidebar — test event injector ─────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<p style="font-size:10px;letter-spacing:.14em;color:{ICE_MID};'
        f'text-transform:uppercase;margin-bottom:8px">Alert Engine</p>',
        unsafe_allow_html=True,
    )

    # Engine status
    rules_enabled = sum(1 for r in ENGINE.rules() if r.enabled)
    unread        = unread_count()
    status_color  = ASCEND if unread > 0 else "rgba(238,242,255,0.25)"

    st.markdown(
        f'<div style="font-size:10px;color:rgba(238,242,255,0.32);letter-spacing:.1em;'
        f'text-transform:uppercase;margin-bottom:6px">Status</div>'
        f'<div style="font-size:22px;font-weight:300;color:{status_color};line-height:1">'
        f'{unread}</div>'
        f'<div style="font-size:10px;color:rgba(238,242,255,0.3)">'
        f'unread alert{"s" if unread != 1 else ""}</div>'
        f'<div style="font-size:10px;color:rgba(238,242,255,0.25);margin-top:4px">'
        f'{rules_enabled} active rule{"s" if rules_enabled != 1 else ""}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        f'<p style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;'
        f'color:rgba(238,242,255,0.22);margin-bottom:8px">Test Event Injector</p>',
        unsafe_allow_html=True,
    )

    TRIGGER_TYPES = [
        "FOMC_RATE_DECISION", "CPI_RELEASE", "NONFARM_PAYROLLS",
        "OPEC_PRODUCTION_DECISION", "EIA_CRUDE_INVENTORY", "EIA_GAS_STORAGE",
        "USDA_WASDE_REPORT", "FED_CHAIR_SPEECH", "DOLLAR_SHOCK",
        "GEOPOLITICAL_SHOCK", "RECESSION_FLAG", "WEATHER_SHOCK",
        "ENERGY_TRANSITION_SIGNAL", "PPI_RELEASE",
    ]

    sel_trigger = st.selectbox(
        "Trigger type", TRIGGER_TYPES, key="inj_trigger_type"
    )
    sel_severity = st.selectbox(
        "Severity", ["HIGH", "CRITICAL"], key="inj_severity"
    )
    sel_direction = st.selectbox(
        "Direction", ["downside_surprise", "upside_surprise", "neutral"],
        key="inj_direction",
    )
    sel_magnitude = st.slider(
        "Magnitude", 0.0, 1.0, 0.75, 0.05, key="inj_magnitude"
    )

    if st.button("Inject Alert Event", type="primary", use_container_width=True):
        # Build a minimal synthetic TriggerSignal and feed it to the engine
        try:
            from services.trigger_classifier import TriggerSignal, SeverityTier as ST
            from services.trigger_config import REGISTRY

            cfg = REGISTRY.get(sel_trigger)
            if cfg is None:
                st.error(f"Trigger type '{sel_trigger}' not found in registry.")
            else:
                sev_map = {"HIGH": ST.HIGH, "CRITICAL": ST.CRITICAL}
                sev     = sev_map[sel_severity]
                now     = datetime.now(timezone.utc)

                from models.config import MODELING_COMMODITIES
                from datetime import timedelta as _td
                all_comms = tuple(MODELING_COMMODITIES.keys())

                sig = TriggerSignal(
                    signal_id=str(uuid.uuid4()),
                    trigger_type=sel_trigger,
                    family_name=cfg.family_name,
                    source_event_id=f"inject_{uuid.uuid4().hex[:8]}",
                    magnitude_score=sel_magnitude,
                    severity=sev,
                    direction=sel_direction,
                    release_timestamp=now.isoformat(),
                    classified_at=now.isoformat(),
                    decay_expires_at=(now + _td(minutes=cfg.decay_window_minutes)).isoformat(),
                    affected_commodities=all_comms,
                    affected_clusters=list(cfg.affected_clusters),
                    model_targets=cfg.model_targets,
                    cascade_priority=1 if sel_severity == "CRITICAL" else 2,
                    classification_method="manual_injection",
                    raw_deviation_score=sel_magnitude * 4.0,
                    event_type=sel_trigger,
                    actual_value=sel_magnitude,
                    expected_value=0.0,
                    is_repeat=False,
                    config=cfg,
                )

                fired = ENGINE.evaluate(sig)
                new_count_after = drain_pending(ENGINE)

                if fired:
                    st.success(f"Fired {len(fired)} alert(s). Panel will show them above.")
                else:
                    st.info(
                        "No alerts fired. Either no rules matched, or this "
                        "trigger+rule was already fired within its dedup window. "
                        "Try a different trigger type or use 'Reset Dedup Cache' below."
                    )
                st.rerun()

        except Exception as exc:
            st.error(f"Injection failed: {exc}")

    st.markdown("---")

    if st.button("Reset Dedup Cache", use_container_width=True,
                 help="Flush the alert engine deduplication store so the same trigger can fire again"):
        ENGINE.flush_dedup_cache()
        st.success("Dedup cache cleared.")

    if st.button("↻  Refresh", use_container_width=True):
        st.rerun()


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    f'<h1 style="margin-bottom:2px">Alerts</h1>'
    f'<p style="font-size:12px;color:rgba(238,242,255,0.35);margin-bottom:0">'
    f'HIGH / CRITICAL trigger notifications · rule management · alert history</p>',
    unsafe_allow_html=True,
)
st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_panel, tab_rules, tab_history = st.tabs(["Notifications", "Rules", "History"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE NOTIFICATION PANEL
# ══════════════════════════════════════════════════════════════════════════════

with tab_panel:
    unread_now = st.session_state.get("ac_alerts_unread", 0)

    panel_header(
        "NOTIFICATION PANEL",
        badge=f"{unread_now} new" if unread_now > 0 else "up to date",
        badge_color=DESCEND if unread_now > 0 else "rgba(238,242,255,0.22)",
    )

    col_info, col_max = st.columns([3, 1])
    with col_info:
        st.markdown(
            f'<div style="font-size:10px;color:rgba(238,242,255,0.30);'
            f'padding:4px 0 12px;line-height:1.5">'
            f'Alerts are fired when incoming TriggerSignals match a rule at '
            f'HIGH or CRITICAL severity. Deduplication prevents repeat alerts '
            f'for the same trigger within its decay window.</div>',
            unsafe_allow_html=True,
        )
    with col_max:
        max_disp = st.number_input(
            "Max cards", min_value=5, max_value=100, value=20, step=5,
            key="notif_max_display",
        )

    notification_panel(engine=ENGINE, max_display=int(max_disp), show_detail=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RULE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

with tab_rules:
    panel_header("ALERT RULES", badge=f"{rules_enabled} enabled")

    rules = ENGINE.rules()

    # ── Rule table ─────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:10px;color:rgba(238,242,255,0.30);'
        f'padding:4px 0 12px;line-height:1.5">'
        f'Toggle rules or inspect their conditions. Custom rules can be added '
        f'via the form below (active for this session only).</div>',
        unsafe_allow_html=True,
    )

    for rule in rules:
        is_default = rule.rule_id in {r.rule_id for r in DEFAULT_RULES}
        label_suffix = " (default)" if is_default else " (custom)"

        with st.expander(
            f"{'✓' if rule.enabled else '○'}  {rule.name}{label_suffix}",
            expanded=False,
        ):
            col_en, col_del = st.columns([2, 1])

            with col_en:
                toggled = st.toggle(
                    "Enabled",
                    value=rule.enabled,
                    key=f"rule_toggle_{rule.rule_id}",
                )
                if toggled != rule.enabled:
                    if toggled:
                        ENGINE.enable(rule.rule_id)
                    else:
                        ENGINE.disable(rule.rule_id)
                    st.rerun()

            with col_del:
                if not is_default:
                    if st.button("Remove", key=f"rule_del_{rule.rule_id}"):
                        ENGINE.unregister(rule.rule_id)
                        st.success(f"Removed rule: {rule.name}")
                        st.rerun()

            # Rule conditions detail
            sev_label = rule.min_severity.label() if hasattr(rule.min_severity, "label") else str(rule.min_severity)

            details = [
                ("Rule ID",    rule.rule_id),
                ("Description", rule.description),
                ("Min severity", sev_label),
                ("Min magnitude", f"{rule.min_magnitude:.0%}"),
            ]
            if rule.trigger_types:
                details.append(("Trigger types", ", ".join(rule.trigger_types)))
            else:
                details.append(("Trigger types", "Any"))
            if rule.required_commodities:
                details.append(("Required commodities (any of)", ", ".join(rule.required_commodities)))
            if rule.required_clusters:
                details.append(("Required clusters (any of)", ", ".join(rule.required_clusters)))
            if rule.directions:
                details.append(("Directions", ", ".join(rule.directions)))
            if rule.max_cascade_priority is not None:
                details.append(("Max cascade priority", str(rule.max_cascade_priority)))

            rows_html = "".join(
                f'<tr>'
                f'<td style="font-size:9px;color:rgba(238,242,255,0.30);'
                f'padding:3px 10px 3px 0;white-space:nowrap;vertical-align:top">{k}</td>'
                f'<td style="font-size:10px;color:rgba(238,242,255,0.65);padding:3px 0">{v}</td>'
                f'</tr>'
                for k, v in details
            )
            st.markdown(
                f'<table style="border-collapse:collapse;width:100%;'
                f'margin-top:6px">{rows_html}</table>',
                unsafe_allow_html=True,
            )

    # ── Add custom rule form ───────────────────────────────────────────────────
    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    st.markdown(f'<hr style="border-color:{BORDER};margin:0">', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    panel_header("ADD CUSTOM RULE", badge="session only")

    with st.form("add_rule_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            r_name = st.text_input("Rule name *", placeholder="e.g. Wheat Spike Alert")
        with c2:
            r_desc = st.text_input("Description", placeholder="One-line rationale")

        c3, c4, c5 = st.columns(3)
        with c3:
            r_severity = st.selectbox("Min severity", ["HIGH", "CRITICAL"], key="new_rule_sev")
        with c4:
            r_magnitude = st.slider("Min magnitude", 0.0, 1.0, 0.40, 0.05, key="new_rule_mag")
        with c5:
            r_priority = st.number_input(
                "Max cascade priority", min_value=1, max_value=10,
                value=5, step=1, key="new_rule_pri",
                help="Leave at 10 to match all priorities",
            )

        TRIGGER_TYPES_OPT = [
            "FOMC_RATE_DECISION", "CPI_RELEASE", "NONFARM_PAYROLLS",
            "OPEC_PRODUCTION_DECISION", "EIA_CRUDE_INVENTORY", "EIA_GAS_STORAGE",
            "USDA_WASDE_REPORT", "FED_CHAIR_SPEECH", "DOLLAR_SHOCK",
            "GEOPOLITICAL_SHOCK", "RECESSION_FLAG", "WEATHER_SHOCK",
            "ENERGY_TRANSITION_SIGNAL", "PPI_RELEASE",
        ]
        r_trigger_types = st.multiselect(
            "Trigger types (empty = any)", TRIGGER_TYPES_OPT, key="new_rule_tt"
        )
        r_clusters = st.multiselect(
            "Required clusters (any-of, empty = no filter)",
            ["energy", "metals", "agriculture", "livestock", "digital"],
            key="new_rule_cl",
        )
        r_directions = st.multiselect(
            "Directions (empty = any)",
            ["downside_surprise", "upside_surprise", "neutral"],
            key="new_rule_dir",
        )
        r_commodities_raw = st.text_area(
            "Required commodities (any-of, one per line, empty = no filter)",
            height=80,
            key="new_rule_comms",
            placeholder="Gold (COMEX)\nSilver (COMEX)",
        )

        submitted = st.form_submit_button("Add Rule", type="primary")

        if submitted:
            if not r_name:
                st.error("Rule name is required.")
            else:
                comms = [c.strip() for c in r_commodities_raw.splitlines() if c.strip()]
                sev_map = {"HIGH": SeverityTier.HIGH, "CRITICAL": SeverityTier.CRITICAL}
                new_rule = AlertRule(
                    rule_id=f"custom_{uuid.uuid4().hex[:8]}",
                    name=r_name,
                    description=r_desc or "Custom rule",
                    trigger_types=r_trigger_types or None,
                    min_severity=sev_map[r_severity],
                    min_magnitude=r_magnitude,
                    required_commodities=comms,
                    required_clusters=list(r_clusters),
                    directions=list(r_directions) or None,
                    max_cascade_priority=int(r_priority) if r_priority < 10 else None,
                )
                ENGINE.register(new_rule)
                st.success(f"Rule '{r_name}' registered (session only — not persisted).")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ALERT HISTORY
# ══════════════════════════════════════════════════════════════════════════════

with tab_history:
    panel_header("ALERT HISTORY", badge="session")

    all_alerts: list[Alert] = st.session_state.get("ac_alerts", [])

    if not all_alerts:
        st.markdown(
            f'<div style="text-align:center;padding:32px 16px;'
            f'color:rgba(238,242,255,0.18);font-size:12px;letter-spacing:.07em">'
            f'No alerts in this session. Inject a test event or wait for a '
            f'HIGH/CRITICAL signal from the Trigger Bus.</div>',
            unsafe_allow_html=True,
        )
    else:
        # ── Filter controls ────────────────────────────────────────────────────
        hcol1, hcol2, hcol3 = st.columns([2, 2, 2])
        with hcol1:
            flt_sev = st.multiselect(
                "Severity", ["HIGH", "CRITICAL"], default=[], key="hist_sev"
            )
        with hcol2:
            rule_names = sorted({a.rule_name for a in all_alerts})
            flt_rule = st.multiselect(
                "Rule", rule_names, default=[], key="hist_rule"
            )
        with hcol3:
            flt_crit_only = st.checkbox(
                "Critical only", value=False, key="hist_crit"
            )

        filtered = all_alerts
        if flt_sev:
            filtered = [a for a in filtered if a.severity in flt_sev]
        if flt_rule:
            filtered = [a for a in filtered if a.rule_name in flt_rule]
        if flt_crit_only:
            filtered = [a for a in filtered if a.is_critical]

        st.markdown(
            f'<div style="font-size:9px;color:rgba(238,242,255,0.25);'
            f'padding:4px 0 10px">{len(filtered)} of {len(all_alerts)} alerts</div>',
            unsafe_allow_html=True,
        )

        # ── History table ──────────────────────────────────────────────────────
        _SEV_COLOR_MAP = {
            "CRITICAL": "#D94F4F",
            "HIGH":     "#e07020",
            "MEDIUM":   "#F59E0B",
            "LOW":      "rgba(238,242,255,0.35)",
        }

        header_html = (
            f'<table style="width:100%;border-collapse:collapse;'
            f'font-family:Arial,sans-serif">'
            f'<thead><tr>'
            + "".join(
                f'<th style="font-size:8.5px;letter-spacing:.10em;text-transform:uppercase;'
                f'color:rgba(238,242,255,0.25);text-align:left;'
                f'padding:4px 10px 4px 0;border-bottom:0.5px solid rgba(123,156,255,0.10)">'
                f'{col}</th>'
                for col in ["Time", "Severity", "Trigger", "Rule", "Magnitude", "Direction", "Matched"]
            )
            + "</tr></thead><tbody>"
        )

        rows_html = ""
        for alert in filtered[:200]:
            sev_c   = _SEV_COLOR_MAP.get(alert.severity, "#aaa")
            mag_pct = int(alert.magnitude_score * 100)
            matched = (
                ", ".join(list(alert.matched_commodities)[:2])
                if alert.matched_commodities
                else ", ".join(alert.matched_clusters[:2]) if alert.matched_clusters
                else "—"
            )
            dir_short = {
                "downside_surprise": "↓ Down",
                "upside_surprise":   "↑ Up",
                "neutral":           "→ Neutral",
            }.get(alert.direction, alert.direction)

            from components.notification_panel import _relative_time
            time_str = _relative_time(alert.fired_at)

            rows_html += (
                f'<tr style="border-bottom:0.5px solid rgba(123,156,255,0.06)">'
                f'<td style="font-size:9px;color:rgba(238,242,255,0.35);'
                f'padding:5px 10px 5px 0;font-family:monospace;white-space:nowrap">{time_str}</td>'
                f'<td style="font-size:9px;font-weight:700;color:{sev_c};'
                f'padding:5px 10px 5px 0">{alert.severity}</td>'
                f'<td style="font-size:9px;color:rgba(238,242,255,0.55);'
                f'padding:5px 10px 5px 0;white-space:nowrap">{alert.trigger_type}</td>'
                f'<td style="font-size:9px;color:rgba(123,156,255,0.60);'
                f'padding:5px 10px 5px 0">{alert.rule_name}</td>'
                f'<td style="font-size:9px;color:rgba(238,242,255,0.55);'
                f'padding:5px 10px 5px 0">{mag_pct}%</td>'
                f'<td style="font-size:9px;color:rgba(238,242,255,0.40);'
                f'padding:5px 10px 5px 0;white-space:nowrap">{dir_short}</td>'
                f'<td style="font-size:9px;color:rgba(238,242,255,0.35);'
                f'padding:5px 0 5px 0;max-width:180px;overflow:hidden;'
                f'text-overflow:ellipsis;white-space:nowrap">{matched}</td>'
                f'</tr>'
            )

        st.markdown(
            header_html + rows_html + "</tbody></table>",
            unsafe_allow_html=True,
        )

        if len(filtered) > 200:
            st.markdown(
                f'<div style="font-size:9px;color:rgba(238,242,255,0.22);'
                f'text-align:center;padding-top:8px">'
                f'Showing first 200 of {len(filtered)}</div>',
                unsafe_allow_html=True,
            )


# ── Footer ────────────────────────────────────────────────────────────────────

now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.markdown(
    f'<div style="margin-top:28px;font-size:9px;color:rgba(238,242,255,0.15);'
    f'text-align:center;letter-spacing:.06em">'
    f'Alert engine active · {rules_enabled} rules enabled · '
    f'snapshot at {now_str}</div>',
    unsafe_allow_html=True,
)
