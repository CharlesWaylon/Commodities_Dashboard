"""
Notification Panel Component  (Mission 9)
==========================================
Drains the ``AlertEngine.pending_queue`` on every Streamlit render cycle and
maintains an in-session alert history in ``st.session_state``.

Renders a vertical stack of alert cards with severity-coded badges, dismiss
controls, and a «Clear all» button.  New alerts also fire ``st.toast()`` for
an immediate pop-up independent of page scroll position.

Public API
----------
  drain_pending(engine)      → int          — pull new alerts from queue → session state
  notification_panel(engine) → None         — full panel render
  unread_count()             → int          — badge helper for nav links
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import streamlit as st

from services.alert_engine import Alert, AlertEngine, ENGINE

# ── Session-state key constants ───────────────────────────────────────────────

_KEY_ALERTS   = "ac_alerts"          # List[Alert], newest first
_KEY_UNREAD   = "ac_alerts_unread"   # int


# ── Colour helpers ─────────────────────────────────────────────────────────────

_SEV_BORDER = {
    "LOW":      "rgba(238,242,255,0.18)",
    "MEDIUM":   "#F59E0B",
    "HIGH":     "#e07020",
    "CRITICAL": "#D94F4F",
}
_SEV_BADGE_BG = {
    "LOW":      "rgba(238,242,255,0.06)",
    "MEDIUM":   "rgba(245,158,11,0.14)",
    "HIGH":     "rgba(224,112,32,0.14)",
    "CRITICAL": "rgba(217,79,79,0.18)",
}
_SEV_BADGE_TEXT = {
    "LOW":      "rgba(238,242,255,0.40)",
    "MEDIUM":   "#F59E0B",
    "HIGH":     "#e07020",
    "CRITICAL": "#D94F4F",
}
_SEV_ICON = {
    "LOW":      "○",
    "MEDIUM":   "◆",
    "HIGH":     "⚠",
    "CRITICAL": "🚨",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _relative_time(iso: str) -> str:
    try:
        dt   = _parse_dt(iso)
        diff = int((_utcnow() - dt).total_seconds())
        if diff < 60:
            return "just now"
        if diff < 3600:
            m = diff // 60
            return f"{m} min ago"
        if diff < 86400:
            h = diff // 3600
            return f"{h}h ago"
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return iso[:16]


def _expires_label(iso: str) -> str:
    try:
        exp = _parse_dt(iso)
        remaining = int((exp - _utcnow()).total_seconds())
        if remaining <= 0:
            return "expired"
        if remaining < 3600:
            return f"expires in {remaining // 60}m"
        if remaining < 86400:
            return f"expires in {remaining // 3600}h"
        return f"expires in {remaining // 86400}d"
    except Exception:
        return ""


# ── Session-state helpers ──────────────────────────────────────────────────────

def _init_state() -> None:
    if _KEY_ALERTS not in st.session_state:
        st.session_state[_KEY_ALERTS] = []
    if _KEY_UNREAD not in st.session_state:
        st.session_state[_KEY_UNREAD] = 0


def drain_pending(engine: Optional[AlertEngine] = None) -> int:
    """
    Pull all pending alerts from ``engine.pending_queue`` into session state.

    Returns the number of new alerts added.  Call once at the top of each
    page render before calling ``notification_panel()``.

    Parameters
    ----------
    engine : AlertEngine, optional
        Defaults to the module-level ``ENGINE`` singleton.
    """
    _init_state()
    if engine is None:
        engine = ENGINE

    count = 0
    while True:
        try:
            alert: Alert = engine.pending_queue.get_nowait()
        except Exception:
            break
        st.session_state[_KEY_ALERTS].insert(0, alert)
        st.session_state[_KEY_UNREAD] += 1
        count += 1
        # Toast for immediate feedback (Streamlit 1.29+)
        try:
            icon = "🚨" if alert.is_critical else "⚠️"
            st.toast(alert.headline, icon=icon)
        except Exception:
            pass

    return count


def unread_count() -> int:
    """Return the current unread alert count (badge helper)."""
    return st.session_state.get(_KEY_UNREAD, 0)


# ── CSS ────────────────────────────────────────────────────────────────────────

_PANEL_CSS = """
<style>
/* ── Alert card ─────────────────────────────────────────────────────── */
.ac-alert-card {
  background: #0C1228;
  border: 0.5px solid rgba(123,156,255,0.10);
  border-radius: 8px;
  padding: 12px 14px 10px;
  margin-bottom: 8px;
  font-family: Arial, 'Helvetica Neue', sans-serif;
  position: relative;
}
.ac-alert-card.critical {
  border-left: 3px solid #D94F4F;
  box-shadow: 0 0 10px rgba(217,79,79,0.12);
}
.ac-alert-card.high {
  border-left: 3px solid #e07020;
}
.ac-alert-card.medium {
  border-left: 3px solid #F59E0B;
}
.ac-alert-card.low {
  border-left: 3px solid rgba(238,242,255,0.20);
}

/* ── Header row ──────────────────────────────────────────────────────── */
.ac-alert-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 5px;
  flex-wrap: wrap;
}
.ac-alert-badge {
  font-size: 8.5px;
  font-weight: 700;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  padding: 2px 7px;
  border-radius: 3px;
  flex-shrink: 0;
}
.ac-alert-rule {
  font-size: 9px;
  color: rgba(123,156,255,0.55);
  background: rgba(123,156,255,0.07);
  border: 0.5px solid rgba(123,156,255,0.18);
  border-radius: 3px;
  padding: 1px 6px;
  letter-spacing: 0.06em;
  flex-shrink: 0;
}
.ac-alert-time {
  font-size: 8.5px;
  color: rgba(238,242,255,0.22);
  margin-left: auto;
  letter-spacing: 0.04em;
  font-family: 'Courier New', monospace;
  white-space: nowrap;
}

/* ── Headline ────────────────────────────────────────────────────────── */
.ac-alert-headline {
  font-size: 11px;
  font-weight: 500;
  color: #EEF2FF;
  margin-bottom: 5px;
  line-height: 1.4;
}

/* ── Meta row ────────────────────────────────────────────────────────── */
.ac-alert-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  margin-bottom: 4px;
}
.ac-alert-mag-track {
  flex: 1;
  min-width: 60px;
  height: 2px;
  background: rgba(238,242,255,0.06);
  border-radius: 1px;
  overflow: hidden;
}
.ac-alert-mag-fill {
  height: 100%;
  border-radius: 1px;
}
.ac-alert-scope {
  font-size: 9px;
  color: rgba(238,242,255,0.30);
  letter-spacing: 0.04em;
}
.ac-alert-expires {
  font-size: 8.5px;
  color: rgba(238,242,255,0.20);
  letter-spacing: 0.04em;
  margin-left: auto;
}

/* ── Message body (hidden by default, toggle) ───────────────────────── */
.ac-alert-body {
  font-size: 10.5px;
  color: rgba(238,242,255,0.50);
  line-height: 1.55;
  margin-top: 6px;
  border-top: 0.5px solid rgba(123,156,255,0.07);
  padding-top: 7px;
  white-space: pre-wrap;
}

/* ── Empty state ─────────────────────────────────────────────────────── */
.ac-alerts-empty {
  text-align: center;
  padding: 32px 16px;
  color: rgba(238,242,255,0.18);
  font-size: 12px;
  letter-spacing: 0.07em;
}
</style>
"""


# ── Single card renderer ───────────────────────────────────────────────────────

def _alert_card_html(alert: Alert) -> str:
    sev      = alert.severity
    border   = _SEV_BORDER.get(sev, "#888")
    badge_bg = _SEV_BADGE_BG.get(sev, "rgba(255,255,255,0.05)")
    badge_tx = _SEV_BADGE_TEXT.get(sev, "#aaa")
    icon     = _SEV_ICON.get(sev, "◆")
    mag_pct  = int(alert.magnitude_score * 100)
    n_comms  = len(alert.affected_commodities)
    fired    = _relative_time(alert.fired_at)
    expires  = _expires_label(alert.expires_at)

    matched_label = ""
    if alert.matched_commodities:
        mc = list(alert.matched_commodities)
        if len(mc) <= 2:
            matched_label = " · ".join(mc)
        else:
            matched_label = f"{mc[0]}, {mc[1]} +{len(mc)-2}"
    elif alert.matched_clusters:
        matched_label = " · ".join(c.capitalize() for c in alert.matched_clusters[:2])

    scope_text = f"{n_comms} commodities"
    if matched_label:
        scope_text = matched_label

    sev_lower = sev.lower()
    headline  = alert.headline.replace("🚨 ", "").replace("⚠️ ", "").strip()

    return f"""
<div class="ac-alert-card {sev_lower}">
  <div class="ac-alert-header">
    <span class="ac-alert-badge"
          style="background:{badge_bg};color:{badge_tx}">{icon} {sev}</span>
    <span class="ac-alert-rule">{alert.rule_name}</span>
    <span class="ac-alert-time">{fired}</span>
  </div>
  <div class="ac-alert-headline">{headline}</div>
  <div class="ac-alert-meta">
    <div class="ac-alert-mag-track">
      <div class="ac-alert-mag-fill"
           style="width:{mag_pct}%;background:{border}"></div>
    </div>
    <span class="ac-alert-scope">{scope_text}</span>
    <span class="ac-alert-expires">{expires}</span>
  </div>
</div>"""


# ── Main panel renderer ────────────────────────────────────────────────────────

def notification_panel(
    engine: Optional[AlertEngine] = None,
    max_display: int = 50,
    show_detail: bool = True,
) -> None:
    """
    Render the full notification panel.

    Call ``drain_pending(engine)`` once at page top before calling this.
    The panel itself does not drain the queue — separation of concerns keeps
    the drain idempotent regardless of how many times the panel renders.

    Parameters
    ----------
    engine      : AlertEngine, optional.  Defaults to module-level ENGINE.
    max_display : Maximum number of alert cards to show (newest first).
    show_detail : If True, show a «Detail» expander with the full message body.
    """
    _init_state()
    if engine is None:
        engine = ENGINE

    st.markdown(_PANEL_CSS, unsafe_allow_html=True)

    alerts: list[Alert] = st.session_state.get(_KEY_ALERTS, [])

    # ── Header row with unread badge + clear ──────────────────────────────────
    col_title, col_clear = st.columns([3, 1])
    with col_title:
        unread = st.session_state.get(_KEY_UNREAD, 0)
        badge_html = ""
        if unread > 0:
            badge_html = (
                f'<span style="background:#D94F4F;color:#fff;font-size:8px;'
                f'font-weight:700;border-radius:10px;padding:1px 6px;'
                f'margin-left:8px;vertical-align:middle">{unread} new</span>'
            )
        st.markdown(
            f'<div style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;'
            f'color:rgba(238,242,255,0.35);padding:4px 0 8px">'
            f'Alerts{badge_html}</div>',
            unsafe_allow_html=True,
        )

    with col_clear:
        if alerts:
            if st.button("Clear all", key="ac_clear_all", use_container_width=True):
                st.session_state[_KEY_ALERTS] = []
                st.session_state[_KEY_UNREAD] = 0
                st.rerun()

    # Mark unread as read after render
    st.session_state[_KEY_UNREAD] = 0

    if not alerts:
        st.markdown(
            '<div class="ac-alerts-empty">No alerts · engine monitoring for HIGH/CRITICAL signals</div>',
            unsafe_allow_html=True,
        )
        return

    display = alerts[:max_display]

    for i, alert in enumerate(display):
        # Render card HTML
        st.markdown(_alert_card_html(alert), unsafe_allow_html=True)

        # Detail + dismiss row
        if show_detail:
            col_det, col_dis = st.columns([3, 1])
            with col_det:
                with st.expander("Detail", expanded=False):
                    st.markdown(
                        f'<div class="ac-alert-body">{alert.message}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div style="font-size:8px;color:rgba(238,242,255,0.18);'
                        f'margin-top:6px;font-family:monospace">'
                        f'Alert ID: {alert.alert_id} · Signal: {alert.signal_id[:8]}…'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            with col_dis:
                if st.button("✕", key=f"ac_dismiss_{alert.alert_id}", help="Dismiss"):
                    st.session_state[_KEY_ALERTS] = [
                        a for a in alerts if a.alert_id != alert.alert_id
                    ]
                    st.rerun()
        else:
            # Compact mode: just a dismiss button per card
            if st.button("✕ Dismiss", key=f"ac_dismiss_{alert.alert_id}"):
                st.session_state[_KEY_ALERTS] = [
                    a for a in alerts if a.alert_id != alert.alert_id
                ]
                st.rerun()

    if len(alerts) > max_display:
        st.markdown(
            f'<div style="font-size:9px;color:rgba(238,242,255,0.22);'
            f'text-align:center;padding:8px 0">'
            f'+{len(alerts) - max_display} older alerts not shown</div>',
            unsafe_allow_html=True,
        )
