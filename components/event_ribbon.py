"""
Event Ribbon Component  (Mission 7)
=====================================
Renders a real-time horizontal ribbon of TriggerSignal cards inside a
Streamlit ``components.v1.html()`` iframe.

The JavaScript client connects to the WebSocket broadcast server, receives
serialised TriggerSignal JSON, and inserts animated cards into the ribbon.

Usage
-----
  from components.event_ribbon import event_ribbon

  event_ribbon(ws_url="ws://localhost:8765")           # full ribbon
  event_ribbon(ws_url=url, height=200, max_events=10)  # customised
"""

from __future__ import annotations

import json
import streamlit.components.v1 as components


# ── Severity → brand colours ──────────────────────────────────────────────────
# Mapped to the Accendio palette (see utils/theme.py)
_SEVERITY_BORDER = {
    "LOW":      "rgba(238,242,255,0.18)",
    "MEDIUM":   "#F59E0B",
    "HIGH":     "#e07020",
    "CRITICAL": "#D94F4F",
}
_SEVERITY_BADGE_BG = {
    "LOW":      "rgba(238,242,255,0.06)",
    "MEDIUM":   "rgba(245,158,11,0.15)",
    "HIGH":     "rgba(224,112,32,0.15)",
    "CRITICAL": "rgba(217,79,79,0.20)",
}
_SEVERITY_BADGE_TEXT = {
    "LOW":      "rgba(238,242,255,0.35)",
    "MEDIUM":   "#F59E0B",
    "HIGH":     "#e07020",
    "CRITICAL": "#D94F4F",
}
_SEVERITY_BAR = {
    "LOW":      "rgba(238,242,255,0.2)",
    "MEDIUM":   "#F59E0B",
    "HIGH":     "#e07020",
    "CRITICAL": "#D94F4F",
}


def _build_html(ws_url: str, max_events: int) -> str:
    severity_border    = json.dumps(_SEVERITY_BORDER)
    severity_badge_bg  = json.dumps(_SEVERITY_BADGE_BG)
    severity_badge_txt = json.dumps(_SEVERITY_BADGE_TEXT)
    severity_bar       = json.dumps(_SEVERITY_BAR)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  html, body {{
    background: transparent;
    font-family: 'Arial', 'Helvetica Neue', Helvetica, sans-serif;
    overflow: hidden;
    height: 100%;
  }}

  /* ── Wrapper ─────────────────────────────────────────── */
  .ribbon-wrapper {{
    display: flex;
    flex-direction: column;
    height: 100%;
    background: #060912;
    border: 0.5px solid rgba(123,156,255,0.14);
    border-radius: 8px;
    overflow: hidden;
  }}

  /* ── Header bar ──────────────────────────────────────── */
  .ribbon-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 12px;
    background: #09102A;
    border-bottom: 0.5px solid rgba(123,156,255,0.14);
    flex-shrink: 0;
    min-height: 28px;
  }}
  .header-left {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .ribbon-label {{
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(238,242,255,0.3);
    font-weight: 500;
  }}
  .status-pill {{
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 9px;
    color: rgba(238,242,255,0.38);
    letter-spacing: 0.06em;
  }}
  .status-dot {{
    width: 5px; height: 5px;
    border-radius: 50%;
    background: rgba(238,242,255,0.15);
    transition: background 0.4s;
  }}
  .status-dot.live    {{ background: #3DB87A; box-shadow: 0 0 4px #3DB87A88; }}
  .status-dot.retry   {{ background: #F59E0B; }}
  .status-dot.offline {{ background: #D94F4F; }}

  .event-count {{
    font-size: 9px;
    color: rgba(238,242,255,0.25);
    letter-spacing: 0.06em;
  }}

  /* ── Ribbon scroll area ───────────────────────────────── */
  .ribbon {{
    display: flex;
    flex-direction: row;
    align-items: stretch;
    gap: 8px;
    padding: 8px 10px;
    overflow-x: auto;
    overflow-y: hidden;
    flex: 1;
    scroll-behavior: smooth;
    -webkit-overflow-scrolling: touch;
  }}
  .ribbon::-webkit-scrollbar {{ height: 3px; }}
  .ribbon::-webkit-scrollbar-track {{ background: transparent; }}
  .ribbon::-webkit-scrollbar-thumb {{
    background: rgba(123,156,255,0.2);
    border-radius: 2px;
  }}

  /* ── Empty state ─────────────────────────────────────── */
  .empty-state {{
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: rgba(238,242,255,0.18);
    font-size: 11px;
    letter-spacing: 0.08em;
    gap: 8px;
  }}

  /* ── Event card ──────────────────────────────────────── */
  .card {{
    flex: 0 0 auto;
    width: 186px;
    padding: 9px 11px 8px;
    border-radius: 7px;
    background: #0C1228;
    border: 0.5px solid rgba(123,156,255,0.1);
    border-left: 3px solid rgba(238,242,255,0.15);
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 5px;
    cursor: default;
  }}
  .card.new {{
    animation: slideIn 0.38s cubic-bezier(0.16, 1, 0.3, 1);
  }}
  @keyframes slideIn {{
    from {{ transform: translateX(40px); opacity: 0; }}
    to   {{ transform: translateX(0);   opacity: 1; }}
  }}
  .card.critical-glow {{
    box-shadow: 0 0 10px rgba(217,79,79,0.22);
  }}
  .card:hover {{
    background: rgba(12,18,40,0.92);
    border-color: rgba(123,156,255,0.22);
  }}

  /* ── Card rows ───────────────────────────────────────── */
  .card-top {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 4px;
  }}
  .severity-badge {{
    font-size: 8.5px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 6px;
    border-radius: 3px;
    white-space: nowrap;
  }}
  .direction-icon {{
    font-size: 12px;
    font-weight: 700;
    line-height: 1;
  }}
  .card-name {{
    font-size: 11px;
    font-weight: 500;
    color: #EEF2FF;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    letter-spacing: 0.01em;
  }}
  .card-meta {{
    font-size: 9px;
    color: rgba(238,242,255,0.32);
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    gap: 4px;
    flex-wrap: wrap;
    line-height: 1.4;
  }}
  .meta-dot {{ color: rgba(123,156,255,0.3); }}
  .priority-pill {{
    font-size: 8px;
    letter-spacing: 0.08em;
    color: rgba(123,156,255,0.55);
    background: rgba(123,156,255,0.07);
    padding: 1px 5px;
    border-radius: 3px;
    border: 0.5px solid rgba(123,156,255,0.18);
    white-space: nowrap;
  }}

  /* ── Magnitude bar ───────────────────────────────────── */
  .mag-track {{
    height: 2px;
    background: rgba(238,242,255,0.06);
    border-radius: 1px;
    overflow: hidden;
  }}
  .mag-fill {{
    height: 100%;
    border-radius: 1px;
    transition: width 0.5s ease;
  }}

  /* ── Footer row ──────────────────────────────────────── */
  .card-footer {{
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .card-time {{
    font-size: 8px;
    color: rgba(238,242,255,0.2);
    letter-spacing: 0.04em;
    font-family: 'Courier New', monospace;
  }}
  .repeat-tag {{
    font-size: 8px;
    color: rgba(238,242,255,0.25);
    letter-spacing: 0.06em;
    background: rgba(238,242,255,0.05);
    padding: 1px 4px;
    border-radius: 2px;
  }}
  .cluster-pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
    margin-top: 1px;
  }}
  .cluster-pill {{
    font-size: 7.5px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 1px 4px;
    border-radius: 2px;
    background: rgba(123,156,255,0.07);
    color: rgba(123,156,255,0.55);
    border: 0.5px solid rgba(123,156,255,0.14);
  }}
</style>
</head>
<body>

<div class="ribbon-wrapper">
  <div class="ribbon-header">
    <div class="header-left">
      <span class="ribbon-label">Macro Event Ribbon</span>
      <div class="status-pill">
        <span class="status-dot" id="statusDot"></span>
        <span id="statusText">connecting</span>
      </div>
    </div>
    <span class="event-count" id="eventCount"></span>
  </div>
  <div class="ribbon" id="ribbon">
    <div class="empty-state">
      <span>⟳</span>
      <span>Awaiting macro events</span>
    </div>
  </div>
</div>

<script>
(function() {{
  const WS_URL      = {json.dumps(ws_url)};
  const MAX_EVENTS  = {max_events};
  const SEV_BORDER  = {severity_border};
  const SEV_BADGE_BG  = {severity_badge_bg};
  const SEV_BADGE_TXT = {severity_badge_txt};
  const SEV_BAR     = {severity_bar};

  let events = [];
  let reconnectDelay = 1200;
  let ws = null;

  // ── Status ──────────────────────────────────────────────────────────────
  function setStatus(state) {{
    const dot  = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    dot.className  = 'status-dot ' + state;
    const labels = {{ live: 'live', retry: 'reconnecting', offline: 'offline' }};
    text.textContent = labels[state] || state;
  }}

  // ── Time formatter ───────────────────────────────────────────────────────
  function fmtTime(iso) {{
    try {{
      const d = new Date(iso);
      return d.toLocaleTimeString([], {{ hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }});
    }} catch(e) {{ return ''; }}
  }}

  // ── Direction glyph ──────────────────────────────────────────────────────
  function dirGlyph(d) {{
    if (d === 'upside_surprise')   return {{ g: '▲', c: '#3DB87A' }};
    if (d === 'downside_surprise') return {{ g: '▼', c: '#D94F4F' }};
    return {{ g: '→', c: 'rgba(238,242,255,0.3)' }};
  }}

  // ── Card HTML ────────────────────────────────────────────────────────────
  function buildCard(ev, isNew) {{
    const sev     = ev.severity || 'LOW';
    const border  = SEV_BORDER[sev]   || '#333';
    const bbg     = SEV_BADGE_BG[sev] || 'rgba(255,255,255,0.05)';
    const btxt    = SEV_BADGE_TXT[sev]|| 'rgba(255,255,255,0.3)';
    const barClr  = SEV_BAR[sev]      || '#444';
    const mag     = Math.round((ev.magnitude_score || 0) * 100);
    const dir     = dirGlyph(ev.direction);
    const clusters = (ev.affected_clusters || []).slice(0, 4);
    const isCrit  = sev === 'CRITICAL';

    const clusterHTML = clusters
      .map(c => `<span class="cluster-pill">${{c.substring(0,3)}}</span>`)
      .join('');

    return `
      <div class="card${{isNew ? ' new' : ''}}${{isCrit ? ' critical-glow' : ''}}"
           style="border-left-color:${{border}}">
        <div class="card-top">
          <span class="severity-badge"
                style="background:${{bbg}};color:${{btxt}}">${{sev}}</span>
          <span class="priority-pill">P${{ev.cascade_priority || '?'}}</span>
          <span class="direction-icon" style="color:${{dir.c}}">${{dir.g}}</span>
        </div>
        <div class="card-name" title="${{ev.display_name || ev.trigger_type}}">${{ev.display_name || ev.trigger_type}}</div>
        <div class="cluster-pills">${{clusterHTML}}</div>
        <div class="card-meta">
          <span>${{ev.affected_commodity_count || 0}} commodities</span>
          <span class="meta-dot">·</span>
          <span>${{ev.source || ''}}</span>
        </div>
        <div class="mag-track">
          <div class="mag-fill" style="width:${{mag}}%;background:${{barClr}}"></div>
        </div>
        <div class="card-footer">
          <span class="card-time">${{fmtTime(ev.classified_at)}}</span>
          ${{ev.is_repeat ? '<span class="repeat-tag">↻ repeat</span>' : ''}}
        </div>
      </div>
    `;
  }}

  // ── Render ───────────────────────────────────────────────────────────────
  function render() {{
    const ribbon = document.getElementById('ribbon');
    const counter = document.getElementById('eventCount');

    if (events.length === 0) {{
      ribbon.innerHTML = '<div class="empty-state"><span>⟳</span><span>Awaiting macro events</span></div>';
      counter.textContent = '';
      return;
    }}

    const n = events.length;
    const suffix = n === MAX_EVENTS ? ` (${{MAX_EVENTS}})` : '';
    counter.textContent = n + ' event' + (n !== 1 ? 's' : '') + suffix;

    ribbon.innerHTML = events
      .map((ev, i) => buildCard(ev, i === 0))
      .join('');

    // Scroll newest card into view (rightmost)
    ribbon.scrollLeft = ribbon.scrollWidth - ribbon.clientWidth;
  }}

  // ── Add event ────────────────────────────────────────────────────────────
  function addEvent(ev) {{
    events.push(ev);
    if (events.length > MAX_EVENTS) events = events.slice(events.length - MAX_EVENTS);
    render();
  }}

  // ── WebSocket ────────────────────────────────────────────────────────────
  function connect() {{
    try {{
      ws = new WebSocket(WS_URL);
    }} catch(e) {{
      setStatus('offline');
      return;
    }}

    ws.onopen = function() {{
      setStatus('live');
      reconnectDelay = 1200;
    }};

    ws.onmessage = function(e) {{
      try {{
        const sig = JSON.parse(e.data);
        addEvent(sig);
      }} catch(err) {{
        console.warn('[ribbon] parse error', err);
      }}
    }};

    ws.onclose = function() {{
      setStatus('retry');
      setTimeout(connect, reconnectDelay);
      reconnectDelay = Math.min(reconnectDelay * 1.6, 16000);
    }};

    ws.onerror = function() {{
      ws.close();
    }};
  }}

  connect();
}})();
</script>
</body>
</html>"""


def event_ribbon(
    ws_url: str = "ws://localhost:8765",
    height: int = 195,
    max_events: int = 10,
) -> None:
    """
    Render the real-time event ribbon component inside a Streamlit page.

    Parameters
    ----------
    ws_url     : WebSocket URL of the TriggerBroadcaster server.
    height     : Component iframe height in pixels (default 195).
    max_events : Maximum number of cards displayed (default 10, oldest drop off).
    """
    html = _build_html(ws_url, max_events)
    components.html(html, height=height, scrolling=False)
