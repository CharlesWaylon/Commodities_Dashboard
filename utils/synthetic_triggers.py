"""
Synthetic trigger utilities — shared across Event Ribbon + downstream pages.

Provides:
  - get_active_synthetics()     → list of currently alive synthetic triggers
  - cleanup_expired_synthetics() → delete expired rows from trigger_events
  - clear_all_synthetics()      → nuke all synthetic rows + LIFECYCLE entries
  - render_synthetic_banner()   → Streamlit UI banner with clear button
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import streamlit as st

from utils.theme import DEPTH, BORDER, ASCEND, DESCEND, AMBER, ICE, ICE_MID

SYNTHETIC_TTL_MINUTES = 20
_SYNTHETIC_PREFIX = "[SYNTHETIC]"


def cleanup_expired_synthetics() -> int:
    """Delete synthetic trigger_events rows past their TTL. Returns count deleted."""
    try:
        from database.db import get_db
        from database.models import TriggerEvent

        deleted = 0
        now = datetime.now(timezone.utc)
        with get_db() as db:
            rows = (
                db.query(TriggerEvent)
                .filter(TriggerEvent.rationale.like(f"{_SYNTHETIC_PREFIX}%"))
                .all()
            )
            for row in rows:
                try:
                    meta = json.loads(row.trigger_metadata or "{}")
                    expires = datetime.fromisoformat(meta.get("synthetic_expires_at", ""))
                    if now >= expires:
                        db.delete(row)
                        deleted += 1
                except (ValueError, KeyError):
                    db.delete(row)
                    deleted += 1
            db.commit()
        return deleted
    except Exception:
        return 0


def get_active_synthetics() -> list[dict]:
    """Return currently-alive synthetic triggers from the DB."""
    try:
        from database.db import get_db
        from database.models import TriggerEvent

        now = datetime.now(timezone.utc)
        with get_db() as db:
            rows = (
                db.query(TriggerEvent)
                .filter(TriggerEvent.rationale.like(f"{_SYNTHETIC_PREFIX}%"))
                .all()
            )
            alive = []
            for r in rows:
                meta = json.loads(r.trigger_metadata or "{}")
                try:
                    exp = datetime.fromisoformat(meta["synthetic_expires_at"])
                except (KeyError, ValueError):
                    continue
                if now < exp:
                    alive.append({
                        "family": r.family,
                        "strength": float(r.strength),
                        "display_name": meta.get("display_name", r.family),
                        "direction": meta.get("direction", "neutral"),
                        "clusters": meta.get("affected_clusters", []),
                        "trigger_type": meta.get("trigger_type", ""),
                        "remaining_min": (exp - now).total_seconds() / 60,
                    })
            return alive
    except Exception:
        return []


def clear_all_synthetics():
    """Remove all synthetic triggers from DB and LIFECYCLE."""
    try:
        from database.db import get_db
        from database.models import TriggerEvent

        with get_db() as db:
            synths = (
                db.query(TriggerEvent)
                .filter(TriggerEvent.rationale.like(f"{_SYNTHETIC_PREFIX}%"))
                .all()
            )
            trigger_types = set()
            for row in synths:
                meta = json.loads(row.trigger_metadata or "{}")
                tt = meta.get("trigger_type", "")
                if tt:
                    trigger_types.add(tt)
                db.delete(row)
            db.commit()

        # Also expire from LIFECYCLE
        try:
            from services.trigger_lifecycle import LIFECYCLE
            for tt in trigger_types:
                LIFECYCLE.force_expire(tt)
        except Exception:
            pass

        # Purge synthetic events from the WS replay buffer so reconnecting
        # clients don't get them re-sent, then tell live clients to drop them.
        try:
            from services.ws_broadcast import BROADCASTER
            if BROADCASTER.is_running:
                BROADCASTER.purge_synthetic_replay()
                BROADCASTER.broadcast_raw(
                    json.dumps({"_action": "clear_synthetics"}), replay=False,
                )
        except Exception:
            pass

        # Bump ribbon nonce so the iframe HTML changes and Streamlit recreates it
        st.session_state["_ribbon_nonce"] = st.session_state.get("_ribbon_nonce", 0) + 1

    except Exception:
        pass


def render_synthetic_banner(page_context: str = "", show_details: bool = True):
    """
    Render an inline banner when synthetic triggers are active.

    Call this near the top of any downstream page. It shows what synthetics
    are active, page-specific context, and a clear button.
    Returns the list of active synthetics (empty list if none).
    """
    active = get_active_synthetics()
    if not active:
        return active

    from features.macro_features import family_to_regime

    n = len(active)
    names = ", ".join(s["display_name"] for s in active)
    ttl_min = min(s["remaining_min"] for s in active)

    regime_parts = []
    for s in active:
        regime = family_to_regime(s["family"])
        if regime != "neutral":
            regime_parts.append(regime.replace("_", " ").title())
    regime_str = ", ".join(set(regime_parts)) if regime_parts else "Neutral"

    context_html = ""
    if page_context:
        context_html = (
            f'<div style="font-size:11px;color:rgba(238,242,255,0.55);margin-top:6px">'
            f'{page_context}</div>'
        )

    st.markdown(
        f'<div style="background:rgba(245,166,35,0.06);border:0.5px solid rgba(245,166,35,0.25);'
        f'border-left:3px solid {AMBER};border-radius:6px;padding:12px 16px;margin-bottom:16px">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<div>'
        f'<span style="font-size:10px;letter-spacing:.12em;color:{AMBER};'
        f'text-transform:uppercase;font-weight:500">SCENARIO MODE</span>'
        f'<span style="font-size:11px;color:rgba(238,242,255,0.5);margin-left:12px">'
        f'{n} synthetic trigger{"s" if n != 1 else ""} active · '
        f'{names} · regime: {regime_str} · '
        f'{ttl_min:.0f} min until auto-expiry</span>'
        f'</div>'
        f'</div>'
        f'{context_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button("Clear synthetic triggers", key=f"clear_synth_{page_context[:10]}",
                 use_container_width=False):
        clear_all_synthetics()
        st.rerun()

    return active
