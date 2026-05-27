"""
Macro Trigger Alert Engine  (Mission 9)
=========================================
Watches the Trigger Bus for HIGH and CRITICAL severity signals, evaluates
user-defined ``AlertRule`` objects against each incoming ``TriggerSignal``,
formats human-readable alert messages, and dispatches them to registered
handler callbacks.

Architecture
------------
  TriggerBus → AlertEngine.as_bus_handler()
                    │
                    ├─ rule evaluation (all enabled rules)
                    ├─ deduplication  (per rule × trigger_type × decay window)
                    ├─ message formatting
                    └─ handler dispatch → [pending_queue, custom callbacks, ...]

Thread safety
-------------
``AlertEngine`` is thread-safe.  ``evaluate()`` may be called from any
thread (e.g. a Trigger Bus background consumer thread).  Handlers are
called synchronously inside the lock — keep them fast.

Streamlit integration
---------------------
The engine exposes a ``pending_queue`` (``queue.Queue``).  The notification
panel component drains it on each Streamlit render cycle and appends alerts
to ``st.session_state``.  This sidesteps the Streamlit threading constraint
(session state is only writable from the main render thread).

Public API
----------
  AlertRule           — rule dataclass; define your own or use DEFAULT_RULES
  Alert               — alert output dataclass
  AlertEngine         — evaluation / dedup / dispatch
  DEFAULT_RULES       — six illustrative rules covering common macro scenarios
  ENGINE              — module-level singleton
  evaluate_signal(s)  — convenience wrapper
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

from services.trigger_classifier import SeverityTier, TriggerSignal

log = logging.getLogger(__name__)

# ── Minimum severity to even attempt rule evaluation ──────────────────────────
_MIN_EVAL_SEVERITY = SeverityTier.HIGH   # LOW and MEDIUM signals skip evaluation entirely


# ── Macro feed silence canary ──────────────────────────────────────────────────
# Families that the FRED/AV daemon should be writing into trigger_events on a
# regular cadence. If none of these has arrived in MACRO_FEED_SILENCE_HOURS,
# the daemon is presumed dead and a CRITICAL alert fires. Keep this list in
# sync with FRED_SERIES / AV_INDICATORS in services/macro_ingestion.py.
MACRO_FAMILIES: Tuple[str, ...] = (
    "CPI", "Unemployment", "Fed_Funds_Rate",
    "Yield_Curve_10Y2Y", "Real_Yield_10Y", "Industrial_Production",
    "PPI_All_Commodities", "M2_Money_Supply", "USD_EUR", "WTI_Spot",
    # AV indicator family names
    "CPI_AV", "NFP", "Unemployment_AV", "Fed_Funds_AV",
    "Treasury_10Y", "Inflation_YoY",
)
MACRO_FEED_SILENCE_HOURS = float(os.getenv("MACRO_FEED_SILENCE_HOURS", "24"))
_FEED_SILENCE_RULE_ID    = "macro_feed_silence"
_FEED_SILENCE_RULE_NAME  = "Macro Feed Silent"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _ttl_str(minutes: int) -> str:
    if minutes >= 10080:
        return f"{minutes // 10080} week{'s' if minutes // 10080 != 1 else ''}"
    if minutes >= 1440:
        return f"{minutes // 1440} day{'s' if minutes // 1440 != 1 else ''}"
    if minutes >= 60:
        return f"{minutes // 60} hour{'s' if minutes // 60 != 1 else ''}"
    return f"{minutes} min"


def _direction_text(direction: str) -> str:
    return {
        "upside_surprise":   "upside surprise (stronger than consensus)",
        "downside_surprise": "downside surprise (weaker than consensus)",
        "neutral":           "broadly in line with expectations",
    }.get(direction, direction)


def _short_direction(direction: str) -> str:
    return {
        "upside_surprise":   "↑ upside",
        "downside_surprise": "↓ downside",
        "neutral":           "→ neutral",
    }.get(direction, direction)


# ── Alert Rule ─────────────────────────────────────────────────────────────────

@dataclass
class AlertRule:
    """
    Declarative rule that maps a set of conditions to an alert.

    All non-None conditions must pass for the rule to fire.

    Parameters
    ----------
    rule_id             : Unique identifier (used in dedup keys and logs).
    name                : Human-readable name shown in the notification panel.
    description         : One-line explanation of why this rule exists.
    trigger_types       : Allowlist of trigger_type strings (None = any type).
    min_severity        : Minimum SeverityTier required (inclusive).
    min_magnitude       : Minimum magnitude_score (0–1) required.
    required_commodities: Rule fires only if at least one of these commodity
                          names appears in signal.affected_commodities.
                          Empty list = no commodity filter.
    required_clusters   : Rule fires only if at least one of these cluster
                          names appears in signal.affected_clusters.
                          Empty list = no cluster filter.
    directions          : Allowlist of direction strings (None = any direction).
    max_cascade_priority: Fire only for triggers at or above this priority
                          (1 = highest).  None = no priority filter.
    enabled             : Toggle without deleting the rule.
    """
    rule_id:              str
    name:                 str
    description:          str
    trigger_types:        Optional[List[str]]    = None
    min_severity:         SeverityTier           = SeverityTier.HIGH
    min_magnitude:        float                  = 0.0
    required_commodities: List[str]              = field(default_factory=list)
    required_clusters:    List[str]              = field(default_factory=list)
    directions:           Optional[List[str]]    = None
    max_cascade_priority: Optional[int]          = None
    enabled:              bool                   = True


# ── Alert ──────────────────────────────────────────────────────────────────────

@dataclass
class Alert:
    """Structured, dispatch-ready alert produced by the engine."""
    alert_id:            str                     # UUID4
    rule_id:             str
    rule_name:           str
    signal_id:           str                     # provenance link to TriggerSignal
    trigger_type:        str
    severity:            str                     # "HIGH" | "CRITICAL"
    headline:            str                     # ≤ 100-char single line
    message:             str                     # full human-readable body
    fired_at:            str                     # ISO 8601 UTC
    expires_at:          str                     # signal decay_expires_at
    affected_commodities: Tuple[str, ...]
    magnitude_score:     float
    direction:           str
    is_critical:         bool
    matched_commodities: Tuple[str, ...]         # subset matching rule filter
    matched_clusters:    Tuple[str, ...]         # subset matching rule filter
    metadata:            Dict = field(default_factory=dict)


# ── Message formatter ──────────────────────────────────────────────────────────

def _format_headline(signal: TriggerSignal, rule: AlertRule) -> str:
    sev_glyph = "🚨" if signal.severity == SeverityTier.CRITICAL else "⚠️"
    dir_glyph = _short_direction(signal.direction)
    mag_pct   = int(signal.magnitude_score * 100)
    name      = signal.config.display_name.split("(")[0].strip()
    return (
        f"{sev_glyph} {signal.severity.label()}  {name} · "
        f"{dir_glyph} · {mag_pct}% magnitude  [{rule.name}]"
    )


def _format_message(
    signal: TriggerSignal,
    rule: AlertRule,
    matched_comms: List[str],
    matched_clusters: List[str],
) -> str:
    now_str   = _parse_dt(signal.classified_at).strftime("%H:%M UTC on %Y-%m-%d")
    decay_str = _ttl_str(signal.config.decay_window_minutes)
    expires   = _parse_dt(signal.decay_expires_at).strftime("%Y-%m-%d %H:%M UTC")
    mag_pct   = int(signal.magnitude_score * 100)
    sigma     = round(signal.raw_deviation_score, 2)
    dir_text  = _direction_text(signal.direction)
    sev_label = signal.severity.label()

    # Commodity scope sentence
    if matched_comms:
        n = len(matched_comms)
        if n <= 3:
            comm_list = " and ".join(matched_comms)
        else:
            comm_list = ", ".join(matched_comms[:3]) + f" and {n - 3} more"
        scope_line = (
            f"Your rule \"{rule.name}\" matched because **{comm_list}** "
            f"{'is' if n == 1 else 'are'} in the affected commodity scope."
        )
    elif matched_clusters:
        scope_line = (
            f"Your rule \"{rule.name}\" matched on cluster scope: "
            f"**{', '.join(matched_clusters)}**."
        )
    else:
        scope_line = f"Your rule \"{rule.name}\" matched this trigger type and severity."

    # Severity-specific threshold context
    threshold_label = ""
    thresholds = signal.config.thresholds
    for tier in ("severe", "moderate", "mild"):
        band = thresholds.get(tier)
        if band and band.contains(signal.magnitude_score):
            threshold_label = f" ({band.label})"
            break

    # Cluster summary
    all_clusters  = list(signal.affected_clusters)
    primary_scope = " · ".join(c.capitalize() for c in all_clusters[:3])
    if len(all_clusters) > 3:
        primary_scope += f" + {len(all_clusters) - 3} more"

    # Decay / urgency
    if signal.config.decay_window_minutes >= 10080:
        urgency = "This is a structural event with a multi-week price adjustment window."
    elif signal.config.decay_window_minutes >= 1440:
        urgency = "Monitor for trend continuation over the next several days."
    else:
        urgency = "Short-term volatility window — monitor intraday for follow-through."

    # Description excerpt (first sentence)
    desc = signal.config.description or ""
    first_sentence = desc.split(". ")[0].strip() + "." if desc else ""

    lines = [
        f"A {sev_label} **{signal.config.display_name}** was classified at {now_str}.",
        f"Direction: {dir_text}. Magnitude: {mag_pct}%{threshold_label} (σ = {sigma}).",
        "",
        scope_line,
        "",
    ]

    if first_sentence:
        lines += [f"Context: {first_sentence}", ""]

    lines += [
        f"Affected scope: {primary_scope}.",
        urgency,
        "",
        f"Alert active for {decay_str} (expires {expires}).",
        f"Rule: {rule.name} · Signal ID: {signal.signal_id[:8]}…",
    ]

    return "\n".join(lines)


# ── Alert Engine ───────────────────────────────────────────────────────────────

AlertHandler = Callable[[List["Alert"]], None]


class AlertEngine:
    """
    Evaluates AlertRules against incoming TriggerSignals.

    Thread-safe.  ``evaluate()`` and ``as_bus_handler()`` may be called from
    any thread.  Handlers are invoked synchronously inside the internal lock,
    so keep them non-blocking.
    """

    def __init__(self) -> None:
        self._rules:    Dict[str, AlertRule] = {}
        self._handlers: List[AlertHandler]   = []
        # (rule_id, trigger_type) → expires_at datetime
        self._dedup:    Dict[Tuple[str, str], datetime] = {}
        self._lock = threading.Lock()
        # Thread-safe queue for Streamlit integration (drains on page render)
        self.pending_queue: queue.Queue[Alert] = queue.Queue()
        # Silence-canary dedup — suppress re-fires for one window.
        self._last_silence_alert_at: Optional[datetime] = None

    # ── Rule management ───────────────────────────────────────────────────────

    def register(self, rule: AlertRule) -> None:
        with self._lock:
            self._rules[rule.rule_id] = rule

    def unregister(self, rule_id: str) -> None:
        with self._lock:
            self._rules.pop(rule_id, None)

    def enable(self, rule_id: str) -> None:
        with self._lock:
            if rule_id in self._rules:
                self._rules[rule_id].enabled = True

    def disable(self, rule_id: str) -> None:
        with self._lock:
            if rule_id in self._rules:
                self._rules[rule_id].enabled = False

    def rules(self) -> List[AlertRule]:
        with self._lock:
            return list(self._rules.values())

    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        with self._lock:
            return self._rules.get(rule_id)

    # ── Handler management ────────────────────────────────────────────────────

    def register_handler(self, handler: AlertHandler) -> None:
        with self._lock:
            self._handlers.append(handler)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, signal: TriggerSignal) -> List[Alert]:
        """
        Evaluate all enabled rules against a TriggerSignal.

        Signals below HIGH severity are skipped immediately.
        Returns the list of newly fired (non-deduplicated) alerts.
        """
        if signal.severity < _MIN_EVAL_SEVERITY:
            return []

        now = _utcnow()
        new_alerts: List[Alert] = []

        with self._lock:
            for rule in self._rules.values():
                if not rule.enabled:
                    continue
                matched_comms, matched_clusters = self._matches(signal, rule)
                if matched_comms is None:
                    continue    # rule conditions not satisfied

                dedup_key = (rule.rule_id, signal.trigger_type)
                if self._is_deduplicated(dedup_key, now):
                    log.debug(
                        "Alert deduped: rule=%s trigger=%s",
                        rule.rule_id, signal.trigger_type,
                    )
                    continue

                # Record dedup entry using the signal's decay window
                expires_dt = _parse_dt(signal.decay_expires_at)
                self._dedup[dedup_key] = expires_dt

                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    signal_id=signal.signal_id,
                    trigger_type=signal.trigger_type,
                    severity=signal.severity.label(),
                    headline=_format_headline(signal, rule),
                    message=_format_message(signal, rule, matched_comms, matched_clusters),
                    fired_at=now.isoformat(),
                    expires_at=signal.decay_expires_at,
                    affected_commodities=signal.affected_commodities,
                    magnitude_score=signal.magnitude_score,
                    direction=signal.direction,
                    is_critical=signal.severity == SeverityTier.CRITICAL,
                    matched_commodities=tuple(matched_comms),
                    matched_clusters=tuple(matched_clusters),
                    metadata={
                        "classification_method": signal.classification_method,
                        "raw_deviation_score":   signal.raw_deviation_score,
                        "cascade_priority":      signal.cascade_priority,
                        "source":                signal.config.source,
                    },
                )
                new_alerts.append(alert)
                self.pending_queue.put(alert)
                log.info(
                    "Alert fired: rule=%s trigger=%s severity=%s",
                    rule.name, signal.trigger_type, signal.severity.label(),
                )

        if new_alerts:
            handlers_snapshot: List[AlertHandler]
            with self._lock:
                handlers_snapshot = list(self._handlers)
            for handler in handlers_snapshot:
                try:
                    handler(new_alerts)
                except Exception:
                    log.exception("Alert handler raised")

        return new_alerts

    def flush_dedup_cache(self) -> int:
        """Remove stale dedup entries (past their expiry).  Returns count removed."""
        now = _utcnow()
        with self._lock:
            stale = [k for k, exp in self._dedup.items() if now >= exp]
            for k in stale:
                del self._dedup[k]
        return len(stale)

    # ── Macro feed silence canary ────────────────────────────────────────────

    def check_macro_feed_silence(
        self,
        window_hours: float = MACRO_FEED_SILENCE_HOURS,
        families: Tuple[str, ...] = MACRO_FAMILIES,
    ) -> Optional[Alert]:
        """
        Fire a CRITICAL alert if no trigger_events row whose family is in
        `families` has arrived within the last `window_hours`. This is the
        canary that proves the macro feed daemon is alive end-to-end:
        if the daemon dies, no FRED/AV rows land, and this check fires.

        Returns the Alert if newly fired, None otherwise (no problem detected,
        or already alerted within this window — i.e. deduped).

        Safe to call on a timer (e.g. once per Streamlit page render) — the
        per-window dedup prevents repeated firing while the feed stays silent.
        """
        now = _utcnow()

        # Dedup: if we already raised the silence alarm inside this window,
        # don't raise it again until the window passes.
        with self._lock:
            last = self._last_silence_alert_at
            if last is not None and (now - last) < timedelta(hours=window_hours):
                return None

        # Probe the DB for the most recent row in any of the macro families.
        try:
            from database.db import get_db
            from database.models import TriggerEvent as DBTriggerEvent
            from sqlalchemy import func

            with get_db() as session:
                latest_str = (
                    session.query(func.max(DBTriggerEvent.detected_at))
                    .filter(DBTriggerEvent.family.in_(families))
                    .scalar()
                )
        except Exception as exc:
            log.warning("Macro silence check: DB read failed (%s)", exc)
            return None

        is_silent = True
        latest_dt: Optional[datetime] = None
        if latest_str:
            try:
                dt = datetime.fromisoformat(latest_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                latest_dt = dt
                if (now - dt) < timedelta(hours=window_hours):
                    is_silent = False
            except ValueError:
                pass    # fall through — unparseable timestamp counts as silent

        if not is_silent:
            return None

        # Build the synthetic alert. No TriggerSignal here — this is a
        # health-check fire, not a rule match.
        if latest_dt is None:
            age_str = "ever"
            detail = (
                "No row in trigger_events matches any tracked macro family — "
                "the feed has never written. Confirm pipeline/run_macro_feed.py "
                "is running and FRED_API_KEY / ALPHA_VANTAGE_KEY are set."
            )
        else:
            age_h  = (now - latest_dt).total_seconds() / 3600.0
            age_str = f"{age_h:.1f}h ago"
            detail = (
                f"Last macro trigger_events row: {latest_dt.isoformat()} "
                f"({age_str}). Threshold is {window_hours:.0f}h. "
                "Check launchd: `launchctl list | grep macrofeed`."
            )

        alert = Alert(
            alert_id      = str(uuid.uuid4()),
            rule_id       = _FEED_SILENCE_RULE_ID,
            rule_name     = _FEED_SILENCE_RULE_NAME,
            signal_id     = "",                    # synthetic — no source signal
            trigger_type  = "MACRO_FEED_SILENCE",
            severity      = "CRITICAL",
            headline      = f"🚨 CRITICAL  Macro feed silent · last event {age_str}",
            message       = (
                f"The macro ingestion daemon appears to be silent.\n\n"
                f"{detail}\n\n"
                f"Logs: logs/macro_feed.log · logs/macro_feed.error.log\n"
                f"Restart: launchctl kickstart -k gui/$(id -u)/com.accendio.macrofeed"
            ),
            fired_at      = now.isoformat(),
            expires_at    = (now + timedelta(hours=window_hours)).isoformat(),
            affected_commodities = (),
            magnitude_score = 1.0,
            direction       = "neutral",
            is_critical     = True,
            matched_commodities = (),
            matched_clusters    = (),
            metadata = {
                "kind":                "feed_silence_canary",
                "window_hours":         window_hours,
                "last_macro_event_at":  latest_str,
                "families_checked":     list(families),
            },
        )

        with self._lock:
            self._last_silence_alert_at = now
            self.pending_queue.put(alert)
            handlers_snapshot = list(self._handlers)

        for handler in handlers_snapshot:
            try:
                handler([alert])
            except Exception:
                log.exception("Silence-canary handler raised")

        log.warning(
            "Macro feed silence alert fired: last_event=%s (window=%.0fh)",
            latest_str, window_hours,
        )
        return alert

    # ── TriggerBus integration ────────────────────────────────────────────────

    def as_bus_handler(self) -> Callable[[dict, str, str], None]:
        """
        Return a Handler-signature callable for TriggerBus.subscribe().

        The bus passes ``(payload_dict, channel, model_id)``; we reconstruct
        the TriggerSignal and evaluate rules.
        """
        def _handle(payload: dict, channel: str, model_id: str) -> None:
            from services.trigger_bus import reconstruct_signal
            signal = reconstruct_signal(payload)
            if signal is not None:
                self.evaluate(signal)
        return _handle

    # ── Internal ──────────────────────────────────────────────────────────────

    def _matches(
        self,
        signal: TriggerSignal,
        rule: AlertRule,
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """
        Check all rule conditions.  Returns (matched_commodities, matched_clusters)
        if the rule fires, or (None, None) if any condition fails.
        """
        # Severity gate
        if signal.severity < rule.min_severity:
            return None, None

        # Trigger type filter
        if rule.trigger_types and signal.trigger_type not in rule.trigger_types:
            return None, None

        # Magnitude filter
        if signal.magnitude_score < rule.min_magnitude:
            return None, None

        # Direction filter
        if rule.directions and signal.direction not in rule.directions:
            return None, None

        # Priority filter (lower number = higher priority)
        if rule.max_cascade_priority is not None:
            if signal.cascade_priority > rule.max_cascade_priority:
                return None, None

        # Commodity scope filter (any-of)
        affected_comm_set = set(signal.affected_commodities)
        matched_comms: List[str] = []
        if rule.required_commodities:
            matched_comms = [c for c in rule.required_commodities if c in affected_comm_set]
            if not matched_comms:
                return None, None

        # Cluster scope filter (any-of)
        affected_cluster_set = set(signal.affected_clusters)
        matched_clusters: List[str] = []
        if rule.required_clusters:
            matched_clusters = [c for c in rule.required_clusters if c in affected_cluster_set]
            if not matched_clusters:
                return None, None

        return matched_comms, matched_clusters

    def _is_deduplicated(self, key: Tuple[str, str], now: datetime) -> bool:
        exp = self._dedup.get(key)
        return exp is not None and now < exp


# ── Default alert rules ────────────────────────────────────────────────────────

DEFAULT_RULES: List[AlertRule] = [
    AlertRule(
        rule_id="any_critical",
        name="Any CRITICAL Signal",
        description="Fires for every CRITICAL-severity macro trigger regardless of type.",
        trigger_types=None,
        min_severity=SeverityTier.CRITICAL,
        min_magnitude=0.0,
        required_commodities=[],
        required_clusters=[],
    ),
    AlertRule(
        rule_id="fomc_gold_scope",
        name="FOMC — Gold & Silver in Scope",
        description="FOMC rate decision at HIGH or above with precious metals affected.",
        trigger_types=["FOMC_RATE_DECISION", "FED_CHAIR_SPEECH"],
        min_severity=SeverityTier.HIGH,
        min_magnitude=0.45,
        required_commodities=["Gold (COMEX)", "Silver (COMEX)", "Gold (Physical/London)*"],
        required_clusters=[],
    ),
    AlertRule(
        rule_id="energy_supply_shock",
        name="Energy Supply Shock",
        description="OPEC+ or geopolitical event that significantly elevates the energy cluster.",
        trigger_types=["OPEC_PRODUCTION_DECISION", "GEOPOLITICAL_SHOCK", "EIA_CRUDE_INVENTORY"],
        min_severity=SeverityTier.HIGH,
        min_magnitude=0.50,
        required_commodities=[],
        required_clusters=["energy"],
        directions=["downside_surprise"],
    ),
    AlertRule(
        rule_id="recession_any",
        name="Recession Indicator Active",
        description="NBER USREC=1 regime override — fires at any severity tier.",
        trigger_types=["RECESSION_FLAG"],
        min_severity=SeverityTier.HIGH,
        min_magnitude=0.0,
        required_commodities=[],
        required_clusters=[],
    ),
    AlertRule(
        rule_id="agri_disruption",
        name="Agricultural Disruption",
        description="WASDE or weather shock with CRITICAL magnitude affecting grain/oilseed markets.",
        trigger_types=["USDA_WASDE_REPORT", "WEATHER_SHOCK"],
        min_severity=SeverityTier.HIGH,
        min_magnitude=0.60,
        required_commodities=[
            "Corn (CBOT)", "Wheat (CBOT SRW)", "Wheat (KC HRW)",
            "Soybeans (CBOT)", "Soybean Meal", "Soybean Oil",
        ],
        required_clusters=["agriculture"],
    ),
    AlertRule(
        rule_id="dollar_shock_broad",
        name="USD Dollar Broad Shock",
        description="DXY shock at HIGH+ that reprices all USD-denominated commodities.",
        trigger_types=["DOLLAR_SHOCK"],
        min_severity=SeverityTier.HIGH,
        min_magnitude=0.40,
        required_commodities=[],
        required_clusters=["metals", "energy"],
    ),
]


# ── Module-level singleton ─────────────────────────────────────────────────────

ENGINE = AlertEngine()
for _rule in DEFAULT_RULES:
    ENGINE.register(_rule)


def evaluate_signal(signal: TriggerSignal) -> List[Alert]:
    """Convenience wrapper — evaluates via the module-level ENGINE."""
    return ENGINE.evaluate(signal)


def check_macro_feed_silence(
    window_hours: float = MACRO_FEED_SILENCE_HOURS,
) -> Optional[Alert]:
    """Convenience wrapper — runs the canary against the module-level ENGINE."""
    return ENGINE.check_macro_feed_silence(window_hours=window_hours)
