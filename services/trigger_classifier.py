"""
Trigger Classification Engine
==============================
Receives a normalized ``MacroEvent`` (Mission 1), looks up its config from
the ``TriggerRegistry`` (Mission 2), computes a magnitude score, assigns a
severity tier, and emits a ``TriggerSignal`` ready for dispatch.

Pipeline
--------
  MacroEvent  →  _resolve_trigger_type()
             →  _compute_magnitude()      # [0, 1] normalized score
             →  _classify_severity()      # LOW / MEDIUM / HIGH / CRITICAL
             →  _classify_direction()     # upside / downside / neutral
             →  TriggerSignal             # structured, dispatch-ready object

Magnitude scoring (two methods, blended when both inputs are available)
-----------------------------------------------------------------------
  Z-score method   :  score = clip(|deviation_score| / Z_MAX, 0, 1)
                       Always available — ``deviation_score`` is required on
                       every ``MacroEvent``.

  Surprise method  :  score = clip(|actual − expected| / |expected| × SURPRISE_SCALE, 0, 1)
                       Used when ``expected_value`` is not None. Weighted 40 %
                       in the blend (z-score takes the 60 % anchor role).

  Impact floor     :  ``impact == "high"`` events receive a minimum score of
                       ``IMPACT_FLOOR`` so they never silently fall into LOW.

Severity → tier mapping
-----------------------
  magnitude < mild_band.min            → LOW
  magnitude in [mild_band.min, max]    → MEDIUM
  magnitude in [moderate.min, max]     → HIGH
  magnitude in [severe.min, 1.0]       → CRITICAL

Repeat-event detection
----------------------
  A (trigger_type, release_date) pair is memoised for ``dedup_window_hours``
  (default 24 h). Subsequent arrivals within the window emit a signal with
  ``is_repeat=True`` so the dispatch layer can decide whether to re-fire.

Public API
----------
  TriggerClassifier(registry, dedup_window_hours=24)
    .classify(event)         → Optional[TriggerSignal]
    .classify_batch(events)  → list[TriggerSignal]

  classify_event(event)      → Optional[TriggerSignal]   # module-level shortcut
  CLASSIFIER                 — module-level singleton
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from services.macro_ingestion import MacroEvent
from services.trigger_config import TriggerConfig, TriggerRegistry, REGISTRY

log = logging.getLogger(__name__)

# ── Tuning constants ───────────────────────────────────────────────────────────

Z_MAX          = 3.0    # z-score that maps to magnitude = 1.0 (CRITICAL ceiling)
SURPRISE_SCALE = 5.0    # 20 % surprise → surprise_score = 1.0
IMPACT_FLOOR   = 0.15   # high-impact events never score below this

# ── Enumerations ───────────────────────────────────────────────────────────────

class SeverityTier(IntEnum):
    """
    Four-tier severity scale. IntEnum so tiers are directly comparable:
        SeverityTier.LOW < SeverityTier.MEDIUM < SeverityTier.HIGH < SeverityTier.CRITICAL
    """
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4

    def label(self) -> str:
        return self.name  # "LOW", "MEDIUM", "HIGH", "CRITICAL"


# ── Output object ──────────────────────────────────────────────────────────────

@dataclass
class TriggerSignal:
    """
    Structured, dispatch-ready output of the classification engine.

    Consumers (cascade orchestrator, macro router, alert system) should
    check ``is_repeat`` before acting — they may want to skip or de-duplicate.
    ``decay_expires_at`` tells consumers when to stop treating the event as live.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    signal_id:          str              # UUID4 — unique per TriggerSignal emission
    trigger_type:       str              # SCREAMING_SNAKE_CASE from registry
    family_name:        str              # snake_case for trigger_log / SignalRouter
    source_event_id:    str              # MacroEvent.trigger_id (provenance chain)

    # ── Scores ────────────────────────────────────────────────────────────────
    magnitude_score:    float            # [0, 1] normalized composite score
    severity:           SeverityTier     # LOW / MEDIUM / HIGH / CRITICAL
    direction:          str              # "upside_surprise" | "downside_surprise" | "neutral"

    # ── Timing ────────────────────────────────────────────────────────────────
    release_timestamp:  str              # ISO 8601 UTC — from MacroEvent
    classified_at:      str             # ISO 8601 UTC — when this signal was created
    decay_expires_at:   str             # classified_at + config.decay_window_minutes

    # ── Dispatch targets ──────────────────────────────────────────────────────
    affected_commodities: Tuple[str, ...]
    affected_clusters:    Tuple[str, ...]
    model_targets:        Tuple[str, ...]
    cascade_priority:     int

    # ── Provenance ────────────────────────────────────────────────────────────
    classification_method: str          # "z_score" | "blended_surprise" | suffix "_impact_floor"
    raw_deviation_score:   float        # MacroEvent.deviation_score (unsigned, preserves sign)
    event_type:            str          # original MacroEvent.event_type
    actual_value:          Optional[float]
    expected_value:        Optional[float]
    is_repeat:             bool         # True if same (trigger_type, date) seen in dedup window

    # ── Config snapshot ───────────────────────────────────────────────────────
    config:                TriggerConfig = field(repr=False)

    @property
    def severity_label(self) -> str:
        return self.severity.label()

    @property
    def is_live(self) -> bool:
        """True when the current time is within the decay window."""
        return not self.config.is_expired(
            datetime.fromisoformat(self.classified_at),
        )


# ── Event-type → trigger-type mapping ─────────────────────────────────────────
#
# Maps the strings that appear in MacroEvent.event_type to registry
# trigger_type keys.  Sources:
#   FRED poller    — meta["name"] values from _FRED_SERIES in macro_ingestion.py
#   AV indicators  — meta["name"] values from _AV_INDICATORS
#   Econ calendar  — event names with spaces → underscores
#   Trigger families — existing family names registered in models/triggers.py
#
# The classifier tries exact match first, then keyword substring scan (ordered
# most-specific first), then returns None (event not in scope).

_EXACT_MAP: Dict[str, str] = {
    # ── FRED poller ───────────────────────────────────────────────────────────
    "CPI":                      "CPI_RELEASE",
    "Unemployment":             "NONFARM_PAYROLLS",
    "Fed_Funds_Rate":           "FOMC_RATE_DECISION",
    "PPI_All_Commodities":      "PPI_RELEASE",
    "USD_EUR":                  "DOLLAR_SHOCK",
    "USREC":                    "RECESSION_FLAG",
    # ── Alpha Vantage indicators ──────────────────────────────────────────────
    "CPI_AV":                   "CPI_RELEASE",
    "Inflation_YoY":            "CPI_RELEASE",
    "NFP":                      "NONFARM_PAYROLLS",
    "Nonfarm_Payrolls":         "NONFARM_PAYROLLS",
    "Unemployment_AV":          "NONFARM_PAYROLLS",
    "Fed_Funds_AV":             "FOMC_RATE_DECISION",
    # ── Economic calendar (AV ECONOMIC_CALENDAR) — spaces → underscores ──────
    "CPI_m/m":                  "CPI_RELEASE",
    "Core_CPI_m/m":             "CPI_RELEASE",
    "CPI_y/y":                  "CPI_RELEASE",
    "Core_CPI_y/y":             "CPI_RELEASE",
    "Core_PCE_Price_Index_m/m": "CPI_RELEASE",
    "PCE_Price_Index_m/m":      "CPI_RELEASE",
    "Non-Farm_Employment_Change": "NONFARM_PAYROLLS",
    "Nonfarm_Employment_Change":  "NONFARM_PAYROLLS",
    "ADP_Non-Farm_Employment_Change": "NONFARM_PAYROLLS",
    "ADP_Nonfarm_Employment_Change":  "NONFARM_PAYROLLS",
    "Unemployment_Claims":      "NONFARM_PAYROLLS",
    "FOMC_Statement":           "FOMC_RATE_DECISION",
    "Federal_Funds_Rate":       "FOMC_RATE_DECISION",
    "FOMC_Press_Conference":    "FED_CHAIR_SPEECH",
    "Fed_Chair_Powell_Speaks":  "FED_CHAIR_SPEECH",
    "FOMC_Meeting_Minutes":     "FED_CHAIR_SPEECH",
    "EIA_Crude_Oil_Inventories":       "EIA_CRUDE_INVENTORY",
    "Crude_Oil_Inventories":           "EIA_CRUDE_INVENTORY",
    "EIA_Natural_Gas_Storage":         "EIA_GAS_STORAGE",
    "Natural_Gas_Storage":             "EIA_GAS_STORAGE",
    "WASDE":                    "USDA_WASDE_REPORT",
    "USDA_WASDE":               "USDA_WASDE_REPORT",
    "Crop_Report":              "USDA_WASDE_REPORT",
    "PPI_m/m":                  "PPI_RELEASE",
    "Core_PPI_m/m":             "PPI_RELEASE",
    "PPI_y/y":                  "PPI_RELEASE",
    # ── Existing TriggerFamily names (models/triggers.py) ────────────────────
    "opec_action":              "OPEC_PRODUCTION_DECISION",
    "fed_tightening":           "DOLLAR_SHOCK",
    "weather_shock":            "WEATHER_SHOCK",
    "energy_transition":        "ENERGY_TRANSITION_SIGNAL",
    "fomc_rate_decision":       "FOMC_RATE_DECISION",
    "cpi_release":              "CPI_RELEASE",
    "nonfarm_payrolls":         "NONFARM_PAYROLLS",
    "opec_production_decision": "OPEC_PRODUCTION_DECISION",
    "eia_crude_inventory":      "EIA_CRUDE_INVENTORY",
    "eia_gas_storage":          "EIA_GAS_STORAGE",
    "usda_wasde_report":        "USDA_WASDE_REPORT",
    "fed_chair_speech":         "FED_CHAIR_SPEECH",
    "dollar_shock":             "DOLLAR_SHOCK",
    "geopolitical_shock":       "GEOPOLITICAL_SHOCK",
    "recession_flag":           "RECESSION_FLAG",
    "ppi_release":              "PPI_RELEASE",
}

# Ordered substring rules — checked only when exact match fails.
# Longest / most-specific patterns first to avoid false positives.
_KEYWORD_RULES: List[Tuple[str, str]] = [
    ("non-farm_employment",   "NONFARM_PAYROLLS"),
    ("nonfarm_employment",    "NONFARM_PAYROLLS"),
    ("nonfarm_payroll",       "NONFARM_PAYROLLS"),
    ("unemployment_rate",     "NONFARM_PAYROLLS"),
    ("unemployment_claim",    "NONFARM_PAYROLLS"),
    ("fomc_statement",        "FOMC_RATE_DECISION"),
    ("federal_funds",         "FOMC_RATE_DECISION"),
    ("fed_funds",             "FOMC_RATE_DECISION"),
    ("fomc_press",            "FED_CHAIR_SPEECH"),
    ("jackson_hole",          "FED_CHAIR_SPEECH"),
    ("powell_speaks",         "FED_CHAIR_SPEECH"),
    ("yellen_speaks",         "FED_CHAIR_SPEECH"),
    ("fed_chair",             "FED_CHAIR_SPEECH"),
    ("eia_crude",             "EIA_CRUDE_INVENTORY"),
    ("crude_oil_inventor",    "EIA_CRUDE_INVENTORY"),
    ("eia_natural_gas",       "EIA_GAS_STORAGE"),
    ("natural_gas_storage",   "EIA_GAS_STORAGE"),
    ("gas_storage",           "EIA_GAS_STORAGE"),
    ("wasde",                 "USDA_WASDE_REPORT"),
    ("usda_crop",             "USDA_WASDE_REPORT"),
    ("opec",                  "OPEC_PRODUCTION_DECISION"),
    ("core_pce",              "CPI_RELEASE"),
    ("core_cpi",              "CPI_RELEASE"),
    ("cpi_m",                 "CPI_RELEASE"),
    ("cpi_y",                 "CPI_RELEASE"),
    ("inflation",             "CPI_RELEASE"),
    ("ppi_m",                 "PPI_RELEASE"),
    ("ppi_y",                 "PPI_RELEASE"),
    ("producer_price",        "PPI_RELEASE"),
    ("dxy",                   "DOLLAR_SHOCK"),
    ("dollar_index",          "DOLLAR_SHOCK"),
    ("usd_",                  "DOLLAR_SHOCK"),
    ("usrec",                 "RECESSION_FLAG"),
    ("recession",             "RECESSION_FLAG"),
    ("geopolit",              "GEOPOLITICAL_SHOCK"),
    ("sanction",              "GEOPOLITICAL_SHOCK"),
    ("enso",                  "WEATHER_SHOCK"),
    ("mei_index",             "WEATHER_SHOCK"),
    ("battery_demand",        "ENERGY_TRANSITION_SIGNAL"),
    ("ets_stress",            "ENERGY_TRANSITION_SIGNAL"),
    ("energy_transition",     "ENERGY_TRANSITION_SIGNAL"),
]


def _resolve_trigger_type(event_type: str, registry: TriggerRegistry) -> Optional[str]:
    """
    Map a MacroEvent.event_type string to a registry trigger_type.

    Strategy (in order):
      1. Exact match in _EXACT_MAP
      2. Exact match in registry (trigger_type or family_name)
      3. Substring keyword scan (longest match wins)
      4. None → event is out of scope for this classifier
    """
    # 1. Exact match in hard-coded map
    tt = _EXACT_MAP.get(event_type)
    if tt:
        return tt

    # 2. Exact match against registry keys (trigger_type or family_name)
    if event_type in registry:
        return event_type
    cfg_by_family = registry.get_by_family(event_type)
    if cfg_by_family:
        return cfg_by_family.trigger_type

    # 3. Case-insensitive substring keyword scan
    lower = event_type.lower()
    for keyword, trigger_type in _KEYWORD_RULES:
        if keyword in lower:
            return trigger_type

    return None


# ── Classifier ─────────────────────────────────────────────────────────────────

class TriggerClassifier:
    """
    Stateful classifier that converts MacroEvents into TriggerSignals.

    Parameters
    ----------
    registry : TriggerRegistry
        The loaded trigger config registry (defaults to the module singleton).
    dedup_window_hours : float
        How long (hours) to remember a (trigger_type, release_date) pair for
        repeat-event detection. Default 24 h covers same-day re-polling.
    """

    def __init__(
        self,
        registry: TriggerRegistry = REGISTRY,
        dedup_window_hours: float = 24.0,
    ) -> None:
        self._registry     = registry
        self._dedup_window = timedelta(hours=dedup_window_hours)
        # Maps (trigger_type, date_str) → first-seen datetime (UTC)
        self._seen: Dict[Tuple[str, str], datetime] = {}

    # ── Public interface ───────────────────────────────────────────────────────

    def classify(self, event: MacroEvent) -> Optional["TriggerSignal"]:
        """
        Classify one MacroEvent.

        Returns None when:
          - actual_value is missing (no observable signal)
          - event_type maps to no registry entry (out of scope)

        Returns a TriggerSignal with is_repeat=True when the same
        (trigger_type, release_date) arrived within the dedup window.
        """
        if event.actual_value is None:
            log.debug("classify: skip %s — no actual_value", event.event_type)
            return None

        trigger_type = _resolve_trigger_type(event.event_type, self._registry)
        if trigger_type is None:
            log.debug("classify: no registry match for event_type=%r", event.event_type)
            return None

        config = self._registry.get(trigger_type)
        if config is None:
            log.warning("classify: trigger_type %r resolved but not in registry", trigger_type)
            return None

        magnitude, method = self._compute_magnitude(event, config)
        severity          = self._classify_severity(magnitude, config)
        direction         = self._classify_direction(event)
        now               = datetime.now(timezone.utc)
        is_repeat         = self._check_and_record(trigger_type, event.release_timestamp, now)

        decay_expires = now + timedelta(minutes=config.decay_window_minutes)

        signal = TriggerSignal(
            signal_id             = str(uuid.uuid4()),
            trigger_type          = trigger_type,
            family_name           = config.family_name,
            source_event_id       = event.trigger_id,
            magnitude_score       = magnitude,
            severity              = severity,
            direction             = direction,
            release_timestamp     = event.release_timestamp,
            classified_at         = now.isoformat(),
            decay_expires_at      = decay_expires.isoformat(),
            affected_commodities  = config.affected_commodities,
            affected_clusters     = config.affected_clusters,
            model_targets         = config.model_targets,
            cascade_priority      = config.cascade_priority,
            classification_method = method,
            raw_deviation_score   = event.deviation_score,
            event_type            = event.event_type,
            actual_value          = event.actual_value,
            expected_value        = event.expected_value,
            is_repeat             = is_repeat,
            config                = config,
        )

        log.info(
            "SIGNAL  %s  [%s]  magnitude=%.3f  dir=%s  repeat=%s",
            trigger_type, severity.label(), magnitude, direction, is_repeat,
        )
        return signal

    def classify_batch(self, events: List[MacroEvent]) -> List["TriggerSignal"]:
        """Classify a list of events; None results are silently dropped."""
        signals: List[TriggerSignal] = []
        for ev in events:
            try:
                sig = self.classify(ev)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                log.warning("classify_batch: error on %s — %s", ev.event_type, exc)
        return signals

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _compute_magnitude(
        self,
        event: MacroEvent,
        config: TriggerConfig,
    ) -> Tuple[float, str]:
        """
        Compute a magnitude score in [0, 1] and name the method used.

        Z-score is always available; surprise is blended in (weight 0.40)
        when expected_value is not None and non-zero.
        """
        z_score_norm = min(abs(event.deviation_score) / Z_MAX, 1.0)
        method       = "z_score"

        if event.expected_value is not None and event.actual_value is not None:
            expected = event.expected_value
            actual   = event.actual_value
            denom    = abs(expected) + 1e-8
            if abs(expected) > 1e-6:                           # avoid near-zero division
                pct_surprise   = abs(actual - expected) / denom
                surprise_norm  = min(pct_surprise * SURPRISE_SCALE, 1.0)
                magnitude      = 0.60 * z_score_norm + 0.40 * surprise_norm
                method         = "blended_surprise"
            else:
                magnitude = z_score_norm
        else:
            magnitude = z_score_norm

        # High-impact floor — never let an officially "high" event slip below threshold
        if event.impact == "high" and magnitude < IMPACT_FLOOR:
            magnitude = IMPACT_FLOOR
            method   += "_impact_floor"

        return round(min(magnitude, 1.0), 4), method

    @staticmethod
    def _classify_severity(magnitude: float, config: TriggerConfig) -> SeverityTier:
        """
        Map magnitude [0, 1] to a SeverityTier using the config threshold bands.
        Works highest-tier-first so the highest matching tier always wins.
        """
        for tier_name, st in (
            ("severe",   SeverityTier.CRITICAL),
            ("moderate", SeverityTier.HIGH),
            ("mild",     SeverityTier.MEDIUM),
        ):
            band = config.thresholds.get(tier_name)
            if band and band.contains(magnitude):
                return st
        return SeverityTier.LOW

    @staticmethod
    def _classify_direction(event: MacroEvent) -> str:
        d = event.deviation_score
        if abs(d) < 0.10:
            return "neutral"
        return "upside_surprise" if d > 0 else "downside_surprise"

    def _check_and_record(
        self,
        trigger_type: str,
        release_timestamp: str,
        now: datetime,
    ) -> bool:
        """
        Return True (is_repeat) if (trigger_type, date) was seen within the
        dedup window. Always records the new observation.
        """
        try:
            date_str = release_timestamp[:10]     # YYYY-MM-DD
        except Exception:
            date_str = now.date().isoformat()

        key      = (trigger_type, date_str)
        is_repeat = False

        if key in self._seen:
            age = now - self._seen[key]
            if age <= self._dedup_window:
                is_repeat = True
            else:
                # Stale — update to newest timestamp so window refreshes
                self._seen[key] = now
        else:
            self._seen[key] = now

        return is_repeat

    def flush_dedup_cache(self) -> None:
        """Evict dedup entries older than the window. Call periodically to limit memory use."""
        now   = datetime.now(timezone.utc)
        stale = [k for k, ts in self._seen.items() if (now - ts) > self._dedup_window]
        for k in stale:
            del self._seen[k]


# ── Module-level singleton & convenience wrapper ───────────────────────────────

CLASSIFIER: TriggerClassifier = TriggerClassifier(REGISTRY)


def classify_event(event: MacroEvent) -> Optional[TriggerSignal]:
    """Classify one event using the module-level singleton classifier."""
    return CLASSIFIER.classify(event)
