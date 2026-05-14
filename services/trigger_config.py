"""
Trigger Configuration Registry
================================
Loads, validates, and hydrates the JSON trigger config at
``config/trigger_registry.json`` into typed Python objects.

Each ``TriggerConfig`` describes one macro trigger type:

  trigger_type          — canonical SCREAMING_SNAKE_CASE identifier
  display_name          — human-readable label for UI / logs
  source                — "calendar" | "market" | "fred" | "alpha_vantage" | "research"
  description           — plain-text rationale
  thresholds            — {"mild": ThresholdBand, "moderate": ..., "severe": ...}
  affected_clusters     — sector names (energy, metals, agriculture, livestock, digital)
  affected_commodities  — resolved commodity display names (clusters + explicit overrides)
  cascade_priority      — int 1 (highest) – 5 (lowest)
  decay_window_minutes  — how long after firing a signal remains "live"
  model_targets         — list of model IDs that should respond to this trigger

Public API
----------
  load_trigger_registry(path=None)  → TriggerRegistry
  REGISTRY                          — module-level singleton (loaded once at import)

Integration with models/triggers.py
-------------------------------------
On load, every entry is registered as a ``TriggerFamily`` so the existing
``detect_all()`` and ``SignalRouter.make_event()`` machinery picks it up without
any changes to those modules.

Usage
-----
  from services.trigger_config import REGISTRY

  cfg = REGISTRY.get("CPI_RELEASE")
  severity = REGISTRY.severity("CPI_RELEASE", strength=0.62)   # → "moderate"
  expired   = REGISTRY.is_expired("CPI_RELEASE", fired_at_dt)  # → bool
  high_pri  = REGISTRY.by_priority(max_priority=2)             # → list[TriggerConfig]
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "trigger_registry.json"

VALID_SOURCES   = {"calendar", "market", "fred", "alpha_vantage", "research"}
VALID_CLUSTERS  = {"energy", "metals", "agriculture", "livestock", "digital"}
VALID_MODEL_TARGETS = {
    "macro_router", "xgboost", "random_forest", "lstm",
    "quantum", "meta_predictor", "all",
}
VALID_SEVERITIES = ("mild", "moderate", "severe")

# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThresholdBand:
    """A [min, max] strength interval for a severity tier."""
    min: float
    max: float
    label: str = ""

    def contains(self, strength: float) -> bool:
        return self.min <= strength <= self.max


@dataclass(frozen=True)
class TriggerConfig:
    """
    Fully resolved, validated configuration for one trigger type.
    Immutable after construction; safe to share across threads.
    """
    trigger_type:          str
    family_name:           str                        # snake_case used in triggers.py
    display_name:          str
    source:                str
    description:           str
    thresholds:            Dict[str, ThresholdBand]
    affected_clusters:     Tuple[str, ...]
    affected_commodities:  Tuple[str, ...]            # resolved from clusters + explicit list
    cascade_priority:      int                        # 1 (highest) – 5 (lowest)
    decay_window_minutes:  int
    model_targets:         Tuple[str, ...]

    def severity(self, strength: float) -> Optional[str]:
        """Return the severity tier name for a given strength, or None if below all bands."""
        for tier in reversed(VALID_SEVERITIES):         # severe → moderate → mild
            band = self.thresholds.get(tier)
            if band and band.contains(strength):
                return tier
        return None

    def is_expired(self, fired_at: datetime, now: Optional[datetime] = None) -> bool:
        """Return True if ``decay_window_minutes`` has elapsed since ``fired_at``."""
        if now is None:
            now = datetime.now(timezone.utc)
        elapsed = (now - fired_at).total_seconds() / 60.0
        return elapsed > self.decay_window_minutes

    def notifies(self, model_id: str) -> bool:
        """Return True if this trigger should notify the given model."""
        return "all" in self.model_targets or model_id in self.model_targets


# ── Registry ───────────────────────────────────────────────────────────────────

class TriggerRegistry:
    """
    In-process registry of all loaded TriggerConfig objects.
    Populated by ``load_trigger_registry()``; consumed everywhere else.
    """

    def __init__(self, configs: List[TriggerConfig]) -> None:
        self._by_type:   Dict[str, TriggerConfig] = {c.trigger_type: c for c in configs}
        self._by_family: Dict[str, TriggerConfig] = {c.family_name:  c for c in configs}

    # ── Lookups ────────────────────────────────────────────────────────────────

    def get(self, trigger_type: str) -> Optional[TriggerConfig]:
        """Fetch by SCREAMING_SNAKE_CASE trigger_type."""
        return self._by_type.get(trigger_type)

    def get_by_family(self, family_name: str) -> Optional[TriggerConfig]:
        """Fetch by snake_case family_name (models/triggers.py compatible key)."""
        return self._by_family.get(family_name)

    def all(self) -> List[TriggerConfig]:
        return list(self._by_type.values())

    def __len__(self) -> int:
        return len(self._by_type)

    def __contains__(self, trigger_type: str) -> bool:
        return trigger_type in self._by_type

    # ── Filtered views ─────────────────────────────────────────────────────────

    def by_priority(self, max_priority: int = 5) -> List[TriggerConfig]:
        """Return configs with cascade_priority <= max_priority, sorted ascending."""
        return sorted(
            [c for c in self._by_type.values() if c.cascade_priority <= max_priority],
            key=lambda c: c.cascade_priority,
        )

    def for_cluster(self, cluster: str) -> List[TriggerConfig]:
        """Return configs that affect the given commodity cluster."""
        return [c for c in self._by_type.values() if cluster in c.affected_clusters]

    def for_commodity(self, commodity_name: str) -> List[TriggerConfig]:
        """Return configs whose resolved commodity list includes this name."""
        return [c for c in self._by_type.values() if commodity_name in c.affected_commodities]

    def for_model(self, model_id: str) -> List[TriggerConfig]:
        """Return configs that target a given model (or have 'all')."""
        return [c for c in self._by_type.values() if c.notifies(model_id)]

    def by_source(self, source: str) -> List[TriggerConfig]:
        return [c for c in self._by_type.values() if c.source == source]

    # ── Convenience helpers ────────────────────────────────────────────────────

    def severity(self, trigger_type: str, strength: float) -> Optional[str]:
        cfg = self.get(trigger_type)
        return cfg.severity(strength) if cfg else None

    def is_expired(
        self,
        trigger_type: str,
        fired_at: datetime,
        now: Optional[datetime] = None,
    ) -> bool:
        cfg = self.get(trigger_type)
        return cfg.is_expired(fired_at, now) if cfg else True

    def decay_minutes(self, trigger_type: str) -> Optional[int]:
        cfg = self.get(trigger_type)
        return cfg.decay_window_minutes if cfg else None

    def affected_commodities(self, trigger_type: str) -> Tuple[str, ...]:
        cfg = self.get(trigger_type)
        return cfg.affected_commodities if cfg else ()

    def summary_table(self) -> List[dict]:
        """Return a list of dicts suitable for pd.DataFrame or st.dataframe."""
        rows = []
        for c in sorted(self._by_type.values(), key=lambda x: x.cascade_priority):
            rows.append({
                "trigger_type":        c.trigger_type,
                "display_name":        c.display_name,
                "source":              c.source,
                "priority":            c.cascade_priority,
                "decay_hrs":           round(c.decay_window_minutes / 60, 1),
                "clusters":            ", ".join(c.affected_clusters),
                "n_commodities":       len(c.affected_commodities),
                "model_targets":       ", ".join(c.model_targets),
                "has_severe_band":     "severe" in c.thresholds,
            })
        return rows


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate_entry(entry: dict, idx: int) -> None:
    """Raise ValueError with a descriptive message if entry is invalid."""
    label = entry.get("trigger_type", f"entry[{idx}]")

    required = [
        "trigger_type", "display_name", "source", "description",
        "thresholds", "cascade_priority", "decay_window_minutes", "model_targets",
    ]
    for key in required:
        if key not in entry:
            raise ValueError(f"[{label}] missing required field: '{key}'")

    source = entry["source"]
    if source not in VALID_SOURCES:
        raise ValueError(
            f"[{label}] invalid source '{source}'. Must be one of: {sorted(VALID_SOURCES)}"
        )

    priority = entry["cascade_priority"]
    if not isinstance(priority, int) or not (1 <= priority <= 5):
        raise ValueError(
            f"[{label}] cascade_priority={priority!r} is invalid. Must be int 1–5."
        )

    decay = entry["decay_window_minutes"]
    if not isinstance(decay, int) or decay < 1:
        raise ValueError(
            f"[{label}] decay_window_minutes={decay!r} must be a positive int."
        )

    thresholds = entry["thresholds"]
    if not isinstance(thresholds, dict) or len(thresholds) == 0:
        raise ValueError(f"[{label}] 'thresholds' must be a non-empty dict.")
    if "mild" not in thresholds:
        raise ValueError(f"[{label}] 'thresholds' must define at least a 'mild' band.")
    for tier, band in thresholds.items():
        if tier not in VALID_SEVERITIES:
            raise ValueError(
                f"[{label}] unknown threshold tier '{tier}'. "
                f"Valid tiers: {VALID_SEVERITIES}"
            )
        if "min" not in band or "max" not in band:
            raise ValueError(f"[{label}] threshold '{tier}' must have 'min' and 'max'.")
        lo, hi = band["min"], band["max"]
        if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
            raise ValueError(f"[{label}] threshold '{tier}' min/max must be numeric.")
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(
                f"[{label}] threshold '{tier}' requires 0 ≤ min < max ≤ 1. "
                f"Got min={lo}, max={hi}."
            )

    clusters = entry.get("affected_clusters", [])
    for cl in clusters:
        if cl not in VALID_CLUSTERS:
            raise ValueError(
                f"[{label}] unknown cluster '{cl}'. "
                f"Valid clusters: {sorted(VALID_CLUSTERS)}"
            )

    targets = entry.get("model_targets", [])
    for t in targets:
        if t not in VALID_MODEL_TARGETS:
            raise ValueError(
                f"[{label}] unknown model_target '{t}'. "
                f"Valid targets: {sorted(VALID_MODEL_TARGETS)}"
            )


# ── Commodity cluster resolver ─────────────────────────────────────────────────

def _resolve_commodities(
    clusters: List[str],
    explicit: List[str],
) -> Tuple[str, ...]:
    """
    Build the full affected-commodity list by:
      1. Expanding each cluster to its MODELING_COMMODITIES members.
      2. Merging explicit commodity names (override / extend).
      3. Deduplicating while preserving insertion order.

    Returns an empty tuple if both inputs are empty (receiver covers all).
    """
    try:
        from models.config import COMMODITY_SECTORS, MODELING_COMMODITIES
        cluster_commodities: List[str] = [
            name for name, sector in COMMODITY_SECTORS.items()
            if sector in clusters
        ]
    except ImportError:
        log.warning("_resolve_commodities: models.config unavailable; using explicit list only")
        cluster_commodities = []

    seen:   set        = set()
    result: List[str]  = []
    for name in cluster_commodities + list(explicit):
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return tuple(result)


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_trigger_registry(
    path: Optional[Path] = None,
    register_families: bool = True,
) -> TriggerRegistry:
    """
    Load, validate, and hydrate the trigger config JSON into a TriggerRegistry.

    Parameters
    ----------
    path : Path, optional
        Override the default config location (``config/trigger_registry.json``).
    register_families : bool
        If True (default), register each loaded config as a TriggerFamily in
        ``models.triggers``, making them visible to ``detect_all()`` and the
        SignalRouter without any additional wiring.

    Returns
    -------
    TriggerRegistry
        Thread-safe registry ready for use across the dashboard.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the JSON fails schema validation (descriptive message included).
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Trigger registry not found at {config_path}. "
            "Check that config/trigger_registry.json is present."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Top-level structure
    if "version" not in raw:
        raise ValueError("trigger_registry.json must have a top-level 'version' field.")
    if "triggers" not in raw or not isinstance(raw["triggers"], list):
        raise ValueError("trigger_registry.json must have a top-level 'triggers' list.")

    log.info(
        "Loading trigger registry v%s from %s (%d entries)",
        raw["version"], config_path, len(raw["triggers"]),
    )

    configs: List[TriggerConfig] = []

    for idx, entry in enumerate(raw["triggers"]):
        # Validate — raises ValueError with descriptive message on failure
        _validate_entry(entry, idx)

        trigger_type = entry["trigger_type"]
        family_name  = entry.get("family_name", trigger_type.lower())

        # Build ThresholdBand objects
        thresholds: Dict[str, ThresholdBand] = {}
        for tier, band in entry["thresholds"].items():
            thresholds[tier] = ThresholdBand(
                min=float(band["min"]),
                max=float(band["max"]),
                label=band.get("label", ""),
            )

        # Resolve affected commodities from clusters + explicit overrides
        commodities = _resolve_commodities(
            clusters=entry.get("affected_clusters", []),
            explicit=entry.get("affected_commodities", []),
        )

        cfg = TriggerConfig(
            trigger_type         = trigger_type,
            family_name          = family_name,
            display_name         = entry["display_name"],
            source               = entry["source"],
            description          = entry["description"],
            thresholds           = thresholds,
            affected_clusters    = tuple(entry.get("affected_clusters", [])),
            affected_commodities = commodities,
            cascade_priority     = int(entry["cascade_priority"]),
            decay_window_minutes = int(entry["decay_window_minutes"]),
            model_targets        = tuple(entry.get("model_targets", [])),
        )
        configs.append(cfg)

        if register_families:
            _register_as_family(cfg)

    registry = TriggerRegistry(configs)
    log.info("Trigger registry loaded: %d configs, priorities 1–%d", len(registry), 5)
    return registry


def _register_as_family(cfg: TriggerConfig) -> None:
    """Register a TriggerConfig as a TriggerFamily in models.triggers."""
    try:
        from models.triggers import TriggerFamily, register_trigger_family
        register_trigger_family(TriggerFamily(
            name                 = cfg.family_name,
            description          = cfg.display_name + " — " + cfg.description[:120],
            affected_commodities = cfg.affected_commodities,
            source               = cfg.source,
        ))
    except Exception as exc:
        log.warning("Could not register %s as TriggerFamily: %s", cfg.trigger_type, exc)


# ── Module-level singleton ─────────────────────────────────────────────────────
# Loaded once at import time; any validation error surfaces immediately on startup.

def _init_registry() -> TriggerRegistry:
    try:
        return load_trigger_registry()
    except Exception as exc:
        log.error("Failed to load trigger registry: %s — returning empty registry", exc)
        return TriggerRegistry([])


REGISTRY: TriggerRegistry = _init_registry()
