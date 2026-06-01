"""
Trigger schema — macro events that ripple through the model ecosystem.

A TriggerFamily defines:
  - which commodities are affected when this trigger fires
  - human-readable description and source category

A TriggerEvent is a specific firing of a family with a measured strength.
Detection logic (calendar, market data, proprietary research) emits
TriggerEvents; the SignalRouter consumes them to reweight model confidence.

────────────────────────────────────────────────────────────────────────
Extension point — proprietary research
────────────────────────────────────────────────────────────────────────
The dashboard's long-term goal is to host *patented causal research*:
custom signals from quantified headlines, policy shifts, and macro
chains that are bespoke to this firm. To plug a new family in:

    from models.triggers import TriggerFamily, register_trigger_family

    register_trigger_family(TriggerFamily(
        name="my_research_signal",
        description="Quantified OPEC+ headline sentiment z-score",
        affected_commodities=("WTI Crude Oil", "Brent Crude Oil"),
        source="research",
    ))

The router auto-discovers it. No core code change needed.
"""

import json
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

_REGISTRY_JSON = (
    pathlib.Path(__file__).resolve().parent.parent / "config" / "trigger_registry.json"
)


@dataclass(frozen=True)
class TriggerFamily:
    """
    Definition of a trigger type — frozen because families are registered
    once at import time and shouldn't mutate during a session.
    """
    name: str                                    # canonical id, e.g. "opec_action"
    description: str                             # human-readable
    affected_commodities: Tuple[str, ...]        # tuple for hashability
    source: str = "calendar"                     # "calendar" | "market" | "research"


@dataclass
class TriggerEvent:
    """A single firing of a TriggerFamily with measured strength in [0, 1]."""
    family: str                                  # TriggerFamily.name
    strength: float                              # 0.0 – 1.0
    detected_at: str                             # ISO 8601 timestamp
    affected_commodities: List[str]              # snapshot at detection time
    rationale: str = ""                          # why detected (e.g., "DXY z=2.4 for 3d")
    metadata: Dict = field(default_factory=dict)


# ── Registry ──────────────────────────────────────────────────────────────────
_REGISTRY: Dict[str, TriggerFamily] = {}


def register_trigger_family(family: TriggerFamily) -> None:
    """Register a trigger family. Idempotent; re-registering overwrites."""
    _REGISTRY[family.name] = family


def get_trigger_family(name: str) -> Optional[TriggerFamily]:
    return _REGISTRY.get(name)


def all_trigger_families() -> List[TriggerFamily]:
    return list(_REGISTRY.values())


# ── Representative commodities per cluster ────────────────────────────────────
# Used to populate affected_commodities when the registry JSON entry leaves
# that field empty.  Names must match MODELING_COMMODITIES keys (models/config.py).
_CLUSTER_REPS: Dict[str, Tuple[str, ...]] = {
    "energy":      ("WTI Crude Oil", "Brent Crude Oil", "Natural Gas"),
    "metals":      ("Gold (COMEX)", "Silver (COMEX)", "Copper (COMEX)"),
    "agriculture": ("Corn (CBOT)", "Wheat (CBOT SRW)", "Soybeans (CBOT)"),
    "livestock":   ("Live Cattle", "Lean Hogs"),
    "digital":     ("Bitcoin",),
}

# Primary clusters per family — overrides the catch-all affected_clusters list
# in the registry JSON so the chain's "Affects:" line shows the most relevant
# instruments rather than the same three energy names for every trigger.
_PRIMARY_CLUSTERS: Dict[str, Tuple[str, ...]] = {
    "fomc_rate_decision":    ("metals",),
    "cpi_release":           ("metals", "energy"),
    "nonfarm_payrolls":      ("metals",),
    "opec_action":           ("energy",),
    "eia_crude_inventory":   ("energy",),
    "eia_gas_storage":       ("energy",),
    "usda_wasde_report":     ("agriculture",),
    "fed_chair_speech":      ("metals",),
    "fed_tightening":        ("metals", "energy"),
    "geopolitical_shock":    ("energy", "metals"),
    "recession_flag":        ("energy", "metals", "agriculture"),
    "weather_shock":         ("agriculture", "energy"),
    "energy_transition":     ("metals",),
    "ppi_release":           ("energy", "metals"),
}


# ── Fallback families — used only when trigger_registry.json is unavailable ───
DEFAULT_FAMILIES: Tuple[TriggerFamily, ...] = (
    TriggerFamily(
        name="opec_action",
        description="OPEC+ meeting / production decision affecting crude complex",
        affected_commodities=(
            "WTI Crude Oil", "Brent Crude Oil",
            "Natural Gas", "Gasoline (RBOB)", "Heating Oil",
        ),
        source="calendar",
    ),
    TriggerFamily(
        name="fed_tightening",
        description="Fed hike / DXY shock — pressures USD-sensitive metals",
        affected_commodities=(
            "Gold (COMEX)", "Silver (COMEX)", "Copper (COMEX)",
        ),
        source="market",
    ),
    TriggerFamily(
        name="weather_shock",
        description="Drought / freeze / climate disruption affecting agriculture",
        affected_commodities=(
            "Corn (CBOT)", "Wheat (CBOT SRW)", "Soybeans (CBOT)",
        ),
        source="market",
    ),
    TriggerFamily(
        name="energy_transition",
        description="Renewable capacity / battery demand / EV adoption shift",
        affected_commodities=(
            "Copper (COMEX)", "Silver (COMEX)", "Lithium*", "Uranium*",
        ),
        source="research",
    ),
)

for _f in DEFAULT_FAMILIES:
    register_trigger_family(_f)


# ── Load full registry from config/trigger_registry.json ─────────────────────
# Entries in the JSON overwrite the DEFAULT_FAMILIES above (same family_name
# keys), so the richer registry descriptions and thresholds take precedence.
# Any families not already registered are added fresh.
def _load_registry() -> None:
    try:
        with open(_REGISTRY_JSON, encoding="utf-8") as _fh:
            _data = json.load(_fh)
        for _entry in _data.get("triggers", []):
            _fname = _entry.get("family_name", "").strip()
            if not _fname:
                continue
            # Use explicit affected_commodities if provided; else use the
            # primary clusters for this family (more specific than the catch-all
            # affected_clusters list in the JSON which covers all sectors).
            _affected: Tuple[str, ...] = tuple(_entry.get("affected_commodities") or ())
            if not _affected:
                _primary = _PRIMARY_CLUSTERS.get(_fname)
                _clusters = _primary if _primary else tuple(_entry.get("affected_clusters", []))
                _affected = tuple(
                    _c
                    for _cl in _clusters
                    for _c in _CLUSTER_REPS.get(_cl, ())
                )
            register_trigger_family(TriggerFamily(
                name=_fname,
                description=_entry.get("display_name", _fname.replace("_", " ").title()),
                affected_commodities=_affected,
                source=_entry.get("source", "calendar"),
            ))
    except Exception:
        pass   # silently fall back to DEFAULT_FAMILIES if JSON is missing / malformed

_load_registry()
