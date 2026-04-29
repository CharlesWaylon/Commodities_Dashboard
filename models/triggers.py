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

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


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


# ── Default families (from Model Integration Roadmap) ─────────────────────────
# Commodity names match the canonical display names in CORE_TICKERS / DB.
DEFAULT_FAMILIES: Tuple[TriggerFamily, ...] = (
    TriggerFamily(
        name="opec_action",
        description="OPEC+ meeting / production decision affecting crude complex",
        affected_commodities=(
            "WTI Crude Oil", "Brent Crude Oil",
            "Natural Gas (Henry Hub)", "Gasoline (RBOB)", "Heating Oil",
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
            "Copper (COMEX)", "Silver (COMEX)",
            # Lithium / Rare Earths / Uranium will be added when those tickers
            # land in CORE_TICKERS; the family extends without code changes.
        ),
        source="research",
    ),
)

for _f in DEFAULT_FAMILIES:
    register_trigger_family(_f)
