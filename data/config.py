"""
Data-layer feature flags.

Follows the existing MACRO_TRIGGERS_ENABLED pattern (CLAUDE.md Evolution Rule #2):
every new analytical surface ships behind an env-var flag so the old path stays
default and rollback is a flag flip — no redeploy.
"""

from __future__ import annotations

import os


def load_env() -> None:
    """
    Load .env into the process environment (idempotent, best-effort).

    The ingest runners call this on entry so API keys resolve identically whether
    invoked from an interactive shell or from launchd (which has no exported shell
    environment). Mirrors the load_dotenv() convention in features/macro_overlays.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Gates the new free fundamental feeds (COT / EIA / USDA) and the point-in-time
# fundamental store. Default OFF until the ingestors are proven on real data, so
# the existing price-only pipeline is unaffected by Phase 1 landing.
FUNDAMENTAL_FEEDS_ENABLED: bool = _flag("FUNDAMENTAL_FEEDS_ENABLED", False)

# Gates the data-health console surface (Phase 1 presentation). Independent of
# the feeds flag so the health page can show price-layer health alone.
DATA_HEALTH_ENABLED: bool = _flag("DATA_HEALTH_ENABLED", False)


def fundamental_feeds_enabled() -> bool:
    """Re-read at call time so tests / runtime toggles take effect without reimport."""
    return _flag("FUNDAMENTAL_FEEDS_ENABLED", False)


def data_health_enabled() -> bool:
    return _flag("DATA_HEALTH_ENABLED", False)
