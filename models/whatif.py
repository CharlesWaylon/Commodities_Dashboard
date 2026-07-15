"""In-memory what-if helpers (spec §F3). Pure functions — no DB, no files.

Implements the documented cascade formula (CLAUDE.md, economic-prior
methodology): effective_prior = (1 - alpha) + alpha * prior;
upstream contribution = corr * upstream_forecast * damping * effective_prior.
"""
from __future__ import annotations

from models.config import SECTOR_TRANSMISSION_PRIORS


def blended_prior(economic_prior: float, alpha: float) -> float:
    return (1.0 - alpha) + alpha * economic_prior


def upstream_contribution(corr: float, upstream_forecast: float, damping: float,
                          economic_prior: float, alpha: float) -> float:
    return corr * upstream_forecast * damping * blended_prior(economic_prior, alpha)


def prior_table(alpha: float, damping: float,
                corr: float = 0.4, upstream_forecast: float = 0.01) -> list[dict]:
    """One row per configured transmission edge at the sandbox settings.

    corr/upstream_forecast are illustrative constants so the table isolates
    what alpha and damping change; the page labels them as such.
    """
    rows = []
    for src, targets in SECTOR_TRANSMISSION_PRIORS.items():
        for dst, prior in targets.items():
            rows.append(dict(
                src=src, dst=dst, prior=prior,
                effective=round(blended_prior(prior, alpha), 4),
                contribution=round(
                    upstream_contribution(corr, upstream_forecast, damping, prior, alpha), 6),
            ))
    return rows
