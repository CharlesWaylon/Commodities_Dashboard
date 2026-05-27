"""
measure_trigger_impact.py — quantify how much the macro-trigger surface
moves cascade forecasts today.

Runs the cascade twice — once with ``MACRO_TRIGGERS_ENABLED=false`` and once
with it on — and reports per-sector forecast deltas. This is an interim
proxy for the spec's "IC delta" acceptance bar (Steps 2/3/4): a true IC
measurement would need many days of realised returns, but the per-commodity
forecast diff is a cheap, honest signal of whether triggers are nudging
predictions at all.

Usage
-----
    python -m scripts.measure_trigger_impact

The script writes a markdown table to stdout suitable for dropping into a
PR description. If trigger_events has no high-strength rows today (which is
the case at the time of writing, 2026-05-27), all deltas will be ~0 — that
itself is a useful audit signal.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Make repo root importable when run as `python scripts/measure_trigger_impact.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _run_cascade_with_flag(flag_value: str) -> Dict[str, float]:
    """Run the cascade once with the env flag set, return commodity → final_fc."""
    os.environ["MACRO_TRIGGERS_ENABLED"] = flag_value

    # Re-import to make sure module-level flag reads are honoured for this run.
    for mod in [
        "features.macro_features",
        "models.cascade_orchestrator",
        "models.sector_model",
        "models.macro_router",
    ]:
        if mod in sys.modules:
            del sys.modules[mod]

    from models.cascade_orchestrator import run_cascade

    result = run_cascade(dry_run=True)
    if not result.success:
        raise RuntimeError(f"cascade failed (flag={flag_value!r}): {result.errors}")
    return {c: cf.final_forecast for c, cf in result.commodities.items()}


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(name)s: %(message)s")

    print("Running cascade with MACRO_TRIGGERS_ENABLED=false …", file=sys.stderr)
    off = _run_cascade_with_flag("false")
    print("Running cascade with MACRO_TRIGGERS_ENABLED=true …", file=sys.stderr)
    on  = _run_cascade_with_flag("true")

    from models.config import COMMODITY_SECTORS

    rows = []
    for commodity in sorted(set(off) | set(on)):
        a = off.get(commodity, np.nan)
        b = on.get(commodity, np.nan)
        rows.append({
            "commodity":   commodity,
            "sector":      COMMODITY_SECTORS.get(commodity, "unknown"),
            "flag_off":    a,
            "flag_on":     b,
            "delta_bps":   (b - a) * 1e4 if (np.isfinite(a) and np.isfinite(b)) else np.nan,
        })
    df = pd.DataFrame(rows)

    # Per-sector aggregate
    print()
    print("# Trigger-impact report — cascade forecasts today")
    print()
    print(f"Generated: {pd.Timestamp.utcnow().isoformat()}")
    print(f"Commodities compared: {len(df)}")
    print()
    print("## Per-sector summary")
    print()
    print("| Sector | n | mean Δ (bps) | max |Δ| (bps) | n commodities moved ≥ 1 bp |")
    print("|---|---:|---:|---:|---:|")
    for sector, sub in df.groupby("sector"):
        mean_d  = sub["delta_bps"].mean()
        max_abs = sub["delta_bps"].abs().max()
        n_moved = int((sub["delta_bps"].abs() >= 1.0).sum())
        print(f"| {sector} | {len(sub)} | {mean_d:+.3f} | {max_abs:.3f} | {n_moved} |")

    # Top movers across all sectors
    print()
    print("## Top movers (by |Δ| bps)")
    print()
    print("| Commodity | Sector | flag_off (bps) | flag_on (bps) | Δ (bps) |")
    print("|---|---|---:|---:|---:|")
    top = df.reindex(df["delta_bps"].abs().sort_values(ascending=False).index).head(10)
    for _, r in top.iterrows():
        print(
            f"| {r['commodity']} | {r['sector']} "
            f"| {r['flag_off']*1e4:+.3f} | {r['flag_on']*1e4:+.3f} | {r['delta_bps']:+.4f} |"
        )

    # Empty / zero-shock disclosure
    n_moved_total = int((df["delta_bps"].abs() >= 1.0).sum())
    print()
    if n_moved_total == 0:
        print(
            "> **Note:** every Δ is < 1 bp today, meaning the trigger surface is wired up "
            "but no historical trigger meets the strength thresholds that drive amplification "
            "(0.5), upstream boost (0.5), or regime override (0.8). As the macro-feed daemon "
            "accumulates strong shocks, this report will show real divergence."
        )
    else:
        print(
            f"> **Note:** {n_moved_total} commodity forecast(s) moved by ≥ 1 bp when triggers "
            "are enabled vs disabled. Spec acceptance bar (Steps 2/3/4) is no IC loss of more "
            "than 0.05 on any primary sector — that needs a full walk-forward run with realised "
            "returns; this report is an interim proxy."
        )


if __name__ == "__main__":
    main()
