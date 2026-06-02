#!/usr/bin/env python3
"""
model_health_report.py  —  Live model-performance pull for Claude to analyse.

WHY THIS EXISTS
  Claude runs in a sandbox that cannot reach your Mac's Postgres. This script
  runs *locally* against the live DB (via the repo's own get_engine(), which
  reads DATABASE_URL from .env), prints a compact report, and you copy-paste
  the whole output back to Claude.

HOW TO RUN  (from the repo root, same place you run the dashboard):
    cd ~/Desktop/Future_of_Commodities/Commodities_Dashboard
    python model_health_report.py

  Everything between the BEGIN/END markers is what Claude needs — copy all of it.
"""
from __future__ import annotations
import sys
from datetime import datetime, timezone

try:
    from sqlalchemy import text
    from database.db import get_engine, DATABASE_URL
except Exception as e:  # pragma: no cover
    print("ERROR importing repo DB layer. Run this from the repo root "
          "(~/Desktop/Future_of_Commodities/Commodities_Dashboard).")
    print(f"  {type(e).__name__}: {e}")
    sys.exit(1)


def _redact(url: str) -> str:
    if "@" in url and "//" in url:
        head, tail = url.split("//", 1)
        creds, host = tail.split("@", 1)
        user = creds.split(":", 1)[0]
        return f"{head}//{user}:***@{host}"
    return url


def _rows(conn, sql, **params):
    return list(conn.execute(text(sql), params))


def main() -> None:
    eng = get_engine()
    print("==================== BEGIN MODEL HEALTH REPORT ====================")
    print(f"generated_at_utc : {datetime.now(timezone.utc).isoformat()}")
    print(f"database         : {_redact(DATABASE_URL)}")
    is_pg = DATABASE_URL.startswith("postgres")
    print(f"backend          : {'PostgreSQL (live)' if is_pg else 'SQLite (NOT live — check .env DATABASE_URL)'}")

    with eng.connect() as conn:

        # ---- 1. Latest IC per (commodity, tier), worst first -----------------
        print("\n----- [1] LATEST IC PER (commodity, tier)  — worst first -----")
        print("(IC = Spearman corr of forecast vs realised return; <0 = wrong direction)")
        try:
            # newest row per (commodity, tier) by computed_at
            sql = """
                SELECT t.commodity, t.tier, t.ic_value, t.n_obs, t.regime,
                       t.window_start, t.window_end, t.computed_at
                FROM ic_log t
                JOIN (
                    SELECT commodity, tier, MAX(computed_at) AS mx
                    FROM ic_log GROUP BY commodity, tier
                ) m
                ON t.commodity = m.commodity AND t.tier = m.tier
                   AND t.computed_at = m.mx
                ORDER BY t.ic_value ASC
            """
            rows = _rows(conn, sql)
            if not rows:
                print("  (ic_log is EMPTY — no IC has been logged yet)")
            else:
                print(f"  {'IC':>8}  {'n':>4}  {'tier':<12} {'regime':<10} {'commodity':<22} window")
                for r in rows:
                    reg = str(r.regime) if r.regime is not None else "-"
                    print(f"  {r.ic_value:>+8.4f}  {r.n_obs:>4}  {r.tier:<12} {reg:<10} "
                          f"{r.commodity:<22} {r.window_start}->{r.window_end}")
                print(f"  most_recent_computed_at: {max(r.computed_at for r in rows)}")
        except Exception as e:
            print(f"  [skipped: {type(e).__name__}: {e}]")

        # ---- 2. Avg IC by tier ----------------------------------------------
        print("\n----- [2] AVG IC BY TIER (latest row per commodity/tier) -----")
        try:
            sql = """
                WITH latest AS (
                    SELECT t.tier, t.ic_value
                    FROM ic_log t
                    JOIN (SELECT commodity, tier, MAX(computed_at) mx
                          FROM ic_log GROUP BY commodity, tier) m
                    ON t.commodity=m.commodity AND t.tier=m.tier AND t.computed_at=m.mx
                )
                SELECT tier, AVG(ic_value) avg_ic, MIN(ic_value) min_ic,
                       MAX(ic_value) max_ic, COUNT(*) n
                FROM latest GROUP BY tier ORDER BY avg_ic ASC
            """
            for r in _rows(conn, sql):
                print(f"  {r.tier:<14} avg={r.avg_ic:+.4f}  min={r.min_ic:+.4f}  "
                      f"max={r.max_ic:+.4f}  n={r.n}")
        except Exception as e:
            print(f"  [skipped: {type(e).__name__}: {e}]")

        # ---- 3. Realised forecast error from forecast_log -------------------
        print("\n----- [3] FORECAST_LOG ACCURACY (last 90 days, where actual known) -----")
        print("(hit_rate = % of forecasts with correct sign vs realised return)")
        try:
            sql = """
                SELECT model_name, tier, COUNT(*) n,
                       AVG(ABS(error)) mae,
                       AVG(CASE WHEN forecast_return*actual_return > 0 THEN 1.0
                                ELSE 0.0 END) hit_rate
                FROM forecast_log
                WHERE actual_return IS NOT NULL
                  AND forecast_date >= (CURRENT_DATE - INTERVAL '90 days')
                GROUP BY model_name, tier
                HAVING COUNT(*) >= 5
                ORDER BY hit_rate ASC
            """ if is_pg else """
                SELECT model_name, tier, COUNT(*) n,
                       AVG(ABS(error)) mae,
                       AVG(CASE WHEN forecast_return*actual_return > 0 THEN 1.0
                                ELSE 0.0 END) hit_rate
                FROM forecast_log
                WHERE actual_return IS NOT NULL
                  AND forecast_date >= date('now','-90 day')
                GROUP BY model_name, tier
                HAVING COUNT(*) >= 5
                ORDER BY hit_rate ASC
            """
            rows = _rows(conn, sql)
            if not rows:
                print("  (no scored rows in forecast_log for the window)")
            else:
                print(f"  {'hit%':>6}  {'mae':>10}  {'n':>5}  {'tier':<12} model_name")
                for r in rows:
                    print(f"  {100*r.hit_rate:>5.1f}  {r.mae:>10.5f}  {r.n:>5}  "
                          f"{str(r.tier):<12} {r.model_name}")
        except Exception as e:
            print(f"  [skipped: {type(e).__name__}: {e}]")

        # ---- 4. Trigger threshold IC ----------------------------------------
        print("\n----- [4] THRESHOLD_CONFIG (trigger families; negative IC = anti-predictive) -----")
        try:
            sql = """
                SELECT family, optimal_threshold, best_ic, continuous_ic,
                       n_events_total, n_events_at_threshold, forward_days
                FROM threshold_config
                ORDER BY (CASE WHEN continuous_ic IS NULL THEN 1 ELSE 0 END),
                         continuous_ic ASC
            """
            for r in _rows(conn, sql):
                bi = f"{r.best_ic:+.4f}" if r.best_ic is not None else "  None "
                ci = f"{r.continuous_ic:+.4f}" if r.continuous_ic is not None else "  None "
                print(f"  {r.family:<18} thr={r.optimal_threshold}  best_ic={bi}  "
                      f"cont_ic={ci}  n_total={r.n_events_total}  "
                      f"n_at_thr={r.n_events_at_threshold}  fwd={r.forward_days}d")
        except Exception as e:
            print(f"  [skipped: {type(e).__name__}: {e}]")

        # ---- 5. Last retrain metadata ---------------------------------------
        print("\n----- [5] LATEST MODEL_TRAINING_LOG -----")
        try:
            sql = """
                SELECT retrained_at, n_commodities, n_training_pairs,
                       tier_distribution, tree_n_leaves, top_feature, error
                FROM model_training_log
                ORDER BY retrained_at DESC LIMIT 3
            """
            for r in _rows(conn, sql):
                print(f"  {r.retrained_at} | comms={r.n_commodities} "
                      f"pairs={r.n_training_pairs} leaves={r.tree_n_leaves} "
                      f"top_feature={r.top_feature} tiers={r.tier_distribution} "
                      f"err={r.error!r}")
        except Exception as e:
            print(f"  [skipped: {type(e).__name__}: {e}]")

        # ---- 6. Price freshness ---------------------------------------------
        print("\n----- [6] PRICE_HISTORY FRESHNESS -----")
        try:
            sql = "SELECT MAX(date) latest, MIN(date) earliest, COUNT(*) n FROM price_history"
            for r in _rows(conn, sql):
                print(f"  latest_price_date={r.latest}  earliest={r.earliest}  rows={r.n}")
        except Exception as e:
            print(f"  [skipped: {type(e).__name__}: {e}]")

    print("\n===================== END MODEL HEALTH REPORT =====================")
    print("Copy everything from BEGIN to END and paste it back to Claude.")


if __name__ == "__main__":
    main()
