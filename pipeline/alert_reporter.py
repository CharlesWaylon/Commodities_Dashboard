"""
Post-ingestion alert reporter — for the dashboard designer's eyes only.

Runs automatically at the end of every pipeline/ingest.py run. Produces:
  reports/latest_alert.md      — human-readable report, overwritten each run
  reports/alert_history/       — timestamped archive of every report ever generated

Also fires a native macOS notification so you see it without having to
open a terminal (uses osascript — no extra dependencies).

ALERT CATEGORIES (shown in priority order in the report):

  🚨 CIRCUIT BREAKER
     The validator halted ingestion for a ticker entirely because too many
     rows were uncorrectable. No data was written for that ticker this run.
     Always investigate.

  ⚠️  MARKET EVENT
     Large price move (>±ALERT_RETURN_THRESHOLD) or strong z-score deviation
     in recently ingested data that the validator did NOT auto-correct.
     Two possibilities:
       (a) Real market event — confirm on Bloomberg/Reuters, do nothing.
           The negative-oil / cocoa-crisis / Ukraine-wheat scenarios land here.
       (b) Subtle data error — delete the rows and re-run ingestion.
     You decide.

  🟡 QUARANTINED
     Validator flagged these rows but kept them as-is (typically contract-roll
     discontinuities that roll_adjust.py will handle). Review if the flagged
     move looks disproportionately large for a roll.

  🔴 EXCLUDED
     Rows dropped by the validator — couldn't be corrected or interpolated.
     The ticker now has a gap on those dates. Consider a manual backfill if
     the gap is material.

  ✅ AUTO-CORRECTED
     Scale or interpolation fix applied automatically. Included only for
     auditability — no action needed unless you suspect the fix was wrong.

RUN STANDALONE (check last N days without running a full ingest):
  python -m pipeline.alert_reporter
"""

import logging
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

REPORTS_DIR            = Path(__file__).parent.parent / "reports"
ALERT_RETURN_THRESHOLD = 0.15   # flag 1-day moves > ±15%
ALERT_ZSCORE_THRESHOLD = 4.0    # flag z-score > this from 60-day rolling mean
ROLLING_WINDOW         = 60     # days for rolling mean/std
LOOKBACK_DAYS          = 5      # how many recent days to scan for market events


# ── macOS notification ─────────────────────────────────────────────────────────

def _macos_notify(title: str, message: str, subtitle: str = "") -> None:
    """Fire a native macOS notification via osascript (no-op on non-Mac)."""
    try:
        script = (
            f'display notification "{message}" with title "{title}"'
            + (f' subtitle "{subtitle}"' if subtitle else "")
        )
        subprocess.run(
            ["osascript", "-e", script],
            check=False, timeout=5, capture_output=True,
        )
    except Exception:
        pass


# ── DB loaders ─────────────────────────────────────────────────────────────────

def _load_validation_log(run_id: str) -> list[dict]:
    """All price_validation_log rows for this run."""
    try:
        from database.db import get_engine
        from sqlalchemy import text
        with get_engine().connect() as conn:
            rows = conn.execute(text("""
                SELECT ticker, name, date, raw_close, corrected_close,
                       reason_code, action, details
                FROM   price_validation_log
                WHERE  run_id = :run_id
                ORDER  BY ticker, date
            """), {"run_id": run_id}).fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception as e:
        log.warning("alert_reporter: validation log unavailable: %s", e)
        return []


def _load_ingestion_log(run_id: str) -> list[dict]:
    """All ingestion_log rows for this run — used to detect circuit_breaker status."""
    try:
        from database.db import get_engine
        from sqlalchemy import text
        with get_engine().connect() as conn:
            rows = conn.execute(text("""
                SELECT ticker, name, status, rows_inserted, rows_skipped, error_msg
                FROM   ingestion_log
                WHERE  run_id = :run_id
                ORDER  BY ticker
            """), {"run_id": run_id}).fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception as e:
        log.warning("alert_reporter: ingestion log unavailable: %s", e)
        return []


def _load_recent_prices() -> pd.DataFrame:
    """
    Pull the last (LOOKBACK_DAYS + ROLLING_WINDOW) days of price_history so we
    can compute returns and rolling z-scores for market event detection.
    """
    try:
        from database.db import get_engine
        from sqlalchemy import text
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS + ROLLING_WINDOW)
        ).date()
        with get_engine().connect() as conn:
            rows = conn.execute(text("""
                SELECT c.name, c.ticker, ph.date, ph.close
                FROM   price_history ph
                JOIN   commodities   c ON c.id = ph.commodity_id
                WHERE  ph.date     >= :cutoff
                  AND  ph.interval  = '1d'
                ORDER  BY c.name, ph.date
            """), {"cutoff": str(cutoff)}).fetchall()
        df = pd.DataFrame(rows, columns=["name", "ticker", "date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        log.warning("alert_reporter: price history unavailable: %s", e)
        return pd.DataFrame()


# ── Market event scanner ───────────────────────────────────────────────────────

def _scan_market_events(already_flagged_dates: dict[str, set]) -> list[dict]:
    """
    Find large moves or z-score spikes in recently ingested data that the
    validator did NOT automatically correct (so they land in the DB as-is).

    already_flagged_dates: {ticker: {date_str, ...}} — skip dates that the
    validator already handled so we don't double-report.
    """
    df = _load_recent_prices()
    if df.empty:
        return []

    alerts = []
    scan_cutoff = pd.Timestamp(
        datetime.now() - timedelta(days=LOOKBACK_DAYS)
    )

    for name, grp in df.groupby("name"):
        grp   = grp.sort_values("date").set_index("date")
        close = grp["close"].astype(float)
        ticker = grp["ticker"].iloc[0]
        flagged_for_this = already_flagged_dates.get(ticker, set())

        if len(close) < 3:
            continue

        # ── 1-day return spikes ────────────────────────────────────────────────
        ret    = close.pct_change()
        recent = ret[ret.index >= scan_cutoff].dropna()
        for dt, r in recent.items():
            date_str = str(dt.date())
            if abs(r) > ALERT_RETURN_THRESHOLD and date_str not in flagged_for_this:
                alerts.append({
                    "category":  "MARKET_EVENT",
                    "name":      name,
                    "ticker":    ticker,
                    "date":      date_str,
                    "raw_close": round(float(close.loc[dt]), 4),
                    "detail": (
                        f"1-day return {r * 100:+.1f}% exceeds ±"
                        f"{ALERT_RETURN_THRESHOLD * 100:.0f}% threshold. "
                        f"Confirm: real event or data error?"
                    ),
                })
                flagged_for_this.add(date_str)

        # ── Rolling z-score spikes ─────────────────────────────────────────────
        if len(close) >= ROLLING_WINDOW:
            roll_mean = close.rolling(ROLLING_WINDOW, min_periods=20).mean()
            roll_std  = close.rolling(ROLLING_WINDOW, min_periods=20).std()
            z = (close - roll_mean) / roll_std.replace(0.0, np.nan)
            recent_z = z[z.index >= scan_cutoff].dropna()
            for dt, zval in recent_z.items():
                date_str = str(dt.date())
                if abs(zval) > ALERT_ZSCORE_THRESHOLD and date_str not in flagged_for_this:
                    alerts.append({
                        "category":  "MARKET_EVENT",
                        "name":      name,
                        "ticker":    ticker,
                        "date":      date_str,
                        "raw_close": round(float(close.loc[dt]), 4),
                        "detail": (
                            f"{close.loc[dt]:.4f} is {abs(zval):.1f}σ from "
                            f"{ROLLING_WINDOW}-day rolling mean. "
                            f"Confirm: real event or data error?"
                        ),
                    })
                    flagged_for_this.add(date_str)

    return alerts


# ── Report builder ─────────────────────────────────────────────────────────────

def generate_alert_report(run_id: str) -> Path:
    """
    Build the full alert report for one ingestion run.
    Writes reports/latest_alert.md and an archived copy.
    Fires a macOS notification if anything needs human review.
    Returns the report path.
    """
    REPORTS_DIR.mkdir(exist_ok=True)

    val_log = _load_validation_log(run_id)
    ing_log = _load_ingestion_log(run_id)

    # Categorise validator output
    auto_fixed  = [r for r in val_log if r["action"] in ("rescaled", "interpolated")]
    quarantined = [r for r in val_log if r["action"] == "quarantined"]
    excluded    = [r for r in val_log if r["action"] == "excluded"]
    cb_rows     = [r for r in ing_log if r["status"] == "circuit_breaker"]

    # Build set of dates already handled by validator so we don't double-report
    already_flagged: dict[str, set] = {}
    for r in val_log:
        already_flagged.setdefault(r["ticker"], set()).add(str(r["date"]))

    mkt_evts = _scan_market_events(already_flagged)

    needs_review = len(quarantined) + len(excluded) + len(cb_rows) + len(mkt_evts)
    auto_count   = len(auto_fixed)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L   = []  # report lines

    L.append(f"# Commodities Alert Report\n")
    L.append(f"**Generated:** {now}  \n")
    L.append(f"**Run ID:** `{run_id}`  \n")
    L.append(
        f"**Summary:** {needs_review} item(s) requiring review · "
        f"{auto_count} auto-corrected\n\n"
    )
    L.append("---\n\n")

    # ── 🚨 Circuit breakers ────────────────────────────────────────────────────
    if cb_rows:
        L.append("## 🚨 CIRCUIT BREAKER — Ingestion Halted\n\n")
        L.append(
            "No data was written for these tickers this run. "
            "Investigate before the next scheduled ingest.\n\n"
        )
        for r in cb_rows:
            L.append(f"### {r['name']} (`{r['ticker']}`)\n")
            L.append(f"- **Error:** {r['error_msg']}\n\n")

    # ── ⚠️  Market events ─────────────────────────────────────────────────────
    if mkt_evts:
        L.append("## ⚠️  MARKET EVENTS — Manual Review Required\n\n")
        L.append(
            "Large moves in recently ingested data that the validator kept as-is.\n\n"
            "**If this is a real event** (Ukraine invasion, OPEC cut, sanctions, "
            "crop failure): do nothing — the data is correct.\n\n"
            "**If this looks like a data error**: delete the rows manually and "
            "re-run `python -m pipeline.ingest --backfill`.\n\n"
        )
        for a in sorted(mkt_evts, key=lambda x: x["date"], reverse=True):
            L.append(f"### {a['name']} (`{a['ticker']}`) — {a['date']}\n")
            L.append(f"- **Close:** {a['raw_close']}\n")
            L.append(f"- **Flag:** {a['detail']}\n\n")

    # ── 🟡 Quarantined ────────────────────────────────────────────────────────
    if quarantined:
        L.append("## 🟡 QUARANTINED — Flagged but Kept\n\n")
        L.append(
            "Flagged by the validator and inserted as-is. Usually contract-roll "
            "discontinuities that `roll_adjust.py` handles automatically. "
            "Only act if a move looks disproportionately large for a roll.\n\n"
        )
        by_ticker: dict = {}
        for r in quarantined:
            by_ticker.setdefault(f"{r['name']} ({r['ticker']})", []).append(r)
        for key, recs in sorted(by_ticker.items()):
            L.append(f"### {key}\n")
            for r in recs:
                L.append(
                    f"- {r['date']}  close={r['raw_close']:.4f}  "
                    f"reason=`{r['reason_code']}`\n"
                )
            L.append("\n")

    # ── 🔴 Excluded ───────────────────────────────────────────────────────────
    if excluded:
        L.append("## 🔴 EXCLUDED — Rows Dropped\n\n")
        L.append(
            "These rows failed sanity checks and could not be interpolated. "
            "They were NOT written to the DB — a gap now exists on these dates. "
            "Consider a manual backfill if the gap is material.\n\n"
        )
        for r in excluded:
            L.append(
                f"- **{r['name']}** (`{r['ticker']}`) {r['date']}  "
                f"raw_close={r['raw_close']:.4f}  "
                f"reason=`{r['reason_code']}`\n"
            )
        L.append("\n")

    # ── ✅ Auto-corrected (informational) ─────────────────────────────────────
    if auto_fixed:
        L.append("## ✅ AUTO-CORRECTED — No Action Required\n\n")
        L.append("Included for auditability. The validator fixed these automatically.\n\n")
        by_ticker = {}
        for r in auto_fixed:
            key = f"{r['name']} (`{r['ticker']}`)"
            by_ticker.setdefault(key, []).append(r)
        for key, recs in sorted(by_ticker.items()):
            sample = recs[0]
            L.append(
                f"- {key}: {len(recs)} row(s) — "
                f"`{sample['reason_code']}` → `{sample['action']}`\n"
            )
        L.append("\n")

    if needs_review == 0 and auto_count == 0:
        L.append("## ✅ All Clear\n\n")
        L.append("No anomalies detected in this ingestion run.\n")

    report_text = "".join(L)

    # Write report
    report_path = REPORTS_DIR / "latest_alert.md"
    report_path.write_text(report_text, encoding="utf-8")

    archive_dir = REPORTS_DIR / "alert_history"
    archive_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (archive_dir / f"alert_{ts}.md").write_text(report_text, encoding="utf-8")

    log.info("Alert report written → %s", report_path)

    # ── macOS notification ────────────────────────────────────────────────────
    if cb_rows or mkt_evts or excluded:
        parts = []
        if cb_rows:
            parts.append(f"{len(cb_rows)} circuit breaker(s)")
        if mkt_evts:
            parts.append(f"{len(mkt_evts)} market event(s)")
        if excluded:
            parts.append(f"{len(excluded)} excluded row(s)")
        _macos_notify(
            title="⚠️ Commodities Alert",
            subtitle=", ".join(parts),
            message="Open reports/latest_alert.md to review",
        )
    elif auto_fixed:
        _macos_notify(
            title="Commodities Ingest Complete",
            subtitle=f"{auto_count} row(s) auto-corrected — no review needed",
            message="",
        )
    else:
        _macos_notify(
            title="Commodities Ingest Complete",
            subtitle="All clear — no anomalies",
            message="",
        )

    return report_path


# ── Standalone CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run a market-event scan against the last LOOKBACK_DAYS of DB data
    without triggering a full ingestion. Useful for manual spot-checks.

    Usage:
        python -m pipeline.alert_reporter
    """
    import uuid
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Standalone run: no run_id (no validator or ingest log to pull)
    # Just do the market-event scan directly.
    log.info("Running standalone market-event scan (last %d days)...", LOOKBACK_DAYS)
    events = _scan_market_events({})
    if events:
        print(f"\n{len(events)} potential market event(s) detected:\n")
        for e in sorted(events, key=lambda x: x["date"], reverse=True):
            print(f"  [{e['date']}]  {e['name']} ({e['ticker']})")
            print(f"    Close: {e['raw_close']}  |  {e['detail']}\n")
    else:
        print("\nAll clear — no anomalies in recent price history.\n")

    # Also write a report (with a dummy run_id so the file is generated)
    dummy_id = f"standalone-{uuid.uuid4()}"
    path = generate_alert_report(dummy_id)
    print(f"Report written → {path}")
