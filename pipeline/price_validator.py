"""
Multi-layer price validation and anomaly correction pipeline.

WHY THIS EXISTS:
  Yahoo Finance (via yfinance) occasionally delivers price data that is
  systematically wrong in one of four ways:

    1. UNIT / SCALE ERROR — ZR=F (Rough Rice) is quoted at the exchange in
       cents per hundredweight but some YF contract stitches expose the raw
       cents value (~1,800) instead of the dollar equivalent (~18). The same
       ×100 or ×1000 issue can appear on any contract during a bad feed day.

    2. CONTRACT ROLLOVER ARTIFACT — when the front-month contract expires, YF
       stitches in the next contract, producing a step-change in the price
       series that was not a real market move. These are handled by
       roll_adjust.py AFTER ingestion; the validator records them for audit
       but does NOT auto-remove them (roll_adjust needs the raw jumps).

    3. ISOLATED BAD TICK — a single day's close is clearly wrong relative to
       its neighbors (e.g., a digit transposition or a stale feed). These
       are interpolated from adjacent valid prices.

    4. SYSTEMATIC FEED CORRUPTION — many rows in a single fetch are bad.
       The circuit breaker halts ingestion to prevent corrupted data from
       silently entering the analytics pipeline.

CORRECTION PRIORITY ORDER:
  1. rescale     — series-wide scale factor applied (e.g., ÷100 for Rough Rice)
  2. interpolate — individual outlier replaced by average of nearest neighbors
  3. exclude     — row dropped if interpolation impossible (edge of series, etc.)
  4. quarantine  — row flagged and logged but kept as-is (rollover artifact,
                   possible extreme but real market event)

CIRCUIT BREAKER:
  If the fraction of excluded rows (rows we could not fix) exceeds
  CIRCUIT_BREAKER_PCT (default 5%) of the fetched batch, the entire ticker's
  batch is rejected — clean_df is returned empty and the caller should log
  status='circuit_breaker' without upserting any rows.
  Quarantined rows do NOT count toward the circuit breaker because they are
  kept and roll_adjust.py may correctly handle them.

ROUGH RICE NOTE:
  CBOT ZR=F is priced in USD per hundredweight (cwt). The realistic range is
  roughly $10–$20/cwt. When Yahoo Finance reports ~$1,800, it is delivering
  the exchange's cents-per-cwt quote without the ÷100 conversion.
  The sanity band (5, 50) and scale detector catch this automatically.

INTEGRATION:
  Called from pipeline/ingest.py immediately after the yfinance download,
  before any row is written to price_history.
  Anomalies are persisted to price_validation_log for long-term audit.

RETROACTIVE FIX:
  For data already in the DB, call:
    python -c "from pipeline.price_validator import retroactive_fix_scaling; print(retroactive_fix_scaling('ZR=F'))"
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date as date_type
from typing import Optional

log = logging.getLogger(__name__)


# ── Per-ticker absolute sanity bands ──────────────────────────────────────────
# (floor, ceiling) in the unit stored in price_history.close.
# Only tickers listed here get hard-floor/ceiling enforcement.
# Bands are intentionally wide — they catch systematic errors, not normal moves.
SANITY_BANDS: dict[str, tuple[float, float]] = {
    # Agriculture — grains quoted in USc/bu on CBOT; realistic floors/ceilings
    "ZC=F":  (100.0,  2000.0),   # Corn (USc/bu)
    "ZW=F":  (150.0,  2500.0),   # CBOT Wheat SRW (USc/bu)
    "KE=F":  (150.0,  2500.0),   # KC Wheat HRW (USc/bu)
    "ZS=F":  (300.0,  3000.0),   # Soybeans (USc/bu)
    "ZL=F":  (15.0,   200.0),    # Soybean Oil (USc/lb)
    "ZM=F":  (80.0,   800.0),    # Soybean Meal (USD/ton)
    "ZO=F":  (80.0,   1000.0),   # Oats (USc/bu)
    # Rough Rice — the primary problem ticker.
    # YF often returns raw cents/cwt (~1800) instead of USD/cwt (~18).
    # Band is intentionally tight to force scale detection.
    "ZR=F":  (5.0,    50.0),     # Rough Rice (USD/cwt)
    # Softs
    "KC=F":  (40.0,   600.0),    # Coffee Arabica (USc/lb)
    "SB=F":  (3.0,    70.0),     # Sugar No.11 (USc/lb)
    "CT=F":  (25.0,   250.0),    # Cotton No.2 (USc/lb)
    "CC=F":  (800.0,  20000.0),  # Cocoa (USD/MT)
    "OJ=F":  (50.0,   600.0),    # OJ FCOJ-A (USc/lb)
    # Energy
    "CL=F":  (10.0,   200.0),    # WTI Crude (USD/bbl)
    "BZ=F":  (10.0,   200.0),    # Brent Crude (USD/bbl)
    "NG=F":  (0.5,    30.0),     # Natural Gas (USD/MMBtu)
    "RB=F":  (0.5,    6.0),      # RBOB Gasoline (USD/gal)
    "HO=F":  (0.5,    6.0),      # Heating Oil (USD/gal)
    # Metals
    "GC=F":  (200.0,  5000.0),   # Gold (USD/oz)
    "SI=F":  (3.0,    100.0),    # Silver (USD/oz)
    "HG=F":  (0.5,    10.0),     # Copper (USD/lb)
    "ALI=F": (0.5,    5.0),      # Aluminum (USD/lb)
    "PL=F":  (200.0,  2500.0),   # Platinum (USD/oz)
    "PA=F":  (300.0,  4000.0),   # Palladium (USD/oz)
    # Livestock
    "LE=F":  (40.0,   350.0),    # Live Cattle (USc/lb)
    "GF=F":  (80.0,   450.0),    # Feeder Cattle (USc/lb)
    "HE=F":  (25.0,   200.0),    # Lean Hogs (USc/lb)

    # ── ETF proxies ────────────────────────────────────────────────────────────
    # Wide bands — ETF/equity prices move freely; we're only catching obvious
    # data errors (near-zero or implausible magnitudes), not normal volatility.
    "SGOL":  (15.0,   80.0),     # Aberdeen Physical Gold ETF (tracks LBMA gold)
    "SIVR":  (5.0,    60.0),     # Aberdeen Physical Silver ETF
    "URA":   (5.0,    120.0),    # Global X Uranium ETF
    "KRBN":  (3.0,    80.0),     # KraneShares Global Carbon ETF
    "SLX":   (15.0,   200.0),    # VanEck Steel ETF
    "LIT":   (8.0,    150.0),    # Global X Lithium & Battery Tech ETF
    "REMX":  (8.0,    150.0),    # VanEck Rare Earth/Strategic Metals ETF
    "WOOD":  (25.0,   250.0),    # iShares Global Timber & Forestry ETF

    # ── Equity proxies ─────────────────────────────────────────────────────────
    "LNG":   (30.0,   600.0),    # Cheniere Energy (large-cap US equity)
    "BTU":   (0.50,   150.0),    # Peabody Energy (cyclical; survived bankruptcy → wide floor)
    "HCC":   (3.0,    200.0),    # Warrior Met Coal (small-cap; volatile)
    "GLNCY": (1.0,    50.0),     # Glencore ADR (London-listed, ADR price varies)

    # ── Crypto ────────────────────────────────────────────────────────────────
    "BTC-USD": (1000.0, 500000.0),  # Bitcoin — intentionally very wide
}

# Scale factors tried in order when systematic band violation is detected.
# List largest first so we detect ×1000 before ×100 etc. (median is divided by factor).
SCALE_FACTORS = [1000.0, 100.0, 10.0, 0.1, 0.01, 0.001]

# ── Thresholds ─────────────────────────────────────────────────────────────────
MAX_RETURN_THRESHOLD = 0.25    # flag single-period returns > ±25%
ROLLOVER_THRESHOLD   = 0.50    # returns > ±50% are treated as data errors, not rolls
ROLLING_WINDOW       = 60      # days for rolling mean / std
ZSCORE_SIGMA         = 4.0     # flag when |price – rolling_mean| / rolling_std > this
SCALE_OUTLIER_PCT    = 0.30    # if >30% of prices outside band → try scale detection
CIRCUIT_BREAKER_PCT  = 0.05    # halt ingestion if excluded / total > this

# ── Reason codes ───────────────────────────────────────────────────────────────
RC_SCALING    = "scaling_error"             # unit/scale mismatch (e.g., cents vs dollars)
RC_SANITY     = "absolute_sanity_violation" # price outside hard floor/ceiling
RC_OUTLIER    = "outlier_spike"             # extreme single-period move or rolling z-score
RC_ROLLOVER   = "rollover_artifact"         # likely contract-roll discontinuity
RC_CONTINUITY = "missing_contract_continuity"  # gap preventing proper back-adjustment

# ── Correction actions ─────────────────────────────────────────────────────────
ACT_RESCALED     = "rescaled"      # scale factor applied; row inserted with corrected price
ACT_INTERPOLATED = "interpolated"  # replaced with linear interpolation; row inserted
ACT_EXCLUDED     = "excluded"      # row dropped; NOT inserted
ACT_QUARANTINED  = "quarantined"   # flagged but kept as-is; roll_adjust.py may fix it


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class AnomalyRecord:
    """One detected anomaly with its correction outcome."""
    ticker:          str
    name:            str
    date:            date_type
    raw_close:       float
    corrected_close: Optional[float]
    reason_code:     str
    action:          str
    details:         str


@dataclass
class ValidatorOutput:
    """Result of validate_price_series(); carries both raw and cleaned DataFrames."""
    ticker:                    str
    name:                      str
    raw_df:                    pd.DataFrame
    clean_df:                  pd.DataFrame           # empty if circuit_breaker_triggered
    anomalies:                 list[AnomalyRecord] = field(default_factory=list)
    circuit_breaker_triggered: bool = False
    circuit_breaker_reason:    str  = ""
    scale_factor_applied:      Optional[float] = None


# ── Internal helpers ───────────────────────────────────────────────────────────

def _detect_scale_factor(ticker: str, close: pd.Series) -> Optional[float]:
    """
    Test whether dividing the series by a standard factor brings its median
    inside the sanity band for ticker. Returns the divisor (e.g. 100.0) or None.
    """
    band = SANITY_BANDS.get(ticker)
    if band is None:
        return None

    lo, hi = band
    median = float(close.median())

    if lo <= median <= hi:
        return None  # Already in range

    for factor in SCALE_FACTORS:
        scaled = median / factor
        if lo <= scaled <= hi:
            log.info(
                "  [validator] %s: scale mismatch detected — "
                "median=%.2f, factor=÷%.0f → scaled_median=%.4f ∈ [%.1f, %.1f]",
                ticker, median, factor, scaled, lo, hi,
            )
            return factor

    return None


def _check_return_spikes(close: pd.Series) -> pd.Index:
    """Dates where |pct_change| > MAX_RETURN_THRESHOLD."""
    returns = close.pct_change().abs()
    return returns[returns > MAX_RETURN_THRESHOLD].index


def _check_rolling_zscore(close: pd.Series) -> pd.Index:
    """Dates where |(price – 60d_mean) / 60d_std| > ZSCORE_SIGMA."""
    if len(close) < ROLLING_WINDOW:
        return pd.DatetimeIndex([])
    roll_mean = close.rolling(ROLLING_WINDOW, min_periods=10).mean()
    roll_std  = close.rolling(ROLLING_WINDOW, min_periods=10).std()
    z = (close - roll_mean) / roll_std.replace(0.0, np.nan)
    return z[z.abs() > ZSCORE_SIGMA].dropna().index


def _interpolate_point(close: pd.Series, positional_idx: int) -> Optional[float]:
    """
    Linear interpolation from nearest valid neighbours (within ±5 positions).
    Falls back to single-side average when only one neighbour is available.
    Returns None if there are no valid neighbours.
    """
    n = len(close)
    prev_val = next_val = None
    for i in range(1, 6):
        if idx_prev := positional_idx - i:
            if 0 <= idx_prev < n and prev_val is None:
                prev_val = float(close.iloc[idx_prev])
        if idx_next := positional_idx + i:
            if 0 <= idx_next < n and next_val is None:
                next_val = float(close.iloc[idx_next])
        if prev_val is not None and next_val is not None:
            break

    if prev_val is not None and next_val is not None:
        return (prev_val + next_val) / 2.0
    return prev_val if prev_val is not None else next_val


# ── Public API ─────────────────────────────────────────────────────────────────

def validate_price_series(
    ticker: str,
    name:   str,
    raw_df: pd.DataFrame,
) -> ValidatorOutput:
    """
    Run the full multi-layer validation pipeline on a freshly-fetched DataFrame.

    raw_df  — output of yfinance.download(); must have DatetimeIndex and 'Close'.
    Returns ValidatorOutput. If circuit_breaker_triggered, clean_df is empty
    and the caller should skip the upsert.

    Layers applied (in order):
      1. Absolute sanity band + scale factor detection/correction
      2. Single-period return spike flagging
      3. Rolling 60-day z-score flagging
      4. Circuit breaker evaluation
    """
    if raw_df.empty or "Close" not in raw_df.columns:
        return ValidatorOutput(ticker=ticker, name=name, raw_df=raw_df, clean_df=raw_df)

    df    = raw_df.copy().sort_index()
    close = df["Close"].copy().astype(float)
    anomalies: list[AnomalyRecord] = []
    scale_factor_applied: Optional[float] = None
    excluded_count = 0

    def _dt_to_date(dt) -> date_type:
        return dt.date() if hasattr(dt, "date") else dt

    # ── Layer 1a: Absolute sanity band ─────────────────────────────────────────
    band = SANITY_BANDS.get(ticker)
    if band is not None:
        lo, hi = band
        outside_mask = (close < lo) | (close > hi)
        outside_pct  = float(outside_mask.sum()) / len(close)

        if outside_pct > SCALE_OUTLIER_PCT:
            # ── Layer 1b: Scale factor detection ───────────────────────────────
            factor = _detect_scale_factor(ticker, close)
            if factor is not None:
                scale_factor_applied = factor
                corrected_close = close / factor
                for dt in close.index:
                    raw_val  = float(close.loc[dt])
                    corr_val = float(corrected_close.loc[dt])
                    anomalies.append(AnomalyRecord(
                        ticker=ticker, name=name,
                        date=_dt_to_date(dt),
                        raw_close=raw_val,
                        corrected_close=corr_val,
                        reason_code=RC_SCALING,
                        action=ACT_RESCALED,
                        details=(
                            f"Series-wide scale correction ÷{factor:.0f}. "
                            f"Raw median={close.median():.2f}; "
                            f"sanity band=[{lo},{hi}]."
                        ),
                    ))
                close = corrected_close
                df["Close"] = close
                for col in ("Open", "High", "Low"):
                    if col in df.columns:
                        df[col] = df[col].astype(float) / factor
                log.warning(
                    "  [validator] %s: applied ÷%.0f scale correction to %d rows.",
                    ticker, factor, len(close),
                )
            else:
                # Scale unknown — quarantine out-of-band rows; circuit breaker
                # will fire below because excluded_count will be high.
                for dt in close[outside_mask].index:
                    raw_val = float(close.loc[dt])
                    anomalies.append(AnomalyRecord(
                        ticker=ticker, name=name,
                        date=_dt_to_date(dt),
                        raw_close=raw_val,
                        corrected_close=None,
                        reason_code=RC_SANITY,
                        action=ACT_QUARANTINED,
                        details=(
                            f"Price {raw_val:.2f} outside sanity band [{lo},{hi}]. "
                            f"{outside_pct*100:.0f}% of batch violates band; "
                            f"no correctable scale factor found."
                        ),
                    ))
                    excluded_count += 1  # count as un-fixable for circuit breaker
        else:
            # Minority of rows outside band → individual correction
            already_flagged: set[date_type] = set()
            for dt in close[outside_mask].index:
                dt_date = _dt_to_date(dt)
                if dt_date in already_flagged:
                    continue
                raw_val = float(close.loc[dt])
                pos_idx = close.index.get_loc(dt)
                interp  = _interpolate_point(close, pos_idx)

                if interp is not None:
                    action = ACT_INTERPOLATED
                    close.iloc[pos_idx] = interp
                    df.iloc[pos_idx, df.columns.get_loc("Close")] = interp
                else:
                    action = ACT_EXCLUDED
                    excluded_count += 1
                    close = close.drop(dt)
                    df = df.drop(dt)

                anomalies.append(AnomalyRecord(
                    ticker=ticker, name=name,
                    date=dt_date,
                    raw_close=raw_val,
                    corrected_close=interp,
                    reason_code=RC_SANITY,
                    action=action,
                    details=f"Price {raw_val:.4f} outside sanity band [{lo},{hi}].",
                ))
                already_flagged.add(dt_date)

    # ── Layer 2: Single-period return spikes ────────────────────────────────────
    # For series that had a wholesale scale correction, returns are now valid;
    # skip this layer to avoid false positives on the rescaled data.
    if scale_factor_applied is None and len(close) >= 2:
        spike_dates   = _check_return_spikes(close)
        already_dated = {a.date for a in anomalies}

        for dt in spike_dates:
            dt_date = _dt_to_date(dt)
            if dt_date in already_dated:
                continue

            raw_val = float(close.loc[dt])
            ret     = float(close.pct_change().loc[dt])

            if abs(ret) > ROLLOVER_THRESHOLD:
                # ≥50% move in a day — almost certainly a data error, not a real
                # event or a roll (futures rolls are typically 1–15% for most
                # commodities). Attempt interpolation.
                pos_idx = close.index.get_loc(dt)
                interp  = _interpolate_point(close, pos_idx)
                if interp is not None:
                    action = ACT_INTERPOLATED
                    close.iloc[pos_idx] = interp
                    df.iloc[pos_idx, df.columns.get_loc("Close")] = interp
                else:
                    action = ACT_QUARANTINED
                    interp = None
                reason = RC_OUTLIER
            else:
                # 25–50% move: could be a legitimate contract roll that
                # roll_adjust.py will handle. Quarantine for audit, keep raw data.
                action = ACT_QUARANTINED
                interp = None
                reason = RC_ROLLOVER if abs(ret) > 0.30 else RC_OUTLIER

            anomalies.append(AnomalyRecord(
                ticker=ticker, name=name,
                date=dt_date,
                raw_close=raw_val,
                corrected_close=interp,
                reason_code=reason,
                action=action,
                details=(
                    f"Single-period return {ret*100:+.1f}% exceeds "
                    f"±{MAX_RETURN_THRESHOLD*100:.0f}% threshold."
                ),
            ))
            already_dated.add(dt_date)

    # ── Layer 3: Rolling z-score ─────────────────────────────────────────────────
    # Flag prices that deviate sharply from their 60-day rolling context.
    # These are quarantined (logged but kept) — they may be real extreme events.
    # If they were also caught by Layer 2, that anomaly already logged them.
    if scale_factor_applied is None and len(close) >= ROLLING_WINDOW:
        zscore_dates  = _check_rolling_zscore(close)
        already_dated = {a.date for a in anomalies}

        for dt in zscore_dates:
            dt_date = _dt_to_date(dt)
            if dt_date in already_dated:
                continue

            raw_val = float(close.loc[dt])
            anomalies.append(AnomalyRecord(
                ticker=ticker, name=name,
                date=dt_date,
                raw_close=raw_val,
                corrected_close=None,
                reason_code=RC_OUTLIER,
                action=ACT_QUARANTINED,
                details=(
                    f"Price {raw_val:.4f} deviates >{ZSCORE_SIGMA}σ from "
                    f"{ROLLING_WINDOW}-day rolling mean. Kept for roll_adjust review."
                ),
            ))
            already_dated.add(dt_date)

    # ── Layer 4: Circuit breaker ─────────────────────────────────────────────────
    total_rows = len(raw_df)
    cb_triggered = False
    cb_reason    = ""

    if total_rows > 0 and excluded_count / total_rows > CIRCUIT_BREAKER_PCT:
        cb_triggered = True
        cb_reason = (
            f"{excluded_count}/{total_rows} rows "
            f"({excluded_count / total_rows * 100:.1f}%) are uncorrectable; "
            f"threshold is {CIRCUIT_BREAKER_PCT * 100:.0f}%. "
            f"Ingestion halted to prevent corrupted data from propagating."
        )
        log.error(
            "  [validator] CIRCUIT BREAKER tripped for %s — %s", ticker, cb_reason
        )

    clean_df = pd.DataFrame() if cb_triggered else df

    if anomalies:
        actions_summary = {}
        for a in anomalies:
            actions_summary[a.action] = actions_summary.get(a.action, 0) + 1
        log.info(
            "  [validator] %s: %d anomalies %s",
            ticker, len(anomalies), actions_summary,
        )

    return ValidatorOutput(
        ticker=ticker,
        name=name,
        raw_df=raw_df,
        clean_df=clean_df,
        anomalies=anomalies,
        circuit_breaker_triggered=cb_triggered,
        circuit_breaker_reason=cb_reason,
        scale_factor_applied=scale_factor_applied,
    )


# ── Retroactive DB fix ─────────────────────────────────────────────────────────

def retroactive_fix_scaling(ticker: str) -> int:
    """
    One-time corrective pass for a ticker whose existing price_history rows
    are stored at the wrong scale (e.g., ZR=F close ~1,100 instead of ~11).

    IMPORTANT: This targets only rows where close is outside the sanity band.
    Rows already at the correct scale are untouched. adjusted_close is also
    left alone — roll_adjust.py computes it from ratios (which are scale-invariant),
    so it is usually already correct even when raw close is wrong.

    After running this, re-run roll_adjust to resync adjusted_close with the
    corrected close values:
        python -m pipeline.roll_adjust

    Returns the number of rows updated.

    Run from the terminal:
        python -c "
        from pipeline.price_validator import retroactive_fix_scaling
        n = retroactive_fix_scaling('ZR=F')
        print(f'Updated {n} rows.')
        "
    """
    from database.db import get_db
    from database.models import Commodity, PriceHistory

    band = SANITY_BANDS.get(ticker)
    if band is None:
        log.warning("retroactive_fix_scaling: no sanity band defined for %s", ticker)
        return 0

    lo, hi = band

    with get_db() as db:
        commodity = db.query(Commodity).filter_by(ticker=ticker).first()
        if commodity is None:
            log.warning("retroactive_fix_scaling: ticker %s not in DB", ticker)
            return 0

        rows = db.query(PriceHistory).filter_by(commodity_id=commodity.id).all()
        if not rows:
            return 0

        # Only examine rows whose close is currently outside the sanity band
        bad_rows = [r for r in rows if r.close is not None and not (lo <= r.close <= hi)]
        if not bad_rows:
            log.info(
                "retroactive_fix_scaling: %s — all %d rows already within [%.1f, %.1f].",
                ticker, len(rows), lo, hi,
            )
            return 0

        bad_closes = pd.Series([r.close for r in bad_rows])
        factor = _detect_scale_factor(ticker, bad_closes)
        if factor is None:
            log.warning(
                "retroactive_fix_scaling: %s — %d out-of-band rows found but no "
                "correctable scale factor detected. Manual review required.",
                ticker, len(bad_rows),
            )
            return 0

        log.info(
            "retroactive_fix_scaling: %s — applying ÷%.0f to %d out-of-band rows "
            "(%d rows already correct; adjusted_close left as-is).",
            ticker, factor, len(bad_rows), len(rows) - len(bad_rows),
        )

        updated = 0
        for row in bad_rows:
            if row.close is not None:
                row.close = row.close / factor
            if row.open is not None:
                row.open = row.open / factor
            if row.high is not None:
                row.high = row.high / factor
            if row.low is not None:
                row.low = row.low / factor
            # Do NOT touch adjusted_close — roll_adjust.py computes it from
            # proportional ratios which are scale-invariant; it is likely already
            # correct. Re-run roll_adjust after this function to resync.
            updated += 1

    log.info("retroactive_fix_scaling: committed %d updates for %s.", updated, ticker)
    return updated
