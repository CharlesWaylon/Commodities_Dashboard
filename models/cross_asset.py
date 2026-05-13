"""
Cross-asset learning utilities — rolling correlations, relative IC, and
forecast consistency checking.

This module is the glue between the per-commodity model stack and the
five cross-asset goals:

  Goal 1 — Rolling correlation matrices stored daily in the DB.
  Goal 2 — Regime-conditional relative IC ("Brent XGBoost 26% better than WTI").
  Goal 3 — Cluster-level prior propagation (use better commodity as prior).
  Goal 4 — Post-forecast consistency checker (flag divergent correlated pairs).
  Goal 5 — (Future) Feed joint distribution to QAOA portfolio optimizer.

Entry points
────────────
  store_correlation_snapshot(db_path, prices_df, date, window)
      Compute and persist today's pairwise correlations.  Call once per day
      after aligned_prices is updated.

  load_correlation_matrix(db_path, date)
      Load a stored snapshot as a wide commodity × commodity DataFrame.

  relative_ic_comparison(db_path, days, min_correlation, min_obs)
      For every pair (A, B) whose rolling correlation exceeds min_correlation,
      compare IC and return a DataFrame with relative_ic and a human-readable
      description ("Brent's XGBoost IC is 26% higher than WTI's in Bull regime").

  check_forecast_consistency(forecasts, corr_matrix, min_correlation,
                              max_sign_divergence, max_magnitude_ratio)
      After all N forecasts are generated, flag pairs where strongly-correlated
      commodities have divergent directional calls or implausible magnitude gaps.

  apply_cross_asset_prior(forecasts, corr_matrix, ic_df, db_path)
      Goal 3: for each commodity, blend its raw forecast with a weighted prior
      from its most IC-dominant correlated peer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from sqlalchemy import text
from database.db import get_engine

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Pearson correlation window — matches config.CORRELATION_WINDOW
CORRELATION_WINDOW = 21

# Sector labels used for cluster display (mirrors config.py COMMODITY_SECTORS)
_SECTOR_MAP: Dict[str, str] = {
    "WTI Crude Oil": "Energy", "Brent Crude Oil": "Energy",
    "Natural Gas": "Energy", "Gasoline (RBOB)": "Energy",
    "Heating Oil": "Energy", "Carbon Credits*": "Energy",
    "LNG / Intl Gas*": "Energy", "Metallurgical Coal*": "Energy",
    "Thermal Coal*": "Energy", "Uranium*": "Energy",
    "Gold (COMEX)": "Metals", "Silver (COMEX)": "Metals",
    "Copper (COMEX)": "Metals", "Platinum": "Metals",
    "Palladium": "Metals", "Aluminum (COMEX)": "Metals",
    "HRC Steel": "Metals", "Gold (Physical/London)*": "Metals",
    "Silver (Physical)*": "Metals", "Iron Ore / Steel*": "Metals",
    "Lithium*": "Metals", "Rare Earths*": "Metals",
    "Zinc & Cobalt*": "Metals",
    "Corn (CBOT)": "Agriculture", "Wheat (CBOT SRW)": "Agriculture",
    "Wheat (KC HRW)": "Agriculture", "Soybeans (CBOT)": "Agriculture",
    "Soybean Meal": "Agriculture", "Soybean Oil": "Agriculture",
    "Coffee": "Agriculture", "Cocoa": "Agriculture",
    "Sugar": "Agriculture", "Cotton": "Agriculture",
    "Orange Juice (FCOJ-A)": "Agriculture", "Oats (CBOT)": "Agriculture",
    "Rough Rice (CBOT)": "Agriculture", "Lumber*": "Agriculture",
    "Live Cattle": "Livestock", "Feeder Cattle": "Livestock",
    "Lean Hogs": "Livestock",
    "Bitcoin": "Digital Assets",
}



def _load_aligned_prices() -> pd.DataFrame:
    """
    Read aligned_prices joined with commodity names.
    Returns a wide DataFrame: index=date, columns=commodity display names.
    """
    sql = text("""
        SELECT ap.date, c.name, ap.adjusted_close
        FROM   aligned_prices ap
        JOIN   commodities    c  ON c.id = ap.commodity_id
        ORDER  BY ap.date
    """)
    with get_engine().connect() as _conn:
        result = _conn.execute(sql)
        raw = pd.DataFrame(result.fetchall(), columns=result.keys())
    if raw.empty:
        return pd.DataFrame()
    wide = raw.pivot(index="date", columns="name", values="adjusted_close")
    wide.index = pd.to_datetime(wide.index)
    return wide


# ── Goal 1: Rolling correlation computation & storage ─────────────────────────

def compute_rolling_correlations(
    prices_df: pd.DataFrame,
    window: int = CORRELATION_WINDOW,
) -> pd.DataFrame:
    """
    Compute 21-day rolling Pearson correlation for all commodity pairs.

    Parameters
    ----------
    prices_df : wide DataFrame (index=date, columns=commodity names) of
                adjusted close prices; typically from aligned_prices.
    window    : rolling window in trading days (default 21).

    Returns
    -------
    pd.DataFrame with columns [date, commodity_a, commodity_b, correlation,
    window_days].  Only the LAST date's correlations are returned (the
    "today" snapshot).  Use store_correlation_snapshot() to persist them.
    """
    if prices_df.empty or len(prices_df) < window:
        return pd.DataFrame()

    log_returns = np.log(prices_df / prices_df.shift(1)).dropna(how="all")
    if len(log_returns) < window:
        return pd.DataFrame()

    # Rolling correlation matrix for the last window rows
    tail = log_returns.tail(window)
    corr_matrix = tail.corr(method="pearson")

    snapshot_date = log_returns.index[-1].date()
    records: List[dict] = []
    commodities = corr_matrix.columns.tolist()
    for i, a in enumerate(commodities):
        for b in commodities[i + 1:]:          # upper triangle only; avoid duplicates
            val = corr_matrix.loc[a, b]
            if np.isnan(val):
                continue
            records.append({
                "date":        snapshot_date,
                "commodity_a": a,
                "commodity_b": b,
                "correlation": round(float(val), 6),
                "window_days": window,
            })

    return pd.DataFrame(records)


def store_correlation_snapshot(
    prices_df: Optional[pd.DataFrame] = None,
    snapshot_date: Optional[date] = None,
    window: int = CORRELATION_WINDOW,
) -> int:
    """
    Compute today's rolling correlations and persist to correlation_snapshots.

    If prices_df is None, loads aligned_prices from the DB automatically.
    If snapshot_date is provided, only rows for that date are written
    (useful for back-filling historical snapshots).

    Returns the number of rows inserted.
    """
    if prices_df is None:
        prices_df = _load_aligned_prices()

    corr_df = compute_rolling_correlations(prices_df, window=window)
    if corr_df.empty:
        log.warning("store_correlation_snapshot: no correlations computed")
        return 0

    if snapshot_date is not None:
        corr_df = corr_df[corr_df["date"] == snapshot_date]

    rows = [
        {
            "date":        str(r["date"]),
            "commodity_a": r["commodity_a"],
            "commodity_b": r["commodity_b"],
            "correlation": r["correlation"],
            "window_days": r["window_days"],
        }
        for _, r in corr_df.iterrows()
    ]

    sql = text("""
        INSERT INTO correlation_snapshots
            (date, commodity_a, commodity_b, correlation, window_days)
        VALUES (:date, :commodity_a, :commodity_b, :correlation, :window_days)
        ON CONFLICT (date, commodity_a, commodity_b) DO UPDATE SET
            correlation = EXCLUDED.correlation,
            window_days = EXCLUDED.window_days
    """)
    with get_engine().connect() as conn:
        conn.execute(sql, rows)
        conn.commit()
    log.info("Stored %d correlation pairs for %s", len(rows),
             corr_df["date"].iloc[0] if not corr_df.empty else "unknown")
    return len(rows)


def load_correlation_matrix(
    as_of: Optional[date] = None,
) -> pd.DataFrame:
    """
    Load the most recent correlation snapshot as a symmetric wide matrix.

    Parameters
    ----------
    as_of : specific date to load; defaults to latest available.

    Returns
    -------
    pd.DataFrame (commodity × commodity), symmetric, diagonal = 1.0.
    Empty DataFrame if no snapshots exist yet.
    """
    engine = get_engine()
    if as_of is not None:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT commodity_a, commodity_b, correlation FROM correlation_snapshots WHERE date = :d"),
                {"d": str(as_of)},
            )
            rows = pd.DataFrame(result.fetchall(), columns=result.keys())
    else:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT commodity_a, commodity_b, correlation FROM correlation_snapshots WHERE date = (SELECT MAX(date) FROM correlation_snapshots)"
            ))
            rows = pd.DataFrame(result.fetchall(), columns=result.keys())

    if rows.empty:
        return pd.DataFrame()

    # Build symmetric matrix
    commodities = sorted(set(rows["commodity_a"]) | set(rows["commodity_b"]))
    mat = pd.DataFrame(1.0, index=commodities, columns=commodities)
    for _, r in rows.iterrows():
        mat.loc[r["commodity_a"], r["commodity_b"]] = r["correlation"]
        mat.loc[r["commodity_b"], r["commodity_a"]] = r["correlation"]
    return mat


# ── Covariance matrix ─────────────────────────────────────────────────────────

def compute_rolling_covariance(
    prices_df: pd.DataFrame,
    window: int = CORRELATION_WINDOW,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute the rolling covariance matrix of log returns for all commodity pairs.

    Covariance preserves magnitude information that correlation discards:
      Cov(r_a, r_b) = σ_a · σ_b · ρ_{ab}
    The diagonal entries are per-commodity variances (σ²), which are used by
    portfolio optimisers and risk sizing routines.

    Parameters
    ----------
    prices_df : wide DataFrame (index=date, columns=commodity names) of
                adjusted close prices.
    window    : rolling window in trading days (default 21, matching correlation).
    annualize : if True, multiply by 252 so units are ann. return² (default True).

    Returns
    -------
    pd.DataFrame with columns [date, commodity_a, commodity_b, covariance,
    window_days, annualized].  Only the last date's values are returned.
    Use store_covariance_snapshot() to persist them.
    """
    if prices_df.empty or len(prices_df) < window:
        return pd.DataFrame()

    log_returns = np.log(prices_df / prices_df.shift(1)).dropna(how="all")
    if len(log_returns) < window:
        return pd.DataFrame()

    tail = log_returns.tail(window)
    cov_matrix = tail.cov()
    if annualize:
        cov_matrix = cov_matrix * 252

    snapshot_date = log_returns.index[-1].date()
    records: List[dict] = []
    commodities = cov_matrix.columns.tolist()
    for i, a in enumerate(commodities):
        for b in commodities[i:]:           # include diagonal (variance terms)
            val = cov_matrix.loc[a, b]
            if np.isnan(val):
                continue
            records.append({
                "date":        snapshot_date,
                "commodity_a": a,
                "commodity_b": b,
                "covariance":  round(float(val), 8),
                "window_days": window,
                "annualized":  int(annualize),
            })

    return pd.DataFrame(records)


def store_covariance_snapshot(
    prices_df: Optional[pd.DataFrame] = None,
    snapshot_date: Optional[date] = None,
    window: int = CORRELATION_WINDOW,
    annualize: bool = True,
) -> int:
    """
    Compute today's rolling covariance matrix and persist to covariance_snapshots.

    If prices_df is None, loads aligned_prices from the DB automatically.

    Returns the number of rows inserted.
    """
    if prices_df is None:
        prices_df = _load_aligned_prices()

    cov_df = compute_rolling_covariance(prices_df, window=window, annualize=annualize)
    if cov_df.empty:
        log.warning("store_covariance_snapshot: no covariance computed")
        return 0

    if snapshot_date is not None:
        cov_df = cov_df[cov_df["date"] == snapshot_date]

    rows = [
        {
            "date":        str(r["date"]),
            "commodity_a": r["commodity_a"],
            "commodity_b": r["commodity_b"],
            "covariance":  r["covariance"],
            "window_days": r["window_days"],
            "annualized":  r["annualized"],
        }
        for _, r in cov_df.iterrows()
    ]

    sql = text("""
        INSERT INTO covariance_snapshots
            (date, commodity_a, commodity_b, covariance, window_days, annualized)
        VALUES (:date, :commodity_a, :commodity_b, :covariance, :window_days, :annualized)
        ON CONFLICT (date, commodity_a, commodity_b) DO UPDATE SET
            covariance  = EXCLUDED.covariance,
            window_days = EXCLUDED.window_days,
            annualized  = EXCLUDED.annualized
    """)
    with get_engine().connect() as conn:
        conn.execute(sql, rows)
        conn.commit()
    log.info(
        "Stored %d covariance entries for %s (annualized=%s)",
        len(rows),
        cov_df["date"].iloc[0] if not cov_df.empty else "unknown",
        bool(annualize),
    )
    return len(rows)


def load_covariance_matrix(
    as_of: Optional[date] = None,
    annualized: bool = True,
) -> pd.DataFrame:
    """
    Load the most recent covariance snapshot as a symmetric wide matrix.

    Diagonal entries are per-commodity annualised variances; off-diagonal
    entries are annualised covariances between pairs.

    Parameters
    ----------
    as_of      : specific date to load; defaults to latest available.
    annualized : filter to annualized=1 rows (default True).

    Returns
    -------
    pd.DataFrame (commodity × commodity), symmetric.
    Empty DataFrame if no snapshots exist yet.
    """
    ann_flag = int(annualized)
    engine = get_engine()

    if as_of is not None:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT commodity_a, commodity_b, covariance FROM covariance_snapshots WHERE date = :d AND annualized = :a"),
                {"d": str(as_of), "a": ann_flag},
            )
            rows = pd.DataFrame(result.fetchall(), columns=result.keys())
    else:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT commodity_a, commodity_b, covariance
                    FROM   covariance_snapshots
                    WHERE  date = (SELECT MAX(date) FROM covariance_snapshots WHERE annualized = :a)
                      AND  annualized = :a
                """),
                {"a": ann_flag},
            )
            rows = pd.DataFrame(result.fetchall(), columns=result.keys())

    if rows.empty:
        return pd.DataFrame()

    # Build symmetric matrix; diagonal stored as a=b pairs
    commodities = sorted(set(rows["commodity_a"]) | set(rows["commodity_b"]))
    mat = pd.DataFrame(np.nan, index=commodities, columns=commodities)
    for _, r in rows.iterrows():
        mat.loc[r["commodity_a"], r["commodity_b"]] = r["covariance"]
        mat.loc[r["commodity_b"], r["commodity_a"]] = r["covariance"]
    return mat


def covariance_to_correlation(cov_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a covariance matrix to a correlation matrix.

    Useful for validating that stored covariances are consistent with the
    separately-stored correlation snapshots.

    Corr(a, b) = Cov(a, b) / sqrt(Var(a) * Var(b))

    Parameters
    ----------
    cov_matrix : symmetric DataFrame with variances on the diagonal.

    Returns
    -------
    pd.DataFrame — same shape, values in [−1, 1].
    """
    std = np.sqrt(np.diag(cov_matrix.values))
    std_outer = np.outer(std, std)
    # Guard against zero variance (degenerate columns)
    std_outer = np.where(std_outer < 1e-12, np.nan, std_outer)
    corr_values = cov_matrix.values / std_outer
    return pd.DataFrame(corr_values, index=cov_matrix.index, columns=cov_matrix.columns)


def volatility_from_covariance(
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Extract per-commodity annualised volatility (σ) from the diagonal of the
    covariance matrix.

    Returns
    -------
    pd.Series indexed by commodity name, values = annualised σ (e.g. 0.25 = 25%).
    """
    if cov_matrix.empty:
        return pd.Series(dtype=float)
    variances = pd.Series(
        np.diag(cov_matrix.values), index=cov_matrix.index
    )
    return np.sqrt(variances.clip(lower=0)).rename("annualised_vol")


# ── Goal 2: Relative IC comparison ────────────────────────────────────────────

@dataclass
class RelativeICResult:
    """IC comparison for a correlated commodity pair."""
    commodity_a:     str
    commodity_b:     str
    tier:            str
    ic_a:            float
    ic_b:            float
    correlation:     float
    relative_pct:    float     # (ic_a - ic_b) / abs(ic_b) * 100; + means A > B
    regime:          str       # "" = regime-agnostic
    description:     str       # human-readable one-liner


def relative_ic_comparison(
    days: int = 90,
    min_correlation: float = 0.60,
    min_obs: int = 20,
) -> pd.DataFrame:
    """
    For every commodity pair whose 21-day rolling correlation exceeds
    min_correlation, compare their per-tier IC and express the gap as a
    percentage ("Brent's XGBoost IC is 26% higher than WTI's").

    Parameters
    ----------
    days            : IC look-back window (matches ic_tracker.recent_ic_scores).
    min_correlation : only consider pairs with |correlation| >= this threshold.
    min_obs         : skip IC entries with fewer than min_obs observations.

    Returns
    -------
    pd.DataFrame with columns:
        commodity_a, commodity_b, tier, ic_a, ic_b, correlation,
        relative_pct, regime, description
    Sorted by abs(relative_pct) DESC.  Empty DataFrame if insufficient data.
    """
    # 1. Load latest correlation matrix
    corr_mat = load_correlation_matrix()
    if corr_mat.empty:
        log.info("relative_ic_comparison: no correlation snapshots yet")
        return pd.DataFrame()

    # 2. Load IC scores
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with get_engine().connect() as conn:
        result = conn.execute(
            text("""
                SELECT commodity, tier, regime,
                       AVG(ic_value) AS ic_value,
                       SUM(n_obs)    AS n_obs
                FROM   ic_log
                WHERE  computed_at >= :cutoff
                GROUP  BY commodity, tier, regime
            """),
            {"cutoff": cutoff},
        )
        ic_raw = pd.DataFrame(result.fetchall(), columns=result.keys())

    if ic_raw.empty:
        return pd.DataFrame()

    ic_raw = ic_raw[ic_raw["n_obs"] >= min_obs]
    if ic_raw.empty:
        return pd.DataFrame()

    # Build lookup: (commodity, tier, regime) → avg IC
    ic_lookup: Dict[Tuple[str, str, str], float] = {}
    for _, row in ic_raw.iterrows():
        regime_key = row["regime"] if pd.notna(row["regime"]) else ""
        ic_lookup[(row["commodity"], row["tier"], regime_key)] = row["ic_value"]

    results: List[RelativeICResult] = []
    commodities = corr_mat.columns.tolist()
    seen_pairs: set = set()

    for i, a in enumerate(commodities):
        for b in commodities[i + 1:]:
            corr = corr_mat.loc[a, b]
            if abs(corr) < min_correlation:
                continue

            pair_key = (min(a, b), max(a, b))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Compare for each (tier, regime) combination present for both
            tiers_regimes = {(t, rg) for (c, t, rg) in ic_lookup if c == a} & \
                            {(t, rg) for (c, t, rg) in ic_lookup if c == b}

            for tier, regime in tiers_regimes:
                ic_a = ic_lookup.get((a, tier, regime))
                ic_b = ic_lookup.get((b, tier, regime))
                if ic_a is None or ic_b is None:
                    continue
                if abs(ic_b) < 1e-9:
                    continue

                rel_pct = (ic_a - ic_b) / abs(ic_b) * 100.0
                winner, loser = (a, b) if rel_pct >= 0 else (b, a)
                winner_ic = ic_a if rel_pct >= 0 else ic_b
                pct_str = f"{abs(rel_pct):.0f}%"
                tier_label = tier.capitalize()
                regime_part = f" in {regime.replace('_', ' ').title()} regime" if regime else ""
                desc = (
                    f"{winner}'s {tier_label} IC ({winner_ic:.3f}) is "
                    f"{pct_str} more reliable than {loser}'s{regime_part} "
                    f"(correlation={corr:.2f})"
                )
                results.append(RelativeICResult(
                    commodity_a=a, commodity_b=b, tier=tier,
                    ic_a=round(ic_a, 6), ic_b=round(ic_b, 6),
                    correlation=round(corr, 4),
                    relative_pct=round(rel_pct, 2),
                    regime=regime, description=desc,
                ))

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([vars(r) for r in results])
    df = df.sort_values("relative_pct", key=abs, ascending=False).reset_index(drop=True)
    return df


# ── Goal 3: Cross-asset prior propagation ─────────────────────────────────────

def apply_cross_asset_prior(
    forecasts: Dict[str, float],
    corr_matrix: pd.DataFrame,
    ic_df: Optional[pd.DataFrame] = None,
    min_correlation: float = 0.70,
    prior_weight: float = 0.20,
) -> Dict[str, float]:
    """
    Blend each commodity's raw forecast with a weighted prior from the
    most IC-dominant correlated peer.

    For a pair (A, B) with correlation r ≥ min_correlation:
      - If A's IC is higher, use A's forecast as a prior for B.
      - blended_B = (1 - α) * raw_B + α * raw_A
        where α = prior_weight * (|relative_ic| / 100) clamped to [0, 0.35].
      - Inverse-correlation pairs (r < 0) flip the sign of the prior.

    This implements Goal 3 without requiring a full meta-predictor refit.

    Parameters
    ----------
    forecasts       : {commodity: forecast_return} from the current day's run.
    corr_matrix     : wide correlation matrix from load_correlation_matrix().
    ic_df           : output of relative_ic_comparison(); loaded from DB if None.
    min_correlation : threshold for "strongly correlated" pairs.
    prior_weight    : base blending weight before IC-scaling.

    Returns
    -------
    dict : same keys as forecasts, with prior-adjusted values.
    """
    if corr_matrix.empty or not forecasts:
        return forecasts

    if ic_df is None or ic_df.empty:
        ic_df = relative_ic_comparison()

    adjusted = dict(forecasts)

    # Build a quick lookup: (winner, loser) → relative_pct (positive = winner better)
    prior_map: Dict[str, Tuple[str, float]] = {}  # loser → (winner, relative_pct)
    if not ic_df.empty:
        for _, row in ic_df[ic_df["tier"] == "ml"].iterrows():
            rel = row["relative_pct"]
            if abs(rel) < 5:
                continue  # not material enough to use as prior
            winner = row["commodity_a"] if rel >= 0 else row["commodity_b"]
            loser  = row["commodity_b"] if rel >= 0 else row["commodity_a"]
            # Keep only the strongest prior per loser
            if loser not in prior_map or abs(rel) > abs(prior_map[loser][1]):
                prior_map[loser] = (winner, rel)

    for commodity, raw_fc in forecasts.items():
        if commodity not in prior_map:
            continue
        winner, rel_pct = prior_map[commodity]
        if winner not in forecasts:
            continue
        if corr_matrix.empty or commodity not in corr_matrix.index \
                or winner not in corr_matrix.index:
            continue

        corr = corr_matrix.loc[commodity, winner]
        if abs(corr) < min_correlation:
            continue

        winner_fc = forecasts[winner]
        alpha = min(prior_weight * abs(rel_pct) / 100.0, 0.35)
        # Negative correlation: prior is the inverse
        signed_prior = winner_fc if corr >= 0 else -winner_fc
        adjusted[commodity] = (1.0 - alpha) * raw_fc + alpha * signed_prior
        log.debug(
            "cross_asset_prior: %s adjusted %.4f → %.4f (prior=%s α=%.2f r=%.2f)",
            commodity, raw_fc, adjusted[commodity], winner, alpha, corr,
        )

    return adjusted


# ── Goal 4: Forecast consistency checker ──────────────────────────────────────

@dataclass
class ConsistencyFlag:
    """A flagged inconsistency between two correlated commodity forecasts."""
    commodity_a:   str
    commodity_b:   str
    correlation:   float
    forecast_a:    float
    forecast_b:    float
    flag_type:     str    # "sign_divergence" | "magnitude_mismatch"
    severity:      str    # "warning" | "critical"
    message:       str


def check_forecast_consistency(
    forecasts: Dict[str, float],
    corr_matrix: Optional[pd.DataFrame] = None,
    min_correlation: float = 0.70,
    max_magnitude_ratio: float = 3.0,
) -> List[ConsistencyFlag]:
    """
    After all forecasts are generated, scan for cross-commodity inconsistencies.

    Two types of flags:
      sign_divergence  — correlated pair predicting opposite directions
                         (e.g. WTI ↑ 3% but Brent ↓ 1%)
      magnitude_mismatch — correlated pair with implausible magnitude gap
                           (e.g. WTI ↑ 3% but Brent only ↑ 0.1%)

    Parameters
    ----------
    forecasts           : {commodity: forecast_return} for the day.
    corr_matrix         : wide correlation matrix; loaded from DB if None.
    min_correlation     : only flag pairs with |correlation| >= this.
    max_magnitude_ratio : flag if |fc_a| / |fc_b| > this (both non-zero).

    Returns
    -------
    List[ConsistencyFlag], sorted by severity then correlation magnitude.
    Empty list if no inconsistencies found.
    """
    if corr_matrix is None or corr_matrix.empty:
        corr_matrix = load_correlation_matrix()

    if corr_matrix.empty or not forecasts:
        return []

    flags: List[ConsistencyFlag] = []
    commodities = [c for c in forecasts if c in corr_matrix.index]

    for i, a in enumerate(commodities):
        for b in commodities[i + 1:]:
            corr = corr_matrix.loc[a, b]
            if abs(corr) < min_correlation:
                continue

            fc_a = forecasts[a]
            fc_b = forecasts[b]

            # For negatively-correlated pairs, expected directions are opposite
            expected_same_sign = corr >= 0

            # Sign divergence: correlated pair pointing different ways
            actual_same_sign = (fc_a * fc_b) >= 0
            if expected_same_sign and not actual_same_sign:
                severity = "critical" if abs(corr) >= 0.85 else "warning"
                flags.append(ConsistencyFlag(
                    commodity_a=a, commodity_b=b,
                    correlation=round(corr, 4),
                    forecast_a=round(fc_a, 4), forecast_b=round(fc_b, 4),
                    flag_type="sign_divergence",
                    severity=severity,
                    message=(
                        f"{a} ({'+' if fc_a > 0 else ''}{fc_a:.2%}) vs "
                        f"{b} ({'+' if fc_b > 0 else ''}{fc_b:.2%}): "
                        f"correlation broke ({corr:.2f}); flag for review"
                    ),
                ))

            # Magnitude mismatch: same direction but wildly different sizes
            if abs(fc_b) > 1e-9 and abs(fc_a / fc_b) > max_magnitude_ratio:
                flags.append(ConsistencyFlag(
                    commodity_a=a, commodity_b=b,
                    correlation=round(corr, 4),
                    forecast_a=round(fc_a, 4), forecast_b=round(fc_b, 4),
                    flag_type="magnitude_mismatch",
                    severity="warning",
                    message=(
                        f"{a} ({fc_a:.2%}) is {abs(fc_a / fc_b):.1f}x larger "
                        f"than {b} ({fc_b:.2%}); unusual for r={corr:.2f}"
                    ),
                ))
            elif abs(fc_a) > 1e-9 and abs(fc_b / fc_a) > max_magnitude_ratio:
                flags.append(ConsistencyFlag(
                    commodity_a=a, commodity_b=b,
                    correlation=round(corr, 4),
                    forecast_a=round(fc_a, 4), forecast_b=round(fc_b, 4),
                    flag_type="magnitude_mismatch",
                    severity="warning",
                    message=(
                        f"{b} ({fc_b:.2%}) is {abs(fc_b / fc_a):.1f}x larger "
                        f"than {a} ({fc_a:.2%}); unusual for r={corr:.2f}"
                    ),
                ))

    flags.sort(key=lambda f: (f.severity != "critical", -abs(f.correlation)))
    return flags


# ── Forecast log helpers ───────────────────────────────────────────────────────

def log_forecasts(
    forecasts: Dict[str, Dict],
    forecast_date: Optional[date] = None,
) -> int:
    """
    Persist today's forecasts to forecast_log for future IC analysis.

    Parameters
    ----------
    forecasts : {commodity: {model_name, tier, forecast_return, confidence,
                              regime (optional)}}
    forecast_date : date of the forecast (defaults to today).

    Returns number of rows written.
    """
    fd = forecast_date or datetime.now(timezone.utc).date()
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    for commodity, info in forecasts.items():
        rows.append({
            "forecast_date":   str(fd),
            "commodity":       commodity,
            "model_name":      info.get("model_name", ""),
            "tier":            info.get("tier", ""),
            "regime":          info.get("regime") or None,
            "forecast_return": info["forecast_return"],
            "actual_return":   None,
            "error":           None,
            "confidence":      info.get("confidence") or None,
            "inserted_at":     now_iso,
        })

    sql = text("""
        INSERT INTO forecast_log
            (forecast_date, commodity, model_name, tier, regime,
             forecast_return, actual_return, error, confidence, inserted_at)
        VALUES (:forecast_date, :commodity, :model_name, :tier, :regime,
                :forecast_return, :actual_return, :error, :confidence, :inserted_at)
        ON CONFLICT (forecast_date, commodity, model_name) DO UPDATE SET
            forecast_return = EXCLUDED.forecast_return,
            regime          = EXCLUDED.regime,
            confidence      = EXCLUDED.confidence,
            inserted_at     = EXCLUDED.inserted_at
    """)
    with get_engine().connect() as conn:
        conn.execute(sql, rows)
        conn.commit()
    log.info("Logged %d forecasts for %s", len(rows), fd)
    return len(rows)


def realize_forecasts(
    actuals: Dict[str, float],
    forecast_date: Optional[date] = None,
) -> int:
    """
    Back-fill actual_return and error for the previous day's forecasts.

    Call once per day after prices are ingested.

    Parameters
    ----------
    actuals       : {commodity: realized_log_return} for forecast_date.
    forecast_date : the date whose forecasts to update (defaults to yesterday).

    Returns number of rows updated.
    """
    if forecast_date is None:
        forecast_date = (datetime.now(timezone.utc).date() - timedelta(days=1))

    updated = 0
    sql = text("""
        UPDATE forecast_log
        SET    actual_return = :actual,
               error         = ABS(forecast_return - :actual)
        WHERE  forecast_date = :fd AND commodity = :commodity
          AND  actual_return IS NULL
    """)
    with get_engine().connect() as conn:
        for commodity, actual in actuals.items():
            result = conn.execute(sql, {"actual": actual, "fd": str(forecast_date), "commodity": commodity})
            updated += result.rowcount
        conn.commit()
    log.info("Realized %d forecast rows for %s", updated, forecast_date)
    return updated


def regime_ic_table(
    days: int = 180,
    min_obs: int = 10,
) -> pd.DataFrame:
    """
    Compute per-regime IC from forecast_log for every (commodity, model_name).

    This is the direct implementation of Goal 2: regime-conditional IC.
    Requires forecast_log rows with both forecast_return and actual_return filled.

    Returns
    -------
    pd.DataFrame with columns:
        commodity, model_name, tier, regime, ic_value, n_obs
    Sorted by commodity, model_name, regime.
    Empty DataFrame if insufficient realized forecasts exist.
    """
    cutoff = str(datetime.now(timezone.utc).date() - timedelta(days=days))
    with get_engine().connect() as conn:
        result = conn.execute(
            text("""
                SELECT commodity, model_name, tier, regime,
                       forecast_return, actual_return
                FROM   forecast_log
                WHERE  forecast_date >= :cutoff
                  AND  actual_return IS NOT NULL
            """),
            {"cutoff": cutoff},
        )
        rows = pd.DataFrame(result.fetchall(), columns=result.keys())

    if rows.empty:
        return pd.DataFrame()

    results = []
    for (commodity, model, tier, regime), grp in rows.groupby(
        ["commodity", "model_name", "tier", "regime"], dropna=False
    ):
        if len(grp) < min_obs:
            continue
        fc = grp["forecast_return"].values
        ac = grp["actual_return"].values
        # Spearman via rank (no scipy needed)
        n = len(fc)
        fc_rank = pd.Series(fc).rank().values
        ac_rank = pd.Series(ac).rank().values
        cov = np.mean((fc_rank - fc_rank.mean()) * (ac_rank - ac_rank.mean()))
        denom = fc_rank.std(ddof=0) * ac_rank.std(ddof=0)
        ic = float(cov / denom) if denom > 1e-12 else float("nan")
        if np.isnan(ic):
            continue
        results.append({
            "commodity":  commodity,
            "model_name": model,
            "tier":       tier,
            "regime":     regime if pd.notna(regime) else "all",
            "ic_value":   round(ic, 6),
            "n_obs":      n,
        })

    if not results:
        return pd.DataFrame()
    return (
        pd.DataFrame(results)
        .sort_values(["commodity", "model_name", "regime"])
        .reset_index(drop=True)
    )
