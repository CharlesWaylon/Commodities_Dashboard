"""
Macro-to-Sector Routing Coefficients
=====================================
Learns how macro variables (DXY, VIX, TLT) impact each commodity sector's
returns, conditioned on the current market regime.

Methodology
-----------
For each (macro_variable, sector, regime) triple, run an OLS regression over a
rolling 2-year window:

    sector_return[t] = α + β · macro_var[t] + ε[t]

where:
  • macro_var is the daily log-return of DXY / VIX 5-day return / TLT daily return
  • sector_return is the equal-weighted daily log-return of all sector members
  • regime ∈ {bull, bear, high_vol, neutral} filters the date universe

β (the slope) is the routing coefficient: how many bps a sector moves per unit
of macro shock.  R² measures the explanatory power.

Regime classification (applied per day, based on the 2-year lookback window)
  bull     : VIX < 20  AND  WTI 63-day log-return > +5 %
  bear     : VIX < 20  AND  WTI 63-day log-return < −5 %
  high_vol : VIX ≥ 20  (includes crisis; VIX ≥ 30 days are not split further
             because they are too sparse for reliable regression)
  neutral  : VIX < 20  AND  −5% ≤ WTI 63-day return ≤ +5%

Output — models/macro_routes.pkl
---------------------------------
A pickled dict with keys:
  _metadata     : run timestamp, data range, n_obs per regime
  coefficients  : {macro_var → {regime → {sector → slope_float}}}
  r_squared     : {macro_var → {regime → {sector → r2_float}}}
  p_values      : {macro_var → {regime → {sector → p_value_float}}}
  n_obs         : {macro_var → {regime → {sector → n_int}}}
  routing_matrix: simplified {label → {sector → slope}} (regime = "all")
  validation    : domain-knowledge checks + pass/fail results

Entry points
------------
  build_macro_routes(prices=None, macro_df=None, save=True) → MacroRouter
      Full pipeline.  Loads data from DB + yfinance if not provided.

  load_macro_routes(path=None) → dict
      Load the saved pkl as a plain dict.

  apply_macro_adjustment(sector_forecasts, macro_snapshot, regime, routes) → dict
      Adjust sector-level forecasts using the learned coefficients.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).resolve().parent          # models/
_PKL_PATH  = _MODEL_DIR / "macro_routes.pkl"

# ── Regression window ──────────────────────────────────────────────────────────
BACKTEST_DAYS   = 504    # ~2 trading years
MIN_OBS_REGIME  = 30     # minimum observations to fit regime-specific regression

# VIX regime thresholds (mirrors macro_overlays.py)
VIX_RISK_OFF = 20.0
VIX_CRISIS   = 30.0

# WTI momentum threshold for bull / bear classification
WTI_MOM_BULL =  0.05    # +5 % 63-day log-return
WTI_MOM_BEAR = -0.05    # −5 % 63-day log-return
WTI_MOM_WINDOW = 63     # ~3 trading months

REGIME_LABELS: List[str] = ["bull", "bear", "high_vol", "neutral"]
REGIME_ALL    = "all"    # sentinel for regime-agnostic fit

# ── Macro variables ────────────────────────────────────────────────────────────
# Keys must match column names produced by features.macro_overlays.macro_features()
MACRO_VARS: Dict[str, str] = {
    "dxy_ret":         "DXY daily log-return   (dollar strength shock)",
    "vix_ret5d":       "VIX 5-day log-return   (fear gauge spike)",
    "tlt_ret":         "TLT daily log-return   (bond price / rate proxy)",
    "tlt_yield_proxy": "−TLT return            (rising-rates proxy)",
}

# Convenience aliases used in the routing_matrix output
ROUTING_ALIASES = {
    "dxy_spike":  "dxy_ret",
    "vix_spike":  "vix_ret5d",
    "tlt_drop":   "tlt_yield_proxy",   # TLT falling = rates rising
}

# ── Domain-knowledge validation checks ────────────────────────────────────────
# (macro_var, sector, expected_sign, description)
#   expected_sign:  +1 → slope should be positive
#                   −1 → slope should be negative
VALIDATION_CHECKS: List[Tuple[str, str, int, str]] = [
    # DXY ↑ → commodity prices ↓ (most commodities are USD-priced)
    ("dxy_ret",  "energy",      -1, "DXY↑ → WTI/oil↓  (USD-pricing effect)"),
    ("dxy_ret",  "metals",      -1, "DXY↑ → Gold↓     (strong inverse for precious metals)"),
    ("dxy_ret",  "agriculture", -1, "DXY↑ → Grains↓   (mild negative, weather offsets)"),
    # VIX ↑ → risk-off → base metals collapse, digital assets sell off
    ("vix_ret5d", "metals",     -1, "VIX↑ → Metals↓   (Copper risk-off drag outweighs Gold bid)"),
    ("vix_ret5d", "digital",    -1, "VIX↑ → Bitcoin↓  (risk-on asset, high VIX sensitivity)"),
    ("vix_ret5d", "agriculture", 0, "VIX↑ → Grains≈0  (supply-driven; weaker macro link)"),
    # TLT ↑ (rates ↓) → Gold ↑ (lower opportunity cost of holding gold)
    ("tlt_ret",  "metals",      +1, "TLT↑ → Gold↑     (lower rates boost precious metals)"),
    # Rate spike (TLT ↓) → storage costs rise → slight commodity headwind
    ("tlt_yield_proxy", "energy",       -1, "Rising rates → Energy↓  (inventory cost, USD strength)"),
    ("tlt_yield_proxy", "agriculture",  -1, "Rising rates → Grains↓  (storage cost headwind)"),
]

# Zero-tolerance margin: checks with expected_sign==0 pass if |slope| < this
NEAR_ZERO_THRESHOLD = 0.05


# ── Sector membership ─────────────────────────────────────────────────────────
# Derived at import time from models.config.COMMODITY_SECTORS so it stays in sync.

def _build_sector_members() -> Dict[str, List[str]]:
    try:
        from models.config import COMMODITY_SECTORS
        members: Dict[str, List[str]] = {}
        for commodity, sector in COMMODITY_SECTORS.items():
            members.setdefault(sector, []).append(commodity)
        return members
    except Exception:
        return {}

SECTOR_MEMBERS: Dict[str, List[str]] = _build_sector_members()
SECTORS: List[str] = sorted(SECTOR_MEMBERS.keys())


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class RegressionResult:
    """Single OLS result for one (macro_var, sector, regime) triple."""
    macro_var:  str
    sector:     str
    regime:     str
    slope:      float
    intercept:  float
    r_squared:  float
    p_value:    float
    n_obs:      int

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05 and self.n_obs >= MIN_OBS_REGIME


@dataclass
class ValidationResult:
    """Outcome of one domain-knowledge check."""
    name:           str
    macro_var:      str
    sector:         str
    expected_sign:  int    # +1, -1, or 0
    actual_slope:   float
    r_squared:      float
    p_value:        float
    passed:         bool
    note:           str = ""


@dataclass
class MacroRoutesSummary:
    """Human-readable summary of a build run."""
    fitted_at:       str
    backtest_days:   int
    n_obs_total:     int
    n_obs_by_regime: Dict[str, int]
    n_checks:        int
    n_passed:        int
    failed_checks:   List[str]
    save_path:       str

    def pretty(self) -> str:
        lines = [
            "=" * 60,
            "  Macro Router — Route Coefficient Build",
            "=" * 60,
            f"  Fitted at      : {self.fitted_at}",
            f"  Window         : {self.backtest_days} trading days (~2 yr)",
            f"  Observations   : {self.n_obs_total} total",
        ]
        for regime, n in sorted(self.n_obs_by_regime.items()):
            lines.append(f"    {regime:<10}: {n}")
        status = "ALL PASSED" if self.n_passed == self.n_checks else (
            f"{self.n_passed}/{self.n_checks} passed"
        )
        lines.append(f"  Validation     : {status}")
        for msg in self.failed_checks:
            lines.append(f"    ✗  {msg}")
        if self.save_path:
            lines.append(f"  Saved to       : {self.save_path}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Core class ────────────────────────────────────────────────────────────────

class MacroRouter:
    """
    Learns and stores macro-to-sector routing coefficients.

    Parameters
    ----------
    backtest_days : int
        Rolling lookback window in trading days.
    min_obs_regime : int
        Minimum observations in a regime bucket to fit a reliable regression.
        If fewer observations are available, falls back to the 'all' bucket.
    """

    def __init__(
        self,
        backtest_days:  int = BACKTEST_DAYS,
        min_obs_regime: int = MIN_OBS_REGIME,
    ):
        self.backtest_days   = backtest_days
        self.min_obs_regime  = min_obs_regime

        self._results:    List[RegressionResult]   = []
        self._validation: List[ValidationResult]   = []
        self._metadata:   Dict                     = {}
        self._fitted_at:  str                      = ""

    # ── Regime classification ─────────────────────────────────────────────────

    def _classify_regimes(
        self,
        prices:   pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Return a Series of regime labels aligned to `prices.index`.

        Priority:
          1. high_vol: VIX ≥ 20
          2. bear:     VIX < 20 AND WTI 63-day return < −5%
          3. bull:     VIX < 20 AND WTI 63-day return > +5%
          4. neutral:  everything else
        """
        idx     = prices.index
        regimes = pd.Series("neutral", index=idx, dtype=str)

        # ── VIX threshold ──────────────────────────────────────────────────────
        if "vix" in macro_df.columns:
            vix = macro_df["vix"].reindex(idx).ffill(limit=5)
            regimes[vix >= VIX_RISK_OFF] = "high_vol"
            low_vix_mask = vix < VIX_RISK_OFF
        else:
            low_vix_mask = pd.Series(True, index=idx)

        # ── WTI momentum for low-VIX days ─────────────────────────────────────
        if "WTI Crude Oil" in prices.columns:
            wti     = prices["WTI Crude Oil"].ffill(limit=5)
            mom63   = np.log(wti / wti.shift(WTI_MOM_WINDOW))
            regimes[low_vix_mask & (mom63 < WTI_MOM_BEAR)] = "bear"
            regimes[low_vix_mask & (mom63 > WTI_MOM_BULL)] = "bull"

        return regimes

    # ── Sector return index ───────────────────────────────────────────────────

    def _sector_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Equal-weighted daily log-return for each sector.

        Members with >20% NaN are excluded to avoid ETF-proxy gaps biasing
        the sector index.

        Returns
        -------
        pd.DataFrame
            Columns = sector names, index = dates.
        """
        log_ret = np.log(prices / prices.shift(1))

        sector_idx: Dict[str, pd.Series] = {}
        for sector, members in SECTOR_MEMBERS.items():
            present = [c for c in members if c in log_ret.columns]
            if not present:
                continue
            # Drop members with excessive NaN
            clean = [c for c in present if log_ret[c].isna().mean() < 0.20]
            if not clean:
                clean = present   # keep them all if all are gappy
            sector_idx[sector] = log_ret[clean].mean(axis=1)

        return pd.DataFrame(sector_idx).dropna(how="all")

    # ── Macro variable columns ────────────────────────────────────────────────

    @staticmethod
    def _extract_macro_vars(macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pull the four routing macro columns from the full macro feature frame.
        Computes tlt_yield_proxy = −tlt_ret if missing.
        """
        wanted = list(MACRO_VARS.keys())
        available: Dict[str, pd.Series] = {}

        for col in wanted:
            if col in macro_df.columns:
                available[col] = macro_df[col]

        # Derive tlt_yield_proxy if raw tlt_ret is present but proxy isn't
        if "tlt_yield_proxy" not in available and "tlt_ret" in available:
            available["tlt_yield_proxy"] = -available["tlt_ret"]

        if not available:
            return pd.DataFrame()

        return pd.DataFrame(available)

    # ── OLS regression ────────────────────────────────────────────────────────

    @staticmethod
    def _fit_ols(
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float, float, float, int]:
        """
        OLS via scipy.stats.linregress.

        Returns (slope, intercept, r2, p_value, n_obs).
        Returns all-zero tuple if inputs are too short or degenerate.
        """
        mask = np.isfinite(X) & np.isfinite(y)
        X, y = X[mask], y[mask]
        n = len(X)
        if n < 10:
            return 0.0, 0.0, 0.0, 1.0, n
        try:
            result = linregress(X, y)
            r2     = float(result.rvalue ** 2)
            return (
                float(result.slope),
                float(result.intercept),
                r2,
                float(result.pvalue),
                n,
            )
        except Exception:
            return 0.0, 0.0, 0.0, 1.0, n

    # ── Main fit ──────────────────────────────────────────────────────────────

    def fit(
        self,
        prices:   pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> "MacroRouter":
        """
        Run all (macro_var × sector × regime) regressions on the rolling
        2-year window and store the results internally.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily closing prices from load_price_matrix_from_db().
            Must cover at least BACKTEST_DAYS rows.
        macro_df : pd.DataFrame
            Macro overlay features from macro_overlays.macro_features().
            Must contain columns: dxy, vix, tlt, dxy_ret, vix_ret5d, tlt_ret.
        """
        self._results  = []
        self._fitted_at = datetime.now(timezone.utc).isoformat()

        # ── 1. Truncate to backtest window ────────────────────────────────────
        prices   = prices.iloc[-self.backtest_days :].copy()
        macro_df = macro_df.copy()

        # ── 2. Build regime labels ────────────────────────────────────────────
        regimes = self._classify_regimes(prices, macro_df)

        n_by_regime = {r: int((regimes == r).sum()) for r in REGIME_LABELS}
        n_by_regime[REGIME_ALL] = len(regimes)
        log.info("Regime distribution over %d days: %s", len(regimes), n_by_regime)

        # ── 3. Build sector return index ──────────────────────────────────────
        sect_ret = self._sector_returns(prices)

        # ── 4. Extract macro variable series ─────────────────────────────────
        macro_vars_df = self._extract_macro_vars(macro_df)
        if macro_vars_df.empty:
            raise RuntimeError(
                "No macro variable columns found in macro_df. "
                "Expected: dxy_ret, vix_ret5d, tlt_ret, tlt_yield_proxy"
            )

        # ── 5. Align on common date index ─────────────────────────────────────
        common_idx = sect_ret.index.intersection(macro_vars_df.index)
        common_idx = common_idx.intersection(regimes.index)
        if len(common_idx) < MIN_OBS_REGIME * 2:
            raise RuntimeError(
                f"Only {len(common_idx)} overlapping dates between prices, "
                "macro, and regime series. Need at least "
                f"{MIN_OBS_REGIME * 2}."
            )

        sect_ret    = sect_ret.loc[common_idx]
        macro_v     = macro_vars_df.reindex(common_idx).ffill(limit=3)
        reg_aligned = regimes.reindex(common_idx)

        # ── 6. Run regressions ────────────────────────────────────────────────
        for macro_col in macro_v.columns:
            X_all = macro_v[macro_col].values

            for sector in sect_ret.columns:
                y_all = sect_ret[sector].values

                # Regime-agnostic fit (full window)
                slope, intercept, r2, pval, n = self._fit_ols(X_all, y_all)
                self._results.append(RegressionResult(
                    macro_var=macro_col, sector=sector, regime=REGIME_ALL,
                    slope=slope, intercept=intercept,
                    r_squared=r2, p_value=pval, n_obs=n,
                ))

                # Per-regime fits
                for regime in REGIME_LABELS:
                    mask     = (reg_aligned == regime).values
                    X_r, y_r = X_all[mask], y_all[mask]

                    if mask.sum() < self.min_obs_regime:
                        # Not enough data → copy the "all" coefficient
                        self._results.append(RegressionResult(
                            macro_var=macro_col, sector=sector, regime=regime,
                            slope=slope, intercept=intercept,
                            r_squared=r2, p_value=pval,
                            n_obs=int(mask.sum()),
                        ))
                        log.debug(
                            "Regime '%s' has only %d obs for %s/%s — "
                            "using full-window coefficient.",
                            regime, mask.sum(), macro_col, sector,
                        )
                        continue

                    s, i, r2r, p, nr = self._fit_ols(X_r, y_r)
                    self._results.append(RegressionResult(
                        macro_var=macro_col, sector=sector, regime=regime,
                        slope=s, intercept=i,
                        r_squared=r2r, p_value=p, n_obs=int(nr),
                    ))

            log.info(
                "Regressions done for macro_var=%s across %d sectors × %d regimes",
                macro_col, len(sect_ret.columns), len(REGIME_LABELS) + 1,
            )

        # ── 7. Store metadata ─────────────────────────────────────────────────
        self._metadata = {
            "fitted_at":      self._fitted_at,
            "backtest_days":  self.backtest_days,
            "date_start":     str(common_idx.min().date()),
            "date_end":       str(common_idx.max().date()),
            "n_obs_total":    len(common_idx),
            "n_obs_by_regime": n_by_regime,
            "macro_vars":     list(macro_v.columns),
            "sectors":        list(sect_ret.columns),
        }

        # ── 8. Run domain-knowledge validation ────────────────────────────────
        self._run_validation()

        return self

    # ── Lookup helpers ────────────────────────────────────────────────────────

    def get_slope(self, macro_var: str, sector: str, regime: str = REGIME_ALL) -> float:
        """Return the regression slope for (macro_var, sector, regime)."""
        for r in self._results:
            if r.macro_var == macro_var and r.sector == sector and r.regime == regime:
                return r.slope
        return 0.0

    def get_r2(self, macro_var: str, sector: str, regime: str = REGIME_ALL) -> float:
        for r in self._results:
            if r.macro_var == macro_var and r.sector == sector and r.regime == regime:
                return r.r_squared
        return 0.0

    # ── Routing matrix ────────────────────────────────────────────────────────

    def routing_matrix(self, regime: str = REGIME_ALL) -> Dict[str, Dict[str, float]]:
        """
        Return a simplified {macro_alias → {sector → slope}} dict for the
        given regime.  Uses the ROUTING_ALIASES mapping for human-readable keys.

        Example output::

            {
                "dxy_spike": {"energy": -0.27, "metals": -0.24, ...},
                "vix_spike": {"energy": -0.18, "metals": -0.31, ...},
                "tlt_drop":  {"energy": -0.05, "metals": +0.14, ...},
            }
        """
        matrix: Dict[str, Dict[str, float]] = {}
        for alias, macro_col in ROUTING_ALIASES.items():
            matrix[alias] = {
                s: round(self.get_slope(macro_col, s, regime), 4)
                for s in SECTORS
            }
        return matrix

    # ── Domain-knowledge validation ───────────────────────────────────────────

    def _run_validation(self) -> None:
        """Check learned coefficients against economic intuition."""
        self._validation = []

        for macro_var, sector, expected_sign, description in VALIDATION_CHECKS:
            slope = self.get_slope(macro_var, sector, REGIME_ALL)
            r2    = self.get_r2(macro_var, sector, REGIME_ALL)
            pval  = next(
                (r.p_value for r in self._results
                 if r.macro_var == macro_var and r.sector == sector
                 and r.regime == REGIME_ALL),
                1.0,
            )

            if expected_sign == 0:
                passed = abs(slope) < NEAR_ZERO_THRESHOLD
                note   = f"|slope| = {abs(slope):.3f} (threshold {NEAR_ZERO_THRESHOLD})"
            elif expected_sign > 0:
                passed = slope > 0
                note   = f"slope = {slope:+.4f}"
            else:
                passed = slope < 0
                note   = f"slope = {slope:+.4f}"

            if not passed:
                log.warning(
                    "Validation FAILED: %s — expected %s, got slope=%.4f (R²=%.3f)",
                    description,
                    "positive" if expected_sign > 0 else
                    "negative" if expected_sign < 0 else "≈ zero",
                    slope, r2,
                )

            self._validation.append(ValidationResult(
                name=description,
                macro_var=macro_var,
                sector=sector,
                expected_sign=expected_sign,
                actual_slope=round(slope, 4),
                r_squared=round(r2, 4),
                p_value=round(pval, 4),
                passed=passed,
                note=note,
            ))

    def validation_report(self) -> str:
        """Human-readable validation report."""
        if not self._validation:
            return "No validation results — call fit() first."
        lines = ["Macro Router — Domain Knowledge Validation", "-" * 50]
        for v in self._validation:
            tick = "✓" if v.passed else "✗"
            lines.append(
                f"  [{tick}] {v.name}\n"
                f"       slope={v.actual_slope:+.4f}  R²={v.r_squared:.3f}"
                f"  p={v.p_value:.3f}  ({v.note})"
            )
        n_pass = sum(1 for v in self._validation if v.passed)
        lines.append(f"\n  {n_pass}/{len(self._validation)} checks passed.")
        return "\n".join(lines)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Convert all learned data to a plain Python dict for pickling.

        Structure::

            {
                "_metadata":     {...},
                "coefficients":  {macro_var → {regime → {sector → slope}}},
                "r_squared":     {macro_var → {regime → {sector → r2}}},
                "p_values":      {macro_var → {regime → {sector → p}}},
                "n_obs":         {macro_var → {regime → {sector → n}}},
                "routing_matrix":{alias     → {sector → slope}},   # regime='all'
                "validation":    {"passed": bool, "checks": [...]},
            }
        """
        all_regimes = REGIME_LABELS + [REGIME_ALL]
        macro_cols  = {r.macro_var for r in self._results}

        coefficients: Dict = {mv: {rg: {} for rg in all_regimes} for mv in macro_cols}
        r_squared:    Dict = {mv: {rg: {} for rg in all_regimes} for mv in macro_cols}
        p_values:     Dict = {mv: {rg: {} for rg in all_regimes} for mv in macro_cols}
        n_obs:        Dict = {mv: {rg: {} for rg in all_regimes} for mv in macro_cols}

        for res in self._results:
            if res.regime not in all_regimes:
                continue
            coefficients[res.macro_var][res.regime][res.sector] = round(res.slope, 6)
            r_squared   [res.macro_var][res.regime][res.sector] = round(res.r_squared, 4)
            p_values    [res.macro_var][res.regime][res.sector] = round(res.p_value, 4)
            n_obs       [res.macro_var][res.regime][res.sector] = res.n_obs

        validation_checks = [
            {
                "name":          v.name,
                "macro_var":     v.macro_var,
                "sector":        v.sector,
                "expected_sign": v.expected_sign,
                "actual_slope":  v.actual_slope,
                "r_squared":     v.r_squared,
                "p_value":       v.p_value,
                "passed":        v.passed,
                "note":          v.note,
            }
            for v in self._validation
        ]

        return {
            "_metadata":     self._metadata,
            "coefficients":  coefficients,
            "r_squared":     r_squared,
            "p_values":      p_values,
            "n_obs":         n_obs,
            "routing_matrix": self.routing_matrix(REGIME_ALL),
            "validation": {
                "n_checks":  len(self._validation),
                "n_passed":  sum(1 for v in self._validation if v.passed),
                "all_passed": all(v.passed for v in self._validation),
                "checks":    validation_checks,
            },
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """Pickle the route dict to disk. Returns the saved path."""
        out = Path(path) if path else _PKL_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(self.to_dict(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Macro routes saved → %s", out)
        return out

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "MacroRouter":
        """Load a previously saved MacroRouter from pickle."""
        src = Path(path) if path else _PKL_PATH
        with open(src, "rb") as fh:
            data = pickle.load(fh)

        router = cls()
        router._metadata   = data.get("_metadata", {})
        router._fitted_at  = router._metadata.get("fitted_at", "")
        router._results    = []   # raw results not stored; reconstruct from dict

        # Reconstruct RegressionResult list from the nested dicts
        coefficients = data.get("coefficients", {})
        r_sq         = data.get("r_squared",    {})
        pvs          = data.get("p_values",     {})
        nobs         = data.get("n_obs",        {})

        for macro_var, regimes in coefficients.items():
            for regime, sectors in regimes.items():
                for sector, slope in sectors.items():
                    router._results.append(RegressionResult(
                        macro_var=macro_var, sector=sector, regime=regime,
                        slope=slope,
                        intercept=0.0,   # not stored separately
                        r_squared=r_sq.get(macro_var, {}).get(regime, {}).get(sector, 0.0),
                        p_value=pvs.get(macro_var, {}).get(regime, {}).get(sector, 1.0),
                        n_obs=nobs.get(macro_var, {}).get(regime, {}).get(sector, 0),
                    ))

        checks = data.get("validation", {}).get("checks", [])
        router._validation = [
            ValidationResult(
                name=c["name"], macro_var=c["macro_var"], sector=c["sector"],
                expected_sign=c["expected_sign"], actual_slope=c["actual_slope"],
                r_squared=c["r_squared"], p_value=c["p_value"],
                passed=c["passed"], note=c.get("note", ""),
            )
            for c in checks
        ]

        return router

    def summary(self) -> MacroRoutesSummary:
        n_obs = self._metadata.get("n_obs_by_regime", {})
        failed = [v.name for v in self._validation if not v.passed]
        return MacroRoutesSummary(
            fitted_at=self._fitted_at,
            backtest_days=self.backtest_days,
            n_obs_total=self._metadata.get("n_obs_total", 0),
            n_obs_by_regime={k: v for k, v in n_obs.items() if k != REGIME_ALL},
            n_checks=len(self._validation),
            n_passed=sum(1 for v in self._validation if v.passed),
            failed_checks=failed,
            save_path="",
        )


# ── Data loaders (standalone, no Streamlit dependency) ────────────────────────

def _load_prices_for_router(period: str = "3y") -> pd.DataFrame:
    """Load 41-commodity price matrix from SQLite (DB-first, no yfinance)."""
    try:
        from models.data_loader import load_price_matrix_from_db
        df = load_price_matrix_from_db()
        log.info("MacroRouter: prices loaded from DB (%d rows × %d cols)",
                 len(df), df.shape[1])
        return df
    except Exception as exc:
        log.warning("MacroRouter: DB price load failed (%s).", exc)
        return pd.DataFrame()


def _load_macro_for_router(period: str = "2y") -> pd.DataFrame:
    """Fetch DXY / VIX / TLT features from yfinance."""
    try:
        from features.macro_overlays import macro_features
        df = macro_features(period=period)
        if df.empty:
            raise RuntimeError("macro_features() returned empty DataFrame")
        log.info("MacroRouter: macro features loaded (%d rows × %d cols)",
                 len(df), df.shape[1])
        return df
    except Exception as exc:
        log.warning("MacroRouter: macro feature load failed (%s).", exc)
        return pd.DataFrame()


# ── Staleness check ────────────────────────────────────────────────────────────

def _is_stale(path: Optional[Path] = None, max_age_days: int = 7) -> bool:
    """Return True if the pkl is missing or older than max_age_days."""
    p = Path(path) if path else _PKL_PATH
    if not p.exists():
        return True
    age = datetime.now().timestamp() - p.stat().st_mtime
    return age > max_age_days * 86400


# ── Public API ─────────────────────────────────────────────────────────────────

def build_macro_routes(
    prices:    Optional[pd.DataFrame] = None,
    macro_df:  Optional[pd.DataFrame] = None,
    save:      bool = True,
    path:      Optional[Path] = None,
) -> MacroRouter:
    """
    Full pipeline: load data → fit regressions → validate → (optionally) save.

    Parameters
    ----------
    prices   : pre-loaded price DataFrame (skips DB fetch if provided).
    macro_df : pre-loaded macro feature DataFrame (skips yfinance if provided).
    save     : whether to pickle the result.
    path     : custom pkl path. Defaults to models/macro_routes.pkl.

    Returns
    -------
    MacroRouter
        Fitted router with all coefficients populated.
    """
    if prices is None:
        prices = _load_prices_for_router()
    if prices.empty:
        raise RuntimeError("build_macro_routes: price matrix is empty.")

    if macro_df is None:
        macro_df = _load_macro_for_router()
    if macro_df.empty:
        raise RuntimeError(
            "build_macro_routes: macro features are empty. "
            "Check yfinance connectivity for DXY/VIX/TLT."
        )

    router = MacroRouter()
    router.fit(prices, macro_df)

    if save:
        out  = router.save(path)
        smry = router.summary()
        smry.save_path = str(out)
        print(smry.pretty())
        print()
        print(router.validation_report())

    return router


def load_macro_routes(path: Optional[Path] = None) -> dict:
    """
    Load the saved macro routes as a plain dict.

    Returns an empty dict if the pkl does not exist yet.
    """
    p = Path(path) if path else _PKL_PATH
    if not p.exists():
        log.warning("macro_routes.pkl not found at %s — run build_macro_routes().", p)
        return {}
    with open(p, "rb") as fh:
        return pickle.load(fh)


def get_current_regime(macro_df: pd.DataFrame) -> str:
    """
    Classify the most recent date in macro_df into a regime label.

    Parameters
    ----------
    macro_df : pd.DataFrame
        Output of macro_features() — must contain 'vix' column.

    Returns
    -------
    str
        One of: "bull", "bear", "high_vol", "neutral"
    """
    if macro_df.empty or "vix" not in macro_df.columns:
        return "neutral"

    latest = macro_df.dropna(subset=["vix"]).iloc[-1]
    vix    = float(latest.get("vix", 15.0))

    if vix >= VIX_RISK_OFF:
        return "high_vol"

    # Approximate momentum from dxy_mom21 as commodity-trend proxy
    # (WTI not directly available in macro_df — use DXY momentum as inverse proxy)
    dxy_mom = float(latest.get("dxy_mom21", 0.0))
    # DXY rising → commodities falling → "bear" for commodities
    if dxy_mom > 0.025:
        return "bear"
    if dxy_mom < -0.025:
        return "bull"
    return "neutral"


def apply_macro_adjustment(
    sector_forecasts: Dict[str, float],
    macro_snapshot:   Dict[str, float],
    regime:           str = "neutral",
    routes:           Optional[dict] = None,
    path:             Optional[Path] = None,
) -> Dict[str, float]:
    """
    Adjust sector-level return forecasts using learned macro coefficients.

    Parameters
    ----------
    sector_forecasts : {sector: base_forecast_return}
        Raw model forecasts before macro adjustment.
    macro_snapshot   : {macro_var: observed_value}
        Current values of macro variables (e.g. today's DXY log-return).
        Keys: "dxy_ret", "vix_ret5d", "tlt_ret", "tlt_yield_proxy"
    regime           : current market regime label.
    routes           : pre-loaded routes dict (loaded from pkl if None).

    Returns
    -------
    dict
        {sector: adjusted_forecast}
    """
    if routes is None:
        routes = load_macro_routes(path)
    if not routes:
        return sector_forecasts.copy()

    coefficients = routes.get("coefficients", {})
    adjusted     = {}

    for sector, base_fc in sector_forecasts.items():
        total_adj = 0.0
        for macro_var, macro_val in macro_snapshot.items():
            if macro_var not in coefficients:
                continue
            regime_dict = coefficients[macro_var]
            # Use requested regime; fall back to "all" if not present
            sector_slopes = regime_dict.get(regime, regime_dict.get(REGIME_ALL, {}))
            slope          = sector_slopes.get(sector, 0.0)
            total_adj     += slope * float(macro_val)

        adjusted[sector] = round(base_fc + total_adj, 6)

    return adjusted


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    """python -m models.macro_router  [--dry-run] [--force]"""
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Build macro-to-sector routing coefficients.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fit but do not save the pkl.")
    parser.add_argument("--force", action="store_true",
                        help="Refit even if pkl is fresh (< 7 days old).")
    args = parser.parse_args()

    if not args.force and not _is_stale():
        age_h = (_PKL_PATH.stat().st_mtime - datetime.now().timestamp()) / -3600
        print(f"macro_routes.pkl is {age_h:.1f}h old — skipping (use --force to override).")
        sys.exit(0)

    try:
        router = build_macro_routes(save=not args.dry_run)
    except Exception as exc:
        log.error("MacroRouter build failed: %s", exc, exc_info=True)
        sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] Routing matrix (regime=all):")
        import json
        mat = router.routing_matrix()
        print(json.dumps(mat, indent=2))


if __name__ == "__main__":
    main()
