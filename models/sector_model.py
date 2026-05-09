"""
Sector-Specific Model Wrapper
==============================
Wraps XGBoostForecaster with three sector-aware layers:

  1. Sector-filtered feature matrix  — only the commodity columns for this
     sector (+ macro signals) are used, reducing feature dimensionality from
     ~1,968 to ~100–150 and cutting cross-sector noise.

  2. Macro routing adjustment        — learned OLS coefficients from
     macro_router.py translate DXY/VIX/TLT observations into a regime-
     conditioned sector return adjustment.

  3. Upstream shock propagation      — forecasts from causally upstream
     commodities/sectors (e.g. WTI → Heating Oil) are propagated via pairwise
     correlations from the last correlation snapshot, then damped before
     being added to the base forecast.

The class inherits SignalBroadcaster and emits ModelSignal objects through
both its standard predict_with_signal() and the richer predict_with_context()
method that returns a full breakdown dict alongside the signal.

Usage
-----
    from models.sector_model import SectorSpecificModel
    from models.data_loader import load_price_matrix_from_db

    prices = load_price_matrix_from_db()
    model  = SectorSpecificModel(sector="energy", commodity="WTI Crude Oil")
    model.fit(prices)

    # Standard signal (Streamlit / meta-predictor compatible)
    signal = model.predict_with_signal(prices)

    # Full contextual breakdown
    result = model.predict_with_context(
        prices,
        macro_state={"dxy_ret": 0.004, "vix_ret5d": 0.12},
        upstream_shocks={"Natural Gas": 0.018},
    )
    print(result["breakdown"])
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.broadcaster import SignalBroadcaster
from models.config import COMMODITY_SECTORS
from models.features import build_feature_matrix, build_target
from models.ml.xgboost_shap import XGBoostForecaster
from models.model_signal import ModelSignal

log = logging.getLogger(__name__)

# ── Tuning constants ───────────────────────────────────────────────────────────

# Upstream shocks are multiplied by correlation × this factor before adding to
# the base forecast.  Keeps upstream propagation from dominating the signal.
UPSTREAM_DAMPING = 0.25

# Minimum IC to treat as a useable base confidence
MIN_IC = 0.0

# Default intra-sector correlation used when the DB snapshot is unavailable
_DEFAULT_INTRA_SECTOR_CORR  = 0.50
_DEFAULT_CROSS_SECTOR_CORR  = 0.10

# Key aliases: normalise caller-friendly names → macro_router canonical names
_MACRO_KEY_ALIASES: Dict[str, str] = {
    # DXY variants
    "DXY_zscore":       "dxy_ret",
    "dxy_zscore":       "dxy_ret",
    "DXY_ret":          "dxy_ret",
    "DXY":              "dxy_ret",
    # VIX variants
    "VIX_zscore":       "vix_ret5d",
    "vix_zscore":       "vix_ret5d",
    "VIX_ret5d":        "vix_ret5d",
    "VIX_level":        "vix_ret5d",
    "VIX":              "vix_ret5d",
    "vix":              "vix_ret5d",
    # TLT / rates variants
    "TLT_ret":          "tlt_ret",
    "tlt_mom21":        "tlt_ret",
    "TLT":              "tlt_ret",
    "tlt_yield_proxy":  "tlt_yield_proxy",
    "rates_proxy":      "tlt_yield_proxy",
    "rising_rates":     "tlt_yield_proxy",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_macro_state(macro_state: Dict[str, float]) -> Dict[str, float]:
    """
    Remap caller-friendly macro key names to canonical macro_router keys.

    IMPORTANT — value scale: the OLS regression was fit on daily log-returns
    (dxy_ret ~ 0.002–0.008, vix_ret5d ~ 0.02–0.15, tlt_ret ~ 0.002–0.010).
    Pass values in the same units as the raw macro_features() columns.
    Passing z-scores directly will produce over-scaled adjustments.
    """
    out: Dict[str, float] = {}
    for k, v in macro_state.items():
        canonical = _MACRO_KEY_ALIASES.get(k, k)
        val = float(v)
        # Heuristic: if the caller passed a z-score key (contains "zscore") and
        # the magnitude looks like a z-score (>0.10), scale it to log-return units.
        # Typical daily return sigma: DXY ~0.004, VIX5d ~0.08, TLT ~0.005.
        if "zscore" in k.lower() and abs(val) > 0.10:
            if "dxy" in k.lower():
                val = val * 0.004      # z=1 → ~0.4% DXY move
            elif "vix" in k.lower():
                val = val * 0.08       # z=1 → ~8% VIX 5-day move
            elif "tlt" in k.lower():
                val = val * 0.005      # z=1 → ~0.5% TLT move
            log.debug(
                "_normalize_macro_state: '%s' looks like a z-score — scaled to %.6f", k, val
            )
        out[canonical] = val
    return out


def _load_corr_matrix() -> pd.DataFrame:
    """Load the latest pairwise correlation snapshot. Returns empty DF on failure."""
    try:
        from models.cross_asset import load_correlation_matrix
        return load_correlation_matrix()
    except Exception as exc:
        log.debug("Correlation matrix unavailable (%s) — using defaults.", exc)
        return pd.DataFrame()


def _load_routes() -> dict:
    """Lazy-load macro routes pkl once per process. Returns empty dict on failure."""
    try:
        from models.macro_router import load_macro_routes
        return load_macro_routes()
    except Exception as exc:
        log.debug("Macro routes unavailable (%s).", exc)
        return {}


# Module-level lazy caches so repeated calls in one session don't reload from disk
_routes_cache:  Optional[dict]         = None
_corr_cache:    Optional[pd.DataFrame] = None


def _get_routes() -> dict:
    global _routes_cache
    if _routes_cache is None:
        _routes_cache = _load_routes()
    return _routes_cache


def _get_corr() -> pd.DataFrame:
    global _corr_cache
    if _corr_cache is None:
        _corr_cache = _load_corr_matrix()
    return _corr_cache


# ── Main class ─────────────────────────────────────────────────────────────────

class SectorSpecificModel(SignalBroadcaster):
    """
    Sector-aware XGBoost wrapper that supports macro-adjusted and
    upstream-shock-propagated forecasts.

    Parameters
    ----------
    sector : str
        One of: 'energy', 'metals', 'agriculture', 'livestock', 'digital'.
    commodity : str
        Target commodity display name (must be in COMMODITY_SECTORS).
    upstream_damping : float
        Scaling factor applied to upstream shock contributions before adding
        to base forecast.  Default 0.25 — keeps propagation from dominating.
    """

    def __init__(
        self,
        sector: str,
        commodity: str,
        upstream_damping: float = UPSTREAM_DAMPING,
    ):
        super().__init__(
            commodity=commodity,
            model_type="ml",
            model_name="SectorSpecificModel",
        )
        self.sector           = sector
        self.upstream_damping = upstream_damping

        # Inner XGBoost model — not exposed directly; access via this wrapper
        self._xgb: XGBoostForecaster = XGBoostForecaster(commodity=commodity)

        # Stored during fit() for feature re-construction at predict time
        self._prices:   Optional[pd.DataFrame] = None
        self._feat_df:  Optional[pd.DataFrame] = None
        self._target:   Optional[pd.Series]    = None
        self._ic_score: float                  = 0.0

        # Validate sector membership
        if commodity and commodity in COMMODITY_SECTORS:
            actual_sector = COMMODITY_SECTORS[commodity]
            if actual_sector != sector:
                log.warning(
                    "SectorSpecificModel: '%s' belongs to sector '%s' in config "
                    "but was passed sector='%s'. Using '%s'.",
                    commodity, actual_sector, sector, actual_sector,
                )
                self.sector = actual_sector

    # ── Feature construction ───────────────────────────────────────────────────

    def _build_sector_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build sector-filtered features. Falls back to full matrix if sector filter fails."""
        try:
            return build_feature_matrix(prices, sector=self.sector)
        except ValueError as exc:
            log.warning(
                "Sector filter failed for '%s' (%s) — falling back to full feature matrix.",
                self.sector, exc,
            )
            return build_feature_matrix(prices)

    def _ensure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Accept either raw prices or a pre-built feature matrix.
        Detection: if any fitted feature name exists as a column, treat as features;
        otherwise assume raw prices and apply sector filtering.
        """
        if self._xgb._feature_names is None:
            return self._build_sector_features(data)

        # If at least one expected feature column is present, assume features
        feature_set = set(self._xgb._feature_names)
        if any(c in feature_set for c in data.columns):
            return data

        # Raw prices → build sector features
        return self._build_sector_features(data)

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        prices: pd.DataFrame,
        target: Optional[pd.Series] = None,
    ) -> "SectorSpecificModel":
        """
        Build sector-specific features and fit the wrapped XGBoostForecaster.

        Parameters
        ----------
        prices : pd.DataFrame
            Full price matrix from load_price_matrix_from_db().
        target : pd.Series, optional
            Next-day log-return of self.commodity.  Built automatically if None.
        """
        self._prices = prices

        # Sector-filtered feature matrix (reduced dimensionality)
        self._feat_df = self._build_sector_features(prices)

        if target is None:
            target = build_target(prices, self.commodity)
        self._target = target

        self._xgb.fit(self._feat_df, target)
        self._ic_score = self._xgb._ic_score

        self.set_metadata(
            sector=self.sector,
            n_sector_features=self._feat_df.shape[1],
            ic=self._ic_score,
        )
        log.info(
            "SectorSpecificModel fitted: %s / %s — %d sector features, IC=%.4f",
            self.sector, self.commodity, self._feat_df.shape[1], self._ic_score,
        )
        return self

    # ── Standard signal interface (ABC) ───────────────────────────────────────

    def predict_with_signal(
        self,
        features: pd.DataFrame,
        horizon: int = 1,
    ) -> ModelSignal:
        """
        Standard ModelSignal output. Accepts either raw prices or sector features.

        This satisfies the SignalBroadcaster ABC. For the full contextual breakdown
        (macro + upstream adjustments) use predict_with_context() instead.
        """
        if self._xgb._model is None:
            raise RuntimeError("Call fit() first.")

        feat = self._ensure_features(features)
        signal = self._xgb.predict_with_signal(feat, horizon=horizon)

        # Re-stamp with this wrapper's identity
        from dataclasses import replace
        signal = replace(
            signal,
            model_name="SectorSpecificModel",
            metadata={
                **signal.metadata,
                "sector": self.sector,
                "wrapped_model": "XGBoostForecaster",
            },
        )
        return signal

    # ── Contextual forecast ────────────────────────────────────────────────────

    def predict_with_context(
        self,
        prices: pd.DataFrame,
        macro_state: Optional[Dict[str, float]] = None,
        upstream_shocks: Optional[Dict[str, float]] = None,
        regime: Optional[str] = None,
        horizon: int = 1,
    ) -> Dict:
        """
        Full layered forecast combining base XGBoost prediction with macro
        routing adjustments and upstream shock propagation.

        Parameters
        ----------
        prices : pd.DataFrame
            Raw price matrix. Sector filtering is applied internally.
        macro_state : dict, optional
            Current macro observations.  Accepts both canonical keys
            ('dxy_ret', 'vix_ret5d', 'tlt_ret', 'tlt_yield_proxy') and
            friendly aliases ('DXY_zscore', 'VIX_zscore', etc.).
            Example::

                {"DXY_zscore": 0.45, "VIX_zscore": 1.2, "tlt_ret": -0.003}

        upstream_shocks : dict, optional
            Forecast returns from causally upstream commodities/sectors.
            These are propagated via pairwise correlation before being added.
            Example::

                {"Natural Gas": 0.018, "Brent Crude Oil": 0.012}

            When dependency_graph.py is available, pass its output here directly.

        regime : str, optional
            Current market regime ('bull', 'bear', 'high_vol', 'neutral').
            If None, the macro routing uses regime='all' (regime-agnostic slopes).

        horizon : int
            Forecast horizon in days.

        Returns
        -------
        dict with keys:
            signal    : ModelSignal — the adjusted, ready-to-emit signal
            breakdown : dict with keys:
                base_forecast       (float) — raw XGBoost point prediction
                macro_adjustment    (float) — delta from macro routing
                upstream_adjustment (float) — delta from upstream shocks
                final_forecast      (float) — sum of all three components
                final_confidence    (float) — regime/validation-penalized IC
                regime              (str)   — regime used for routing
                macro_detail        (dict)  — {macro_var: contribution}
                upstream_detail     (dict)  — {commodity: contribution}
        """
        if self._xgb._model is None:
            raise RuntimeError("Call fit() first.")

        # ── 1. Base XGBoost forecast ───────────────────────────────────────────
        feat       = self._build_sector_features(prices)
        base_wf    = self._xgb.waterfall_data(feat, idx=-1)
        base_fc    = float(base_wf["forecast"])

        # ── 2. Macro routing adjustment ────────────────────────────────────────
        macro_adj, macro_detail = self._compute_macro_adjustment(
            macro_state=macro_state or {},
            regime=regime or "all",
        )

        # ── 3. Upstream shock propagation ─────────────────────────────────────
        upstream_adj, upstream_detail = self._compute_upstream_adjustment(
            upstream_shocks=upstream_shocks or {},
            prices=prices,
        )

        # ── 4. Final forecast ──────────────────────────────────────────────────
        final_fc = base_fc + macro_adj + upstream_adj

        # Clamp to a plausible intraday range (±15% log-return)
        final_fc = float(np.clip(final_fc, -0.15, 0.15))

        # ── 5. Confidence adjustment ───────────────────────────────────────────
        final_conf = self._compute_confidence(regime=regime or "all")

        # ── 6. Regime label for output ─────────────────────────────────────────
        regime_out = regime if regime and regime != "all" else self._infer_regime(macro_state)

        # ── 7. Assemble ModelSignal ────────────────────────────────────────────
        bullish = [(n, s) for n, s, _ in base_wf["top_positive"]]
        bearish = [(n, s) for n, s, _ in base_wf["top_negative"]]

        # Annotate signal drivers with macro/upstream context
        reasoning = [f"base_xgb_{base_fc:+.3%}"]
        if abs(macro_adj) > 1e-6:
            reasoning.append(f"macro_adj_{macro_adj:+.4f}")
        if abs(upstream_adj) > 1e-6:
            reasoning.append(f"upstream_adj_{upstream_adj:+.4f}")
        top_macro = max(macro_detail, key=lambda k: abs(macro_detail[k]), default=None)
        if top_macro:
            reasoning.append(f"top_macro_driver:{top_macro}")

        recent_preds = self._xgb.predict(feat).dropna().iloc[-60:]
        pred_std     = float(recent_preds.std()) if len(recent_preds) > 1 else 0.02
        ub = (final_fc - 2 * pred_std, final_fc + 2 * pred_std)

        self.set_reasoning(reasoning)
        self.set_drivers(bullish, bearish)
        self.set_regime(regime_out)
        self.set_dependencies(["price_history", "macro_routes", "correlation_snapshots"])
        self.set_metadata(
            sector=self.sector,
            base_forecast=round(base_fc, 6),
            macro_adjustment=round(macro_adj, 6),
            upstream_adjustment=round(upstream_adj, 6),
            final_confidence=round(final_conf, 4),
            macro_detail=macro_detail,
            upstream_detail=upstream_detail,
            wrapped_model="XGBoostForecaster",
        )

        signal = self._make_signal(
            forecast_return=final_fc,
            confidence=final_conf,
            horizon=horizon,
            uncertainty_band=ub,
            regime=regime_out,
        )

        breakdown = {
            "base_forecast":       round(base_fc, 6),
            "macro_adjustment":    round(macro_adj, 6),
            "upstream_adjustment": round(upstream_adj, 6),
            "final_forecast":      round(final_fc, 6),
            "final_confidence":    round(final_conf, 4),
            "regime":              regime_out,
            "macro_detail":        {k: round(v, 6) for k, v in macro_detail.items()},
            "upstream_detail":     {k: round(v, 6) for k, v in upstream_detail.items()},
        }
        return {"signal": signal, "breakdown": breakdown}

    # ── Component computations ─────────────────────────────────────────────────

    def _compute_macro_adjustment(
        self,
        macro_state: Dict[str, float],
        regime: str,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute the macro-routing adjustment for this sector.

        Returns
        -------
        (total_adjustment, {macro_var: contribution}) tuple.
        """
        if not macro_state:
            return 0.0, {}

        routes  = _get_routes()
        if not routes:
            return 0.0, {}

        coefficients = routes.get("coefficients", {})
        normalized   = _normalize_macro_state(macro_state)
        detail: Dict[str, float] = {}

        from models.macro_router import REGIME_ALL
        for macro_var, macro_val in normalized.items():
            if macro_var not in coefficients:
                continue
            regime_dict   = coefficients[macro_var]
            # Prefer requested regime; fall back to 'all'
            sector_slopes = regime_dict.get(regime, regime_dict.get(REGIME_ALL, {}))
            slope         = sector_slopes.get(self.sector, 0.0)
            if slope == 0.0:
                continue
            contribution = slope * macro_val
            detail[macro_var] = round(contribution, 6)

        total = sum(detail.values())
        return total, detail

    def _compute_upstream_adjustment(
        self,
        upstream_shocks: Dict[str, float],
        prices: pd.DataFrame,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Propagate upstream commodity forecasts via pairwise correlation.

        For each upstream commodity with a forecast, the contribution is:
            corr(this_commodity, upstream_commodity) × forecast × upstream_damping

        Uses the most recent correlation snapshot from the DB.  Falls back to
        default intra/cross-sector correlations when the snapshot is unavailable.

        NOTE: When models/dependency_graph.py is built, pass its sorted causal
        chain as upstream_shocks to get DAG-ordered propagation automatically.
        """
        if not upstream_shocks:
            return 0.0, {}

        corr_mat = _get_corr()
        detail:   Dict[str, float] = {}

        for upstream_commodity, upstream_forecast in upstream_shocks.items():
            if upstream_commodity == self.commodity:
                continue

            # Resolve pairwise correlation
            if (
                not corr_mat.empty
                and self.commodity in corr_mat.index
                and upstream_commodity in corr_mat.columns
            ):
                corr = float(corr_mat.loc[self.commodity, upstream_commodity])
            else:
                # Default: same-sector = 0.50, cross-sector = 0.10
                upstream_sector = COMMODITY_SECTORS.get(upstream_commodity, "")
                corr = (
                    _DEFAULT_INTRA_SECTOR_CORR
                    if upstream_sector == self.sector
                    else _DEFAULT_CROSS_SECTOR_CORR
                )
                log.debug(
                    "No correlation data for (%s, %s) — using default %.2f",
                    self.commodity, upstream_commodity, corr,
                )

            contribution = corr * float(upstream_forecast) * self.upstream_damping
            detail[upstream_commodity] = round(contribution, 6)

        total = sum(detail.values())
        return total, detail

    def _compute_confidence(self, regime: str) -> float:
        """
        Penalize raw IC when the macro evidence is weak.

        Penalties applied:
          • Regime has < 30 observations in the routes pkl → −10%
          • Any macro validation check failed for this sector → −10% per failure
          • No macro routes available → no adjustment
        """
        base = max(MIN_IC, self._ic_score)
        routes = _get_routes()
        if not routes:
            return float(np.clip(base, 0.0, 1.0))

        meta      = routes.get("_metadata", {})
        n_by_reg  = meta.get("n_obs_by_regime", {})
        checks    = routes.get("validation", {}).get("checks", [])

        # Penalty 1: regime has too few observations for reliable sector slopes
        n_regime = n_by_reg.get(regime, 999)
        if n_regime < 30:
            base *= 0.90

        # Penalty 2: any validation check failed for this sector
        sector_failures = sum(
            1 for c in checks
            if c.get("sector") == self.sector and not c.get("passed", True)
        )
        base *= (0.90 ** sector_failures)

        return float(np.clip(base, 0.0, 1.0))

    def _infer_regime(self, macro_state: Optional[Dict[str, float]]) -> str:
        """
        Best-effort regime label from available macro data.
        Uses VIX first, falls back to 'neutral'.
        """
        if not macro_state:
            return "neutral"
        normalized = _normalize_macro_state(macro_state)
        vix_val    = normalized.get("vix_ret5d", None)
        if vix_val is None:
            # Try absolute VIX level from caller
            vix_val = macro_state.get("vix", None)

        if vix_val is not None:
            try:
                vix = float(vix_val)
                if vix >= 20.0:
                    return "high_vol"
                if vix >= 0.15:   # 5-day VIX return of +15% ≈ spiking
                    return "high_vol"
            except (TypeError, ValueError):
                pass
        return "neutral"

    # ── Convenience ───────────────────────────────────────────────────────────

    def ic_score(self) -> float:
        """Return the cached Spearman IC from the last fit()."""
        return self._ic_score

    def sector_feature_names(self) -> List[str]:
        """Return the feature columns actually used for training."""
        return list(self._xgb._feature_names or [])

    def macro_routing_summary(self, regime: str = "all") -> Dict[str, float]:
        """
        Return the macro slope coefficients for this sector in a given regime.
        Useful for inspection without calling predict_with_context.
        """
        routes = _get_routes()
        if not routes:
            return {}
        coefficients = routes.get("coefficients", {})
        from models.macro_router import REGIME_ALL
        reg = regime if regime != "all" else REGIME_ALL
        out = {}
        for macro_var, regimes in coefficients.items():
            sector_slopes = regimes.get(reg, regimes.get(REGIME_ALL, {}))
            slope = sector_slopes.get(self.sector)
            if slope is not None:
                out[macro_var] = round(slope, 6)
        return out

    def invalidate_route_cache(self) -> None:
        """Force reload of macro routes and correlation matrix on next call."""
        global _routes_cache, _corr_cache
        _routes_cache = None
        _corr_cache   = None
