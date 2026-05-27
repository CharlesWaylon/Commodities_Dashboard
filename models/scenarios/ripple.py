"""
CausalRipple — propagate a ScenarioBand from one commodity to its
correlated neighbours via Vector Autoregression impulse responses.

When a bear scenario fires for, say, WTI, we don't believe Brent will be
unaffected. The VAR(p) fitted on a commodity complex emits orthogonalised
Impulse Response Functions (IRFs): IRF[h, response, shock] gives the
response of one series at horizon h to a one-standard-deviation shock to
another. We treat IRF[h, target, source] as a *coupling coefficient* and
scale the source band's per-step log returns by it:

    band_target[h] = IRF[h, target, source] · band_source[h]      (all of bear/mean/bull)

This is a linear, first-order approximation — fine for visualising how
much of a source scenario "leaks" into related commodities, not a joint
distribution. The IRF is computed once at fit time, so propagation across
many targets is essentially free.

HistoricalTriggerReplay — Step 7 of the macro_features spec.
Builds a ScenarioBand by REPLAYING historical days where a chosen trigger
family fired above a strength threshold, rather than synthesising shocks
from a model. The resulting envelope is the empirical bear/mean/bull range
of what actually happened after similar shocks in the past.
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from models.scenarios.band import ScenarioBand

log = logging.getLogger(__name__)


class CausalRipple:
    """
    Propagate scenarios across commodities in the same VAR-fitted complex.

    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix containing both source and target columns.
    group : str
        COMMODITY_GROUPS key (e.g. "energy", "grains"). Both the source
        and any target commodity must live in this group.
    """

    def __init__(self, prices: pd.DataFrame, group: str = "energy"):
        from models.statistical.var_vecm import CommodityVAR, COMMODITY_GROUPS

        self.group = group
        self._members = list(COMMODITY_GROUPS.get(group, []))
        self._members_in_prices = [c for c in self._members if c in prices.columns]

        self._var = CommodityVAR(group=group, use_returns=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._var.fit(prices)

        # Precompute IRF up to a long horizon so we don't refit per-call
        self._irf = self._var.impulse_response(steps=30)
        # MultiIndex columns: (shock, response). Reshape to a 3-D array
        # [step, response_idx, shock_idx] for fast lookup.
        cols = self._var._cols
        idx = {c: i for i, c in enumerate(cols)}
        n = len(cols)
        steps = len(self._irf)
        cube = np.zeros((steps, n, n))
        for (shock, response), series in self._irf.items():
            cube[:, idx[response], idx[shock]] = series.values
        self._cube = cube
        self._idx = idx
        self._cols = cols

    # ── public API ────────────────────────────────────────────────────────────

    def members(self) -> list:
        return list(self._cols)

    def coupling(self, source: str, target: str, horizon: int) -> np.ndarray:
        """
        Per-step IRF coupling from ``source`` to ``target``, normalised so the
        source's own contemporaneous response is 1.0.

        We use the orthogonalised IRF and divide by ``IRF[0, source, source]``
        (= σ of source's Cholesky innovation). This makes the coefficient a
        dimensionless "fraction of source move that lands in target," which
        is the form needed to scale a log-return path directly.

        Step h (1-indexed) uses IRF lag (h-1) — treating each forecast step as
        a fresh shock propagated through the system.
        """
        if source not in self._idx or target not in self._idx:
            raise KeyError(f"Both source ({source}) and target ({target}) must be in {self.group}.")
        i_target = self._idx[target]
        i_source = self._idx[source]
        denom = self._cube[0, i_source, i_source]
        if abs(denom) < 1e-12:
            denom = 1.0
        path = self._cube[:horizon, i_target, i_source] / denom
        if len(path) < horizon:
            path = np.concatenate([path, np.repeat(path[-1], horizon - len(path))])
        return path

    def propagate(self, source_band: ScenarioBand, target: str) -> Optional[ScenarioBand]:
        """
        Propagate a source ScenarioBand into a derived band on ``target``.

        Each step of the source path is treated as a fresh innovation that
        propagates forward through the IRF — so the target's response at
        step h is the **convolution**
            target[h] = Σ_{j ≤ h}  coupling[h − j] · source[j]
        not the element-wise product. With a near-zero-lag IRF (most of the
        response is contemporaneous) the convolution and element-wise form
        differ dramatically for sustained scenarios.

        Returns None if the target is not in this VAR group.
        """
        source = source_band.commodity
        if source not in self._idx or target not in self._idx:
            return None
        H = source_band.horizon
        coup = self.coupling(source, target, H)

        def _conv(x: np.ndarray) -> np.ndarray:
            return np.convolve(x, coup, mode="full")[:H]

        return ScenarioBand(
            commodity=target,
            source_model=f"Ripple[{source} → {target}]",
            asof=source_band.asof,
            horizon=H,
            bear_q=source_band.bear_q,
            bull_q=source_band.bull_q,
            mean=_conv(source_band.mean),
            bear=_conv(source_band.bear),
            bull=_conv(source_band.bull),
            diagnostics={
                "source": source,
                "group": self.group,
                "coupling": coup.round(4).tolist(),
                "var_lag_order": self._var._lag_order,
                "n_series_in_group": len(self._cols),
            },
        )

    def coupling_matrix(self, horizon: int) -> pd.DataFrame:
        """
        Normalised coupling matrix (responses × shocks), averaged over the
        forecast horizon. Each column is divided by the source's own h=0 IRF
        (= σ of its Cholesky innovation) so values are comparable across
        commodities.
        """
        n = len(self._cols)
        out = np.zeros((n, n))
        diag0 = np.diag(self._cube[0])
        for j in range(n):
            denom = diag0[j] if abs(diag0[j]) > 1e-12 else 1.0
            out[:, j] = self._cube[:horizon, :, j].mean(axis=0) / denom
        return pd.DataFrame(out, index=self._cols, columns=self._cols).round(3)


# ── Step 7: Historical Trigger Replay ─────────────────────────────────────────

class HistoricalTriggerReplay:
    """
    Build a ScenarioBand from real history.

    Instead of synthesising shocks via a model (ARIMA / VAR / XGBoost), this
    class queries ``trigger_events`` for historical days where a chosen family
    fired at or above ``min_strength``, then aggregates the realised next-H
    daily log returns of the target commodity into a bear/mean/bull envelope.

    The result is the empirical shock-and-decay profile from past episodes —
    no model assumptions, just "here's what actually happened last time."

    Example
    -------
    >>> replay = HistoricalTriggerReplay(prices)
    >>> band = replay.build_band(
    ...     commodity="WTI Crude Oil",
    ...     family="opec_action",
    ...     min_strength=0.8,
    ...     horizon=10,
    ... )
    >>> if band is not None:
    ...     print(band.diagnostics["n_events"], "historical events used")
    """

    def __init__(self, prices: pd.DataFrame):
        if prices is None or prices.empty:
            raise ValueError("HistoricalTriggerReplay: prices DataFrame is empty.")
        self._prices = prices.sort_index()
        # Daily log returns (vectorised across all columns).
        self._logret = np.log(self._prices / self._prices.shift(1))

    # ── trigger lookup ────────────────────────────────────────────────────────

    def _fetch_event_dates(self, family: str, min_strength: float) -> List[pd.Timestamp]:
        """Pull trigger_events rows matching (family, ≥ strength). [] on DB error."""
        try:
            from database.db import get_db
            from database.models import TriggerEvent
            with get_db() as db:
                rows = (
                    db.query(TriggerEvent.trigger_date)
                    .filter(TriggerEvent.family == family)
                    .filter(TriggerEvent.strength >= min_strength)
                    .all()
                )
            return [pd.Timestamp(r[0]) for r in rows]
        except Exception as exc:
            log.warning("HistoricalTriggerReplay: trigger lookup failed (%s)", exc)
            return []

    # ── public API ────────────────────────────────────────────────────────────

    def build_band(
        self,
        commodity:    str,
        family:       str,
        min_strength: float = 0.8,
        horizon:      int   = 10,
        bear_q:       float = 0.10,
        bull_q:       float = 0.90,
    ) -> Optional[ScenarioBand]:
        """
        Build a ScenarioBand from historical trigger occurrences.

        Returns
        -------
        ScenarioBand or None when no events match (so the caller can fall back
        to a synthetic model gracefully).
        """
        if commodity not in self._logret.columns:
            log.info("HistoricalTriggerReplay: %r not in prices", commodity)
            return None

        event_dates = self._fetch_event_dates(family, min_strength)
        if not event_dates:
            return None

        series = self._logret[commodity].dropna()
        if series.empty:
            return None

        # For each event date, take the H business days that follow.
        paths: List[np.ndarray] = []
        for ev in event_dates:
            after = series[series.index > ev].iloc[:horizon]
            if len(after) == horizon:
                paths.append(after.values)

        if not paths:
            return None

        path_mat = np.vstack(paths)            # shape (n_events, horizon)
        mean_path = np.median(path_mat, axis=0)
        bear_path = np.quantile(path_mat, bear_q, axis=0)
        bull_path = np.quantile(path_mat, bull_q, axis=0)

        return ScenarioBand(
            commodity=commodity,
            source_model=f"HistoricalTriggerReplay[{family}≥{min_strength:.2f}]",
            asof=self._prices.index[-1] if len(self._prices.index) else pd.Timestamp.utcnow(),
            horizon=horizon,
            bear_q=bear_q,
            bull_q=bull_q,
            mean=mean_path,
            bear=bear_path,
            bull=bull_path,
            diagnostics={
                "family":       family,
                "min_strength": min_strength,
                "n_events":     len(paths),
                "n_events_db":  len(event_dates),
                "horizon":      horizon,
            },
        )
