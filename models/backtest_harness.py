"""
Backtest harness — Phase 3 Part 2 of the Model Integration Roadmap.

Purpose
───────
Generate the training data that MetaPredictor.fit() needs.

For every date in the test window the harness asks:
  "Given the macro environment on this date, which model tier made the most
  accurate one-day-ahead forecast for this commodity?"

That (market_state → winning_tier) pairing is one training record.  After
running across enough commodities and dates, MetaPredictor learns a
market-regime → tier-trust mapping that goes beyond equal weights.

Architecture
────────────

    prices (full history)
    macro_df (full history)
    commodity list
          │
          ▼
    BacktestHarness.run()
          │
    ┌─────┴──────────────────────────────────────┐
    │ For each commodity:                        │
    │   _split_indices(n) → fold list            │
    │   For each fold (expanding train window):  │
    │   1. Fit each TierAdapter on train         │
    │   2. Generate out-of-sample forecasts      │
    │   3. Compare each tier to actual_return    │
    │   4. Tag date with winning_tier            │
    │   5. Look up macro state → MetaFeatures    │
    └─────┬──────────────────────────────────────┘
          │
          ▼
    List[BacktestRecord]  →  meta.fit()  →  meta.save()

Walk-forward cross-validation (Phase 5 upgrade)
────────────────────────────────────────────────
`_split_indices(n)` now returns a list of (test_start, test_end) pairs that
implement a standard **expanding-window TimeSeriesSplit**.  Each fold's
training window grows to include all history before the test window, so
adapters see progressively more data as time advances — exactly how they
would be used in production.

With the default n_splits=5 on ~500 trading days of history this produces
~5× more labelled records than the old single 80/20 split, making
MetaPredictor weights substantially more robust.

Fold structure (n=500, min_train_rows=120, n_splits=5):
  fold 0: train [0:196],   test [196:272]
  fold 1: train [0:272],   test [272:348]
  fold 2: train [0:348],   test [348:424]
  fold 3: train [0:424],   test [424:499]   ← last row reserved for returns

Setting n_splits=1 degrades gracefully to the original test_fraction
single-split behaviour, so any caller that pins n_splits=1 is unaffected.

TierAdapter plug-in
───────────────────
Any new model tier is a three-method class:

    class MyTierAdapter(TierAdapter):
        @property
        def tier(self) -> str:
            return "my_new_tier"  # must be in KNOWN_TIERS

        def fit(self, prices_train: pd.DataFrame, commodity: str) -> None:
            ...

        def predict_series(
            self, prices_full: pd.DataFrame, commodity: str, test_idx: pd.DatetimeIndex
        ) -> pd.Series:
            ...  # return Series indexed by test_idx, values = forecast returns

Deep / quantum adapters (LSTM, Prophet, Quantum kernel) are not included by
default because they are slow.  Pass them explicitly via
`BacktestHarness(extra_adapters=[...])`.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from models.config import TEST_FRACTION
from models.meta_predictor import (
    MetaFeatures,
    MetaPredictor,
    collect_meta_features,
    KNOWN_TIERS,
)

log = logging.getLogger(__name__)

# ── BacktestRecord ─────────────────────────────────────────────────────────────

@dataclass
class BacktestRecord:
    """
    One row of backtest output — a single (date, commodity) observation.

    Fields
    ──────
    date            : pd.Timestamp — forecast date (when prediction was made)
    commodity       : str          — commodity display name
    meta_features   : MetaFeatures — market state snapshot for this date
    actual_return   : float        — realised next-day log-return
    tier_forecasts  : dict         — {tier: forecast_return}
    tier_errors     : dict         — {tier: abs(forecast - actual)}
    winning_tier    : str          — tier with min abs error
    """
    date: pd.Timestamp
    commodity: str
    meta_features: MetaFeatures
    actual_return: float
    tier_forecasts: Dict[str, float] = field(default_factory=dict)
    tier_errors: Dict[str, float] = field(default_factory=dict)
    winning_tier: str = ""

    def to_training_pair(self) -> Tuple[MetaFeatures, str]:
        """Return the (MetaFeatures, winning_tier) pair MetaPredictor.fit() needs."""
        return (self.meta_features, self.winning_tier)


# ── TierAdapter ABC ────────────────────────────────────────────────────────────

class TierAdapter(ABC):
    """
    Wraps a model-tier class into the uniform backtest interface.

    Subclasses must implement:
      - `tier`              : str property — one of KNOWN_TIERS
      - `fit()`             : train on the training slice
      - `predict_series()`  : generate forecasts on the test slice
    """

    @property
    @abstractmethod
    def tier(self) -> str:
        """One of KNOWN_TIERS: 'statistical', 'ml', 'deep', 'quantum'."""
        ...

    @abstractmethod
    def fit(self, prices_train: pd.DataFrame, commodity: str) -> None:
        """
        Train on the train-split price history.

        Parameters
        ----------
        prices_train : pd.DataFrame
            Full price matrix up to the split date (n_train × n_commodities).
        commodity : str
            Target commodity display name (column in prices_train).
        """
        ...

    @abstractmethod
    def predict_series(
        self,
        prices_full: pd.DataFrame,
        commodity: str,
        test_idx: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Generate one-day-ahead return forecasts for every date in test_idx.

        Parameters
        ----------
        prices_full : pd.DataFrame
            Full price matrix (train + test, needed for feature context).
        commodity : str
            Target commodity.
        test_idx : pd.DatetimeIndex
            Dates to forecast (the test split dates).

        Returns
        -------
        pd.Series
            Index = test_idx.  Values = predicted next-day log-return.
            Dates with no valid forecast should be NaN.
        """
        ...


# ── Statistical tier adapter: ARIMA ───────────────────────────────────────────

class ARIMAAdapter(TierAdapter):
    """
    Wraps `models.statistical.arima.ARIMAForecaster` for backtesting.

    Uses a single model fitted on the train series, then calls
    `model_fit.forecast(steps=n)` for the full test horizon.  This is an
    optimistic estimate (no re-fitting per step), but is fast and gives the
    harness a valid statistical baseline.

    Parameters
    ----------
    fixed_order : tuple (p, d, q), optional
        Bypass AIC grid search with a fixed order.  Useful in tests and when
        speed matters more than perfect order selection.
    """

    def __init__(self, fixed_order: Optional[Tuple[int, int, int]] = None):
        self._model_fit = None
        self._fixed_order = fixed_order

    @property
    def tier(self) -> str:
        return "statistical"

    def fit(self, prices_train: pd.DataFrame, commodity: str) -> None:
        from models.statistical.arima import ARIMAForecaster
        returns_train = np.log(prices_train[commodity] / prices_train[commodity].shift(1)).dropna()
        fc = ARIMAForecaster(commodity=commodity)
        if self._fixed_order is not None:
            fc._order = self._fixed_order  # skip grid search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc.fit(returns_train)
        self._model_fit = fc._model_fit

    def predict_series(
        self,
        prices_full: pd.DataFrame,
        commodity: str,
        test_idx: pd.DatetimeIndex,
    ) -> pd.Series:
        if self._model_fit is None:
            return pd.Series(np.nan, index=test_idx)
        n = len(test_idx)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fcst = self._model_fit.forecast(steps=n)
            return pd.Series(fcst.values, index=test_idx)
        except Exception as exc:
            log.warning("ARIMAAdapter.predict_series failed: %s", exc)
            return pd.Series(np.nan, index=test_idx)


# ── ML tier adapter: XGBoost ──────────────────────────────────────────────────

class XGBoostAdapter(TierAdapter):
    """
    Wraps `models.ml.xgboost_shap.XGBoostForecaster` for backtesting.

    Builds the feature matrix from the full price history (train + test
    combined) so rolling window features at the test boundary are correct,
    then trains on the train-window slice and predicts the test slice.

    If `_climate_df` is set by the harness before fit(), PDSI and ENSO
    features are appended for climate-sensitive commodities.
    """

    def __init__(self):
        self._model = None
        self._commodity = None
        self._climate_df: Optional[pd.DataFrame] = None

    @property
    def tier(self) -> str:
        return "ml"

    def fit(self, prices_train: pd.DataFrame, commodity: str) -> None:
        from models.ml.xgboost_shap import XGBoostForecaster
        from models.features import build_feature_matrix, build_target, augment_with_climate
        feat_train = build_feature_matrix(prices_train)
        feat_train = augment_with_climate(feat_train, self._climate_df, commodity)
        target_train = build_target(prices_train, commodity)
        aligned = feat_train.join(target_train).dropna()
        X_tr = aligned.drop(columns=["target"])
        y_tr = aligned["target"]
        mdl = XGBoostForecaster(commodity=commodity)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl.fit(X_tr, y_tr)
        self._model = mdl
        self._commodity = commodity

    def predict_series(
        self,
        prices_full: pd.DataFrame,
        commodity: str,
        test_idx: pd.DatetimeIndex,
    ) -> pd.Series:
        if self._model is None:
            return pd.Series(np.nan, index=test_idx)
        from models.features import build_feature_matrix, augment_with_climate
        feat_full = build_feature_matrix(prices_full)
        feat_full = augment_with_climate(feat_full, self._climate_df, commodity)
        test_feats = feat_full.loc[feat_full.index.isin(test_idx)].dropna()
        if test_feats.empty:
            return pd.Series(np.nan, index=test_idx)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = self._model.predict(test_feats)
            return preds.reindex(test_idx)
        except Exception as exc:
            log.warning("XGBoostAdapter.predict_series failed: %s", exc)
            return pd.Series(np.nan, index=test_idx)


# ── ML tier adapter: ElasticNet ───────────────────────────────────────────────

class ElasticNetAdapter(TierAdapter):
    """
    Wraps `models.ml.elastic_net.ElasticNetFactorModel` for backtesting.
    Same feature-matrix approach as XGBoostAdapter, including climate augmentation.
    """

    def __init__(self):
        self._model = None
        self._climate_df: Optional[pd.DataFrame] = None

    @property
    def tier(self) -> str:
        return "ml"

    def fit(self, prices_train: pd.DataFrame, commodity: str) -> None:
        from models.ml.elastic_net import ElasticNetFactorModel
        from models.features import build_feature_matrix, build_target, augment_with_climate
        feat = build_feature_matrix(prices_train)
        feat = augment_with_climate(feat, self._climate_df, commodity)
        target = build_target(prices_train, commodity)
        aligned = feat.join(target).dropna()
        X_tr = aligned.drop(columns=["target"])
        y_tr = aligned["target"]
        mdl = ElasticNetFactorModel(commodity=commodity)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl.fit(X_tr, y_tr)
        self._model = mdl

    def predict_series(
        self,
        prices_full: pd.DataFrame,
        commodity: str,
        test_idx: pd.DatetimeIndex,
    ) -> pd.Series:
        if self._model is None:
            return pd.Series(np.nan, index=test_idx)
        from models.features import build_feature_matrix, augment_with_climate
        feat_full = build_feature_matrix(prices_full)
        feat_full = augment_with_climate(feat_full, self._climate_df, commodity)
        test_feats = feat_full.loc[feat_full.index.isin(test_idx)].dropna()
        if test_feats.empty:
            return pd.Series(np.nan, index=test_idx)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = self._model.predict(test_feats)
            return preds.reindex(test_idx)
        except Exception as exc:
            log.warning("ElasticNetAdapter.predict_series failed: %s", exc)
            return pd.Series(np.nan, index=test_idx)


# ── BacktestHarness ───────────────────────────────────────────────────────────

# Default adapters — fast tiers only.  Pass extra_adapters to include deep/quantum.
# ElasticNet is included alongside ARIMA and XGBoost: it trains in milliseconds,
# often outperforms XGBoost on short histories, and gives the MetaPredictor a
# third regression-based vote to balance against the tree-based ml tier.
DEFAULT_ADAPTERS: List[TierAdapter] = [
    ARIMAAdapter(fixed_order=(1, 0, 1)),  # fast fixed order for default runs
    XGBoostAdapter(),
    ElasticNetAdapter(),
]


class BacktestHarness:
    """
    Runs a walk-forward backtest across one or more commodities and collects
    (MetaFeatures, winning_tier) records for MetaPredictor.fit().

    Parameters
    ----------
    adapters : list[TierAdapter], optional
        Model-tier adapters to include.  Defaults to [ARIMA, XGBoost].
    test_fraction : float
        Fraction of history used as the test window when n_splits=1
        (single-split legacy mode).  Ignored when n_splits > 1.  Default 0.20.
    min_train_rows : int
        Minimum training rows required before the first fold starts.
        Prevents errors on very short price histories.  Default 120.
    n_splits : int
        Number of walk-forward CV folds.  Default 5 (recommended).
        Set to 1 to revert to the original single test_fraction split.
    """

    def __init__(
        self,
        adapters: Optional[List[TierAdapter]] = None,
        test_fraction: float = TEST_FRACTION,
        min_train_rows: int = 120,
        n_splits: int = 5,
        climate_df: Optional[pd.DataFrame] = None,
    ):
        self.adapters = adapters if adapters is not None else list(DEFAULT_ADAPTERS)
        self.test_fraction = test_fraction
        self.min_train_rows = min_train_rows
        self.n_splits = max(1, int(n_splits))
        self.climate_df = climate_df

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        prices: pd.DataFrame,
        macro_df: pd.DataFrame,
        commodities: Optional[Sequence[str]] = None,
    ) -> List[BacktestRecord]:
        """
        Run the full backtest and return labelled records.

        Parameters
        ----------
        prices : pd.DataFrame
            Full price matrix (n_days × n_commodities).
        macro_df : pd.DataFrame
            Macro overlay DataFrame from load_macro().  Must share the same
            DatetimeIndex as prices (or at least overlap).
        commodities : list of str, optional
            Which commodities to run.  Defaults to all columns in prices that
            are also present in the adapters' expected universe.

        Returns
        -------
        list[BacktestRecord]
        """
        targets = list(commodities) if commodities else list(prices.columns)
        all_records: List[BacktestRecord] = []

        for commodity in targets:
            if commodity not in prices.columns:
                log.warning("Skipping %r — not in price matrix.", commodity)
                continue
            try:
                records = self._run_commodity(prices, macro_df, commodity, self.climate_df)
                all_records.extend(records)
                log.info(
                    "%s: %d backtest records (winning tiers: %s)",
                    commodity,
                    len(records),
                    _tier_counts(records),
                )
            except Exception as exc:
                log.warning("Backtest failed for %r: %s", commodity, exc)

        return all_records

    def collect_training_pairs(
        self,
        prices: pd.DataFrame,
        macro_df: pd.DataFrame,
        commodities: Optional[Sequence[str]] = None,
    ) -> List[Tuple[MetaFeatures, str]]:
        """
        Convenience wrapper: run() → strip to training pairs for MetaPredictor.

        Returns
        -------
        list of (MetaFeatures, winning_tier)
        """
        records = self.run(prices, macro_df, commodities)
        return [r.to_training_pair() for r in records]

    def train_meta_predictor(
        self,
        prices: pd.DataFrame,
        macro_df: pd.DataFrame,
        commodities: Optional[Sequence[str]] = None,
        max_depth: int = 5,
        min_samples_leaf: int = 10,
        save_path=None,
    ) -> MetaPredictor:
        """
        End-to-end: backtest → fit MetaPredictor → optionally save.

        Parameters
        ----------
        save_path : Path-like, optional
            If provided, persist the trained predictor to this path.
            Pass True to use the default data/meta_predictor.pkl path.

        Returns
        -------
        MetaPredictor  (trained)
        """
        pairs = self.collect_training_pairs(prices, macro_df, commodities)
        if not pairs:
            raise RuntimeError(
                "Backtest produced zero training records. "
                "Check that prices and macro_df have sufficient overlap."
            )

        meta = MetaPredictor()
        meta.fit(pairs, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        log.info(
            "MetaPredictor trained on %d records. Top feature: %s",
            len(pairs),
            meta._top_feature(),
        )

        if save_path is not None:
            saved = meta.save(None if save_path is True else save_path)
            log.info("MetaPredictor saved to %s", saved)

        return meta

    def validate_causal_chains(
        self,
        prices: pd.DataFrame,
        macro_df: pd.DataFrame,
        shock_window_days: int = 365,
        forward_days_primary: int = 5,
        forward_days_secondary: int = 10,
        save: bool = True,
        db_path=None,
    ) -> "ValidationReport":
        """
        Validate the causal-chain cascade against historical macro shock events.

        See module-level ``validate_causal_chains()`` for full parameter docs.
        This method is a thin instance-bound wrapper so callers can write::

            harness = BacktestHarness()
            report = harness.validate_causal_chains(prices, macro_df)
            print(report.summary_text())

        The harness instance contributes ``min_train_rows`` as the minimum
        price history required before and after each shock event.
        """
        return validate_causal_chains(
            harness               = self,
            prices                = prices,
            macro_df              = macro_df,
            shock_window_days     = shock_window_days,
            forward_days_primary  = forward_days_primary,
            forward_days_secondary= forward_days_secondary,
            save                  = save,
            db_path               = db_path,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _split_indices(self, n: int) -> List[Tuple[int, int]]:
        """
        Return a list of (test_start, test_end) integer position pairs.

        Training data for fold i is always ``prices.iloc[:test_start]``
        (expanding window — all history before the test window).  Test windows
        are non-overlapping and strictly forward in time, so there is zero
        data leakage.

        Walk-forward example (n=500, min_train_rows=120, n_splits=5):
            fold 0: train [0:196],  test [196:272]
            fold 1: train [0:272],  test [272:348]
            fold 2: train [0:348],  test [348:424]
            fold 3: train [0:424],  test [424:499]

        Single-split fallback (n_splits=1):
            fold 0: train [0:split], test [split:n-1]
            where split = int(n * (1 - test_fraction))

        The last row of prices is always excluded from test windows because
        ``actual_returns.shift(-1)`` is NaN there (no future price exists).

        Parameters
        ----------
        n : int
            Total number of rows in the price series.

        Returns
        -------
        list of (test_start, test_end) tuples, or [] if data is insufficient.
        """
        usable = n - 1  # last row has no actual return

        if self.n_splits == 1:
            # ── Legacy single-split behaviour ─────────────────────────────────
            split = int(n * (1 - self.test_fraction))
            if split < self.min_train_rows or split >= usable:
                return []
            return [(split, usable)]

        # ── Walk-forward: n_splits non-overlapping folds ──────────────────────
        if usable <= self.min_train_rows:
            return []

        available = usable - self.min_train_rows
        test_size = max(1, available // self.n_splits)

        splits: List[Tuple[int, int]] = []
        for i in range(self.n_splits):
            test_start = self.min_train_rows + i * test_size
            test_end   = min(test_start + test_size, usable)
            if test_start >= usable:
                break
            splits.append((test_start, test_end))

        return splits

    def _run_commodity(
        self,
        prices: pd.DataFrame,
        macro_df: pd.DataFrame,
        commodity: str,
        climate_df: Optional[pd.DataFrame] = None,
    ) -> List[BacktestRecord]:
        """
        Run the walk-forward backtest for a single commodity.

        For each fold produced by _split_indices():
          1. Fit every adapter on prices.iloc[:test_start]
          2. Generate out-of-sample forecasts on the test window
          3. Compare each tier's forecast to the realised return
          4. Tag the date with the winning tier and its MetaFeatures
        """
        # ── 1. Pre-compute actual next-day returns for the full window ─────────
        actual_returns = np.log(
            prices[commodity] / prices[commodity].shift(1)
        ).shift(-1)

        # ── 2. Determine fold boundaries ──────────────────────────────────────
        splits = self._split_indices(len(prices))
        if not splits:
            log.warning(
                "Insufficient data for %r (%d rows, need ≥ %d). Skipping.",
                commodity, len(prices), self.min_train_rows,
            )
            return []

        # ── 3. Align macro once for the full price index ───────────────────────
        macro_aligned = _align_macro(macro_df, prices.index)

        # Merge enso_phase into macro_aligned so collect_meta_features() picks it up
        if climate_df is not None and not climate_df.empty and "enso_phase" in climate_df.columns:
            climate_phase = climate_df[["enso_phase"]].reindex(prices.index).ffill(limit=5)
            macro_aligned = macro_aligned.join(climate_phase, how="left")

        all_records: List[BacktestRecord] = []

        # ── 4. Loop over walk-forward folds ────────────────────────────────────
        for fold, (test_start, test_end) in enumerate(splits):
            prices_train = prices.iloc[:test_start]
            test_idx     = prices.index[test_start:test_end]

            if len(test_idx) == 0:
                continue

            # ── 4a. Fit each adapter on this fold's expanding train window ─────
            # Propagate climate data to adapters that support it (XGBoost, ElasticNet)
            for adapter in self.adapters:
                if hasattr(adapter, "_climate_df"):
                    adapter._climate_df = climate_df

            tier_forecasts: Dict[str, pd.Series] = {}
            for adapter in self.adapters:
                try:
                    adapter.fit(prices_train, commodity)
                    preds = adapter.predict_series(prices, commodity, test_idx)
                    tier_forecasts[adapter.tier] = preds
                except Exception as exc:
                    log.warning(
                        "Adapter %s failed (fold %d, %r): %s",
                        adapter.__class__.__name__, fold, commodity, exc,
                    )

            if not tier_forecasts:
                continue

            # ── 4b. Build one BacktestRecord per test date in this fold ────────
            for date in test_idx:
                actual = actual_returns.get(date, float("nan"))
                if np.isnan(actual):
                    continue

                t_fcst: Dict[str, float] = {}
                t_err:  Dict[str, float] = {}
                for tier, series in tier_forecasts.items():
                    fc = float(series.get(date, float("nan")))
                    if np.isnan(fc):
                        continue
                    t_fcst[tier] = fc
                    t_err[tier]  = abs(fc - actual)

                if not t_err:
                    continue

                winning_tier = min(t_err, key=lambda k: t_err[k])
                mf = _meta_features_for_date(macro_aligned, date)

                all_records.append(BacktestRecord(
                    date=date,
                    commodity=commodity,
                    meta_features=mf,
                    actual_return=actual,
                    tier_forecasts=t_fcst,
                    tier_errors=t_err,
                    winning_tier=winning_tier,
                ))

        return all_records


# ── Standalone entry point ────────────────────────────────────────────────────

def run_and_train(
    commodities: Optional[Sequence[str]] = None,
    adapters: Optional[List[TierAdapter]] = None,
    save: bool = True,
    test_fraction: float = TEST_FRACTION,
) -> MetaPredictor:
    """
    One-call convenience function: load data, run backtest, train, save.

    Intended for use in a notebook or CLI:

        from models.backtest_harness import run_and_train
        meta = run_and_train(commodities=["WTI Crude Oil", "Gold (COMEX)"])

    Parameters
    ----------
    commodities : list of str, optional
        Subset of MODELING_COMMODITIES to run.  Defaults to a small fast set.
    save : bool
        Whether to persist the trained predictor to data/meta_predictor.pkl.

    Returns
    -------
    MetaPredictor  (trained)
    """
    from models.data_loader import load_price_matrix
    from features.macro_overlays import load_macro

    if commodities is None:
        # Fast default: 3 liquid commodities from different sectors
        commodities = ["WTI Crude Oil", "Gold (COMEX)", "Corn (CBOT)"]

    log.info("Loading prices for: %s", commodities)
    from models.config import MODELING_COMMODITIES
    ticker_subset = {c: MODELING_COMMODITIES[c] for c in commodities if c in MODELING_COMMODITIES}
    prices = load_price_matrix(commodities=ticker_subset, period="2y")

    log.info("Loading macro overlays …")
    macro_df = load_macro(period="2y", _price_index=prices.index)

    harness = BacktestHarness(
        adapters=adapters,
        test_fraction=test_fraction,
    )
    return harness.train_meta_predictor(
        prices=prices,
        macro_df=macro_df,
        commodities=list(commodities),
        save_path=True if save else None,
    )


# ── Utility functions ─────────────────────────────────────────────────────────

def _align_macro(macro_df: pd.DataFrame, price_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex macro_df to price index, forward-fill up to 5 days."""
    if macro_df is None or macro_df.empty:
        return pd.DataFrame(index=price_idx)
    try:
        return macro_df.reindex(price_idx).ffill(limit=5)
    except Exception:
        return pd.DataFrame(index=price_idx)


def _meta_features_for_date(
    macro_aligned: pd.DataFrame, date: pd.Timestamp
) -> MetaFeatures:
    """
    Extract a MetaFeatures snapshot for a single date from an aligned
    macro DataFrame.  Returns default MetaFeatures if the date is missing.
    """
    if macro_aligned.empty or date not in macro_aligned.index:
        return MetaFeatures()
    row_df = macro_aligned.loc[[date]]
    return collect_meta_features(row_df)


def _tier_counts(records: List[BacktestRecord]) -> Dict[str, int]:
    """Count how many times each tier won across a list of records."""
    counts: Dict[str, int] = {}
    for r in records:
        counts[r.winning_tier] = counts.get(r.winning_tier, 0) + 1
    return counts


# ══════════════════════════════════════════════════════════════════════════════
#  Causal-Chain Validation — Phase 5 extension
#
#  BacktestHarness.validate_causal_chains() tests whether the learned macro
#  dependency structure actually improves forecasts relative to a rolling-
#  momentum baseline, and whether the Energy → Metals 1-2 day lag manifests
#  after historical macro shocks.
#
#  Metrics produced for each shock event:
#    (1) Directional accuracy per sector: did the cascade get the sign right?
#    (2) MAE lift vs baseline: did the cascade reduce forecast error?
#    (3) Lag structure test: did Energy returns lead Metals by 1-2 days?
#
#  Health thresholds (from cascade_validator constants):
#    • Cascade directional accuracy must beat 50% (coin-flip) per sector
#    • Cascade MAE must be ≤ baseline MAE (lift ≥ 0)
#    • Lag confirmed in ≥ 50% of shock events (otherwise structure may have shifted)
# ══════════════════════════════════════════════════════════════════════════════


def validate_causal_chains(
    harness: "BacktestHarness",
    prices: pd.DataFrame,
    macro_df: pd.DataFrame,
    shock_window_days: int = 365,
    forward_days_primary: int = 5,
    forward_days_secondary: int = 10,
    save: bool = True,
    db_path=None,
) -> "ValidationReport":
    """
    Validate the causal-chain cascade against historical macro shock events.

    This is a module-level function that accepts a BacktestHarness instance
    (for API consistency) but does not use any adapter state — it works
    directly from price and macro data.

    Parameters
    ----------
    harness : BacktestHarness
        The harness instance (used for min_train_rows threshold only).
    prices : pd.DataFrame
        Full price matrix — all sector commodities must be columns.
    macro_df : pd.DataFrame
        Macro overlay from build_macro_overlay_features().
    shock_window_days : int
        How far back to search for shock events. Default 365 (1 year).
    forward_days_primary : int
        Primary forecast horizon to compare cascade vs actual. Default 5 (1 week).
    forward_days_secondary : int
        Secondary horizon for the 10-day actual return. Default 10.
    save : bool
        If True, persist the report to the cascade_validation_log SQLite table.
    db_path : Path-like, optional
        Custom DB path; defaults to data/commodities.db.

    Returns
    -------
    ValidationReport
        Aggregated results with per-shock detail, sector accuracy, lift,
        lag structure confirmation, flags, and overall health status.
    """
    from models.cascade_validator import (
        ValidationReport, ShockResult, SectorResult, LagResult,
        SECTOR_COMMODITIES, SECTOR_MACRO_EFFECTS,
        CASCADE_ACCURACY_FLOOR, CASCADE_LIFT_FLOOR,
        detect_macro_shocks,
        _macro_signals_at, _sector_cascade_forecast,
        _sector_baseline_forecast, _sector_actual_returns,
        _lag_test, save_validation_report,
    )
    import math

    run_at = datetime.now(timezone.utc).isoformat()
    cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=shock_window_days)

    # ── 1. Collect shock events ────────────────────────────────────────────────
    # Prefer the trigger log (has family labels); fall back to macro detection.
    shocks = []
    try:
        from models.trigger_log import recent_trigger_events
        log_df = recent_trigger_events(days=shock_window_days, db_path=db_path)
        if not log_df.empty:
            for _, row in log_df.iterrows():
                dt = pd.Timestamp(row["trigger_date"])
                if dt >= cutoff:
                    shocks.append({
                        "date":        dt,
                        "family":      row["family"],
                        "strength":    float(row.get("strength", 0.5)),
                        "description": str(row.get("rationale", "")),
                    })
    except Exception as exc:
        log.warning("Could not read trigger log (%s); falling back to macro detection.", exc)

    if not shocks:
        detected = detect_macro_shocks(macro_df)
        shocks = [e for e in detected if e["date"] >= cutoff]

    log.info("validate_causal_chains: %d shock events in window (%d days).", len(shocks), shock_window_days)

    if not shocks:
        empty = ValidationReport(
            computed_at=run_at,
            shock_window_days=shock_window_days,
            n_shock_events=0,
            shock_results=[],
            sector_directional_accuracy={},
            sector_baseline_accuracy={},
            sector_avg_lift={},
            lag_confirmed_pct=float("nan"),
            avg_lag1_corr=float("nan"),
            flags=["No shock events found in the specified window — run the "
                   "trigger detector (Models → Market Signals) to populate the log."],
            overall_status="insufficient_data",
        )
        if save:
            save_validation_report(empty, db_path=db_path)
        return empty

    # ── 2. Evaluate each shock ─────────────────────────────────────────────────
    shock_results: List[ShockResult] = []

    for shock in shocks:
        dt       = shock["date"]
        family   = shock["family"]
        strength = shock["strength"]

        # Need enough history before the shock and enough future after it
        shock_pos = prices.index.searchsorted(dt)
        if shock_pos < harness.min_train_rows or shock_pos >= len(prices) - forward_days_secondary:
            log.debug("Skipping shock %s %s — insufficient surrounding data.", family, dt.date())
            continue

        # Macro signals as-of the shock date (historical state, not current)
        signals = _macro_signals_at(macro_df, dt)

        # Cascade and baseline forecasts as they would have appeared on shock date
        cascade_fcasts  = _sector_cascade_forecast(signals)
        baseline_fcasts = _sector_baseline_forecast(prices.loc[:dt], dt)

        # Realised sector returns over the primary and secondary horizons
        actual_5d  = _sector_actual_returns(prices, dt, forward_days_primary)
        actual_10d = _sector_actual_returns(prices, dt, forward_days_secondary)

        # Lag test over the post-shock window
        lag_result = _lag_test(prices, dt)

        # Build SectorResult for each sector
        sector_results: Dict[str, SectorResult] = {}
        for sector in SECTOR_COMMODITIES:
            a5   = actual_5d.get(sector, float("nan"))
            a10  = actual_10d.get(sector, float("nan"))
            c_fc = cascade_fcasts.get(sector, 0.0)
            b_fc = baseline_fcasts.get(sector, 0.0)

            if math.isnan(a5) or math.isnan(a10):
                continue  # no realised data yet — skip this sector for this event

            c_dir_ok = (c_fc >= 0) == (a5 >= 0)
            b_dir_ok = (b_fc >= 0) == (a5 >= 0)
            mae_c    = abs(c_fc - a5)
            mae_b    = abs(b_fc - a5)

            sector_results[sector] = SectorResult(
                sector               = sector,
                cascade_forecast_pct = round(c_fc, 4),
                baseline_forecast_pct= round(b_fc, 4),
                actual_return_5d     = round(a5,  4),
                actual_return_10d    = round(a10, 4),
                cascade_dir_correct  = c_dir_ok,
                baseline_dir_correct = b_dir_ok,
                mae_cascade          = round(mae_c, 4),
                mae_baseline         = round(mae_b, 4),
                lift                 = round(mae_b - mae_c, 4),
            )

        if not sector_results:
            continue

        shock_results.append(ShockResult(
            shock_date     = dt.strftime("%Y-%m-%d"),
            shock_family   = family,
            shock_strength = strength,
            sector_results = sector_results,
            lag_result     = lag_result,
        ))

    log.info("validate_causal_chains: %d evaluable shock events.", len(shock_results))

    if not shock_results:
        empty = ValidationReport(
            computed_at=run_at,
            shock_window_days=shock_window_days,
            n_shock_events=len(shocks),
            shock_results=[],
            sector_directional_accuracy={},
            sector_baseline_accuracy={},
            sector_avg_lift={},
            lag_confirmed_pct=float("nan"),
            avg_lag1_corr=float("nan"),
            flags=["All detected shock events lacked sufficient surrounding "
                   "price history for evaluation."],
            overall_status="insufficient_data",
        )
        if save:
            save_validation_report(empty, db_path=db_path)
        return empty

    # ── 3. Aggregate metrics ───────────────────────────────────────────────────
    sector_cascade_hits:   Dict[str, List[int]]   = {s: [] for s in SECTOR_COMMODITIES}
    sector_baseline_hits:  Dict[str, List[int]]   = {s: [] for s in SECTOR_COMMODITIES}
    sector_lifts:          Dict[str, List[float]] = {s: [] for s in SECTOR_COMMODITIES}
    lag_confirmed_flags:   List[bool]             = []
    lag1_corrs:            List[float]            = []

    for sr in shock_results:
        for sector, res in sr.sector_results.items():
            sector_cascade_hits[sector].append(int(res.cascade_dir_correct))
            sector_baseline_hits[sector].append(int(res.baseline_dir_correct))
            sector_lifts[sector].append(res.lift)

        if sr.lag_result is not None:
            lag_confirmed_flags.append(sr.lag_result.confirmed)
            lag1_corrs.append(sr.lag_result.energy_lag1_corr)

    sector_dir_acc  = {}
    sector_base_acc = {}
    sector_avg_lift = {}
    for sector in SECTOR_COMMODITIES:
        hits_c = sector_cascade_hits[sector]
        hits_b = sector_baseline_hits[sector]
        lifts  = sector_lifts[sector]
        if hits_c:
            sector_dir_acc[sector]  = round(float(np.mean(hits_c)), 4)
            sector_base_acc[sector] = round(float(np.mean(hits_b)), 4)
            sector_avg_lift[sector] = round(float(np.mean(lifts)),  4)

    lag_confirmed_pct = float(np.mean(lag_confirmed_flags)) if lag_confirmed_flags else float("nan")
    avg_lag1_corr     = float(np.mean(lag1_corrs))          if lag1_corrs          else float("nan")

    # ── 4. Health flags ────────────────────────────────────────────────────────
    flags: List[str] = []

    for sector, acc in sector_dir_acc.items():
        if not math.isnan(acc) and acc < CASCADE_ACCURACY_FLOOR:
            flags.append(
                f"{sector} cascade directional accuracy {acc*100:.0f}% "
                f"< {CASCADE_ACCURACY_FLOOR*100:.0f}% floor — "
                f"macro coefficients may be stale or regime has shifted."
            )

    for sector, lift in sector_avg_lift.items():
        if not math.isnan(lift) and lift < CASCADE_LIFT_FLOOR:
            flags.append(
                f"{sector} cascade MAE lift is negative ({lift:+.3f}pp) — "
                f"cascade underperforms the momentum baseline; "
                f"consider re-weighting or re-calibrating the macro overlay."
            )

    if not math.isnan(lag_confirmed_pct) and lag_confirmed_pct < 0.50:
        flags.append(
            f"Energy → Metals lag confirmed in only {lag_confirmed_pct*100:.0f}% "
            f"of shock events (< 50%) — "
            f"the 1-2 day propagation may have broken down under current market structure."
        )

    if len(shock_results) < 5:
        flags.append(
            f"Only {len(shock_results)} evaluable shock events — "
            f"accuracy estimates are unreliable. Expand the shock window or "
            f"run the trigger detector more frequently."
        )

    # ── 5. Overall status ──────────────────────────────────────────────────────
    n_critical = sum(
        1 for sector, acc in sector_dir_acc.items()
        if not math.isnan(acc) and acc < CASCADE_ACCURACY_FLOOR
    )
    n_negative_lift = sum(
        1 for lift in sector_avg_lift.values()
        if not math.isnan(lift) and lift < CASCADE_LIFT_FLOOR
    )

    if n_critical >= 2 or n_negative_lift >= 3:
        overall_status = "stale_coefficients"
    elif n_critical >= 1 or n_negative_lift >= 1 or (
        not math.isnan(lag_confirmed_pct) and lag_confirmed_pct < 0.50
    ):
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    report = ValidationReport(
        computed_at                  = run_at,
        shock_window_days            = shock_window_days,
        n_shock_events               = len(shock_results),
        shock_results                = shock_results,
        sector_directional_accuracy  = sector_dir_acc,
        sector_baseline_accuracy     = sector_base_acc,
        sector_avg_lift              = sector_avg_lift,
        lag_confirmed_pct            = lag_confirmed_pct,
        avg_lag1_corr                = avg_lag1_corr,
        flags                        = flags,
        overall_status               = overall_status,
    )

    log.info(
        "Cascade validation complete: status=%s, n_events=%d, flags=%d",
        overall_status, len(shock_results), len(flags),
    )

    if save:
        save_validation_report(report, db_path=db_path)

    return report


# Re-export types for callers that import from backtest_harness
from models.cascade_validator import (  # noqa: E402  (after function def)
    ValidationReport,
    ShockResult,
    SectorResult,
    LagResult,
    save_validation_report,
    recent_validation_reports,
    validation_detail,
)


def run_weekly_validation(
    prices: Optional[pd.DataFrame] = None,
    macro_df: Optional[pd.DataFrame] = None,
    commodities: Optional[Sequence[str]] = None,
    shock_window_days: int = 365,
    save: bool = True,
) -> "ValidationReport":
    """
    Convenience function for the weekly automated validation job.

    Loads prices and macro data if not provided, creates a default
    BacktestHarness, runs validate_causal_chains(), logs the summary,
    and returns the ValidationReport.

    Designed to be called from a cron / launchd job or the daily_retrain
    pipeline:

        from models.backtest_harness import run_weekly_validation
        report = run_weekly_validation()
        print(report.summary_text())

    Parameters
    ----------
    prices, macro_df : optional — loaded from DB/yfinance if not supplied.
    shock_window_days : look-back for shock events (default 365 = 1 year).
    save : whether to persist to SQLite.
    """
    from models.cascade_validator import SECTOR_COMMODITIES

    if prices is None:
        try:
            from models.data_loader import load_price_matrix
            from models.config import MODELING_COMMODITIES
            all_sector_comms = [c for v in SECTOR_COMMODITIES.values() for c in v]
            ticker_subset = {c: MODELING_COMMODITIES[c]
                             for c in all_sector_comms if c in MODELING_COMMODITIES}
            prices = load_price_matrix(commodities=ticker_subset, period="3y")
        except Exception as exc:
            log.error("run_weekly_validation: could not load prices: %s", exc)
            raise

    if macro_df is None:
        try:
            from features.macro_overlays import load_macro
            macro_df = load_macro(period="3y", _price_index=prices.index)
        except Exception as exc:
            log.warning("run_weekly_validation: macro load failed (%s); using empty df.", exc)
            macro_df = pd.DataFrame()

    harness = BacktestHarness(min_train_rows=60)
    report  = validate_causal_chains(
        harness           = harness,
        prices            = prices,
        macro_df          = macro_df,
        shock_window_days = shock_window_days,
        save              = save,
    )
    log.info("\n%s", report.summary_text())
    return report
