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
    │   1. Split at TEST_FRACTION                │
    │   2. Fit each TierAdapter on train         │
    │   3. Generate in-sample test forecasts     │
    │   4. Compare each tier to actual_return    │
    │   5. Tag date with winning_tier            │
    │   6. Look up macro state → MetaFeatures    │
    └─────┬──────────────────────────────────────┘
          │
          ▼
    List[BacktestRecord]  →  meta.fit()  →  meta.save()

Walk-forward note
─────────────────
This initial implementation uses a **single 80/20 train/test split** per
commodity.  This is fast and gives ~100 labelled dates per commodity.
Phase 5 will upgrade to TimeSeriesSplit walk-forward cross-validation for
more robust label generation; the harness API stays identical — only
`_split_indices()` changes.

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
    """

    def __init__(self):
        self._model = None
        self._commodity = None

    @property
    def tier(self) -> str:
        return "ml"

    def fit(self, prices_train: pd.DataFrame, commodity: str) -> None:
        from models.ml.xgboost_shap import XGBoostForecaster
        from models.features import build_feature_matrix, build_target
        feat_train = build_feature_matrix(prices_train)
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
        from models.features import build_feature_matrix
        feat_full = build_feature_matrix(prices_full)
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
    Same feature-matrix approach as XGBoostAdapter.
    """

    def __init__(self):
        self._model = None

    @property
    def tier(self) -> str:
        return "ml"

    def fit(self, prices_train: pd.DataFrame, commodity: str) -> None:
        from models.ml.elastic_net import ElasticNetFactorModel
        from models.features import build_feature_matrix, build_target
        feat = build_feature_matrix(prices_train)
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
        from models.features import build_feature_matrix
        feat_full = build_feature_matrix(prices_full)
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
DEFAULT_ADAPTERS: List[TierAdapter] = [
    ARIMAAdapter(fixed_order=(1, 0, 1)),  # fast fixed order for default runs
    XGBoostAdapter(),
]


class BacktestHarness:
    """
    Runs a train/test backtest across one or more commodities and collects
    (MetaFeatures, winning_tier) records for MetaPredictor.fit().

    Parameters
    ----------
    adapters : list[TierAdapter], optional
        Model-tier adapters to include.  Defaults to [ARIMA, XGBoost].
    test_fraction : float
        Fraction of history to hold out as the test window. Default 0.20.
    min_train_rows : int
        Minimum training rows before attempting a fit.  Prevents errors on
        very short price histories.
    """

    def __init__(
        self,
        adapters: Optional[List[TierAdapter]] = None,
        test_fraction: float = TEST_FRACTION,
        min_train_rows: int = 120,
    ):
        self.adapters = adapters if adapters is not None else list(DEFAULT_ADAPTERS)
        self.test_fraction = test_fraction
        self.min_train_rows = min_train_rows

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
                records = self._run_commodity(prices, macro_df, commodity)
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

    # ── Private helpers ────────────────────────────────────────────────────────

    def _run_commodity(
        self,
        prices: pd.DataFrame,
        macro_df: pd.DataFrame,
        commodity: str,
    ) -> List[BacktestRecord]:
        """Run the backtest for a single commodity."""
        # ── 1. Compute actual next-day returns for the full window ─────────────
        actual_returns = np.log(prices[commodity] / prices[commodity].shift(1)).shift(-1)

        # ── 2. Split at test_fraction ─────────────────────────────────────────
        n = len(prices)
        split = int(n * (1 - self.test_fraction))
        if split < self.min_train_rows:
            log.warning(
                "Insufficient training data for %r (%d rows, need %d). Skipping.",
                commodity, split, self.min_train_rows,
            )
            return []

        prices_train = prices.iloc[:split]
        test_idx = prices.index[split:]
        # Drop last row (no actual return available due to shift(-1))
        test_idx = test_idx[:-1]

        if len(test_idx) == 0:
            return []

        # ── 3. Fit each adapter on train, predict on test ──────────────────────
        tier_forecasts: Dict[str, pd.Series] = {}
        for adapter in self.adapters:
            try:
                adapter.fit(prices_train, commodity)
                preds = adapter.predict_series(prices, commodity, test_idx)
                tier_forecasts[adapter.tier] = preds
            except Exception as exc:
                log.warning(
                    "Adapter %s failed for %r: %s",
                    adapter.__class__.__name__, commodity, exc,
                )

        if not tier_forecasts:
            return []

        # ── 4. Build one BacktestRecord per test date ──────────────────────────
        # Align macro_df to test dates for MetaFeatures lookup
        macro_aligned = _align_macro(macro_df, prices.index)

        records: List[BacktestRecord] = []
        for date in test_idx:
            actual = actual_returns.get(date, float("nan"))
            if np.isnan(actual):
                continue

            # Per-tier forecasts for this date
            t_fcst: Dict[str, float] = {}
            t_err: Dict[str, float] = {}
            for tier, series in tier_forecasts.items():
                fc = float(series.get(date, float("nan")))
                if np.isnan(fc):
                    continue
                t_fcst[tier] = fc
                t_err[tier] = abs(fc - actual)

            if not t_err:
                continue

            winning_tier = min(t_err, key=lambda k: t_err[k])

            # Market state on this date
            mf = _meta_features_for_date(macro_aligned, date)

            records.append(BacktestRecord(
                date=date,
                commodity=commodity,
                meta_features=mf,
                actual_return=actual,
                tier_forecasts=t_fcst,
                tier_errors=t_err,
                winning_tier=winning_tier,
            ))

        return records


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
