"""
Random Forest regressor with rolling feature importance.

Primary use: feature importance ranking — which signals actually drive each
commodity's next-day return? The RF aggregates many decorrelated trees, so
importance scores reflect genuine predictive signal rather than overfitting
to a single decision boundary.

Rolling re-fits reveal how importance shifts over time. Example questions
this answers for a club discussion:
  - Did DXY sensitivity in Gold increase after the Fed pivot?
  - Did WTI-NatGas correlation strengthen post Ukraine invasion?
  - Is corn momentum more or less predictive than soybean cross-correlation?

Information Coefficient (IC) — Spearman rank correlation between predictions
and actual next-day returns — is reported instead of R². IC in [0.03, 0.06]
is considered "good" for daily commodity forecasting (Grinold & Kahn).

Usage
-----
    from models.ml.random_forest import CommodityRF, run_all

    prices = load_price_matrix()
    feat_df = build_feature_matrix(prices)
    target  = build_target(prices, "WTI Crude Oil")

    rf = CommodityRF(commodity="WTI Crude Oil")
    rf.fit(feat_df, target)

    importance = rf.feature_importance()     # sorted DataFrame
    rolling_imp = rf.rolling_importance()    # time-varying importance
    ic = rf.ic_score(feat_df, target)        # Spearman IC
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from typing import Optional

from models.config import MODELING_COMMODITIES, TEST_FRACTION, RANDOM_SEED
from models.features import build_feature_matrix, build_target

# Rolling window: re-fit on the past N trading days
ROLLING_FIT_WINDOW = 252      # ~1 year
ROLLING_STEP       = 21       # re-fit monthly

# RF hyperparameters — tuned for financial time series
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=10,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_SEED,
)


class CommodityRF:
    """
    Random Forest forecaster and feature importance ranker for one commodity.

    Parameters
    ----------
    commodity : str
        Display name of the target commodity.
    """

    def __init__(self, commodity: str = "WTI Crude Oil"):
        self.commodity = commodity
        self._rf: Optional[RandomForestRegressor] = None
        self._feature_names: Optional[list] = None
        self._scaler: Optional[StandardScaler] = None

    # ── private ────────────────────────────────────────────────────────────────

    def _prepare(
        self,
        feat_df: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        combined = pd.concat([feat_df, target], axis=1).dropna()
        X_raw = combined.drop(columns=[target.name]).values
        y = combined[target.name].values
        names = combined.drop(columns=[target.name]).columns.tolist()
        return X_raw, y, names

    # ── public interface ───────────────────────────────────────────────────────

    def fit(
        self,
        feat_df: pd.DataFrame,
        target: pd.Series,
    ) -> "CommodityRF":
        """
        Fit on the full dataset (train split).

        Parameters
        ----------
        feat_df : pd.DataFrame
            Feature matrix from build_feature_matrix().
        target : pd.Series
            Next-day log-return from build_target().
        """
        X, y, names = self._prepare(feat_df, target)
        split = int(len(X) * (1 - TEST_FRACTION))
        X_train, y_train = X[:split], y[:split]

        self._scaler = StandardScaler()
        X_train_sc = self._scaler.fit_transform(X_train)

        self._rf = RandomForestRegressor(**RF_PARAMS)
        self._rf.fit(X_train_sc, y_train)
        self._feature_names = names
        return self

    def predict(self, feat_df: pd.DataFrame) -> pd.Series:
        """
        Predict next-day log-returns.

        Returns a pd.Series aligned to feat_df's index (NaN where features are NaN).
        """
        if self._rf is None:
            raise RuntimeError("Call fit() first.")
        X = feat_df[self._feature_names].values
        valid = ~np.isnan(X).any(axis=1)
        preds = np.full(len(X), np.nan)
        preds[valid] = self._rf.predict(self._scaler.transform(X[valid]))
        return pd.Series(preds, index=feat_df.index, name=f"rf_{self.commodity}")

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Gini-based feature importances, sorted descending.

        Returns
        -------
        pd.DataFrame
            Columns: feature, importance, importance_pct.
            Top `top_n` rows.
        """
        if self._rf is None:
            raise RuntimeError("Call fit() first.")
        imp = self._rf.feature_importances_
        df = pd.DataFrame({
            "feature":    self._feature_names,
            "importance": imp,
        }).sort_values("importance", ascending=False).head(top_n)
        df["importance_pct"] = (df["importance"] / df["importance"].sum() * 100).round(2)
        return df.reset_index(drop=True)

    def ic_score(
        self,
        feat_df: pd.DataFrame,
        target: pd.Series,
    ) -> float:
        """
        Information Coefficient — Spearman rank correlation on the test set.

        IC > 0.05 is considered actionable for daily commodity signals.
        """
        if self._rf is None:
            raise RuntimeError("Call fit() first.")
        X, y, _ = self._prepare(feat_df, target)
        split = int(len(X) * (1 - TEST_FRACTION))
        X_test = self._scaler.transform(X[split:])
        y_test = y[split:]
        preds = self._rf.predict(X_test)
        ic, _ = spearmanr(preds, y_test)
        return round(float(ic), 4)

    def rolling_importance(
        self,
        feat_df: pd.DataFrame,
        target: pd.Series,
        window: int = ROLLING_FIT_WINDOW,
        step: int = ROLLING_STEP,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Walk-forward feature importance: re-fit on each rolling window of
        `window` days, stepping forward by `step` days.

        Returns
        -------
        pd.DataFrame
            MultiIndex (date, feature) → importance. Pivot to get a
            wide DataFrame with one column per feature.
        """
        X, y, names = self._prepare(feat_df, target)
        n = len(X)
        records = []

        for start in range(0, n - window - 1, step):
            end = start + window
            X_win = X[start:end]
            y_win = y[start:end]

            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X_win)

            rf = RandomForestRegressor(**RF_PARAMS)
            rf.fit(X_sc, y_win)

            imp = rf.feature_importances_
            date = feat_df.index[end] if end < len(feat_df) else feat_df.index[-1]
            for fname, fval in zip(names, imp):
                records.append({"date": date, "feature": fname, "importance": fval})

        df = pd.DataFrame(records)
        if df.empty:
            return df

        # Keep only the top-N features by average importance across all windows
        avg = df.groupby("feature")["importance"].mean().nlargest(top_n).index
        df = df[df["feature"].isin(avg)]
        return df.pivot(index="date", columns="feature", values="importance")


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_all(
    prices: pd.DataFrame,
    commodities: Optional[dict] = None,
) -> dict:
    """
    Fit a Random Forest for every commodity and return importances + IC scores.

    Returns
    -------
    dict
        Keys = commodity display names.
        Values = dicts: rf, importance (DataFrame), ic, rolling_importance (DataFrame).
    """
    if commodities is None:
        commodities = MODELING_COMMODITIES

    feat_df = build_feature_matrix(prices)
    results = {}

    for name in commodities:
        if name not in prices.columns:
            continue
        target = build_target(prices, name)
        rf = CommodityRF(commodity=name)
        rf.fit(feat_df, target)
        results[name] = {
            "rf":                 rf,
            "importance":         rf.feature_importance(top_n=20),
            "ic":                 rf.ic_score(feat_df, target),
            "rolling_importance": rf.rolling_importance(feat_df, target),
        }

    return results
