"""
XGBoost forecaster with SHAP Shapley value explainability.

XGBoost is the primary forecasting engine for this dashboard. On daily
financial series it consistently produces the highest Information Coefficient
(IC) of any non-deep-learning model, largely because:
  - Gradient boosting captures non-linear interactions between features
    (e.g. momentum matters more during low-vol regimes)
  - Early stopping prevents overfitting to noise in short financial series
  - Feature subsampling at each tree provides implicit regularisation

SHAP (SHapley Additive exPlanations) assigns exact credit for each prediction
to each input feature. For tree models, TreeExplainer computes exact Shapley
values in O(TLD) time (T trees, L leaves, D depth) — milliseconds per sample.

The key output for the dashboard is the waterfall: for a given day's
forecast, show which features pushed the prediction up or down from the
base rate. This is the interpretability centrepiece — a non-technical member
can read "WTI is predicted to fall 0.3% because: NatGas momentum (-0.18%),
21d vol drag (-0.12%), but Copper z-score support (+0.08%)."

Usage
-----
    from models.ml.xgboost_shap import XGBoostForecaster, run_all

    prices = load_price_matrix()
    feat_df = build_feature_matrix(prices)
    target  = build_target(prices, "WTI Crude Oil")

    xgb = XGBoostForecaster(commodity="WTI Crude Oil")
    xgb.fit(feat_df, target)

    preds  = xgb.predict(feat_df)
    ic     = xgb.ic_score(feat_df, target)
    shap   = xgb.shap_values(feat_df)          # pd.DataFrame, same shape as feat_df
    wf     = xgb.waterfall_data(feat_df, idx=-1)  # dict for Plotly waterfall chart

    all_results = run_all(prices)
"""

import warnings
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from typing import Optional

from models.config import MODELING_COMMODITIES, TEST_FRACTION, RANDOM_SEED
from models.features import build_feature_matrix, build_target

# XGBoost hyperparameters — conservative defaults for financial series
XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

# Early stopping rounds on a validation slice
EARLY_STOPPING_ROUNDS = 30
VALIDATION_FRACTION   = 0.15


class XGBoostForecaster:
    """
    XGBoost regressor with integrated SHAP explainability.

    Parameters
    ----------
    commodity : str
        Display name of the target commodity.
    """

    def __init__(self, commodity: str = "WTI Crude Oil"):
        self.commodity = commodity
        self._model: Optional[xgb.XGBRegressor] = None
        self._explainer: Optional[shap.TreeExplainer] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: Optional[list] = None
        self._base_value: Optional[float] = None

    # ── private ────────────────────────────────────────────────────────────────

    def _prepare(
        self,
        feat_df: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        combined = pd.concat([feat_df, target], axis=1).dropna()
        X = combined.drop(columns=[target.name])
        y = combined[target.name]
        return X, y

    # ── public interface ───────────────────────────────────────────────────────

    def fit(
        self,
        feat_df: pd.DataFrame,
        target: pd.Series,
    ) -> "XGBoostForecaster":
        """
        Fit XGBoost with early stopping on a held-out validation slice.

        Chronological split only — no shuffling. The validation slice is
        taken from the middle of the training set to avoid contaminating
        the final test period.

        Parameters
        ----------
        feat_df : pd.DataFrame
            Feature matrix from build_feature_matrix().
        target : pd.Series
            Next-day log-return from build_target().
        """
        X_df, y = self._prepare(feat_df, target)
        self._feature_names = X_df.columns.tolist()

        split = int(len(X_df) * (1 - TEST_FRACTION))
        X_train_all = X_df.iloc[:split]
        y_train_all = y.iloc[:split]

        # Further split training into fit / early-stop validation
        val_cut = int(len(X_train_all) * (1 - VALIDATION_FRACTION))
        X_fit, X_val = X_train_all.iloc[:val_cut], X_train_all.iloc[val_cut:]
        y_fit, y_val = y_train_all.iloc[:val_cut], y_train_all.iloc[val_cut:]

        self._scaler = StandardScaler()
        X_fit_sc = self._scaler.fit_transform(X_fit)
        X_val_sc = self._scaler.transform(X_val)

        self._model = xgb.XGBRegressor(**XGB_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            self._model.fit(
                X_fit_sc, y_fit,
                eval_set=[(X_val_sc, y_val)],
                verbose=False,
            )

        # Build SHAP explainer on the full training set
        X_train_sc = self._scaler.transform(X_train_all.values)
        self._explainer = shap.TreeExplainer(self._model)
        self._base_value = float(self._model.predict(X_train_sc).mean())
        return self

    def predict(self, feat_df: pd.DataFrame) -> pd.Series:
        """Predict next-day log-returns, returning a DatetimeIndex Series."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        X = feat_df[self._feature_names].ffill()
        valid = ~X.isna().any(axis=1)
        preds = np.full(len(X), np.nan)
        preds[valid] = self._model.predict(
            self._scaler.transform(X[valid].values)
        )
        return pd.Series(preds, index=feat_df.index, name=f"xgb_{self.commodity}")

    def ic_score(
        self,
        feat_df: pd.DataFrame,
        target: pd.Series,
    ) -> float:
        """Spearman IC on the held-out test period."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        X_df, y = self._prepare(feat_df, target)
        split = int(len(X_df) * (1 - TEST_FRACTION))
        X_test_sc = self._scaler.transform(X_df.iloc[split:].values)
        preds = self._model.predict(X_test_sc)
        ic, _ = spearmanr(preds, y.iloc[split:].values)
        return round(float(ic), 4)

    def shap_values(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SHAP values for all rows of feat_df.

        Returns
        -------
        pd.DataFrame
            Same shape as feat_df (after dropping NaN rows).
            Each value is the SHAP contribution of that feature to the
            prediction for that date. Units are log-return points.
        """
        if self._explainer is None:
            raise RuntimeError("Call fit() first.")
        X = feat_df[self._feature_names].dropna()
        X_sc = self._scaler.transform(X.values)
        sv = self._explainer.shap_values(X_sc)
        return pd.DataFrame(sv, index=X.index, columns=self._feature_names)

    def waterfall_data(
        self,
        feat_df: pd.DataFrame,
        idx: int = -1,
    ) -> dict:
        """
        Prepare data for a Plotly waterfall chart explaining one prediction.

        Parameters
        ----------
        feat_df : pd.DataFrame
            Feature matrix (any date range).
        idx : int
            Row index into feat_df.dropna(). -1 = most recent date.

        Returns
        -------
        dict with keys:
            date         — string date for the predicted day
            forecast     — the model's point prediction
            base_value   — mean training prediction (the "anchor")
            features     — list of (feature_name, shap_value, feature_value)
                           sorted by abs(shap_value) descending
            top_positive — top 3 features pushing forecast up
            top_negative — top 3 features pushing forecast down
        """
        if self._explainer is None:
            raise RuntimeError("Call fit() first.")

        X = feat_df[self._feature_names].dropna()
        row = X.iloc[idx]
        date_str = str(X.index[idx].date())
        X_sc = self._scaler.transform(row.values.reshape(1, -1))

        sv = self._explainer.shap_values(X_sc)[0]
        forecast = float(self._model.predict(X_sc)[0])

        features = sorted(
            [(name, float(s), float(v))
             for name, s, v in zip(self._feature_names, sv, row.values)],
            key=lambda t: abs(t[1]),
            reverse=True,
        )

        return {
            "date":         date_str,
            "forecast":     round(forecast, 6),
            "base_value":   round(self._base_value, 6),
            "features":     features,
            "top_positive": [f for f in features if f[1] > 0][:3],
            "top_negative": [f for f in features if f[1] < 0][:3],
        }

    def global_shap_importance(
        self,
        feat_df: pd.DataFrame,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Mean absolute SHAP value per feature — model-consistent global importance.

        Unlike RF's Gini importance, SHAP importance is not biased toward
        high-cardinality or correlated features.

        Returns
        -------
        pd.DataFrame
            Columns: feature, mean_abs_shap. Sorted descending.
        """
        sv_df = self.shap_values(feat_df)
        mean_abs = sv_df.abs().mean().sort_values(ascending=False).head(top_n)
        return pd.DataFrame({
            "feature":       mean_abs.index,
            "mean_abs_shap": mean_abs.values.round(6),
        }).reset_index(drop=True)


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_all(
    prices: pd.DataFrame,
    commodities: Optional[dict] = None,
) -> dict:
    """
    Fit XGBoost + SHAP for every commodity. Returns forecasts and explainability.

    Returns
    -------
    dict
        Keys = commodity display names.
        Values = dicts: model, ic, shap_importance (DataFrame),
                        latest_waterfall (dict), predictions (pd.Series).
    """
    if commodities is None:
        commodities = MODELING_COMMODITIES

    feat_df = build_feature_matrix(prices)
    results = {}

    for name in commodities:
        if name not in prices.columns:
            continue
        target = build_target(prices, name)
        model = XGBoostForecaster(commodity=name)
        model.fit(feat_df, target)
        results[name] = {
            "model":             model,
            "ic":                model.ic_score(feat_df, target),
            "shap_importance":   model.global_shap_importance(feat_df),
            "latest_waterfall":  model.waterfall_data(feat_df, idx=-1),
            "predictions":       model.predict(feat_df),
        }

    return results
