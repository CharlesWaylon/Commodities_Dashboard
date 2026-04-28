"""
Elastic Net / Lasso sparse cross-commodity factor model.

Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties:

    min  ||y - Xβ||² + α[ρ||β||₁ + (1-ρ)||β||²]
    β

  α    — overall regularisation strength (tuned by CV)
  ρ    — mixing ratio: ρ=1 → pure Lasso (maximum sparsity)
                       ρ=0 → pure Ridge (no sparsity)

The L1 penalty drives most coefficients exactly to zero, producing a sparse
factor model. The survivors are the signals that genuinely drive this commodity
above and beyond what OLS would retain. Reading the coefficient table is the
explanation — no SHAP required.

Example interpretation for WTI (from a typical fit):
  "WTI 5-day return is driven by:
    NatGas_mom10       +0.12   (nat gas leads oil in storage-shock events)
    DXY_zscore         -0.08   (strong dollar suppresses USD-priced oil)
    WTI_vol21          -0.05   (high vol → mean-reversion, vol drag)
    Copper_ret_1d      +0.04   (copper leads industrial demand)
    ... (all other 40+ features → zero)"

The l1_ratio CV path also tells you whether the commodity is better described
by a sparse model (Lasso wins) or a diffuse factor model (Ridge wins) — itself
an interesting structural result.

Usage
-----
    from models.ml.elastic_net import ElasticNetFactorModel, run_all

    prices = load_price_matrix()
    feat_df = build_feature_matrix(prices)
    target  = build_target(prices, "WTI Crude Oil")

    en = ElasticNetFactorModel(commodity="WTI Crude Oil")
    en.fit(feat_df, target)

    table  = en.coefficient_table()      # non-zero factors sorted by magnitude
    path   = en.regularisation_path()    # alpha/IC trade-off curve
    ic     = en.ic_score(feat_df, target)

    all_results = run_all(prices)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from typing import Optional, Literal

from models.config import MODELING_COMMODITIES, TEST_FRACTION, RANDOM_SEED
from models.features import build_feature_matrix, build_target

# L1 ratios to search in cross-validation
# 1.0 = pure Lasso; 0.5 = balanced; 0.1 = close to Ridge
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

# Alpha grid — broad search, CV selects the best
N_ALPHAS = 100

# Cross-validation: time-series aware (no shuffling)
CV_FOLDS = 5


class ElasticNetFactorModel:
    """
    Elastic Net cross-commodity factor model for a single commodity.

    Parameters
    ----------
    commodity : str
        Display name of the target commodity.
    mode : {"elastic_net", "lasso"}
        "lasso" fixes l1_ratio=1.0 and skips the ratio CV (faster).
        "elastic_net" searches across L1_RATIOS.
    """

    def __init__(
        self,
        commodity: str = "WTI Crude Oil",
        mode: Literal["elastic_net", "lasso"] = "elastic_net",
    ):
        self.commodity = commodity
        self.mode = mode
        self._model = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: Optional[list] = None

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
    ) -> "ElasticNetFactorModel":
        """
        Fit the factor model with cross-validated alpha (and optionally l1_ratio).

        Time-series CV: folds are formed as expanding windows, preserving
        chronological order. No data from the test period leaks into fitting.

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
        X_train = X_df.iloc[:split]
        y_train = y.iloc[:split]

        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X_train.values)
        y_train_vals = y_train.values

        # Time-series cross-validation: sklearn's CV with no shuffling
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

        # Cap alpha search at 50% of alpha_max to prevent the degenerate all-zeros
        # solution that ElasticNetCV's auto path selects on low-SNR financial data.
        # alpha_max is the smallest alpha that zeros all Lasso coefficients.
        alpha_max = float(np.max(np.abs(X_sc.T @ y_train_vals)) / len(y_train_vals))
        alphas = np.logspace(
            np.log10(max(alpha_max * 1e-4, 1e-8)),
            np.log10(alpha_max * 0.5),
            N_ALPHAS,
        )

        if self.mode == "lasso":
            self._model = LassoCV(
                alphas=alphas,
                cv=tscv,
                max_iter=5000,
                random_state=RANDOM_SEED,
            )
        else:
            self._model = ElasticNetCV(
                l1_ratio=L1_RATIOS,
                alphas=alphas,
                cv=tscv,
                max_iter=5000,
                random_state=RANDOM_SEED,
            )

        self._model.fit(X_sc, y_train.values)
        return self

    def predict(self, feat_df: pd.DataFrame) -> pd.Series:
        """Predict next-day log-returns."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        X = feat_df[self._feature_names].ffill()
        valid = ~X.isna().any(axis=1)
        preds = np.full(len(X), np.nan)
        preds[valid] = self._model.predict(
            self._scaler.transform(X[valid].values)
        )
        return pd.Series(preds, index=feat_df.index, name=f"en_{self.commodity}")

    def coefficient_table(self, include_zeros: bool = False) -> pd.DataFrame:
        """
        Non-zero model coefficients sorted by absolute magnitude.

        These are the surviving factors — the sparse representation of what
        drives this commodity's next-day return.

        Parameters
        ----------
        include_zeros : bool
            If True, include zeroed-out features (shows what was eliminated).

        Returns
        -------
        pd.DataFrame
            Columns: feature, coefficient, direction.
            Only non-zero rows unless include_zeros=True.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        coef = self._model.coef_
        df = pd.DataFrame({
            "feature":     self._feature_names,
            "coefficient": coef.round(6),
        })
        if not include_zeros:
            df = df[df["coefficient"] != 0]
        df = df.reindex(df["coefficient"].abs().sort_values(ascending=False).index)
        df["direction"] = df["coefficient"].apply(
            lambda c: "positive" if c > 0 else ("negative" if c < 0 else "zero")
        )
        return df.reset_index(drop=True)

    def sparsity(self) -> dict:
        """
        Summary of model sparsity: how many features survived regularisation?
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        n_total = len(self._feature_names)
        n_nonzero = int((self._model.coef_ != 0).sum())
        return {
            "n_features_total":   n_total,
            "n_features_nonzero": n_nonzero,
            "sparsity_pct":       round(100 * (1 - n_nonzero / n_total), 1),
            "alpha":              round(float(self._model.alpha_), 6),
            "l1_ratio":           round(
                float(getattr(self._model, "l1_ratio_", 1.0)), 4
            ),
        }

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
        if np.std(preds) == 0:
            return 0.0
        ic, _ = spearmanr(preds, y.iloc[split:].values)
        return round(float(ic), 4)

    def regularisation_path(self, feat_df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """
        Alpha vs. number of non-zero coefficients path (for the selected l1_ratio).

        Shows the trade-off between sparsity and model complexity. Useful for
        understanding whether the CV-selected alpha is in a stable region.

        Returns
        -------
        pd.DataFrame
            Columns: alpha, n_nonzero_coefs, ic_train.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        from sklearn.linear_model import enet_path

        X_df, y = self._prepare(feat_df, target)
        split = int(len(X_df) * (1 - TEST_FRACTION))
        X_sc = self._scaler.transform(X_df.iloc[:split].values)
        y_tr = y.iloc[:split].values

        l1 = float(getattr(self._model, "l1_ratio_", 1.0))
        alphas, coef_path, _ = enet_path(X_sc, y_tr, l1_ratio=l1, n_alphas=50)

        rows = []
        for alpha, coefs in zip(alphas, coef_path.T):
            rows.append({
                "alpha":          round(float(alpha), 6),
                "n_nonzero":      int((coefs != 0).sum()),
            })
        return pd.DataFrame(rows)

    def factor_narrative(self, top_n: int = 5) -> str:
        """
        Plain-English summary of the top surviving factors.

        Example output:
          "WTI 5-day return is driven by:
            NatGas_mom10 (+0.12), WTI_vol21 (-0.05), Copper_ret_1d (+0.04)"
        """
        ct = self.coefficient_table().head(top_n)
        parts = []
        for _, row in ct.iterrows():
            sign = "+" if row["coefficient"] > 0 else ""
            parts.append(f"{row['feature']} ({sign}{row['coefficient']:.4f})")
        return f"{self.commodity} next-day return is driven by:\n  " + ", ".join(parts)


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_all(
    prices: pd.DataFrame,
    commodities: Optional[dict] = None,
    mode: Literal["elastic_net", "lasso"] = "elastic_net",
) -> dict:
    """
    Fit an Elastic Net factor model for every commodity.

    Returns
    -------
    dict
        Keys = commodity display names.
        Values = dicts: model, ic, coefficient_table (DataFrame),
                        sparsity (dict), narrative (str).
    """
    if commodities is None:
        commodities = MODELING_COMMODITIES

    feat_df = build_feature_matrix(prices)
    results = {}

    for name in commodities:
        if name not in prices.columns:
            continue
        target = build_target(prices, name)
        en = ElasticNetFactorModel(commodity=name, mode=mode)
        en.fit(feat_df, target)
        results[name] = {
            "model":             en,
            "ic":                en.ic_score(feat_df, target),
            "coefficient_table": en.coefficient_table(),
            "sparsity":          en.sparsity(),
            "narrative":         en.factor_narrative(),
        }

    return results
