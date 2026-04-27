"""
Hidden Markov Model regime detector for commodity markets.

Fits a Gaussian HMM on the joint (return, volatility) observation sequence.
Each hidden state represents a distinct market regime — the HMM learns the
state-switching dynamics from data rather than imposing arbitrary threshold rules.

Regime labeling heuristic (applied after fitting):
  1. Rank states by mean return (ascending).
  2. The lowest-return state is "Bear"; the highest is "Bull".
  3. Among remaining states, the one with the highest observation standard
     deviation is "Crisis"; the other is "Neutral".

With 3 states: Bear, Neutral, Bull (no Crisis label).
With 5 states: Bear, Soft Bear, Neutral, Soft Bull, Bull.

Why 4 states for commodities?
  Energy markets exhibit at least four regimes: trending bull (supply-driven),
  trending bear, low-vol mean-reversion, and crisis spikes (LME squeeze,
  Covid demand collapse, Ukraine invasion). 4 states gives meaningful
  segmentation without overfitting on ~500 trading days.

Usage
-----
    from models.ml.hmm_regime import HMMRegimeDetector, fit_commodity

    prices = load_price_matrix()
    returns = np.log(prices / prices.shift(1)).dropna()

    detector = HMMRegimeDetector(n_states=4)
    detector.fit(returns["WTI Crude Oil"])

    labels  = detector.regime_series()        # pd.Series of "Bull"/"Bear"/...
    probs   = detector.state_probabilities()  # pd.DataFrame, cols = state names
    trans   = detector.transition_matrix()    # pd.DataFrame
    summary = detector.regime_summary()       # stats per regime
    ic_map  = detector.regime_conditional_ic(forecasts, actuals)
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional
from hmmlearn.hmm import GaussianHMM
from scipy.stats import spearmanr

from models.config import MODELING_COMMODITIES, ROLLING_VOL_WINDOW

# Regime label sets indexed by number of states
_REGIME_LABELS = {
    3: ["Bear", "Neutral", "Bull"],
    4: ["Bear", "Neutral", "High-Vol", "Bull"],
    5: ["Bear", "Soft Bear", "Neutral", "Soft Bull", "Bull"],
}

PALETTE = {
    "Bull":      "#2ecc71",
    "Soft Bull":  "#82e0aa",
    "Neutral":   "#f0b429",
    "High-Vol":  "#e67e22",
    "Soft Bear": "#ec7063",
    "Bear":      "#e74c3c",
    "Crisis":    "#8e44ad",
}


class HMMRegimeDetector:
    """
    Gaussian HMM regime detector for a single commodity's return series.

    Parameters
    ----------
    n_states : int
        Number of hidden states. 4 is recommended for most commodities.
    n_iter : int
        Maximum EM iterations.
    covariance_type : str
        Passed to hmmlearn.GaussianHMM. "full" fits a full covariance matrix
        per state; "diag" is faster and often equally good.
    """

    def __init__(
        self,
        n_states: int = 4,
        n_iter: int = 200,
        covariance_type: str = "full",
    ):
        if n_states not in _REGIME_LABELS:
            raise ValueError(f"n_states must be one of {list(_REGIME_LABELS.keys())}")
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type

        self._hmm: Optional[GaussianHMM] = None
        self._hidden_states: Optional[np.ndarray] = None
        self._state_to_label: Optional[dict] = None
        self._index: Optional[pd.DatetimeIndex] = None
        self._obs: Optional[np.ndarray] = None

    # ── private ────────────────────────────────────────────────────────────────

    def _build_observations(self, returns: pd.Series) -> np.ndarray:
        vol = returns.rolling(ROLLING_VOL_WINDOW).std() * np.sqrt(252)
        obs = pd.concat([returns, vol], axis=1).dropna()
        obs.columns = ["return", "vol"]
        return obs

    def _assign_labels(self) -> dict:
        """
        Map integer state indices → human-readable regime labels.

        Logic: rank states by mean return. Inject "High-Vol" label at the
        position of the highest-variance state among the middle states (only
        for n_states == 4).
        """
        means = self._hmm.means_[:, 0]    # mean return for each state
        stds  = np.sqrt(self._hmm.covars_[:, 0, 0] if self.covariance_type == "full"
                        else self._hmm.covars_[:, 0])

        order_by_return = np.argsort(means)    # state indices sorted low → high return

        if self.n_states == 4:
            # Identify the high-vol state among the two middle states
            middle = order_by_return[1:3]
            hv_state = middle[np.argmax(stds[middle])]
            neutral_state = middle[np.argmin(stds[middle])]

            label_map = {
                int(order_by_return[0]):  "Bear",
                int(neutral_state):       "Neutral",
                int(hv_state):            "High-Vol",
                int(order_by_return[-1]): "Bull",
            }
        else:
            labels = _REGIME_LABELS[self.n_states]
            label_map = {int(order_by_return[i]): labels[i]
                         for i in range(self.n_states)}

        return label_map

    # ── public interface ───────────────────────────────────────────────────────

    def fit(self, returns: pd.Series) -> "HMMRegimeDetector":
        """
        Fit the HMM via the Baum-Welch EM algorithm.

        Parameters
        ----------
        returns : pd.Series
            Daily log-returns. DatetimeIndex.
        """
        obs_df = self._build_observations(returns)
        self._index = obs_df.index
        self._obs = obs_df.values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=42,
            )
            hmm.fit(self._obs)

        self._hmm = hmm
        self._hidden_states = hmm.predict(self._obs)
        self._state_to_label = self._assign_labels()
        return self

    def regime_series(self) -> pd.Series:
        """Time series of regime label strings. DatetimeIndex."""
        if self._hidden_states is None:
            raise RuntimeError("Call fit() first.")
        labels = [self._state_to_label[s] for s in self._hidden_states]
        return pd.Series(labels, index=self._index, name="regime")

    def state_probabilities(self) -> pd.DataFrame:
        """
        Posterior state probabilities (soft assignments) for each date.

        Returns
        -------
        pd.DataFrame
            Columns = regime label strings. Each row sums to 1.
        """
        if self._hmm is None:
            raise RuntimeError("Call fit() first.")
        log_probs = self._hmm.predict_proba(self._obs)
        cols = [self._state_to_label[i] for i in range(self.n_states)]
        return pd.DataFrame(log_probs, index=self._index, columns=cols)

    def transition_matrix(self) -> pd.DataFrame:
        """
        Regime transition probability matrix.

        Entry [i, j] = probability of transitioning from regime i to regime j
        on the next trading day.
        """
        if self._hmm is None:
            raise RuntimeError("Call fit() first.")
        labels = [self._state_to_label[i] for i in range(self.n_states)]
        return pd.DataFrame(
            self._hmm.transmat_.round(4),
            index=labels,
            columns=labels,
        )

    def regime_summary(self) -> pd.DataFrame:
        """
        Per-regime descriptive statistics on observed returns.

        Returns
        -------
        pd.DataFrame
            Index = regime labels. Columns: count, freq, mean_ret_ann,
            vol_ann, sharpe, min_ret, max_ret.
        """
        if self._hidden_states is None:
            raise RuntimeError("Call fit() first.")
        returns_arr = self._obs[:, 0]
        rows = []
        for state_idx, label in self._state_to_label.items():
            mask = self._hidden_states == state_idx
            r = returns_arr[mask]
            if len(r) == 0:
                continue
            mean_ann = float(r.mean() * 252)
            vol_ann  = float(r.std() * np.sqrt(252))
            rows.append({
                "regime":       label,
                "count":        int(mask.sum()),
                "freq_%":       round(100 * mask.mean(), 1),
                "mean_ret_ann": round(mean_ann, 4),
                "vol_ann":      round(vol_ann, 4),
                "sharpe":       round(mean_ann / vol_ann if vol_ann > 0 else np.nan, 3),
                "min_ret":      round(float(r.min()), 4),
                "max_ret":      round(float(r.max()), 4),
            })
        df = pd.DataFrame(rows).set_index("regime")
        return df.sort_values("mean_ret_ann")

    def regime_conditional_ic(
        self,
        forecasts: pd.Series,
        actuals: pd.Series,
    ) -> pd.DataFrame:
        """
        Information Coefficient (Spearman rank correlation) of a forecast
        series, computed separately within each detected regime.

        This tells you: "does this model work in Bull markets but fail in Crisis?"

        Parameters
        ----------
        forecasts : pd.Series
            Model predictions aligned to the same DatetimeIndex as actuals.
        actuals : pd.Series
            Realised returns.

        Returns
        -------
        pd.DataFrame
            Columns: regime, n_obs, IC, p_value.
        """
        if self._hidden_states is None:
            raise RuntimeError("Call fit() first.")

        regime_s = self.regime_series()
        aligned = pd.concat([forecasts, actuals, regime_s], axis=1).dropna()
        aligned.columns = ["forecast", "actual", "regime"]

        rows = []
        for label in aligned["regime"].unique():
            sub = aligned[aligned["regime"] == label]
            if len(sub) < 5:
                continue
            ic, pval = spearmanr(sub["forecast"], sub["actual"])
            rows.append({
                "regime":  label,
                "n_obs":   len(sub),
                "IC":      round(float(ic), 4),
                "p_value": round(float(pval), 4),
            })
        return pd.DataFrame(rows).sort_values("IC", ascending=False)

    def color_map(self) -> dict:
        """Returns {date_str: hex_color} for chart annotation."""
        regimes = self.regime_series()
        return {str(d.date()): PALETTE.get(r, "#aaaaaa") for d, r in regimes.items()}


# ── Convenience runner ─────────────────────────────────────────────────────────

def fit_commodity(
    prices: pd.DataFrame,
    commodity: str,
    n_states: int = 4,
) -> HMMRegimeDetector:
    """Fit and return a detector for a single commodity column in prices."""
    returns = np.log(prices[commodity] / prices[commodity].shift(1)).dropna()
    detector = HMMRegimeDetector(n_states=n_states)
    detector.fit(returns)
    return detector


def run_all(
    prices: pd.DataFrame,
    commodities: Optional[dict] = None,
    n_states: int = 4,
) -> dict:
    """
    Fit HMM regime detector for every commodity in `prices`.

    Returns
    -------
    dict
        Keys = commodity display names.
        Values = dicts with: detector, regimes (pd.Series), summary (pd.DataFrame),
                             transition (pd.DataFrame), palette.
    """
    if commodities is None:
        commodities = MODELING_COMMODITIES

    results = {}
    for name in commodities:
        if name not in prices.columns:
            continue
        det = fit_commodity(prices, name, n_states=n_states)
        results[name] = {
            "detector":   det,
            "regimes":    det.regime_series(),
            "summary":    det.regime_summary(),
            "transition": det.transition_matrix(),
            "palette":    PALETTE,
        }
    return results
