"""
Historical analog finder — "the last time we saw a setup like this, here's
what actually happened over the next N days."

Method
------
1. Build a fingerprint of the most recent ``window`` daily log returns.
2. Slide a window of the same length across history; at each past date,
   compute the Pearson correlation between that window and the fingerprint.
3. Return the top-K matches (sorted by similarity), each carrying the
   subsequent ``horizon`` days of realised log returns starting from the
   day *after* the match — that's the "what actually happened next" path.
4. Optionally exclude matches that overlap the fingerprint window.

The output is a list of price paths anchored at today's spot, ready to be
overlaid on the fan chart as a sanity check: "if the consensus says +2 %
over 10 days, do past lookalikes confirm or contradict that?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class HistoricalAnalog:
    """One match: when it occurred, how similar it was, what happened next."""

    match_end_date: pd.Timestamp
    similarity: float                    # Pearson r in [-1, 1]
    forward_log_returns: np.ndarray      # length = horizon
    forward_price_path: np.ndarray       # length = horizon + 1, anchored at spot

    @property
    def forward_total_return(self) -> float:
        return float(np.exp(np.sum(self.forward_log_returns)) - 1.0)


def find_analogs(
    log_returns: pd.Series,
    horizon: int,
    spot: float,
    window: int = 20,
    k: int = 5,
    min_lookback_days: int = 0,
) -> List[HistoricalAnalog]:
    """
    Find the top-K historical periods whose recent return pattern best
    matches the most recent ``window`` days, and return what happened in
    the ``horizon`` days that followed.

    Parameters
    ----------
    log_returns : pd.Series
        Daily log returns for the target commodity, DatetimeIndex.
    horizon : int
        Forward window to project — how many days "after the match."
    spot : float
        Today's price, used to anchor each forward path.
    window : int
        Length (in days) of the fingerprint window.
    k : int
        Number of analogs to return.
    min_lookback_days : int
        Refuse matches whose end date is within this many days of "now" —
        prevents trivially returning the present window itself.

    Returns
    -------
    list[HistoricalAnalog]
        Sorted by similarity descending. Empty if data insufficient.
    """
    s = log_returns.dropna()
    n = len(s)
    if n < window + horizon + 5:
        return []

    arr = s.to_numpy(dtype=float)
    fingerprint = arr[-window:]
    fp_mean = fingerprint.mean()
    fp_std = fingerprint.std()
    if fp_std < 1e-12:
        return []
    fp_centered = (fingerprint - fp_mean) / fp_std

    # Latest valid match end-index: needs `horizon` days after AND must end
    # at least `min_lookback_days` before "now". Latest match's forward path
    # ends exactly at len(arr)-1 (yesterday), so match_end_idx <= n-horizon-1.
    last_match_idx = n - horizon - 1 - min_lookback_days
    # Earliest match end-index: needs `window` days *before* it (inclusive).
    first_match_idx = window - 1

    if last_match_idx <= first_match_idx:
        return []

    scores = []
    for end in range(first_match_idx, last_match_idx + 1):
        w = arr[end - window + 1: end + 1]
        w_std = w.std()
        if w_std < 1e-12:
            continue
        w_centered = (w - w.mean()) / w_std
        r = float(np.dot(fp_centered, w_centered) / window)
        scores.append((end, r))

    if not scores:
        return []

    scores.sort(key=lambda kv: kv[1], reverse=True)
    top = scores[:k]

    analogs: List[HistoricalAnalog] = []
    for end_idx, sim in top:
        forward_rets = arr[end_idx + 1: end_idx + 1 + horizon]
        if len(forward_rets) < horizon:
            continue
        cum = np.exp(np.cumsum(forward_rets))
        price_path = np.concatenate([[1.0], cum]) * spot
        analogs.append(HistoricalAnalog(
            match_end_date=s.index[end_idx],
            similarity=sim,
            forward_log_returns=forward_rets,
            forward_price_path=price_path,
        ))

    return analogs
