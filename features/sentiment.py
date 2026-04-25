"""
NLP sentiment scoring and EIA inventory surprise factor.

FinBERT sentiment
-----------------
FinBERT (Araci 2019, ProsusAI/finbert on HuggingFace) is a BERT model
fine-tuned on 10,000 financial news sentences from Reuters and Bloomberg.
It outperforms general-purpose BERT on financial sentiment by ~15 F1 points.

We route each headline through FinBERT and assign it to commodities via
keyword matching. The per-commodity daily signal is:

    sentiment_score = (n_positive − n_negative) / (n_total + 1) ∈ (−1, +1)

This is the "net sentiment ratio" — a value near +1 means nearly all
headlines about this commodity were bullish; near −1 means bearish.

Performance note: FinBERT inference takes ~0.05-0.5s per headline on CPU.
For a daily batch of 50-100 headlines this is 5-50s total — acceptable for
a scheduled daily job. The module caches results to a Parquet file
(SENTIMENT_CACHE_PATH) so the dashboard reads from cache, not live inference.

EIA inventory surprise
----------------------
The EIA Weekly Petroleum Status Report is published every Wednesday at
10:30 AM ET. It contains:
  - US commercial crude oil stocks (WCRSTUS1) — the headline number
  - Cushing, OK stocks (WCSSTUS1) — WTI delivery point, most price-sensitive

The "surprise" is the observed weekly change minus the trailing 4-week
average change (consensus proxy). On report days, surprise > 0 (inventory
build) is bearish for WTI; surprise < 0 (draw) is bullish.

EIA API key: free at https://www.eia.gov/opendata/register.php
Store as EIA_API_KEY in environment or pass directly.

Usage
-----
    from features.sentiment import SentimentFeatures

    sf = SentimentFeatures(eia_key="YOUR_KEY")

    # One-shot daily update (run after market close)
    sf.update_daily(headlines_df)   # from services/news_data.py fetch_news()

    # Load cached scores
    scores = sf.load_cache()        # pd.DataFrame indexed by date

    # EIA inventory surprise
    eia = sf.eia_inventory_surprise(weeks=52)
    # DataFrame: date, crude_stocks, stocks_change, surprise, surprise_z
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime, date
from pathlib import Path
from typing import Optional

try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

# ── Configuration ──────────────────────────────────────────────────────────────
FINBERT_MODEL       = "ProsusAI/finbert"
SENTIMENT_CACHE_PATH = Path("data/sentiment_cache.parquet")
EIA_BASE_URL        = "https://api.eia.gov/v2"

# EIA series identifiers
_CRUDE_STOCKS_SERIES = "PET.WCRSTUS1.W"     # US commercial crude stocks (kb)
_CUSHING_STOCKS_SERIES = "PET.WCSSTUS1.W"   # Cushing, OK stocks (kb)

# Consensus proxy: 4-week trailing mean of weekly changes
_EIA_CONSENSUS_WINDOW = 4

# ── Commodity keyword routing ──────────────────────────────────────────────────
COMMODITY_KEYWORDS = {
    "WTI Crude Oil":          ["crude", "wti", "oil price", "brent", "opec", "petroleum"],
    "Natural Gas (Henry Hub)":["natural gas", "lng", "henry hub", "gas storage", "natgas"],
    "Gold (COMEX)":           ["gold", "bullion", "precious metal", "safe haven", "xau"],
    "Silver (COMEX)":         ["silver", "xag", "precious"],
    "Copper (COMEX)":         ["copper", "base metal", "dr copper"],
    "Corn (CBOT)":            ["corn", "maize", "ethanol", "grains", "usda"],
    "Wheat (CBOT SRW)":       ["wheat", "grain", "black sea", "ukraine export"],
    "Soybeans (CBOT)":        ["soybean", "soy", "china import", "oilseed"],
    "Live Cattle":            ["cattle", "beef", "livestock", "herd"],
    "Uranium*":               ["uranium", "nuclear", "u3o8", "enrichment"],
    "Lithium*":               ["lithium", "ev battery", "battery metal", "electric vehicle"],
    "Carbon Credits*":        ["carbon credit", "ets", "emission", "co2", "eua"],
    "Coffee (Arabica)":       ["coffee", "arabica", "brazil crop", "colombia"],
    "Sugar (Raw #11)":        ["sugar", "ethanol", "brazil sugar", "sweetener"],
    "Cocoa (ICE)":            ["cocoa", "chocolate", "ivory coast", "ghana crop"],
}


def _route_headline(text: str) -> list:
    """Return list of commodity names whose keywords appear in `text`."""
    text_lower = text.lower()
    matched = []
    for commodity, keywords in COMMODITY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(commodity)
    return matched


# ── FinBERT inference ──────────────────────────────────────────────────────────

def _load_finbert():
    """Lazily load FinBERT pipeline. Raises ImportError if transformers absent."""
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers is required for FinBERT sentiment. "
            "Install with: pip install transformers"
        )
    return hf_pipeline(
        "text-classification",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        truncation=True,
        max_length=512,
    )


def score_headlines(
    headlines: list,
    finbert=None,
) -> list:
    """
    Run FinBERT on a list of headline strings.

    Parameters
    ----------
    headlines : list[str]
    finbert : pipeline, optional
        Pre-loaded FinBERT pipeline. If None, loads it (takes ~30s first time).

    Returns
    -------
    list[dict]
        Each dict: {text, label ("positive"/"negative"/"neutral"), score (confidence)}.
    """
    if finbert is None:
        finbert = _load_finbert()

    results = []
    for text in headlines:
        try:
            out = finbert(text[:512])[0]
            results.append({
                "text":  text,
                "label": out["label"].lower(),
                "score": float(out["score"]),
            })
        except Exception:
            results.append({"text": text, "label": "neutral", "score": 0.5})

    return results


# ── Daily sentiment aggregation ───────────────────────────────────────────────

def aggregate_daily_sentiment(
    headlines_df: pd.DataFrame,
    date_col:    str = "published",
    text_col:    str = "title",
    finbert=None,
) -> pd.DataFrame:
    """
    Compute per-commodity per-day net sentiment ratio from a headlines DataFrame.

    Parameters
    ----------
    headlines_df : pd.DataFrame
        Must contain `date_col` (datetime) and `text_col` (headline string).
        Typically from services.news_data.fetch_news().

    Returns
    -------
    pd.DataFrame
        Index = date (day-level). Columns = commodity display names.
        Values = net sentiment ratio ∈ (−1, +1). NaN = no headlines that day.
    """
    if headlines_df.empty:
        return pd.DataFrame()

    if finbert is None:
        try:
            finbert = _load_finbert()
        except ImportError as e:
            warnings.warn(str(e))
            return pd.DataFrame()

    df = headlines_df.copy()
    df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
    df["text"] = df[text_col].fillna("").astype(str)

    # Score all headlines in one batch
    texts = df["text"].tolist()
    scored = score_headlines(texts, finbert=finbert)
    df["label"] = [s["label"] for s in scored]
    df["commodities"] = df["text"].apply(_route_headline)

    # Explode commodities column
    df_exploded = df.explode("commodities").dropna(subset=["commodities"])

    all_commodities = list(COMMODITY_KEYWORDS.keys())
    all_dates = sorted(df["date"].unique())

    sentiment_matrix = pd.DataFrame(np.nan, index=all_dates, columns=all_commodities)
    sentiment_matrix.index = pd.DatetimeIndex(sentiment_matrix.index)

    for (d, c), group in df_exploded.groupby(["date", "commodities"]):
        if c not in all_commodities:
            continue
        n_pos = (group["label"] == "positive").sum()
        n_neg = (group["label"] == "negative").sum()
        n_tot = len(group)
        ratio = (n_pos - n_neg) / (n_tot + 1)
        sentiment_matrix.loc[d, c] = ratio

    # Rename columns for feature matrix compatibility
    sentiment_matrix.columns = [
        f"sentiment_{c.split('(')[0].strip().replace(' ', '_').lower()}"
        for c in sentiment_matrix.columns
    ]
    return sentiment_matrix


# ── Sentiment cache ────────────────────────────────────────────────────────────

class SentimentCache:
    """
    Simple Parquet-backed cache for daily sentiment scores.

    Avoids re-running FinBERT inference on historical headlines. The cache
    is append-only: each call to `update()` scores only new dates.
    """

    def __init__(self, cache_path: Path = SENTIMENT_CACHE_PATH):
        self.cache_path = cache_path
        self._finbert = None

    def _get_finbert(self):
        if self._finbert is None:
            self._finbert = _load_finbert()
        return self._finbert

    def load(self) -> pd.DataFrame:
        """Load all cached sentiment scores."""
        if self.cache_path.exists():
            return pd.read_parquet(self.cache_path)
        return pd.DataFrame()

    def update(self, headlines_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score any dates in headlines_df not already in the cache, append, save.

        Returns the full updated cache DataFrame.
        """
        existing = self.load()

        if not existing.empty:
            existing_dates = set(existing.index.normalize())
            new_headlines = headlines_df[
                ~pd.to_datetime(headlines_df["published"]).dt.normalize().isin(existing_dates)
            ]
        else:
            new_headlines = headlines_df

        if new_headlines.empty:
            return existing

        try:
            fb = self._get_finbert()
            new_scores = aggregate_daily_sentiment(new_headlines, finbert=fb)
        except Exception as e:
            warnings.warn(f"FinBERT scoring failed: {e}")
            return existing

        if new_scores.empty:
            return existing

        combined = pd.concat([existing, new_scores]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.cache_path)
        return combined

    def rolling_sentiment(self, window: int = 5) -> pd.DataFrame:
        """
        Return 5-day rolling mean sentiment — smooths single-day noise.
        """
        cached = self.load()
        if cached.empty:
            return pd.DataFrame()
        return cached.rolling(window, min_periods=1).mean()


# ── EIA inventory surprise ────────────────────────────────────────────────────

def fetch_eia_crude_stocks(
    eia_key: str,
    weeks: int = 52,
) -> pd.DataFrame:
    """
    Fetch US commercial crude oil inventory data from the EIA API v2.

    Parameters
    ----------
    eia_key : str
        EIA API key (register free at https://www.eia.gov/opendata/register.php).
    weeks : int
        How many weekly observations to retrieve.

    Returns
    -------
    pd.DataFrame
        Columns: crude_stocks (kb), stocks_change, surprise, surprise_z.
        DatetimeIndex at weekly frequency.
    """
    url = f"{EIA_BASE_URL}/petroleum/stoc/wstk/data/"
    params = {
        "api_key":              eia_key,
        "frequency":            "weekly",
        "data[0]":              "value",
        "facets[series][]":     "WCRSTUS1",
        "sort[0][column]":      "period",
        "sort[0][direction]":   "desc",
        "length":               weeks,
        "offset":               0,
    }

    try:
        resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()["response"]["data"]
    except Exception as e:
        warnings.warn(f"EIA crude stocks fetch failed: {e}")
        return pd.DataFrame()

    if not data:
        warnings.warn("EIA API returned no data.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"]        = pd.to_datetime(df["period"])
    df["crude_stocks"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date").sort_index()[["crude_stocks"]]

    df["stocks_change"] = df["crude_stocks"].diff()
    # Consensus proxy: 4-week trailing mean of prior changes
    consensus = df["stocks_change"].shift(1).rolling(_EIA_CONSENSUS_WINDOW).mean()
    df["surprise"] = df["stocks_change"] - consensus

    mu  = df["surprise"].rolling(26).mean()
    sig = df["surprise"].rolling(26).std().clip(lower=1e-9)
    df["surprise_z"] = (df["surprise"] - mu) / sig

    return df


class SentimentFeatures:
    """
    Unified interface for FinBERT sentiment and EIA inventory surprise.

    Parameters
    ----------
    eia_key : str, optional
        EIA API key. Falls back to EIA_API_KEY environment variable.
    cache_path : Path, optional
        Override the default Parquet cache location.
    """

    def __init__(
        self,
        eia_key: Optional[str] = None,
        cache_path: Path = SENTIMENT_CACHE_PATH,
    ):
        self._eia_key = eia_key or os.environ.get("EIA_API_KEY")
        self._cache   = SentimentCache(cache_path)

    def update_sentiment(self, headlines_df: pd.DataFrame) -> pd.DataFrame:
        """Score new headlines and update the cache. Returns full cache."""
        return self._cache.update(headlines_df)

    def load_sentiment(self, rolling_window: int = 5) -> pd.DataFrame:
        """Load cached sentiment, optionally smoothed."""
        if rolling_window > 1:
            return self._cache.rolling_sentiment(rolling_window)
        return self._cache.load()

    def eia_inventory_surprise(self, weeks: int = 52) -> pd.DataFrame:
        """Fetch EIA crude stocks and compute the inventory surprise factor."""
        if not self._eia_key:
            warnings.warn(
                "No EIA_API_KEY found. EIA inventory surprise unavailable. "
                "Register free at https://www.eia.gov/opendata/register.php"
            )
            return pd.DataFrame()
        return fetch_eia_crude_stocks(self._eia_key, weeks=weeks)

    def build_all(
        self,
        headlines_df: Optional[pd.DataFrame] = None,
        rolling_window: int = 5,
        eia_weeks: int = 52,
    ) -> pd.DataFrame:
        """
        Assemble sentiment and EIA surprise into one aligned daily DataFrame.

        Parameters
        ----------
        headlines_df : pd.DataFrame, optional
            New headlines to score before loading. If None, uses cache only.

        Returns
        -------
        pd.DataFrame
            All sentiment columns + eia_surprise + eia_surprise_z on a
            daily DatetimeIndex (EIA data forward-filled from weekly).
        """
        if headlines_df is not None and not headlines_df.empty:
            self.update_sentiment(headlines_df)

        sentiment = self.load_sentiment(rolling_window=rolling_window)
        eia_df    = self.eia_inventory_surprise(weeks=eia_weeks)

        parts = []
        if not sentiment.empty:
            parts.append(sentiment)

        if not eia_df.empty:
            # Resample weekly EIA data to daily (forward fill through week)
            eia_daily = eia_df[["surprise", "surprise_z"]].rename(columns={
                "surprise":   "eia_crude_surprise",
                "surprise_z": "eia_crude_surprise_z",
            })
            daily_idx = pd.date_range(eia_daily.index.min(), eia_daily.index.max(), freq="B")
            eia_daily = eia_daily.reindex(daily_idx).ffill(limit=5)
            parts.append(eia_daily)

        if not parts:
            return pd.DataFrame()

        combined = parts[0]
        for df in parts[1:]:
            combined = combined.join(df, how="outer")

        return combined.sort_index()
