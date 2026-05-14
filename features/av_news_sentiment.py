"""
Alpha Vantage News Sentiment

Fetches pre-computed, relevance-weighted per-ticker sentiment scores from the
AV NEWS_SENTIMENT endpoint and maps them back to commodity names via ETF /
equity proxy tickers.

Why AV over FinBERT alone
--------------------------
FinBERT scores raw RSS headlines from a narrow source set (Reuters, FT, CNBC).
AV NEWS_SENTIMENT covers Bloomberg, Reuters Pro, WSJ, Seeking Alpha, and ~50
other sources not publicly available via RSS — and delivers relevance_score per
ticker so we can weight each article's signal quality.  The two sources are
complementary: AV is broader; FinBERT is fully on-premise and free to run.

Proxy ticker logic
------------------
AV NEWS_SENTIMENT requires stock/ETF tickers (not futures codes).  Each
commodity is mapped to 1–3 thematically pure proxies (e.g. Gold → GLD, GDX,
NEM) whose news coverage closely tracks commodity fundamentals.  For each
article the API returns a `ticker_sentiment_score` ∈ [−1, +1] and a
`relevance_score` ∈ [0, 1] per matched ticker.  We compute a relevance-weighted
average across all proxy tickers for a commodity to get the commodity score.

Rate limits — free tier: 25 req/day, 5 req/min
  All proxy tickers are batched into ONE API call per daily update.
  A Parquet cache prevents re-fetching data already stored for the day.

Output columns
--------------
  av_sentiment_<commodity_slug>   — weighted daily score ∈ [−1, +1]

These are designed to sit alongside FinBERT's `sentiment_<slug>` columns in
the feature matrix.  SentimentFeatures.build_all() combines both into
`combined_sentiment_<slug>` ensemble columns.

Usage
-----
    from features.av_news_sentiment import AVSentimentClient

    client = AVSentimentClient()
    df = client.update()        # fetch today, merge cache, return full history
    df = client.load()          # return cached (no API call)
    df = client.backfill(30)    # walk back 30 days (uses ~5 API calls)
"""

import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger(__name__)

AV_BASE_URL           = "https://www.alphavantage.co/query"
AV_SENTIMENT_CACHE    = Path("data/av_sentiment_cache.parquet")
AV_RATE_LIMIT_DELAY   = 13.0    # seconds between calls — free tier: 5/min
AV_MIN_RELEVANCE      = 0.10    # skip ticker_sentiment entries below this


# ── Commodity → proxy ticker map ──────────────────────────────────────────────

AV_COMMODITY_TICKERS: Dict[str, List[str]] = {
    # Energy
    "WTI Crude Oil":           ["USO", "XOM", "CVX"],
    "Brent Crude Oil":         ["XOM", "CVX", "BP"],
    "Natural Gas (Henry Hub)": ["UNG", "LNG"],
    "RBOB Gasoline":           ["VLO", "PSX"],
    "Heating Oil":             ["VLO", "XOM"],
    "Carbon Credits*":         ["KRBN"],
    "LNG*":                    ["LNG", "GLNG"],
    "Uranium*":                ["URA", "CCJ"],
    "Coal*":                   ["ARCH", "BTU"],
    # Metals
    "Gold (COMEX)":            ["GLD", "GDX", "NEM"],
    "Silver (COMEX)":          ["SLV", "PAAS"],
    "Copper (COMEX)":          ["FCX", "COPX"],
    "Platinum*":               ["PPLT"],
    "Palladium*":              ["PALL"],
    "Aluminum*":               ["AA", "CENX"],
    "Steel*":                  ["X", "CLF", "STLD"],
    "Lithium*":                ["LIT", "ALB", "SQM"],
    "Rare Earths*":            ["MP", "REMX"],
    "Zinc*":                   ["FCX"],
    # Agriculture
    "Corn (CBOT)":             ["CORN", "ADM"],
    "Wheat (CBOT SRW)":        ["WEAT", "ADM"],
    "Soybeans (CBOT)":         ["SOYB", "ADM"],
    "Coffee (Arabica)":        ["JO"],
    "Cocoa (ICE)":             ["NIB"],
    "Sugar (Raw #11)":         ["SGG"],
    "Cotton*":                 ["BAL", "MOO"],
    "Orange Juice*":           ["FDP", "MOO"],
    "Oats*":                   ["ADM", "MOO"],
    "Rice (Rough)*":           ["ADM", "MOO"],
    "Lumber*":                 ["WOOD", "WY"],
    # Livestock
    "Live Cattle":             ["COW", "TSN"],
    "Feeder Cattle":           ["COW", "TSN"],
    "Lean Hogs":               ["COW", "TSN", "HRL"],
    # Digital
    "Bitcoin*":                ["COIN", "MSTR", "GBTC"],
}

# Reverse map: proxy ticker → list of commodity names
_TICKER_TO_COMMODITIES: Dict[str, List[str]] = {}
for _comm, _tickers in AV_COMMODITY_TICKERS.items():
    for _t in _tickers:
        _TICKER_TO_COMMODITIES.setdefault(_t, []).append(_comm)

ALL_PROXY_TICKERS: List[str] = sorted(set(
    t for tickers in AV_COMMODITY_TICKERS.values() for t in tickers
))


# ── Slug helper ───────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    """Commodity display name → snake_case column suffix."""
    return (
        name.replace("*", "").replace("(", "").replace(")", "")
        .replace(" ", "_").strip("_").lower()
    )


# ── API fetch ─────────────────────────────────────────────────────────────────

def fetch_av_sentiment(
    api_key: str,
    tickers: Optional[List[str]] = None,
    time_from: Optional[str] = None,
    time_to:   Optional[str] = None,
    limit:     int = 200,
) -> List[dict]:
    """
    Single call to AV NEWS_SENTIMENT.  Returns the raw 'feed' list.

    Parameters
    ----------
    api_key   : AV API key (from ALPHA_VANTAGE_KEY env var)
    tickers   : proxy tickers to request.  Defaults to ALL_PROXY_TICKERS.
    time_from : YYYYMMDDTHHMM.  Defaults to 24 h ago.
    time_to   : YYYYMMDDTHHMM.  Defaults to now.
    limit     : max articles returned (200 gives good coverage on free tier)
    """
    if not api_key:
        return []

    if time_from is None:
        time_from = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%dT%H%M")
    if time_to is None:
        time_to = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")

    tickers_str = ",".join(tickers or ALL_PROXY_TICKERS)

    try:
        resp = requests.get(
            AV_BASE_URL,
            params={
                "function":  "NEWS_SENTIMENT",
                "tickers":   tickers_str,
                "time_from": time_from,
                "time_to":   time_to,
                "sort":      "RELEVANCE",
                "limit":     limit,
                "apikey":    api_key,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        # AV returns an error message in the JSON body (not HTTP 4xx) on bad key
        if "Note" in data or "Information" in data:
            msg = data.get("Note") or data.get("Information", "")
            log.warning("AV NEWS_SENTIMENT API message: %s", msg)
            return []

        articles = data.get("feed", [])
        log.info("AV NEWS_SENTIMENT: %d articles fetched (%s → %s)",
                 len(articles), time_from[:8], time_to[:8])
        return articles

    except Exception as exc:
        log.warning("fetch_av_sentiment failed: %s", exc)
        return []


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_to_commodity(articles: List[dict]) -> pd.DataFrame:
    """
    Map raw AV articles to per-commodity per-day sentiment scores.

    Scoring
    -------
    For each article → each (ticker, commodity) pair:
        contribution = ticker_sentiment_score × relevance_score

    Per commodity per day:
        score = Σ(contributions) / Σ(relevance_scores)   [relevance-weighted avg]

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (day-level).
        Columns: av_sentiment_<commodity_slug>.
        Values ∈ [−1, +1].
    """
    if not articles:
        return pd.DataFrame()

    # {date → {commodity → [sum_weighted, sum_relevance]}}
    day_commodity: dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))

    for article in articles:
        time_pub = article.get("time_published", "")
        try:
            dt = datetime.strptime(time_pub[:8], "%Y%m%d").date()
        except (ValueError, IndexError):
            continue

        for ts in article.get("ticker_sentiment", []):
            ticker = ts.get("ticker", "")
            if ticker not in _TICKER_TO_COMMODITIES:
                continue
            try:
                sent  = float(ts.get("ticker_sentiment_score", 0))
                relev = float(ts.get("relevance_score", 0))
            except (TypeError, ValueError):
                continue

            if relev < AV_MIN_RELEVANCE:
                continue

            for commodity in _TICKER_TO_COMMODITIES[ticker]:
                day_commodity[dt][commodity][0] += sent * relev
                day_commodity[dt][commodity][1] += relev

    if not day_commodity:
        return pd.DataFrame()

    dates = sorted(day_commodity.keys())
    all_commodities = list(AV_COMMODITY_TICKERS.keys())
    matrix: Dict[str, dict] = {}

    for commodity in all_commodities:
        col = f"av_sentiment_{_slug(commodity)}"
        vals: dict = {}
        for d in dates:
            acc = day_commodity[d].get(commodity)
            if acc and acc[1] > 0:
                vals[d] = acc[0] / acc[1]
        if vals:
            matrix[col] = vals

    if not matrix:
        return pd.DataFrame()

    df = pd.DataFrame(matrix)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index().clip(-1.0, 1.0)


# ── Cache ─────────────────────────────────────────────────────────────────────

class AVSentimentCache:
    """Parquet-backed store for AV sentiment scores."""

    def __init__(self, path: Path = AV_SENTIMENT_CACHE):
        self.path = path

    def load(self) -> pd.DataFrame:
        if self.path.exists():
            try:
                return pd.read_parquet(self.path)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def save(self, df: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.path)

    def is_current(self) -> bool:
        """Return True if cache already contains today's or yesterday's data."""
        cached = self.load()
        if cached.empty:
            return False
        today     = pd.Timestamp.today().normalize()
        yesterday = today - pd.Timedelta("1D")
        return bool((cached.index >= yesterday).any())

    def merge(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """Append new_df to existing cache, deduplicate by date, and save."""
        existing = self.load()
        if existing.empty:
            combined = new_df
        else:
            # Align columns — new batches may cover different instruments
            combined = pd.concat([existing, new_df]).sort_index()
            combined = combined.groupby(level=0).last()  # keep latest values per date
        self.save(combined)
        return combined


# ── Client ────────────────────────────────────────────────────────────────────

class AVSentimentClient:
    """
    Orchestrates AV NEWS_SENTIMENT fetches with caching and rate-limiting.

    Designed for free-tier constraints (25 req/day, 5 req/min).
    All proxy tickers are batched into a single API call per update.
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        cache_path: Path = AV_SENTIMENT_CACHE,
    ):
        self._key   = api_key or os.getenv("ALPHA_VANTAGE_KEY", "")
        self._cache = AVSentimentCache(cache_path)

    def update(self, days_back: int = 2, force: bool = False) -> pd.DataFrame:
        """
        Fetch recent articles, aggregate to commodities, merge into cache.

        Parameters
        ----------
        days_back : int
            Calendar days of history to fetch (default 2 covers today + yesterday).
        force : bool
            Fetch even if today's data is already cached.

        Returns
        -------
        pd.DataFrame — full updated cache (all dates).
        """
        if not force and self._cache.is_current():
            log.info("AVSentimentClient: cache is current — skipping API call")
            return self._cache.load()

        if not self._key:
            log.warning("AVSentimentClient: ALPHA_VANTAGE_KEY not set")
            return self._cache.load()

        now       = datetime.now(timezone.utc)
        time_from = (now - timedelta(days=days_back)).strftime("%Y%m%dT%H%M")
        time_to   = now.strftime("%Y%m%dT%H%M")

        articles = fetch_av_sentiment(
            self._key,
            tickers=ALL_PROXY_TICKERS,
            time_from=time_from,
            time_to=time_to,
            limit=200,
        )

        if not articles:
            return self._cache.load()

        new_df = aggregate_to_commodity(articles)
        if new_df.empty:
            return self._cache.load()

        return self._cache.merge(new_df)

    def load(self, rolling_window: int = 3) -> pd.DataFrame:
        """
        Load cached sentiment scores.

        Parameters
        ----------
        rolling_window : int
            Days to smooth (default 3).  Set to 1 for raw daily values.
        """
        cached = self._cache.load()
        if cached.empty or rolling_window <= 1:
            return cached
        return cached.rolling(rolling_window, min_periods=1).mean()

    def backfill(self, days: int = 30) -> pd.DataFrame:
        """
        Walk backward in 7-day windows to build historical sentiment.

        Each window = 1 API call.  With 25 req/day limit, cap at ~20 windows
        (140 days) spread over multiple days to avoid exhausting quota.

        Parameters
        ----------
        days : int
            Total calendar days to backfill.
        """
        if not self._key:
            log.warning("AVSentimentClient: ALPHA_VANTAGE_KEY not set for backfill")
            return self._cache.load()

        n_windows = max(1, days // 7)
        log.info("AVSentimentClient: backfilling %d days (%d 7-day windows)", days, n_windows)

        cursor = datetime.now(timezone.utc)
        for i in range(n_windows):
            t_from = (cursor - timedelta(days=7)).strftime("%Y%m%dT%H%M")
            t_to   = cursor.strftime("%Y%m%dT%H%M")

            articles = fetch_av_sentiment(
                self._key,
                tickers=ALL_PROXY_TICKERS,
                time_from=t_from,
                time_to=t_to,
                limit=200,
            )
            if articles:
                new_df = aggregate_to_commodity(articles)
                if not new_df.empty:
                    self._cache.merge(new_df)
                    log.info("Backfill window %d/%d: %s→%s (%d days)",
                             i + 1, n_windows, t_from[:8], t_to[:8], len(new_df))

            cursor -= timedelta(days=7)
            if i < n_windows - 1:
                time.sleep(AV_RATE_LIMIT_DELAY)

        return self._cache.load()


# ── Ensemble helper ───────────────────────────────────────────────────────────

def combine_with_finbert(
    av_df: pd.DataFrame,
    finbert_df: pd.DataFrame,
    weight_av: float = 0.5,
) -> pd.DataFrame:
    """
    Produce ensemble `combined_sentiment_<slug>` columns by averaging AV and
    FinBERT scores where both are available, falling back to whichever exists.

    Parameters
    ----------
    av_df      : output of AVSentimentClient.load()  — columns: av_sentiment_*
    finbert_df : output of SentimentCache.load()     — columns: sentiment_*
    weight_av  : weight on AV score (FinBERT gets 1 − weight_av)

    Returns
    -------
    pd.DataFrame with columns: combined_sentiment_<slug>
    """
    if av_df.empty and finbert_df.empty:
        return pd.DataFrame()

    # Align to the same date union
    combined_index = (
        av_df.index.union(finbert_df.index)
        if not av_df.empty and not finbert_df.empty
        else (av_df.index if not av_df.empty else finbert_df.index)
    )

    result: dict = {}

    for commodity in AV_COMMODITY_TICKERS:
        sl = _slug(commodity)
        av_col  = f"av_sentiment_{sl}"
        fb_col  = f"sentiment_{sl}"
        out_col = f"combined_sentiment_{sl}"

        av_series  = av_df[av_col].reindex(combined_index)  if (not av_df.empty  and av_col  in av_df.columns)  else pd.Series(np.nan, index=combined_index)
        fb_series  = finbert_df[fb_col].reindex(combined_index) if (not finbert_df.empty and fb_col in finbert_df.columns) else pd.Series(np.nan, index=combined_index)

        has_av = av_series.notna()
        has_fb = fb_series.notna()
        both   = has_av & has_fb

        out = pd.Series(np.nan, index=combined_index)
        out[both]            = weight_av * av_series[both] + (1 - weight_av) * fb_series[both]
        out[has_av & ~has_fb] = av_series[has_av & ~has_fb]
        out[has_fb & ~has_av] = fb_series[has_fb & ~has_av]

        if out.notna().any():
            result[out_col] = out

    if not result:
        return pd.DataFrame()
    return pd.DataFrame(result).sort_index()
