"""
News data service using free RSS feeds from financial publications.

RSS (Really Simple Syndication) is an open standard that news sites publish
for free — no API key required. feedparser parses these XML feeds into Python
dicts. We pull from multiple sources and merge them into a single sorted feed.
"""

import feedparser
import streamlit as st
import pandas as pd
from datetime import datetime
import time

# Free RSS feeds relevant to commodities markets
RSS_FEEDS = {
    "Reuters Commodities": "https://feeds.reuters.com/reuters/businessNews",
    "Bloomberg Markets":   "https://feeds.bloomberg.com/markets/news.rss",
    "FT Markets":          "https://www.ft.com/rss/home/uk",
    "Seeking Alpha Commodities": "https://seekingalpha.com/tag/commodities.xml",
    "Oil Price":           "https://oilprice.com/rss/main",
    "Mining.com":          "https://www.mining.com/feed/",
    "AgWeb":               "https://www.agweb.com/rss",
    "CNBC Commodities":    "https://www.cnbc.com/id/19832390/device/rss/rss.html",
}

# Keyword filters to surface commodity-relevant stories
COMMODITY_KEYWORDS = [
    "oil", "crude", "natural gas", "lng", "opec",
    "gold", "silver", "copper", "platinum", "palladium", "metals",
    "corn", "wheat", "soybeans", "coffee", "sugar", "cotton", "cocoa",
    "commodity", "commodities", "futures", "energy",
    "inflation", "fed", "interest rate", "dollar",  # macro drivers
]


@st.cache_data(ttl=600)  # Cache 10 minutes — news doesn't change that fast
def fetch_news(max_per_feed: int = 10, filter_keywords: bool = True) -> pd.DataFrame:
    """
    Fetch and merge news from all RSS sources.
    Returns DataFrame: title, summary, link, published, source.
    """
    articles = []

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title   = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "")
                link    = getattr(entry, "link", "")

                # Parse publish date — feedparser normalises to time.struct_time
                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub:
                    pub_dt = datetime(*pub[:6])
                else:
                    pub_dt = datetime.utcnow()

                # Optional: filter to commodity-relevant articles only
                if filter_keywords:
                    text = (title + " " + summary).lower()
                    if not any(kw in text for kw in COMMODITY_KEYWORDS):
                        continue

                articles.append({
                    "title":     title,
                    "summary":   summary[:300] + "..." if len(summary) > 300 else summary,
                    "link":      link,
                    "published": pub_dt,
                    "source":    source,
                })
        except Exception:
            # Silently skip feeds that are temporarily unavailable
            continue

    if not articles:
        return _mock_news()

    df = pd.DataFrame(articles)
    df = df.drop_duplicates(subset="title")
    df = df.sort_values("published", ascending=False).reset_index(drop=True)
    return df


def _mock_news() -> pd.DataFrame:
    """Fallback articles when all RSS feeds are unavailable."""
    mock = [
        {
            "title":     "OPEC+ Holds Output Cuts as Oil Demand Outlook Improves",
            "summary":   "OPEC+ members agreed to maintain current production cuts through Q3 amid signs of recovering global demand, supporting crude prices above $80/bbl.",
            "link":      "https://reuters.com",
            "published": datetime(2026, 4, 22, 9, 0),
            "source":    "Reuters (Mock)",
        },
        {
            "title":     "Gold Hits Record High on Fed Rate Cut Expectations",
            "summary":   "Gold futures surged past $2,400/oz as softer-than-expected CPI data renewed bets on Federal Reserve rate cuts in the second half of the year.",
            "link":      "https://bloomberg.com",
            "published": datetime(2026, 4, 22, 8, 30),
            "source":    "Bloomberg (Mock)",
        },
        {
            "title":     "Wheat Prices Jump on Ukraine Export Concerns",
            "summary":   "Chicago wheat futures rose 2% after renewed tensions disrupted Black Sea shipping lanes, raising concerns about global grain supply.",
            "link":      "https://ft.com",
            "published": datetime(2026, 4, 21, 14, 0),
            "source":    "FT (Mock)",
        },
    ]
    return pd.DataFrame(mock)
