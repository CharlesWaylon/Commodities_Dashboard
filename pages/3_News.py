"""
News page — live commodity market news pulled from free RSS feeds.

RSS feeds are polled and filtered to commodity-relevant stories.
Each card links directly to the original article.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from services.news_data import fetch_news, RSS_FEEDS

st.set_page_config(page_title="News | Commodities", page_icon="📰", layout="wide")

with st.sidebar:
    st.title("📈 Commodities Hub")
    st.caption("Future of Commodities Club")
    st.divider()
    st.markdown("""
**Navigation**
- 🏠 [Overview](/)
- 💰 [Pricing](/Pricing)
- 📊 [Charts](/Charts)
- 📰 **News** ← you are here
- 🤖 [Models](/Models)
    """)
    st.divider()
    filter_keywords = st.toggle("Filter to commodity topics", value=True,
                                help="Show only articles containing commodity-related keywords")
    max_per_feed = st.slider("Max articles per source", 5, 20, 10)
    st.divider()
    if st.button("🔄 Refresh News"):
        st.cache_data.clear()
        st.rerun()

st.title("📰 Market News")
st.caption("Live news from Reuters, Bloomberg, FT, CNBC, OilPrice, Mining.com & more")

# ── Load News ──────────────────────────────────────────────────────────────────
with st.spinner("Fetching news feeds..."):
    df = fetch_news(max_per_feed=max_per_feed, filter_keywords=filter_keywords)

if df.empty:
    st.warning("No articles found. Try disabling the keyword filter.")
    st.stop()

# ── Filter Controls ────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])
with col1:
    sources = ["All Sources"] + sorted(df["source"].unique().tolist())
    selected_source = st.selectbox("Filter by Source", sources)
with col2:
    search = st.text_input("🔍 Search headlines", placeholder="e.g. OPEC, gold, wheat...")

filtered = df.copy()
if selected_source != "All Sources":
    filtered = filtered[filtered["source"] == selected_source]
if search:
    mask = filtered["title"].str.contains(search, case=False, na=False) | \
           filtered["summary"].str.contains(search, case=False, na=False)
    filtered = filtered[mask]

st.caption(f"Showing **{len(filtered)}** articles")
st.divider()

# ── News Cards ─────────────────────────────────────────────────────────────────
if filtered.empty:
    st.info("No articles match your filters. Try broadening your search.")
else:
    for _, row in filtered.iterrows():
        with st.container():
            # Time ago
            try:
                now    = datetime.utcnow()
                pub    = row["published"]
                if pub.tzinfo is not None:
                    pub = pub.replace(tzinfo=None)
                delta  = now - pub
                if delta.days > 0:
                    age = f"{delta.days}d ago"
                elif delta.seconds > 3600:
                    age = f"{delta.seconds // 3600}h ago"
                else:
                    age = f"{delta.seconds // 60}m ago"
            except Exception:
                age = ""

            col_text, col_meta = st.columns([5, 1])
            with col_text:
                st.markdown(f"### [{row['title']}]({row['link']})")
                if row["summary"]:
                    st.markdown(f"<p style='color:#AAAAAA;font-size:0.9em'>{row['summary']}</p>",
                                unsafe_allow_html=True)
            with col_meta:
                st.markdown(f"**{row['source']}**")
                st.caption(age)

            st.divider()

# ── Sources ────────────────────────────────────────────────────────────────────
with st.expander("📡 News Sources"):
    st.markdown("The following free RSS feeds are monitored:")
    for name, url in RSS_FEEDS.items():
        st.markdown(f"- **{name}**: `{url}`")
    st.caption("To add a source, edit `services/news_data.py` → `RSS_FEEDS` dict.")
