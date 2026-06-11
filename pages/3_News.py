"""
News page — live commodity market news pulled from free RSS feeds.

RSS feeds are polled and filtered to commodity-relevant stories.
Each card links directly to the original article.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from services.news_data import fetch_news, RSS_FEEDS
from utils.theme import apply_theme, render_topbar, render_sidebar_nav
from models.headline_insight import HeadlineInsightModel, NEWS_INSIGHT_ENABLED

st.set_page_config(page_title="Accendio | News", page_icon="assets/accendio_icon_transparent_32.png", layout="wide")
apply_theme()
render_topbar()
render_sidebar_nav()

st.title("Market News")
st.caption("Live news from Reuters, Bloomberg, FT, CNBC, OilPrice, Mining.com & more")

# ── Feed controls ──────────────────────────────────────────────────────────────
_fc1, _fc2, _fc3 = st.columns([2, 2, 1])
with _fc1:
    filter_keywords = st.toggle(
        "Filter to commodity topics", value=True,
        help="Show only articles containing commodity-related keywords",
    )
with _fc2:
    max_per_feed = st.slider("Max articles per source", 5, 20, 10)
with _fc3:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh News", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── Load News ──────────────────────────────────────────────────────────────────
with st.spinner("Fetching news feeds..."):
    df = fetch_news(max_per_feed=max_per_feed, filter_keywords=filter_keywords)

if df.empty:
    st.warning("No articles found. Try disabling the keyword filter.")
    st.stop()

# ── Headline insight weighting (background model, flag-gated) ──────────────────
# All computation lives in models/headline_insight.py; this hook only feeds the
# corpus and renders scores. Invisible unless NEWS_INSIGHT_ENABLED=true.
if NEWS_INSIGHT_ENABLED:
    _insight_model = HeadlineInsightModel()
    _corpus_n = _insight_model.append_corpus(df)   # idempotent — brews the training corpus
    df = _insight_model.score_headlines(df)
    with st.expander("🧪 Insight weighting (experimental — internal only)"):
        st.caption(
            f"Corpus: **{_corpus_n}** headlines · scoring mode: "
            f"**{df['scoring_mode'].iloc[0]}** · weights are unverified placeholders"
        )
        st.dataframe(
            df.sort_values("insight_weight", ascending=False)[
                ["insight_weight", "title", "insight_sim", "generic_sim",
                 "commodity_specificity", "novelty", "source"]
            ],
            use_container_width=True, hide_index=True,
        )

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
