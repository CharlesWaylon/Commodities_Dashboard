"""
Headline Insight Weighting — Word2Vec-based "signal vs. noise" scorer for news.

Goal
----
Most commodity headlines are generic market wrap ("Oil edges higher as traders
weigh data") that confuses more than it informs. A small minority connect a
*specific* commodity market to a *real-world sociopolitical or physical shift*
(sanctions, export bans, strikes, droughts, coups, infrastructure failures).
This model assigns every headline an **insight weight ∈ [0, 1]**: how much of a
genuine heads-up it gives an investor, independent of bullish/bearish tone
(that is FinBERT / AV sentiment's job — see features/sentiment.py).

Status: SKELETON / BACKGROUND-ONLY (2026-06-11)
-----------------------------------------------
Gated behind NEWS_INSIGHT_ENABLED (default false). Not surfaced publicly.
Component weights and anchor lexicons are first-pass placeholders and are
**not yet verified against outside sources** — see MODEL_VERIFICATION_LOG.md.
Do not promote to a visible dashboard surface until the verification rule
has been satisfied and the scorer beats the heuristic baseline on a labelled
holdout of historical headlines.

Architecture (per the layered-architecture rule)
------------------------------------------------
- This module owns ALL computation. pages/3_News.py only calls
  `score_headlines(df)` behind the flag and renders the returned column.
- Corpus: every fetch_news() batch can be appended via `append_corpus(df)`;
  headlines accumulate in a Parquet file so Word2Vec has something to train
  on after a few weeks of brewing.
- Word2Vec model is (re)trained by `train()` (intended to be called from
  models/daily_retrain.py once the corpus is large enough) and persisted to
  MODEL_PATH. Scoring loads the persisted model lazily.

Scoring decomposition
---------------------
    weight = clip( W_INSIGHT  * insight_sim        # near sociopolitical/supply-shock anchors
                 - W_GENERIC  * generic_sim        # near market-wrap boilerplate anchors
                 + W_SPECIFIC * commodity_specificity   # names an actual market
                 + W_NOVELTY  * novelty,           # not redundant with the rest of the batch
                 0, 1)

- insight_sim / generic_sim: cosine similarity between the headline's mean
  word vector and the centroid of each anchor lexicon, in the trained
  Word2Vec space. Word2Vec generalises beyond the literal anchor words —
  e.g. "blockade" scores high even if only "embargo" is in the lexicon,
  once the corpus is big enough to place them nearby.
- commodity_specificity: reuses features.sentiment.COMMODITY_KEYWORDS routing.
- novelty: 1 − max cosine similarity to any other headline in the batch
  (penalises ten outlets rewriting the same wire story).

Fallback behaviour
------------------
If gensim is unavailable or no trained model exists yet (cold start), the
scorer degrades to a pure lexicon-overlap heuristic using the same anchors,
so the flag can be flipped on at any time without crashing the page.

Usage
-----
    from models.headline_insight import HeadlineInsightModel, NEWS_INSIGHT_ENABLED

    model = HeadlineInsightModel()
    model.append_corpus(df)          # df from services.news_data.fetch_news()
    model.train()                    # no-op until corpus >= MIN_CORPUS_SIZE
    scored = model.score_headlines(df)   # adds insight_weight + components
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from features.sentiment import COMMODITY_KEYWORDS

logger = logging.getLogger(__name__)

# ── Feature flag (default OFF — background brew only) ─────────────────────────
NEWS_INSIGHT_ENABLED = os.environ.get("NEWS_INSIGHT_ENABLED", "false").lower() in (
    "1", "true", "yes",
)

# ── Persistence paths ──────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_PATH = _DATA_DIR / "headline_corpus.parquet"
MODEL_PATH = _DATA_DIR / "headline_w2v.model"

# ── Anchor lexicons ─────────────────────────────────────────────────────────────
# PLACEHOLDER lexicons — to be expanded/verified against event-study literature
# (e.g. which event classes actually move commodity prices) before any public
# surface uses them. Word2Vec generalises beyond these seeds once trained.
INSIGHT_ANCHORS = [
    # Sociopolitical / geopolitical shifts
    "sanctions", "embargo", "tariff", "war", "invasion", "coup", "election",
    "blockade", "ceasefire", "nationalize", "expropriation", "unrest",
    # Supply-side physical shocks
    "strike", "outage", "shutdown", "drought", "flood", "frost", "hurricane",
    "pipeline", "refinery", "harvest", "disease", "shortage", "disruption",
    # Policy / structural regime changes
    "ban", "quota", "subsidy", "opec", "stockpile", "reserve", "mandate",
    "export", "restriction", "curb",
]
GENERIC_ANCHORS = [
    # Market-wrap boilerplate that carries no forward-looking information
    "edges", "ticks", "wavers", "steadies", "mixed", "wrap", "recap",
    "weekly", "outlook", "watch", "ahead", "session", "settles", "hovers",
    "investors", "traders", "weigh", "eye", "await", "digest", "muted",
]

# ── Scoring component weights ───────────────────────────────────────────────────
# PLACEHOLDERS pending verification + tuning on a labelled headline set.
W_INSIGHT = 0.45
W_GENERIC = 0.30
W_SPECIFIC = 0.20
W_NOVELTY = 0.15

# ── Word2Vec hyperparameters ───────────────────────────────────────────────────
W2V_PARAMS = dict(
    vector_size=100,
    window=5,
    min_count=3,        # headline corpora are small; drop hapaxes only
    sg=1,               # skip-gram — better for rare, information-bearing words
    epochs=15,
    workers=2,
    seed=42,
)
MIN_CORPUS_SIZE = 500   # headlines required before first training run

_TOKEN_RE = re.compile(r"[a-z][a-z\-']+")
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have in is it its of on or "
    "say says said that the to was were will with would".split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, drop stopwords. Shared by train + score."""
    if not isinstance(text, str):
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class HeadlineInsightModel:
    """Word2Vec-based insight scorer with a lexicon-heuristic cold-start fallback."""

    def __init__(self, model_path: Path = MODEL_PATH, corpus_path: Path = CORPUS_PATH):
        self.model_path = Path(model_path)
        self.corpus_path = Path(corpus_path)
        self._w2v = None          # lazy-loaded gensim Word2Vec
        self._w2v_failed = False  # remember import/load failures, don't retry per call

    # ── Corpus management ──────────────────────────────────────────────────────

    def append_corpus(self, df: pd.DataFrame) -> int:
        """Append a fetch_news() batch to the persistent corpus, dedup by title.

        Returns the total corpus size after the append. Safe to call on every
        page load / ingest run — idempotent for already-seen titles.
        """
        if df is None or df.empty or "title" not in df.columns:
            return self.corpus_size()
        batch = df[["title"]].copy()
        batch["summary"] = df.get("summary", "")
        batch["source"] = df.get("source", "")
        batch["seen_at"] = datetime.now(timezone.utc).isoformat()

        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        if self.corpus_path.exists():
            corpus = pd.read_parquet(self.corpus_path)
            corpus = pd.concat([corpus, batch], ignore_index=True)
        else:
            corpus = batch
        corpus = corpus.drop_duplicates(subset="title", keep="first")
        corpus.to_parquet(self.corpus_path, index=False)
        return len(corpus)

    def corpus_size(self) -> int:
        if not self.corpus_path.exists():
            return 0
        return len(pd.read_parquet(self.corpus_path))

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, force: bool = False) -> bool:
        """Train Word2Vec on the accumulated corpus and persist it.

        No-op (returns False) when gensim is unavailable or the corpus is still
        below MIN_CORPUS_SIZE, unless force=True. Intended to be called from
        models/daily_retrain.py so the model keeps brewing unattended.
        """
        n = self.corpus_size()
        if n < MIN_CORPUS_SIZE and not force:
            logger.info("headline_insight: corpus %d < %d, skipping train", n, MIN_CORPUS_SIZE)
            return False
        try:
            from gensim.models import Word2Vec
        except ImportError as exc:
            logger.warning("headline_insight: gensim unavailable, cannot train (%s)", exc)
            return False

        corpus = pd.read_parquet(self.corpus_path)
        sentences = [
            tokenize(f"{row.title} {row.summary}")
            for row in corpus.itertuples()
        ]
        sentences = [s for s in sentences if len(s) >= 2]
        if not sentences:
            return False

        w2v = Word2Vec(sentences=sentences, **W2V_PARAMS)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        w2v.save(str(self.model_path))
        self._w2v = w2v
        logger.info(
            "headline_insight: trained on %d headlines, vocab=%d",
            len(sentences), len(w2v.wv),
        )
        return True

    def _load_w2v(self):
        """Lazy-load the persisted Word2Vec model; cache failures."""
        if self._w2v is not None or self._w2v_failed:
            return self._w2v
        try:
            from gensim.models import Word2Vec
            if self.model_path.exists():
                self._w2v = Word2Vec.load(str(self.model_path))
            else:
                self._w2v_failed = True
        except Exception as exc:
            logger.warning("headline_insight: Word2Vec load failed (%s); using heuristic", exc)
            self._w2v_failed = True
        return self._w2v

    # ── Vector helpers ─────────────────────────────────────────────────────────

    def _mean_vector(self, tokens: List[str]) -> Optional[np.ndarray]:
        """Mean of in-vocabulary word vectors; None if nothing is in-vocab."""
        w2v = self._load_w2v()
        if w2v is None:
            return None
        vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
        if not vecs:
            return None
        return np.mean(vecs, axis=0)

    @staticmethod
    def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 0.0
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # ── Component scores ───────────────────────────────────────────────────────

    def _anchor_similarity(self, tokens: List[str], anchors: List[str]) -> float:
        """Semantic proximity to an anchor lexicon ∈ [0, 1].

        Word2Vec path: cosine(headline centroid, anchor centroid), rescaled
        from [−1, 1] to [0, 1]. Heuristic fallback: lexical overlap fraction.
        """
        vec = self._mean_vector(tokens)
        if vec is not None:
            anchor_vec = self._mean_vector(anchors)
            if anchor_vec is not None:
                return (self._cosine(vec, anchor_vec) + 1.0) / 2.0
        # Cold-start / no-gensim fallback: fraction of tokens hitting the lexicon
        if not tokens:
            return 0.0
        anchor_set = set(anchors)
        return min(1.0, sum(t in anchor_set for t in tokens) / 3.0)

    @staticmethod
    def _commodity_specificity(text: str) -> float:
        """1.0 if the headline names a specific commodity market, scaled by count."""
        text_lower = text.lower()
        hits = sum(
            any(kw in text_lower for kw in kws)
            for kws in COMMODITY_KEYWORDS.values()
        )
        return min(1.0, hits / 2.0)

    def _novelty(self, token_lists: List[List[str]]) -> np.ndarray:
        """1 − max similarity to any other headline in the batch (redundancy penalty)."""
        n = len(token_lists)
        if n <= 1:
            return np.ones(n)
        vecs = [self._mean_vector(t) for t in token_lists]
        if all(v is None for v in vecs):
            # Heuristic fallback: Jaccard overlap on token sets
            sets = [set(t) for t in token_lists]
            novelty = np.ones(n)
            for i in range(n):
                if not sets[i]:
                    continue
                best = max(
                    (len(sets[i] & sets[j]) / len(sets[i] | sets[j])
                     for j in range(n) if j != i and sets[j]),
                    default=0.0,
                )
                novelty[i] = 1.0 - best
            return novelty
        novelty = np.ones(n)
        for i in range(n):
            best = max(
                (self._cosine(vecs[i], vecs[j]) for j in range(n) if j != i),
                default=0.0,
            )
            novelty[i] = 1.0 - max(0.0, best)
        return novelty

    # ── Public API ─────────────────────────────────────────────────────────────

    def score_headlines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with insight_weight ∈ [0, 1] and component columns.

        Added columns: insight_weight, insight_sim, generic_sim,
        commodity_specificity, novelty, scoring_mode ('word2vec' | 'heuristic').
        Never raises on bad input — empty/odd frames come back unchanged with
        the columns absent, so the page hook can render unconditionally.
        """
        out = df.copy()
        if out.empty or "title" not in out.columns:
            return out

        texts = (
            out["title"].fillna("") + " " + out.get("summary", pd.Series("", index=out.index)).fillna("")
        ).tolist()
        token_lists = [tokenize(t) for t in texts]

        insight_sim = np.array([self._anchor_similarity(t, INSIGHT_ANCHORS) for t in token_lists])
        generic_sim = np.array([self._anchor_similarity(t, GENERIC_ANCHORS) for t in token_lists])
        specificity = np.array([self._commodity_specificity(t) for t in texts])
        novelty = self._novelty(token_lists)

        weight = (
            W_INSIGHT * insight_sim
            - W_GENERIC * generic_sim
            + W_SPECIFIC * specificity
            + W_NOVELTY * novelty
        )
        out["insight_sim"] = insight_sim
        out["generic_sim"] = generic_sim
        out["commodity_specificity"] = specificity
        out["novelty"] = novelty
        out["insight_weight"] = np.clip(weight, 0.0, 1.0)
        out["scoring_mode"] = "word2vec" if self._load_w2v() is not None else "heuristic"
        return out
