"""Tests for models/headline_insight.py — skeleton behaviour + fallbacks."""

import numpy as np
import pandas as pd
import pytest

from models.headline_insight import (
    HeadlineInsightModel,
    MIN_CORPUS_SIZE,
    tokenize,
)

STRONG = "Russia imposes wheat export ban as Black Sea blockade halts grain shipments"
GENERIC = "Stocks mixed as investors weigh data ahead of weekly session wrap"


@pytest.fixture
def model(tmp_path):
    """Model with isolated tmp paths — never touches the real corpus/model."""
    return HeadlineInsightModel(
        model_path=tmp_path / "w2v.model",
        corpus_path=tmp_path / "corpus.parquet",
    )


def _frame(titles):
    return pd.DataFrame({
        "title": titles,
        "summary": ["" for _ in titles],
        "link": ["http://x" for _ in titles],
        "source": ["Test" for _ in titles],
    })


def test_tokenize_strips_stopwords_and_punctuation():
    toks = tokenize("OPEC said it will cut output, and prices rose!")
    assert "opec" in toks and "output" in toks
    assert "and" not in toks and "will" not in toks


def test_score_adds_columns_and_bounds(model):
    scored = model.score_headlines(_frame([STRONG, GENERIC]))
    for col in ("insight_weight", "insight_sim", "generic_sim",
                "commodity_specificity", "novelty", "scoring_mode"):
        assert col in scored.columns
    assert ((scored["insight_weight"] >= 0) & (scored["insight_weight"] <= 1)).all()


def test_heuristic_ranks_sociopolitical_above_generic(model):
    """No trained model on disk → heuristic mode must still order sensibly."""
    scored = model.score_headlines(_frame([STRONG, GENERIC]))
    assert scored["scoring_mode"].iloc[0] == "heuristic"
    by_title = scored.set_index("title")["insight_weight"]
    assert by_title[STRONG] > by_title[GENERIC]


def test_empty_frame_passes_through(model):
    out = model.score_headlines(pd.DataFrame())
    assert out.empty


def test_append_corpus_dedups_by_title(model):
    n1 = model.append_corpus(_frame([STRONG, GENERIC]))
    n2 = model.append_corpus(_frame([STRONG]))  # duplicate
    assert n1 == 2 and n2 == 2


def test_train_skips_below_min_corpus(model):
    model.append_corpus(_frame([STRONG]))
    assert model.corpus_size() < MIN_CORPUS_SIZE
    assert model.train() is False


def test_train_and_score_word2vec_path(model):
    """force=True trains on a tiny synthetic corpus; scoring switches modes."""
    gensim = pytest.importorskip("gensim")
    rng = np.random.default_rng(0)
    base = (STRONG + " " + GENERIC).split()
    titles = [" ".join(rng.choice(base, size=8)) + f" filler{i % 7}" for i in range(60)]
    model.append_corpus(_frame(titles))
    assert model.train(force=True) is True
    scored = model.score_headlines(_frame([STRONG, GENERIC]))
    assert scored["scoring_mode"].iloc[0] == "word2vec"
    assert ((scored["insight_weight"] >= 0) & (scored["insight_weight"] <= 1)).all()
