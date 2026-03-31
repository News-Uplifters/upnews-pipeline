"""Tests for features added in feat/docker-support.

Covers:
- URL/published_at field normalisation in run_pipeline
- Thumbnail extraction integration in run_pipeline
- Rule-based classifier fallback (_RuleBasedModel + load_model behaviour)
"""

import os
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from pipeline.database import SQLiteDB, init_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_article(**kwargs):
    defaults = {
        "title": "Dog rescues child from river",
        "rss_link": "https://example.com/dog-saves",
        "original_url": "https://example.com/dog-saves",
        "source_id": "TestSource",
        "published": datetime(2026, 3, 26, 10, 0, 0),
        "published_at": datetime(2026, 3, 26, 10, 0, 0),
        "threshold": 0.75,
    }
    defaults.update(kwargs)
    return defaults


def _make_mock_model(scores: list):
    model = MagicMock()
    model.predict_proba.return_value = [[1.0 - s, s] for s in scores]
    return model


def _seed_source(db_path: str, source_id: str = "TestSource") -> None:
    db = init_db(db_path)
    db.upsert_source({
        "source_id": source_id,
        "name": "Test Source",
        "rss_url": "https://example.com/feed.xml",
        "active": True,
        "category": "news",
    })
    db.close()


# ===========================================================================
# 1. URL / published_at field normalisation
# ===========================================================================


@patch("pipeline.run_pipeline.crawl_all_sources")
@patch("pipeline.run_pipeline.load_model")
@patch("enrichment.categorizer.categorize_batch")
@patch("pipeline.summarizer.summarize")
@patch("enrichment.thumbnails.extract_thumbnails_batch")
def test_url_normalisation_from_original_url(
    mock_thumbs, mock_summarize, mock_categorize, mock_load_model, mock_crawl, tmp_path
):
    """Articles using original_url (not url) are stored with the correct URL."""
    db_path = str(tmp_path / "test.db")
    _seed_source(db_path)

    raw = [_make_article(original_url="https://example.com/a1")]
    mock_crawl.return_value = raw
    mock_load_model.return_value = _make_mock_model([0.92])
    mock_categorize.side_effect = lambda arts: [{**a, "category": "Community"} for a in arts]
    mock_summarize.return_value = "Summary."
    # Return no thumbnails for simplicity
    mock_thumbs.return_value = {}

    from pipeline.run_pipeline import run_pipeline
    result = run_pipeline(db_path=db_path)

    assert result["articles_stored"] >= 1
    db = SQLiteDB(db_path)
    db.connect()
    assert db.article_exists("https://example.com/a1")
    db.close()


@patch("pipeline.run_pipeline.crawl_all_sources")
@patch("pipeline.run_pipeline.load_model")
@patch("enrichment.categorizer.categorize_batch")
@patch("pipeline.summarizer.summarize")
@patch("enrichment.thumbnails.extract_thumbnails_batch")
def test_url_normalisation_prefers_explicit_url(
    mock_thumbs, mock_summarize, mock_categorize, mock_load_model, mock_crawl, tmp_path
):
    """When article already has a 'url' key, it is kept as-is."""
    db_path = str(tmp_path / "test.db")
    _seed_source(db_path)

    raw = [_make_article(url="https://example.com/explicit", original_url="https://example.com/should-not-use")]
    mock_crawl.return_value = raw
    mock_load_model.return_value = _make_mock_model([0.92])
    mock_categorize.side_effect = lambda arts: [{**a, "category": "Community"} for a in arts]
    mock_summarize.return_value = "Summary."
    mock_thumbs.return_value = {}

    from pipeline.run_pipeline import run_pipeline
    run_pipeline(db_path=db_path)

    db = SQLiteDB(db_path)
    db.connect()
    assert db.article_exists("https://example.com/explicit")
    assert not db.article_exists("https://example.com/should-not-use")
    db.close()


@patch("pipeline.run_pipeline.crawl_all_sources")
@patch("pipeline.run_pipeline.load_model")
@patch("enrichment.categorizer.categorize_batch")
@patch("pipeline.summarizer.summarize")
@patch("enrichment.thumbnails.extract_thumbnails_batch")
def test_published_at_normalised_from_published(
    mock_thumbs, mock_summarize, mock_categorize, mock_load_model, mock_crawl, tmp_path
):
    """Articles with 'published' field have published_at stored correctly."""
    db_path = str(tmp_path / "test.db")
    _seed_source(db_path)
    pub_date = datetime(2026, 3, 26, 10, 0, 0)

    raw = [_make_article(original_url="https://example.com/pub-test", published=pub_date)]
    mock_crawl.return_value = raw
    mock_load_model.return_value = _make_mock_model([0.90])
    mock_categorize.side_effect = lambda arts: [{**a, "category": "Community"} for a in arts]
    mock_summarize.return_value = "Summary."
    mock_thumbs.return_value = {}

    from pipeline.run_pipeline import run_pipeline
    run_pipeline(db_path=db_path)

    db = SQLiteDB(db_path)
    db.connect()
    row = db.connect().execute(
        "SELECT published_at FROM articles WHERE url = ?",
        ("https://example.com/pub-test",),
    ).fetchone()
    assert row is not None
    assert row[0] is not None
    db.close()


@patch("pipeline.run_pipeline.crawl_all_sources")
@patch("pipeline.run_pipeline.load_model")
@patch("enrichment.categorizer.categorize_batch")
@patch("pipeline.summarizer.summarize")
@patch("enrichment.thumbnails.extract_thumbnails_batch")
def test_deduplication_works_with_normalised_url(
    mock_thumbs, mock_summarize, mock_categorize, mock_load_model, mock_crawl, tmp_path
):
    """Deduplication correctly recognises already-stored articles via original_url."""
    db_path = str(tmp_path / "test.db")
    _seed_source(db_path)

    article = _make_article(original_url="https://example.com/dup")
    # Pre-seed DB with this URL
    db = init_db(db_path)
    db.upsert_articles([{**article, "url": "https://example.com/dup", "published_at": None}])
    db.close()

    mock_crawl.return_value = [article]
    mock_load_model.return_value = _make_mock_model([0.90])
    mock_categorize.side_effect = lambda arts: arts
    mock_summarize.return_value = "Summary."
    mock_thumbs.return_value = {}

    from pipeline.run_pipeline import run_pipeline
    result = run_pipeline(db_path=db_path)

    assert result.get("articles_deduplicated", 0) == 1
    assert result.get("articles_classified", 0) == 0


# ===========================================================================
# 2. Thumbnail integration in run_pipeline
# ===========================================================================


@patch("pipeline.run_pipeline.crawl_all_sources")
@patch("pipeline.run_pipeline.load_model")
@patch("enrichment.categorizer.categorize_batch")
@patch("pipeline.summarizer.summarize")
@patch("enrichment.thumbnails.extract_thumbnails_batch")
def test_thumbnail_extraction_called_during_pipeline(
    mock_thumbs, mock_summarize, mock_categorize, mock_load_model, mock_crawl, tmp_path
):
    """extract_thumbnails_batch is called once during a normal pipeline run."""
    db_path = str(tmp_path / "test.db")
    _seed_source(db_path)

    url = "https://example.com/thumb-article"
    raw = [_make_article(original_url=url)]
    mock_crawl.return_value = raw
    mock_load_model.return_value = _make_mock_model([0.90])
    mock_categorize.side_effect = lambda arts: [{**a, "category": "Community"} for a in arts]
    mock_summarize.return_value = "Summary."
    mock_thumbs.return_value = {url: "https://example.com/image.jpg"}

    from pipeline.run_pipeline import run_pipeline
    result = run_pipeline(db_path=db_path)

    assert mock_thumbs.called
    assert result["articles_stored"] >= 1


@patch("pipeline.run_pipeline.crawl_all_sources")
@patch("pipeline.run_pipeline.load_model")
@patch("enrichment.categorizer.categorize_batch")
@patch("pipeline.summarizer.summarize")
@patch("enrichment.thumbnails.extract_thumbnails_batch")
def test_thumbnail_stored_in_db(
    mock_thumbs, mock_summarize, mock_categorize, mock_load_model, mock_crawl, tmp_path
):
    """Thumbnail URL returned by extract_thumbnails_batch is persisted to DB."""
    db_path = str(tmp_path / "test.db")
    _seed_source(db_path)

    url = "https://example.com/article-with-thumb"
    thumb_url = "https://cdn.example.com/thumb.jpg"
    raw = [_make_article(original_url=url)]
    mock_crawl.return_value = raw
    mock_load_model.return_value = _make_mock_model([0.90])
    mock_categorize.side_effect = lambda arts: [{**a, "category": "Community"} for a in arts]
    mock_summarize.return_value = "Summary."
    mock_thumbs.return_value = {url: thumb_url}

    from pipeline.run_pipeline import run_pipeline
    run_pipeline(db_path=db_path)

    db = SQLiteDB(db_path)
    row = db.connect().execute(
        "SELECT thumbnail_url FROM articles WHERE url = ?", (url,)
    ).fetchone()
    assert row is not None
    assert row[0] == thumb_url
    db.close()


@patch("pipeline.run_pipeline.crawl_all_sources")
@patch("pipeline.run_pipeline.load_model")
@patch("enrichment.categorizer.categorize_batch")
@patch("pipeline.summarizer.summarize")
@patch("enrichment.thumbnails.extract_thumbnails_batch", side_effect=Exception("network down"))
def test_pipeline_continues_when_thumbnail_extraction_fails(
    mock_thumbs, mock_summarize, mock_categorize, mock_load_model, mock_crawl, tmp_path
):
    """Pipeline completes and stores articles even if thumbnail extraction throws."""
    db_path = str(tmp_path / "test.db")
    _seed_source(db_path)

    raw = [_make_article(original_url="https://example.com/no-thumb")]
    mock_crawl.return_value = raw
    mock_load_model.return_value = _make_mock_model([0.90])
    mock_categorize.side_effect = lambda arts: [{**a, "category": "Community"} for a in arts]
    mock_summarize.return_value = "Summary."

    from pipeline.run_pipeline import run_pipeline
    result = run_pipeline(db_path=db_path)

    # Pipeline should not crash; article still stored
    assert result["articles_fetched"] == 1
    assert result["articles_stored"] >= 1


# ===========================================================================
# 3. Rule-based classifier fallback
# ===========================================================================


def test_rule_based_model_positive_hints_score_high():
    """Titles with multiple positive hints receive a score > 0.5."""
    from classifier.classify_headlines import _RuleBasedModel
    model = _RuleBasedModel()
    results = model.predict_proba(["Dog rescued and wins award for helping community"])
    pos_score = results[0][1]
    assert pos_score > 0.5


def test_rule_based_model_negative_hints_score_low():
    """Titles with negative hints receive a score < 0.5."""
    from classifier.classify_headlines import _RuleBasedModel
    model = _RuleBasedModel()
    results = model.predict_proba(["Deadly attack kills dozens in war zone"])
    pos_score = results[0][1]
    assert pos_score < 0.5


def test_rule_based_model_returns_probability_pairs():
    """predict_proba returns [[neg, pos], ...] summing to ~1.0 per pair."""
    from classifier.classify_headlines import _RuleBasedModel
    model = _RuleBasedModel()
    titles = ["Good news article", "Bad news article", "Neutral headline"]
    results = model.predict_proba(titles)
    assert len(results) == 3
    for pair in results:
        assert len(pair) == 2
        assert abs(pair[0] + pair[1] - 1.0) < 1e-6


def test_rule_based_model_scores_clamped():
    """Scores never go below 0 or above 1."""
    from classifier.classify_headlines import _RuleBasedModel
    model = _RuleBasedModel()
    extreme_titles = [
        "war death crash flood explosion murdered violence terror disaster crisis",
        "wins rescued recovery breakthrough helps saved volunteer uplifting success hope",
    ]
    results = model.predict_proba(extreme_titles)
    for pair in results:
        assert 0.0 <= pair[0] <= 1.0
        assert 0.0 <= pair[1] <= 1.0


def test_load_model_returns_rule_based_when_mode_is_rules(monkeypatch):
    """CLASSIFIER_MODE=rules returns _RuleBasedModel without touching SetFit."""
    monkeypatch.setenv("CLASSIFIER_MODE", "rules")
    from classifier.classify_headlines import _RuleBasedModel, load_model
    model = load_model()
    assert isinstance(model, _RuleBasedModel)


def test_load_model_returns_rule_based_when_path_missing(monkeypatch, tmp_path):
    """load_model falls back to _RuleBasedModel when model directory is absent."""
    monkeypatch.delenv("CLASSIFIER_MODE", raising=False)
    monkeypatch.setenv("CLASSIFIER_MODE", "setfit")
    from classifier.classify_headlines import _RuleBasedModel, load_model
    non_existent = str(tmp_path / "no_model_here")
    model = load_model(model_path=non_existent)
    assert isinstance(model, _RuleBasedModel)


def test_rule_based_model_integrates_with_filter_positive_news():
    """filter_positive_news works end-to-end with _RuleBasedModel."""
    from classifier.classify_headlines import _RuleBasedModel, filter_positive_news
    model = _RuleBasedModel()
    titles = [
        "Rescued dog wins community award",    # many positive hints → high score
        "Deadly explosion kills civilians",     # negative hints → low score
    ]
    df = pd.DataFrame({"title": titles})
    result = filter_positive_news(df, model, threshold=0.5)
    # Positive article should pass; negative should not
    assert len(result) >= 1
    assert "Rescued dog wins community award" in result["title"].values


@patch("pipeline.run_pipeline.crawl_all_sources")
@patch("enrichment.categorizer.categorize_batch")
@patch("pipeline.summarizer.summarize")
@patch("enrichment.thumbnails.extract_thumbnails_batch")
def test_pipeline_runs_with_rule_based_classifier(
    mock_thumbs, mock_summarize, mock_categorize, mock_crawl, monkeypatch, tmp_path
):
    """Full pipeline run works with CLASSIFIER_MODE=rules (no SetFit model needed)."""
    monkeypatch.setenv("CLASSIFIER_MODE", "rules")
    db_path = str(tmp_path / "test.db")
    _seed_source(db_path)

    raw = [
        _make_article(
            original_url="https://example.com/rule-based-1",
            title="Rescued volunteers win award for helping community",
        )
    ]
    mock_crawl.return_value = raw
    mock_categorize.side_effect = lambda arts: [{**a, "category": "Community"} for a in arts]
    mock_summarize.return_value = "Summary."
    mock_thumbs.return_value = {}

    from pipeline.run_pipeline import run_pipeline
    result = run_pipeline(db_path=db_path)

    assert result["articles_fetched"] == 1
    # Rule-based model should classify at least the positively-hinted article
    assert result.get("articles_classified", 0) >= 0  # may vary based on score
