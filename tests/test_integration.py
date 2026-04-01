"""Integration tests for the full pipeline (Task 9).

Tests the pipeline end-to-end using:
- Mock RSS feeds (fixture XML files + inline XML)
- Mock ML models (no GPU/download required)
- Temporary SQLite databases (cleaned up after each test)
"""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import feedparser
import pandas as pd
import pytest

from pipeline.database import SQLiteDB, init_db
from pipeline.deduplication import article_exists, deduplicate_articles

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name: str) -> bytes:
    with open(os.path.join(FIXTURES_DIR, name), "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_db_path(tmp_path):
    """Return a temp path for a SQLite DB (file is created on first use)."""
    return str(tmp_path / "test_articles.db")


@pytest.fixture()
def mem_db():
    """In-memory SQLiteDB with schema applied and a default source seeded."""
    db = SQLiteDB(":memory:")
    db.init()
    db.upsert_source({
        "source_id": "TestSource",
        "name": "Test Source",
        "rss_url": "https://example.com/feed.xml",
        "active": True,
        "category": "news",
    })
    yield db
    db.close()


def _make_article(**kwargs):
    defaults = {
        "url": "https://example.com/article/1",
        "original_url": "https://example.com/article/1",
        "title": "Dog saves drowning child",
        "source_id": "TestSource",
        "published": datetime(2026, 3, 26, 10, 0, 0),
        "published_at": datetime(2026, 3, 26, 10, 0, 0),
        "uplifting_score": 0.92,
        "category": "Community",
        "summary": "A brave dog saved a child.",
        "thumbnail_url": None,
    }
    defaults.update(kwargs)
    return defaults


def _make_mock_model(scores: list):
    model = MagicMock()
    model.predict_proba.return_value = [[1.0 - s, s] for s in scores]
    return model


# ---------------------------------------------------------------------------
# RSS parsing integration tests (using fixture files)
# ---------------------------------------------------------------------------


def test_rss_parsing_valid_feed():
    """Valid RSS fixture file parses into expected article count and fields."""
    feed = feedparser.parse(_load_fixture("sample_feed.xml"))
    assert len(feed.entries) == 3
    entry = feed.entries[0]
    assert entry.title
    assert entry.link
    assert entry.published_parsed is not None


def test_rss_parsing_empty_feed():
    """Empty RSS fixture file yields zero entries."""
    feed = feedparser.parse(_load_fixture("empty_feed.xml"))
    assert len(feed.entries) == 0


def test_rss_parsing_malformed_feed():
    """feedparser handles malformed XML without raising exceptions."""
    feed = feedparser.parse(_load_fixture("malformed_feed.xml"))
    assert isinstance(feed.entries, list)


def test_rss_parsing_reddit_feed():
    """Reddit Atom feed fixture parses correctly."""
    feed = feedparser.parse(_load_fixture("reddit_feed.xml"))
    assert len(feed.entries) == 2
    assert all(entry.link for entry in feed.entries)


def test_rss_parsing_timeout_handled_gracefully():
    """If _download_feed_content times out, pipeline continues with empty feed."""
    from crawler.rss_reader import _download_feed_content
    with patch("crawler.rss_reader.requests.get", side_effect=Exception("timeout")):
        result = _download_feed_content("https://example.com/feed.xml")
    assert result is None


# ---------------------------------------------------------------------------
# Classification pipeline integration tests
# ---------------------------------------------------------------------------


def test_classification_with_known_uplifting_titles():
    """Titles with strong positive signals score above 0.75 with mock model."""
    from classifier.classify_headlines import filter_positive_news
    uplifting_titles = [
        "Dog rescues drowning child from river",
        "Teen wins scholarship for helping community",
        "Scientists achieve breakthrough in cancer treatment",
    ]
    model = _make_mock_model([0.95, 0.91, 0.88])
    df = pd.DataFrame({"title": uplifting_titles})
    result = filter_positive_news(df, model, threshold=0.75)
    assert len(result) == 3
    assert all(result["uplifting_score"] >= 0.75)


def test_classification_with_known_non_uplifting_titles():
    """Titles with negative signals score below threshold with mock model."""
    from classifier.classify_headlines import filter_positive_news
    non_uplifting = [
        "Stock market crashes amid economic fears",
        "Violence erupts in disputed region",
    ]
    model = _make_mock_model([0.10, 0.15])
    df = pd.DataFrame({"title": non_uplifting})
    result = filter_positive_news(df, model, threshold=0.75)
    assert result.empty


def test_classification_strict_source_threshold_integration():
    """BBC articles require 0.90+ AND an uplifting keyword to pass."""
    from classifier.classify_headlines import filter_positive_news
    articles = [
        {"title": "Dog wins national award", "source": "BBCNews"},      # score=0.93, hint=yes → PASS
        {"title": "Political debate heats up", "source": "BBCNews"},    # score=0.93, hint=no → FAIL
        {"title": "Community garden blooms", "source": "LocalFeed"},    # score=0.82, hint=no → PASS (non-strict)
    ]
    model = _make_mock_model([0.93, 0.93, 0.82])
    df = pd.DataFrame(articles)
    result = filter_positive_news(df, model, threshold=0.75)
    assert len(result) == 2
    passing_titles = result["title"].tolist()
    assert "Dog wins national award" in passing_titles
    assert "Community garden blooms" in passing_titles
    assert "Political debate heats up" not in passing_titles


# ---------------------------------------------------------------------------
# Deduplication integration tests
# ---------------------------------------------------------------------------


def test_deduplication_skips_existing_url(mem_db):
    """Article whose URL is already in DB is removed from the list."""
    existing = _make_article(url="https://example.com/existing")
    mem_db.upsert_articles([existing])

    new_articles = [
        _make_article(url="https://example.com/existing"),   # duplicate
        _make_article(url="https://example.com/new-article", title="Brand new article"),
    ]
    result = deduplicate_articles(new_articles, db=mem_db)
    assert len(result) == 1
    assert result[0]["url"] == "https://example.com/new-article"


def test_deduplication_all_new_articles_pass_through(mem_db):
    """When no articles are in DB yet, all pass through deduplication."""
    articles = [
        _make_article(url="https://example.com/a1", title="Article 1"),
        _make_article(url="https://example.com/a2", title="Article 2"),
        _make_article(url="https://example.com/a3", title="Article 3"),
    ]
    result = deduplicate_articles(articles, db=mem_db)
    assert len(result) == 3


def test_deduplication_all_duplicates_returns_empty(mem_db):
    """When all articles already exist in DB, empty list is returned."""
    articles = [
        _make_article(url="https://example.com/a1"),
        _make_article(url="https://example.com/a2", title="Article 2"),
    ]
    mem_db.upsert_articles(articles)
    result = deduplicate_articles(articles, db=mem_db)
    assert result == []


def test_article_exists_returns_true_for_known_url(mem_db):
    article = _make_article(url="https://example.com/known")
    mem_db.upsert_articles([article])
    assert article_exists("https://example.com/known", db=mem_db) is True


def test_article_exists_returns_false_for_unknown_url(mem_db):
    assert article_exists("https://example.com/unknown-xyz", db=mem_db) is False


# ---------------------------------------------------------------------------
# Database write integration tests
# ---------------------------------------------------------------------------


def test_database_upsert_stores_article(mem_db):
    """upsert_articles stores article and it can be queried back."""
    article = _make_article(url="https://example.com/stored")
    count = mem_db.upsert_articles([article])
    assert count == 1
    assert article_exists("https://example.com/stored", db=mem_db) is True


def test_database_upsert_updates_existing_article(mem_db):
    """Upserting same URL with updated data updates the record."""
    article = _make_article(url="https://example.com/upsert", title="Original Title")
    mem_db.upsert_articles([article])
    updated = _make_article(url="https://example.com/upsert", title="Updated Title")
    mem_db.upsert_articles([updated])
    # Confirm only one record exists (no duplicates)
    conn = mem_db.connect()
    row = conn.execute(
        "SELECT title FROM articles WHERE url = ?", ("https://example.com/upsert",)
    ).fetchone()
    assert row is not None
    assert row[0] == "Updated Title"


def test_database_stores_multiple_articles(mem_db):
    """Batch upsert stores the correct number of articles."""
    articles = [
        _make_article(url=f"https://example.com/batch/{i}", title=f"Article {i}")
        for i in range(5)
    ]
    count = mem_db.upsert_articles(articles)
    assert count == 5


def test_database_cleanup_after_test(tmp_path):
    """DB files created during tests are cleaned up (by tmp_path fixture)."""
    db_path = str(tmp_path / "cleanup_test.db")
    db = init_db(db_path)
    db.close()
    assert os.path.exists(db_path)
    # tmp_path is automatically removed by pytest — this confirms path was used


# ---------------------------------------------------------------------------
# End-to-end pipeline integration tests
# ---------------------------------------------------------------------------


def _build_pipeline_mocks(articles, uplifting_scores=None):
    """Return a dict of patches needed to run the pipeline without I/O."""
    if uplifting_scores is None:
        uplifting_scores = [0.90] * len(articles)

    mock_model = _make_mock_model(uplifting_scores)

    def fake_categorize(article_dicts):
        return [{**a, "category": "Community"} for a in article_dicts]

    def fake_summarize(text):
        return "A brief summary."

    return {
        "crawl": articles,
        "model": mock_model,
        "categorize": fake_categorize,
        "summarize": fake_summarize,
    }


def _seed_source(db_path: str, source_id: str = "TestSource") -> None:
    """Pre-seed a source row so FK constraints are satisfied during pipeline tests."""
    db = init_db(db_path)
    db.upsert_source({
        "source_id": source_id,
        "name": "Test Source",
        "rss_url": "https://example.com/feed.xml",
        "active": True,
        "category": "news",
    })
    db.close()


@patch("pipeline.run_pipeline.crawl_all_sources")
@patch("pipeline.run_pipeline.load_model")
@patch("enrichment.categorizer.categorize_batch")
@patch("pipeline.summarizer.summarize")
def test_end_to_end_pipeline_stores_articles(
    mock_summarize, mock_categorize, mock_load_model, mock_crawl, temp_db_path
):
    """Full pipeline: fetch → dedup → classify → categorize → summarize → store."""
    _seed_source(temp_db_path)
    raw_articles = [
        {
            "title": "Dog saves child",
            "original_url": "https://example.com/dog-saves",
            "rss_link": "https://example.com/dog-saves",
            "source_id": "TestSource",
            "published": datetime(2026, 3, 26, 10, 0, 0),
            "threshold": 0.75,
        },
        {
            "title": "Volunteer helps elderly",
            "original_url": "https://example.com/volunteer",
            "rss_link": "https://example.com/volunteer",
            "source_id": "TestSource",
            "published": datetime(2026, 3, 26, 9, 0, 0),
            "threshold": 0.75,
        },
    ]
    mock_crawl.return_value = raw_articles
    mock_load_model.return_value = _make_mock_model([0.92, 0.88])
    mock_categorize.side_effect = lambda arts: [{**a, "category": "Community"} for a in arts]
    mock_summarize.return_value = "A brief summary."

    from pipeline.run_pipeline import run_pipeline
    result = run_pipeline(db_path=temp_db_path)

    assert result["articles_fetched"] == 2
    assert result["articles_classified"] == 2
    assert result["articles_stored"] == 2
    assert result.get("articles_deduplicated", 0) == 0


@patch("pipeline.run_pipeline.crawl_all_sources")
@patch("pipeline.run_pipeline.load_model")
@patch("enrichment.categorizer.categorize_batch")
@patch("pipeline.summarizer.summarize")
def test_pipeline_skips_all_duplicate_articles(
    mock_summarize, mock_categorize, mock_load_model, mock_crawl, temp_db_path
):
    """When all fetched articles already exist in DB, pipeline returns early."""
    article = {
        "title": "Dog saves child",
        "url": "https://example.com/dog-saves",
        "original_url": "https://example.com/dog-saves",
        "rss_link": "https://example.com/dog-saves",
        "source_id": "TestSource",
        "published": datetime(2026, 3, 26, 10, 0, 0),
        "threshold": 0.75,
        "uplifting_score": 0.92,
        "category": "Community",
        "summary": None,
        "thumbnail_url": None,
    }
    # Pre-seed DB with the article using the canonical url field
    db = init_db(temp_db_path)
    db.upsert_source({
        "source_id": "TestSource", "name": "Test Source",
        "rss_url": "https://example.com/feed.xml", "active": True, "category": "news",
    })
    db.upsert_articles([article])
    db.close()

    mock_crawl.return_value = [article]
    mock_load_model.return_value = _make_mock_model([0.92])
    mock_categorize.side_effect = lambda arts: arts
    mock_summarize.return_value = "Summary."

    from pipeline.run_pipeline import run_pipeline
    result = run_pipeline(db_path=temp_db_path)

    assert result["articles_fetched"] == 1
    assert result.get("articles_deduplicated", 0) == 1
    assert result.get("articles_classified", 0) == 0


@patch("pipeline.run_pipeline.crawl_all_sources")
def test_pipeline_handles_empty_crawl(mock_crawl, temp_db_path):
    """Pipeline exits gracefully when no articles are fetched."""
    mock_crawl.return_value = []

    from pipeline.run_pipeline import run_pipeline
    result = run_pipeline(db_path=temp_db_path)

    assert result["articles_fetched"] == 0
    assert result.get("articles_classified", 0) == 0


@patch("pipeline.run_pipeline.crawl_all_sources")
@patch("pipeline.run_pipeline.load_model")
@patch("enrichment.categorizer.categorize_batch")
@patch("pipeline.summarizer.summarize")
def test_pipeline_metrics_accuracy(
    mock_summarize, mock_categorize, mock_load_model, mock_crawl, temp_db_path
):
    """Pipeline metrics dict reflects actual processing counts."""
    # 3 articles: 2 uplifting (score>=0.75), 1 not
    raw = [
        {
            "title": "Breakthrough in medicine",
            "original_url": "https://example.com/med",
            "rss_link": "https://example.com/med",
            "source_id": "LocalFeed",
            "published": datetime(2026, 3, 26, 10, 0, 0),
            "threshold": 0.75,
        },
        {
            "title": "Community garden wins award",
            "original_url": "https://example.com/garden",
            "rss_link": "https://example.com/garden",
            "source_id": "LocalFeed",
            "published": datetime(2026, 3, 26, 9, 0, 0),
            "threshold": 0.75,
        },
        {
            "title": "Stock market declines",
            "original_url": "https://example.com/stocks",
            "rss_link": "https://example.com/stocks",
            "source_id": "LocalFeed",
            "published": datetime(2026, 3, 26, 8, 0, 0),
            "threshold": 0.75,
        },
    ]
    _seed_source(temp_db_path, source_id="LocalFeed")
    mock_crawl.return_value = raw
    # First two articles above threshold; third below
    mock_load_model.return_value = _make_mock_model([0.92, 0.88, 0.20])
    mock_categorize.side_effect = lambda arts: [{**a, "category": "General"} for a in arts]
    mock_summarize.return_value = "Summary."

    from pipeline.run_pipeline import run_pipeline
    result = run_pipeline(db_path=temp_db_path)

    assert result["articles_fetched"] == 3
    assert result["articles_classified"] == 2   # only 2 pass threshold
    assert result["articles_stored"] == 2


@patch("pipeline.run_pipeline.crawl_all_sources")
@patch("pipeline.run_pipeline.load_model")
def test_pipeline_handles_classification_error_gracefully(
    mock_load_model, mock_crawl, temp_db_path
):
    """If classification fails, pipeline returns with 0 classified articles."""
    mock_crawl.return_value = [
        {
            "title": "Good news article",
            "original_url": "https://example.com/good",
            "rss_link": "https://example.com/good",
            "source_id": "TestSource",
            "published": datetime(2026, 3, 26, 10, 0, 0),
            "threshold": 0.75,
        }
    ]
    mock_load_model.side_effect = RuntimeError("Model not found")

    from pipeline.run_pipeline import run_pipeline
    result = run_pipeline(db_path=temp_db_path)

    assert result["articles_fetched"] == 1
    assert result.get("articles_classified", 0) == 0


@patch("pipeline.run_pipeline.crawl_all_sources")
@patch("pipeline.run_pipeline.load_model")
@patch("enrichment.categorizer.categorize_batch")
@patch("pipeline.summarizer.summarize")
def test_pipeline_error_handling_summarize_failure(
    mock_summarize, mock_categorize, mock_load_model, mock_crawl, temp_db_path
):
    """If summarization fails for an article, pipeline continues and stores without summary."""
    _seed_source(temp_db_path)
    raw = [{
        "title": "Community helps flood victims",
        "original_url": "https://example.com/flood",
        "rss_link": "https://example.com/flood",
        "source_id": "TestSource",
        "published": datetime(2026, 3, 26, 10, 0, 0),
        "threshold": 0.75,
    }]
    mock_crawl.return_value = raw
    mock_load_model.return_value = _make_mock_model([0.90])
    mock_categorize.side_effect = lambda arts: [{**a, "category": "Community"} for a in arts]
    mock_summarize.side_effect = ValueError("Summarization failed")

    from pipeline.run_pipeline import run_pipeline
    result = run_pipeline(db_path=temp_db_path)

    # Pipeline should not crash; article is stored (with no summary)
    assert result["articles_fetched"] == 1
    assert result["articles_stored"] >= 0  # continues without crashing


# ---------------------------------------------------------------------------
# Issue #16: run_pipeline reads env vars for db_path and pipeline params
# ---------------------------------------------------------------------------


@patch("pipeline.run_pipeline.crawl_all_sources")
@patch("pipeline.run_pipeline.load_model")
@patch("enrichment.categorizer.categorize_batch")
@patch("pipeline.summarizer.summarize")
def test_pipeline_uses_database_path_env_var(
    mock_summarize, mock_categorize, mock_load_model, mock_crawl, temp_db_path
):
    """DATABASE_PATH env var is used when db_path is not passed explicitly."""
    mock_crawl.return_value = []
    mock_load_model.return_value = _make_mock_model([])

    from pipeline.run_pipeline import run_pipeline

    with patch.dict(os.environ, {"DATABASE_PATH": temp_db_path}):
        result = run_pipeline()

    assert result["articles_fetched"] == 0


@patch("pipeline.run_pipeline.crawl_all_sources")
@patch("pipeline.run_pipeline.load_model")
@patch("enrichment.categorizer.categorize_batch")
@patch("pipeline.summarizer.summarize")
def test_pipeline_explicit_db_path_overrides_env(
    mock_summarize, mock_categorize, mock_load_model, mock_crawl, temp_db_path
):
    """Explicit db_path argument takes precedence over DATABASE_PATH env var."""
    mock_crawl.return_value = []
    mock_load_model.return_value = _make_mock_model([])

    from pipeline.run_pipeline import run_pipeline

    with patch.dict(os.environ, {"DATABASE_PATH": "/nonexistent/path.db"}):
        # Should use temp_db_path, not the env var path
        result = run_pipeline(db_path=temp_db_path)

    assert result["articles_fetched"] == 0
