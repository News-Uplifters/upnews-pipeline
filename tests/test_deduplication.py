"""Unit tests for pipeline/deduplication.py (Task 7).

All tests use in-memory SQLite so no files are created on disk.
"""

from datetime import datetime, timezone

import pytest

from pipeline.database import SQLiteDB
from pipeline.deduplication import (
    _title_published_dedup,
    _url_dedup,
    article_exists,
    deduplicate_articles,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db():
    """Fresh in-memory SQLiteDB with schema applied and a default source."""
    instance = SQLiteDB(":memory:")
    instance.init()
    instance.upsert_source(
        {
            "source_id": "TestSource",
            "name": "Test Source",
            "rss_url": "https://example.com/feed.xml",
            "active": True,
            "category": "news",
        }
    )
    yield instance
    instance.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_article(**kwargs):
    defaults = {
        "url": "https://example.com/article/1",
        "title": "Wonderful Day Ahead",
        "source_id": "TestSource",
        "published_at": "2026-03-01 12:00:00",
    }
    defaults.update(kwargs)
    return defaults


def _insert(db, **kwargs):
    """Insert an article directly via upsert_article."""
    db.upsert_article(_make_article(**kwargs))


# ---------------------------------------------------------------------------
# article_exists
# ---------------------------------------------------------------------------


class TestArticleExists:
    def test_returns_false_for_unknown_url(self, db):
        assert article_exists("https://example.com/unknown", db=db) is False

    def test_returns_true_after_insert(self, db):
        _insert(db)
        assert article_exists("https://example.com/article/1", db=db) is True

    def test_case_sensitive_url(self, db):
        _insert(db, url="https://example.com/Article/1")
        # Different capitalisation → different URL → should be False
        assert article_exists("https://example.com/article/1", db=db) is False


# ---------------------------------------------------------------------------
# deduplicate_articles – empty / no-op cases
# ---------------------------------------------------------------------------


class TestDeduplicateEmpty:
    def test_empty_list_returns_empty(self, db):
        assert deduplicate_articles([], db=db) == []

    def test_no_duplicates_returns_all(self, db):
        articles = [
            _make_article(url="https://example.com/a"),
            _make_article(url="https://example.com/b"),
        ]
        result = deduplicate_articles(articles, db=db)
        assert len(result) == 2

    def test_all_new_articles_preserved(self, db):
        articles = [_make_article(url=f"https://example.com/{i}") for i in range(5)]
        result = deduplicate_articles(articles, db=db)
        assert result == articles


# ---------------------------------------------------------------------------
# deduplicate_articles – URL dedup (primary)
# ---------------------------------------------------------------------------


class TestURLDedup:
    def test_single_duplicate_removed(self, db):
        _insert(db, url="https://example.com/exists", title="Existing Title")
        articles = [
            _make_article(url="https://example.com/exists", title="Existing Title"),
            _make_article(url="https://example.com/new", title="Brand New Title"),
        ]
        result = deduplicate_articles(articles, db=db)
        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/new"

    def test_all_duplicates_removed(self, db):
        for i in range(3):
            _insert(db, url=f"https://example.com/{i}", title=f"Title {i}")
        articles = [_make_article(url=f"https://example.com/{i}", title=f"Title {i}") for i in range(3)]
        result = deduplicate_articles(articles, db=db)
        assert result == []

    def test_mixed_new_and_existing(self, db):
        _insert(db, url="https://example.com/old", title="Old Article")
        articles = [
            _make_article(url="https://example.com/old", title="Old Article"),
            _make_article(url="https://example.com/new-1", title="New Article One"),
            _make_article(url="https://example.com/new-2", title="New Article Two"),
        ]
        result = deduplicate_articles(articles, db=db)
        assert len(result) == 2
        urls = {a["url"] for a in result}
        assert urls == {"https://example.com/new-1", "https://example.com/new-2"}

    def test_preserves_article_content(self, db):
        article = _make_article(
            url="https://example.com/preserve",
            title="Keep This",
            uplifting_score=0.95,
        )
        result = deduplicate_articles([article], db=db)
        assert result[0]["title"] == "Keep This"
        assert result[0]["uplifting_score"] == 0.95

    def test_zero_false_positives(self, db):
        """Articles not in DB must never be filtered out."""
        articles = [_make_article(url=f"https://example.com/genuine/{i}") for i in range(10)]
        result = deduplicate_articles(articles, db=db)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# deduplicate_articles – title + published_at fallback
# ---------------------------------------------------------------------------


class TestTitlePublishedDedup:
    def test_title_published_duplicate_removed(self, db):
        _insert(
            db,
            url="https://example.com/original",
            title="Matching Title",
            published_at="2026-03-01 12:00:00",
        )
        # Same title + published_at but different URL (e.g. redirect stripped params)
        articles = [
            _make_article(
                url="https://example.com/redirect",
                title="Matching Title",
                published_at="2026-03-01 12:00:00",
            )
        ]
        result = deduplicate_articles(articles, db=db)
        assert result == []

    def test_different_title_not_filtered(self, db):
        _insert(
            db,
            url="https://example.com/original",
            title="Original Title",
            published_at="2026-03-01 12:00:00",
        )
        articles = [
            _make_article(
                url="https://example.com/different",
                title="Different Title",
                published_at="2026-03-01 12:00:00",
            )
        ]
        result = deduplicate_articles(articles, db=db)
        assert len(result) == 1

    def test_different_published_at_not_filtered(self, db):
        _insert(
            db,
            url="https://example.com/original",
            title="Same Title",
            published_at="2026-03-01 12:00:00",
        )
        articles = [
            _make_article(
                url="https://example.com/different-time",
                title="Same Title",
                published_at="2026-03-02 08:00:00",
            )
        ]
        result = deduplicate_articles(articles, db=db)
        assert len(result) == 1

    def test_missing_published_at_not_filtered(self, db):
        """Articles without published_at skip the title+published_at check."""
        _insert(
            db,
            url="https://example.com/original",
            title="Some Title",
            published_at="2026-03-01 12:00:00",
        )
        # No published_at → should NOT be filtered by title+published_at dedup
        articles = [
            _make_article(
                url="https://example.com/no-date",
                title="Some Title",
                published_at=None,
            )
        ]
        result = deduplicate_articles(articles, db=db)
        assert len(result) == 1

    def test_missing_title_not_filtered(self, db):
        """Articles without title skip the title+published_at check."""
        articles = [
            {
                "url": "https://example.com/no-title",
                "source_id": "TestSource",
                "published_at": "2026-03-01 12:00:00",
            }
        ]
        result = deduplicate_articles(articles, db=db)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Internal helpers: _url_dedup
# ---------------------------------------------------------------------------


class TestUrlDedup:
    def test_removes_existing_urls(self, db):
        _insert(db, url="https://example.com/exists")
        articles = [
            _make_article(url="https://example.com/exists"),
            _make_article(url="https://example.com/new"),
        ]
        result = _url_dedup(articles, db)
        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/new"

    def test_empty_input(self, db):
        assert _url_dedup([], db) == []

    def test_no_existing_urls(self, db):
        articles = [_make_article(url="https://example.com/fresh")]
        result = _url_dedup(articles, db)
        assert result == articles


# ---------------------------------------------------------------------------
# Internal helpers: _title_published_dedup
# ---------------------------------------------------------------------------


class TestTitlePublishedDedup:
    def test_duplicate_pair_removed(self, db):
        _insert(
            db,
            url="https://example.com/base",
            title="Title A",
            published_at="2026-03-01 10:00:00",
        )
        articles = [
            _make_article(
                url="https://example.com/dup",
                title="Title A",
                published_at="2026-03-01 10:00:00",
            )
        ]
        result = _title_published_dedup(articles, db)
        assert result == []

    def test_no_candidates_returns_unchanged(self, db):
        articles = [{"url": "https://example.com/x", "source_id": "TestSource"}]
        result = _title_published_dedup(articles, db)
        assert result == articles

    def test_non_duplicate_passes_through(self, db):
        articles = [
            _make_article(
                url="https://example.com/unique",
                title="Unique Title",
                published_at="2026-04-01 00:00:00",
            )
        ]
        result = _title_published_dedup(articles, db)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# db_path convenience (no pre-opened db)
# ---------------------------------------------------------------------------


class TestDbPathConvenience:
    def test_article_exists_with_db_path(self, tmp_path):
        db_path = str(tmp_path / "conv.db")
        db = SQLiteDB(db_path)
        db.init()
        db.upsert_source(
            {"source_id": "S", "name": "S", "rss_url": None, "active": True, "category": None}
        )
        db.upsert_article(
            {"url": "https://example.com/x", "title": "T", "source_id": "S"}
        )
        db.close()

        assert article_exists("https://example.com/x", db_path=db_path) is True
        assert article_exists("https://example.com/y", db_path=db_path) is False

    def test_deduplicate_with_db_path(self, tmp_path):
        db_path = str(tmp_path / "dedup.db")
        db = SQLiteDB(db_path)
        db.init()
        db.upsert_source(
            {"source_id": "S", "name": "S", "rss_url": None, "active": True, "category": None}
        )
        db.upsert_article(
            {"url": "https://example.com/old", "title": "Old", "source_id": "S"}
        )
        db.close()

        articles = [
            {"url": "https://example.com/old", "title": "Old", "source_id": "S"},
            {"url": "https://example.com/new", "title": "New", "source_id": "S"},
        ]
        result = deduplicate_articles(articles, db_path=db_path)
        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/new"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestDeduplicateLogging:
    def test_logs_skipped_count(self, db, caplog):
        import logging

        _insert(db, url="https://example.com/existing", title="Existing Article")
        articles = [
            _make_article(url="https://example.com/existing", title="Existing Article"),
            _make_article(url="https://example.com/fresh", title="Fresh New Article"),
        ]
        with caplog.at_level(logging.INFO, logger="pipeline.deduplication"):
            deduplicate_articles(articles, db=db)

        assert any("skipped" in r.message and "1" in r.message for r in caplog.records)

    def test_logs_all_new(self, db, caplog):
        import logging

        articles = [_make_article(url="https://example.com/brand-new")]
        with caplog.at_level(logging.INFO, logger="pipeline.deduplication"):
            deduplicate_articles(articles, db=db)

        assert any("all" in r.message.lower() and "new" in r.message.lower() for r in caplog.records)


# Allow running directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
