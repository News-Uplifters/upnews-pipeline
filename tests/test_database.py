"""Unit tests for pipeline/database.py (Task 6).

All tests use in-memory SQLite so no files are created on disk.
"""

import sqlite3
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from pipeline.database import (
    SQLiteDB,
    _article_to_row,
    _category_to_slug,
    _coerce_dt,
    _fmt_dt,
    init_db,
    write_articles,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db():
    """Fresh in-memory SQLiteDB with schema applied."""
    instance = SQLiteDB(":memory:")
    instance.init()
    yield instance
    instance.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(**kwargs):
    defaults = {
        "source_id": "TestSource",
        "name": "Test Source",
        "rss_url": "https://example.com/feed.xml",
        "active": True,
        "category": "news",
    }
    defaults.update(kwargs)
    return defaults


def _make_article(**kwargs):
    defaults = {
        "url": "https://example.com/article/1",
        "title": "A Wonderful Article",
        "summary": "Short summary.",
        "content": "Full article body text.",
        "thumbnail_url": "https://example.com/img.jpg",
        "category": "Health",
        "source_id": "TestSource",
        "uplifting_score": 0.92,
        "published_at": datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc),
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# SQLiteDB: connection / lifecycle
# ---------------------------------------------------------------------------


class TestSQLiteDBLifecycle:
    def test_connect_returns_connection(self, db):
        assert isinstance(db.connect(), sqlite3.Connection)

    def test_second_connect_returns_same_connection(self, db):
        c1 = db.connect()
        c2 = db.connect()
        assert c1 is c2

    def test_close_releases_connection(self, db):
        db.connect()
        db.close()
        assert db._conn is None

    def test_context_manager_closes_on_exit(self):
        instance = SQLiteDB(":memory:")
        with instance as d:
            d.init()
            assert d._conn is not None
        assert instance._conn is None

    def test_context_manager_returns_self(self):
        instance = SQLiteDB(":memory:")
        with instance as d:
            d.init()
            assert d is instance


# ---------------------------------------------------------------------------
# SQLiteDB: schema
# ---------------------------------------------------------------------------


class TestInit:
    def test_articles_table_created(self, db):
        conn = db.connect()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "articles" in tables

    def test_articles_table_includes_source_and_external_urls(self, db):
        columns = {
            row["name"]
            for row in db.connect().execute("PRAGMA table_info(articles)").fetchall()
        }
        assert "source_url" in columns
        assert "external_url" in columns
        assert "category_confidence" in columns
        assert "category_scores" in columns

    def test_crawled_articles_table_created(self, db):
        conn = db.connect()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "crawled_articles" in tables

    def test_crawled_articles_table_includes_source_and_external_urls(self, db):
        columns = {
            row["name"]
            for row in db.connect().execute("PRAGMA table_info(crawled_articles)").fetchall()
        }
        assert "source_url" in columns
        assert "external_url" in columns
        assert "category_confidence" in columns
        assert "category_scores" in columns

    def test_sources_table_created(self, db):
        conn = db.connect()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "sources" in tables

    def test_crawl_metrics_table_created(self, db):
        conn = db.connect()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "crawl_metrics" in tables

    def test_categories_table_created(self, db):
        conn = db.connect()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "categories" in tables

    def test_bookmarks_table_created(self, db):
        conn = db.connect()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "bookmarks" in tables

    def test_categories_seeded_on_init(self, db):
        rows = db.connect().execute("SELECT slug FROM categories").fetchall()
        slugs = {r["slug"] for r in rows}
        assert {"health", "environment", "community", "science-tech"} <= slugs

    def test_indexes_created(self, db):
        conn = db.connect()
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_articles_source" in indexes
        assert "idx_articles_published" in indexes
        assert "idx_articles_score" in indexes
        assert "idx_crawled_articles_source" in indexes
        assert "idx_crawled_articles_published" in indexes
        assert "idx_crawled_articles_score" in indexes

    def test_init_idempotent(self, db):
        """Calling init() twice must not raise."""
        db.init()
        db.init()

    def test_init_migrates_legacy_article_tables(self, tmp_path):
        """Existing databases without URL metadata columns are migrated on init."""
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                rss_url TEXT,
                active BOOLEAN DEFAULT 1,
                category TEXT
            );
            CREATE TABLE articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                source_url TEXT,
                external_url TEXT,
                content TEXT,
                summary TEXT,
                thumbnail_url TEXT,
                category TEXT,
                category_confidence REAL,
                category_scores TEXT,
                source_id INTEGER NOT NULL,
                uplifting_score REAL,
                published_at DATETIME,
                crawled_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE crawled_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                source_url TEXT,
                external_url TEXT,
                content TEXT,
                summary TEXT,
                thumbnail_url TEXT,
                category TEXT,
                category_confidence REAL,
                category_scores TEXT,
                source_id INTEGER NOT NULL,
                uplifting_score REAL,
                is_uplifting INTEGER,
                published_at DATETIME,
                crawled_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
        conn.close()

        migrated = SQLiteDB(str(db_path))
        migrated.init()
        for table_name in ("articles", "crawled_articles"):
            columns = {
                row["name"]
                for row in migrated.connect().execute(
                    f"PRAGMA table_info({table_name})"
                ).fetchall()
            }
            assert "source_url" in columns
            assert "external_url" in columns
            assert "category_confidence" in columns
            assert "category_scores" in columns


# ---------------------------------------------------------------------------
# Transaction helper
# ---------------------------------------------------------------------------


class TestTransaction:
    def test_transaction_commits_on_success(self, db):
        db.upsert_source(_make_source())
        src_int_id = db.get_source_int_id("TestSource")
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO articles (url, title, source_id) VALUES (?, ?, ?)",
                ("https://example.com/tx-ok", "TX OK", src_int_id),
            )
        row = db.connect().execute(
            "SELECT url FROM articles WHERE url = ?", ("https://example.com/tx-ok",)
        ).fetchone()
        assert row is not None

    def test_transaction_rolls_back_on_exception(self, db):
        db.upsert_source(_make_source())
        src_int_id = db.get_source_int_id("TestSource")
        with pytest.raises(ValueError):
            with db.transaction() as conn:
                conn.execute(
                    "INSERT INTO articles (url, title, source_id) VALUES (?, ?, ?)",
                    ("https://example.com/tx-fail", "TX Fail", src_int_id),
                )
                raise ValueError("Intentional rollback")

        row = db.connect().execute(
            "SELECT url FROM articles WHERE url = ?",
            ("https://example.com/tx-fail",),
        ).fetchone()
        assert row is None


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class TestUpsertSource:
    def test_insert_new_source(self, db):
        int_id = db.upsert_source(_make_source())
        assert isinstance(int_id, int) and int_id > 0
        row = db.connect().execute(
            "SELECT * FROM sources WHERE source_id = ?", ("TestSource",)
        ).fetchone()
        assert row is not None
        assert row["name"] == "Test Source"

    def test_update_existing_source(self, db):
        db.upsert_source(_make_source())
        db.upsert_source(_make_source(name="Updated Name"))
        row = db.connect().execute(
            "SELECT name FROM sources WHERE source_id = ?", ("TestSource",)
        ).fetchone()
        assert row["name"] == "Updated Name"

    def test_upsert_sources_bulk(self, db):
        sources = [
            _make_source(source_id="S1", name="Source 1"),
            _make_source(source_id="S2", name="Source 2"),
        ]
        count = db.upsert_sources(sources)
        assert count == 2
        rows = db.connect().execute("SELECT source_id FROM sources").fetchall()
        ids = {r["source_id"] for r in rows}
        assert {"S1", "S2"} <= ids

    def test_upsert_sources_updates_existing(self, db):
        db.upsert_sources([_make_source(source_id="S1", name="Old")])
        db.upsert_sources([_make_source(source_id="S1", name="New")])
        row = db.connect().execute(
            "SELECT name FROM sources WHERE source_id = ?", ("S1",)
        ).fetchone()
        assert row["name"] == "New"


# ---------------------------------------------------------------------------
# Articles: upsert_article (single)
# ---------------------------------------------------------------------------


class TestUpsertArticle:
    def test_insert_new_article(self, db):
        db.upsert_source(_make_source())
        db.upsert_article(_make_article())
        row = db.connect().execute(
            "SELECT title FROM articles WHERE url = ?",
            ("https://example.com/article/1",),
        ).fetchone()
        assert row["title"] == "A Wonderful Article"

    def test_update_existing_article(self, db):
        db.upsert_source(_make_source())
        db.upsert_article(_make_article())
        db.upsert_article(_make_article(title="Updated Title"))
        row = db.connect().execute(
            "SELECT title FROM articles WHERE url = ?",
            ("https://example.com/article/1",),
        ).fetchone()
        assert row["title"] == "Updated Title"

    def test_created_at_preserved_on_update(self, db):
        db.upsert_source(_make_source())
        db.upsert_article(_make_article())
        created_before = db.connect().execute(
            "SELECT created_at FROM articles WHERE url = ?",
            ("https://example.com/article/1",),
        ).fetchone()["created_at"]

        db.upsert_article(_make_article(title="New Title"))
        created_after = db.connect().execute(
            "SELECT created_at FROM articles WHERE url = ?",
            ("https://example.com/article/1",),
        ).fetchone()["created_at"]

        assert created_before == created_after

    def test_nullable_fields_stored_as_null(self, db):
        db.upsert_source(_make_source())
        db.upsert_article(_make_article(summary=None, content=None, thumbnail_url=None))
        row = db.connect().execute(
            "SELECT summary, content, thumbnail_url FROM articles WHERE url = ?",
            ("https://example.com/article/1",),
        ).fetchone()
        assert row["summary"] is None
        assert row["content"] is None
        assert row["thumbnail_url"] is None

    def test_source_and_external_urls_stored(self, db):
        db.upsert_source(_make_source())
        db.upsert_article(
            _make_article(
                source_url="https://www.reddit.com/r/science/comments/1sbjly5/vegetation_traps_nearly_3x_more_microplastic_than/",
                external_url="https://example.com/original-story",
                category_confidence=0.82,
                category_scores={"Health": 0.82, "Community": 0.12},
            )
        )
        row = db.connect().execute(
            "SELECT source_url, external_url, category_confidence, category_scores FROM articles WHERE url = ?",
            ("https://example.com/article/1",),
        ).fetchone()
        assert row["source_url"] == "https://example.com/original-story"
        assert row["external_url"] == "https://example.com/original-story"
        assert row["category_confidence"] == pytest.approx(0.82)
        assert row["category_scores"] is not None

    def test_uplifting_score_stored(self, db):
        db.upsert_source(_make_source())
        db.upsert_article(_make_article(uplifting_score=0.87))
        row = db.connect().execute(
            "SELECT uplifting_score FROM articles WHERE url = ?",
            ("https://example.com/article/1",),
        ).fetchone()
        assert abs(row["uplifting_score"] - 0.87) < 1e-9


# ---------------------------------------------------------------------------
# Articles: upsert_articles (bulk)
# ---------------------------------------------------------------------------


class TestUpsertArticles:
    def test_empty_list_returns_zero(self, db):
        count = db.upsert_articles([])
        assert count == 0

    def test_bulk_insert(self, db):
        db.upsert_source(_make_source())
        articles = [
            _make_article(url=f"https://example.com/{i}", title=f"Article {i}")
            for i in range(10)
        ]
        count = db.upsert_articles(articles)
        assert count == 10

    def test_bulk_upsert_updates_existing(self, db):
        db.upsert_source(_make_source())
        db.upsert_articles([_make_article(title="Original")])
        db.upsert_articles([_make_article(title="Updated")])
        row = db.connect().execute(
            "SELECT title FROM articles WHERE url = ?",
            ("https://example.com/article/1",),
        ).fetchone()
        assert row["title"] == "Updated"

    def test_bulk_insert_is_atomic_on_error(self, db):
        """If source_id cannot be resolved, ValueError is raised and nothing is written."""
        # Do NOT insert the source — upsert_articles raises ValueError for unknown source_id.
        articles = [
            _make_article(url="https://example.com/ok", source_id="MISSING"),
            _make_article(url="https://example.com/also-ok", source_id="MISSING"),
        ]
        with pytest.raises(ValueError, match="MISSING"):
            db.upsert_articles(articles)

        count = db.connect().execute(
            "SELECT COUNT(*) FROM articles WHERE url IN (?, ?)",
            ("https://example.com/ok", "https://example.com/also-ok"),
        ).fetchone()[0]
        assert count == 0

    def test_performance_100_articles(self, db):
        """100 articles should be written in under 500ms."""
        import time

        db.upsert_source(_make_source())
        articles = [
            _make_article(url=f"https://example.com/perf/{i}", title=f"Perf {i}")
            for i in range(100)
        ]
        start = time.monotonic()
        db.upsert_articles(articles)
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms < 500, f"Took {elapsed_ms:.1f}ms for 100 articles"


class TestUpsertCrawledArticles:
    def test_bulk_insert_raw_articles(self, db):
        db.upsert_source(_make_source())
        articles = [
            _make_article(url=f"https://example.com/raw/{i}", title=f"Raw {i}", is_uplifting=bool(i % 2))
            for i in range(4)
        ]
        count = db.upsert_crawled_articles(articles)
        assert count == 4
        row = db.connect().execute(
            "SELECT COUNT(*) FROM crawled_articles"
        ).fetchone()[0]
        assert row == 4

    def test_bulk_insert_raw_articles_preserves_urls(self, db):
        db.upsert_source(_make_source())
        article = _make_article(
            source_url="https://www.reddit.com/r/science/comments/1sbjly5/vegetation_traps_nearly_3x_more_microplastic_than/",
            external_url=None,
            category_confidence=0.61,
            category_scores={"Science & Tech": 0.61},
        )
        db.upsert_crawled_articles([article])
        row = db.connect().execute(
            "SELECT source_url, external_url, category_confidence, category_scores FROM crawled_articles WHERE url = ?",
            ("https://example.com/article/1",),
        ).fetchone()
        assert row["source_url"].startswith("https://www.reddit.com/")
        assert row["external_url"] is None
        assert row["category_confidence"] == pytest.approx(0.61)
        assert row["category_scores"] is not None

    def test_purge_non_uplifting_articles(self, db):
        db.upsert_source(_make_source())
        db.upsert_articles([
            _make_article(url="https://example.com/keep", uplifting_score=0.9),
            _make_article(url="https://example.com/drop", uplifting_score=0.2),
        ])
        purged = db.purge_non_uplifting_articles(0.75)
        assert purged == 1
        urls = {
            r["url"]
            for r in db.connect().execute("SELECT url FROM articles").fetchall()
        }
        assert urls == {"https://example.com/keep"}


# ---------------------------------------------------------------------------
# article_exists / get_existing_urls
# ---------------------------------------------------------------------------


class TestArticleExists:
    def test_returns_false_when_not_present(self, db):
        assert db.article_exists("https://example.com/missing") is False

    def test_returns_true_after_insert(self, db):
        db.upsert_source(_make_source())
        db.upsert_article(_make_article())
        assert db.article_exists("https://example.com/article/1") is True

    def test_get_existing_urls_empty_input(self, db):
        assert db.get_existing_urls([]) == set()

    def test_get_existing_urls_filters_correctly(self, db):
        db.upsert_source(_make_source())
        db.upsert_article(_make_article(url="https://example.com/exists"))
        result = db.get_existing_urls(
            ["https://example.com/exists", "https://example.com/new"]
        )
        assert result == {"https://example.com/exists"}

    def test_get_existing_urls_all_new(self, db):
        result = db.get_existing_urls(
            ["https://example.com/a", "https://example.com/b"]
        )
        assert result == set()


# ---------------------------------------------------------------------------
# Crawl metrics
# ---------------------------------------------------------------------------


class TestCrawlMetrics:
    def test_record_inserts_row(self, db):
        start = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 10, 5, 0, tzinfo=timezone.utc)
        row_id = db.record_crawl_metrics(
            crawl_start=start,
            crawl_end=end,
            articles_fetched=100,
            articles_classified=80,
            articles_stored=75,
            avg_classification_score=0.83,
            errors=None,
        )
        assert row_id == 1
        row = db.connect().execute(
            "SELECT * FROM crawl_metrics WHERE id = ?", (row_id,)
        ).fetchone()
        assert row["articles_fetched"] == 100
        assert row["articles_stored"] == 75

    def test_record_multiple_crawls(self, db):
        ts = datetime(2026, 3, 1, tzinfo=timezone.utc)
        db.record_crawl_metrics(ts, ts, 10, 8, 7)
        db.record_crawl_metrics(ts, ts, 20, 15, 14)
        count = db.connect().execute(
            "SELECT COUNT(*) FROM crawl_metrics"
        ).fetchone()[0]
        assert count == 2

    def test_errors_field_stored(self, db):
        ts = datetime(2026, 3, 1, tzinfo=timezone.utc)
        db.record_crawl_metrics(ts, ts, 0, 0, 0, errors="DB write failed")
        row = db.connect().execute("SELECT errors FROM crawl_metrics").fetchone()
        assert row["errors"] == "DB write failed"

    def test_optional_avg_score_is_null_when_not_provided(self, db):
        ts = datetime(2026, 3, 1, tzinfo=timezone.utc)
        db.record_crawl_metrics(ts, ts, 0, 0, 0)
        row = db.connect().execute(
            "SELECT avg_classification_score FROM crawl_metrics"
        ).fetchone()
        assert row["avg_classification_score"] is None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestInitDb:
    def test_init_db_returns_sqlitedb_instance(self):
        db = init_db(":memory:")
        assert isinstance(db, SQLiteDB)
        db.close()

    def test_init_db_schema_applied(self):
        db = init_db(":memory:")
        tables = {
            row[0]
            for row in db.connect()
            .execute("SELECT name FROM sqlite_master WHERE type='table'")
            .fetchall()
        }
        assert {"articles", "crawled_articles", "sources", "crawl_metrics", "categories", "bookmarks"} <= tables
        db.close()


class TestWriteArticles:
    def test_write_articles_convenience(self):
        db = init_db(":memory:")
        db.upsert_source(_make_source())
        count = write_articles([_make_article()], db=db)
        assert count == 1
        db.close()

    def test_write_articles_empty(self):
        db = init_db(":memory:")
        count = write_articles([], db=db)
        assert count == 0
        db.close()

    def test_write_articles_creates_db_when_no_instance(self, tmp_path):
        path = str(tmp_path / "test.db")
        db_instance = init_db(path)
        db_instance.upsert_source(_make_source())
        db_instance.close()

        count = write_articles([_make_article()], db_path=path)
        assert count == 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestArticleToRow:
    def test_required_fields_present(self):
        row = _article_to_row(_make_article())
        assert "url" in row
        assert "title" in row
        assert "source_id" in row

    def test_missing_optional_fields_default_to_none(self):
        row = _article_to_row({"url": "https://x.com", "title": "T", "source_id": "S"})
        assert row["summary"] is None
        assert row["thumbnail_url"] is None
        assert row["content"] is None
        assert row["category"] is None

    def test_datetime_published_at_converted_to_string(self):
        dt = datetime(2026, 1, 15, 9, 30, tzinfo=timezone.utc)
        row = _article_to_row(_make_article(published_at=dt))
        assert isinstance(row["published_at"], str)
        assert "2026-01-15" in row["published_at"]

    def test_none_published_at_stays_none(self):
        row = _article_to_row(_make_article(published_at=None))
        assert row["published_at"] is None

    def test_string_published_at_preserved(self):
        row = _article_to_row(_make_article(published_at="2026-03-01 00:00:00"))
        assert row["published_at"] == "2026-03-01 00:00:00"


class TestCoerceDt:
    def test_none_returns_none(self):
        assert _coerce_dt(None) is None

    def test_datetime_returns_string(self):
        dt = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = _coerce_dt(dt)
        assert isinstance(result, str)
        assert "2026-06-01" in result

    def test_string_returned_as_is(self):
        assert _coerce_dt("2026-01-01") == "2026-01-01"


class TestFmtDt:
    def test_format(self):
        dt = datetime(2026, 3, 25, 14, 30, 45, tzinfo=timezone.utc)
        assert _fmt_dt(dt) == "2026-03-25 14:30:45"


class TestCategoryToSlug:
    def test_new_labels_return_correct_slug(self):
        assert _category_to_slug("Health") == "Health"
        assert _category_to_slug("Science & Tech") == "Science & Tech"
        assert _category_to_slug("Arts & Culture") == "Arts & Culture"

    def test_legacy_labels_mapped_to_slug(self):
        assert _category_to_slug("Health & Wellness") == "Health"
        assert _category_to_slug("Technology & Science") == "Science & Tech"
        assert _category_to_slug("Culture & Arts") == "Arts & Culture"
        assert _category_to_slug("Community & Social Good") == "Community"

    def test_none_returns_none(self):
        assert _category_to_slug(None) is None

    def test_empty_string_returns_none(self):
        assert _category_to_slug("") is None

    def test_unknown_category_falls_back_to_slugified(self):
        result = _category_to_slug("Custom Category")
        assert isinstance(result, str)
        assert result == "Custom Category"


class TestGetSourceIntId:
    def test_returns_integer_id_for_known_source(self, db):
        db.upsert_source(_make_source())
        int_id = db.get_source_int_id("TestSource")
        assert isinstance(int_id, int) and int_id > 0

    def test_returns_none_for_unknown_source(self, db):
        assert db.get_source_int_id("NonExistentSource") is None

    def test_bulk_map_returns_all_known(self, db):
        db.upsert_source(_make_source(source_id="S1", name="Source 1"))
        db.upsert_source(_make_source(source_id="S2", name="Source 2"))
        result = db.get_source_int_id_map(["S1", "S2", "UNKNOWN"])
        assert "S1" in result
        assert "S2" in result
        assert "UNKNOWN" not in result

    def test_upsert_article_raises_for_unknown_source(self, db):
        with pytest.raises(ValueError, match="unknown_src"):
            db.upsert_article(_make_article(source_id="unknown_src"))


# ---------------------------------------------------------------------------
# Issue #17: seed_sources — sources.yaml seeded on init
# ---------------------------------------------------------------------------


class TestSeedSources:
    def test_sources_seeded_on_init(self, db):
        """init() must populate the sources table from sources.yaml."""
        rows = db.connect().execute("SELECT source_id FROM sources").fetchall()
        ids = {r["source_id"] for r in rows}
        # A sample of known source_ids from config/sources.yaml
        assert "BBCNews" in ids
        assert "NPRNews" in ids
        assert "HackerNews" in ids

    def test_seeded_sources_have_required_fields(self, db):
        row = db.connect().execute(
            "SELECT * FROM sources WHERE source_id = ?", ("BBCNews",)
        ).fetchone()
        assert row is not None
        assert row["name"] == "BBC News"
        assert row["rss_url"] is not None
        assert row["active"] in (0, 1)

    def test_seed_sources_idempotent(self, db):
        """Calling init() again must not raise or duplicate rows."""
        db.init()
        count_before = db.connect().execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        db.init()
        count_after = db.connect().execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        assert count_before == count_after

    def test_seeded_source_id_resolves_to_integer(self, db):
        """Articles can be inserted using a seeded source_id string."""
        int_id = db.get_source_int_id("BBCNews")
        assert isinstance(int_id, int) and int_id > 0

    def test_article_insert_with_seeded_source(self, db):
        """upsert_article works with source_ids loaded from sources.yaml."""
        article = _make_article(source_id="BBCNews", url="https://bbc.com/test/1")
        db.upsert_article(article)
        row = db.connect().execute(
            "SELECT title FROM articles WHERE url = ?", ("https://bbc.com/test/1",)
        ).fetchone()
        assert row is not None


# Allow running directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
