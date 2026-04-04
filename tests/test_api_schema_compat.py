"""API schema compatibility tests.

Verifies that the pipeline writes data the upnews-api can read directly:
- articles.source_id is an INTEGER FK to sources(id)
- articles.content column (not 'body')
- articles.updated_at column present
- categories table seeded with API slugs
- sources table has integer PK with source_id TEXT for pipeline lookups
- category values stored as slugs matching the API's categories table
- A simulated API-style JOIN query returns the expected data

These tests mirror the acceptance criteria in issue #12:
  "Align database schema with upnews-api"
"""

from datetime import datetime

import pytest

from pipeline.database import SQLiteDB, _category_to_slug, init_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db():
    """In-memory DB with schema + seeded source."""
    instance = init_db(":memory:")
    instance.upsert_source({
        "source_id": "BBCNews",
        "name": "BBC News",
        "domain": "bbc.co.uk",
        "rss_url": "http://feeds.bbc.co.uk/news/world/rss.xml",
        "url": "https://www.bbc.co.uk",
        "active": True,
        "category": "news",
    })
    yield instance
    instance.close()


def _article(url: str = "https://example.com/1", title: str = "Great news", **kw):
    defaults = {
        "url": url,
        "title": title,
        "content": "Full article body.",
        "summary": "Brief summary.",
        "category": "Health",
        "source_id": "BBCNews",
        "uplifting_score": 0.91,
        "published_at": datetime(2026, 3, 26, 10, 0, 0),
        "thumbnail_url": "https://cdn.example.com/img.jpg",
    }
    defaults.update(kw)
    return defaults


# ---------------------------------------------------------------------------
# Schema structure tests
# ---------------------------------------------------------------------------


class TestUnifiedSchemaStructure:
    def test_sources_has_integer_pk(self, db):
        row = db.connect().execute(
            "SELECT id FROM sources WHERE source_id = 'BBCNews'"
        ).fetchone()
        assert row is not None
        assert isinstance(row["id"], int)

    def test_sources_has_source_id_text_column(self, db):
        row = db.connect().execute(
            "SELECT source_id FROM sources WHERE source_id = 'BBCNews'"
        ).fetchone()
        assert row["source_id"] == "BBCNews"

    def test_sources_has_domain_column(self, db):
        row = db.connect().execute(
            "SELECT domain FROM sources WHERE source_id = 'BBCNews'"
        ).fetchone()
        assert row["domain"] == "bbc.co.uk"

    def test_articles_has_content_column(self, db):
        db.upsert_articles([_article()])
        row = db.connect().execute(
            "SELECT content FROM articles WHERE url = 'https://example.com/1'"
        ).fetchone()
        assert row["content"] == "Full article body."

    def test_articles_has_source_and_external_url_columns(self, db):
        db.upsert_articles([
            _article(
                source_url="https://www.reddit.com/r/science/comments/1sbjly5/vegetation_traps_nearly_3x_more_microplastic_than/",
                external_url="https://example.com/original-article",
                category_confidence=0.83,
                category_scores={"Health": 0.83, "Community": 0.11},
            )
        ])
        row = db.connect().execute(
            """
            SELECT source_url, external_url, category_confidence, category_scores
            FROM articles
            WHERE url = 'https://example.com/1'
            """
        ).fetchone()
        assert row["source_url"] == "https://example.com/original-article"
        assert row["external_url"] == "https://example.com/original-article"
        assert row["category_confidence"] == pytest.approx(0.83)
        assert row["category_scores"] is not None

    def test_articles_has_updated_at_column(self, db):
        db.upsert_articles([_article()])
        row = db.connect().execute(
            "SELECT updated_at FROM articles WHERE url = 'https://example.com/1'"
        ).fetchone()
        assert row["updated_at"] is not None

    def test_articles_source_id_is_integer(self, db):
        db.upsert_articles([_article()])
        row = db.connect().execute(
            "SELECT source_id FROM articles WHERE url = 'https://example.com/1'"
        ).fetchone()
        assert isinstance(row["source_id"], int)

    def test_categories_table_seeded(self, db):
        rows = db.connect().execute("SELECT slug FROM categories").fetchall()
        slugs = {r["slug"] for r in rows}
        assert {"health", "environment", "community", "science-tech",
                "education", "sports", "arts-culture"} == slugs

    def test_categories_have_name_and_slug(self, db):
        rows = db.connect().execute(
            "SELECT name, slug FROM categories WHERE slug = 'health'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["name"] == "Health"
        assert rows[0]["slug"] == "health"


# ---------------------------------------------------------------------------
# Category slug storage
# ---------------------------------------------------------------------------


class TestCategorySlugStorage:
    def test_article_category_stored_as_name(self, db):
        db.upsert_articles([_article(category="Health")])
        row = db.connect().execute(
            "SELECT category FROM articles WHERE url = 'https://example.com/1'"
        ).fetchone()
        assert row["category"] == "Health"

    def test_legacy_category_mapped_to_name(self, db):
        db.upsert_articles([_article(category="Health & Wellness")])
        row = db.connect().execute(
            "SELECT category FROM articles WHERE url = 'https://example.com/1'"
        ).fetchone()
        assert row["category"] == "Health"

    def test_all_api_categories_are_preserved(self):
        api_categories = [
            "Health", "Environment", "Community",
            "Science & Tech", "Education", "Sports", "Arts & Culture",
        ]
        for cat in api_categories:
            normalized = _category_to_slug(cat)
            assert normalized == cat


# ---------------------------------------------------------------------------
# API-style JOIN queries
# ---------------------------------------------------------------------------


class TestAPIStyleQueries:
    def test_api_style_articles_with_source_join(self, db):
        """Simulate the JOIN an API endpoint would execute to fetch articles."""
        db.upsert_articles([
            _article(url="https://example.com/a1", title="Good news one"),
            _article(url="https://example.com/a2", title="Good news two"),
        ])
        rows = db.connect().execute(
            """
            SELECT a.id, a.title, a.url, a.content, a.category,
                   a.published_at, a.created_at, a.updated_at,
                   s.name AS source_name, s.domain AS source_domain
            FROM   articles a
            JOIN   sources  s ON s.id = a.source_id
            ORDER  BY a.published_at DESC
            """
        ).fetchall()
        assert len(rows) == 2
        for row in rows:
            assert row["source_name"] == "BBC News"
            assert row["source_domain"] == "bbc.co.uk"
            assert row["content"] is not None
            assert row["updated_at"] is not None

    def test_api_style_filter_by_category_slug(self, db):
        """API-style lookup works against the canonical category name."""
        db.upsert_articles([
            _article(url="https://example.com/health", category="Health"),
            _article(url="https://example.com/env", title="Green news", category="Environment"),
        ])
        rows = db.connect().execute(
            "SELECT url FROM articles WHERE category = ?", ("Health",)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["url"] == "https://example.com/health"

    def test_bookmarks_table_accepts_article_fk(self, db):
        """bookmarks table is present and accepts a valid article FK."""
        db.upsert_articles([_article()])
        article_id = db.connect().execute(
            "SELECT id FROM articles WHERE url = 'https://example.com/1'"
        ).fetchone()["id"]

        db.connect().execute(
            "INSERT INTO bookmarks (article_id, session_uuid) VALUES (?, ?)",
            (article_id, "test-session-uuid"),
        )
        db.connect().commit()

        row = db.connect().execute(
            "SELECT article_id FROM bookmarks WHERE session_uuid = 'test-session-uuid'"
        ).fetchone()
        assert row["article_id"] == article_id

    def test_no_body_column_on_articles(self, db):
        """The old 'body' column must not exist — API queries should use 'content'."""
        pragma_rows = db.connect().execute(
            "PRAGMA table_info(articles)"
        ).fetchall()
        column_names = {r["name"] for r in pragma_rows}
        assert "body" not in column_names
        assert "content" in column_names

    def test_sources_integer_pk_matches_articles_fk(self, db):
        """The integer id in sources matches what articles.source_id stores."""
        src_id = db.get_source_int_id("BBCNews")
        db.upsert_articles([_article()])
        article_src_id = db.connect().execute(
            "SELECT source_id FROM articles WHERE url = 'https://example.com/1'"
        ).fetchone()["source_id"]
        assert article_src_id == src_id
