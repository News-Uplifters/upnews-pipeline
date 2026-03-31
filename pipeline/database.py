"""SQLite database layer for the upnews pipeline.

Provides:
- Unified schema compatible with upnews-api (sources integer PK, articles.content,
  updated_at, categories table, bookmarks table)
- UPSERT for articles (insert-or-update by URL)
- Bulk write with transaction support
- Source upsert for keeping the sources table in sync with config
- Crawl-metrics recording
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Generator, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL – unified schema shared with upnews-api
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    domain      TEXT,
    rss_url     TEXT,
    url         TEXT,
    active      BOOLEAN DEFAULT 1,
    category    TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS categories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    slug        TEXT NOT NULL UNIQUE,
    description TEXT,
    icon        TEXT,
    color       TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS articles (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    title            TEXT NOT NULL,
    url              TEXT NOT NULL UNIQUE,
    content          TEXT,
    summary          TEXT,
    thumbnail_url    TEXT,
    category         TEXT,
    source_id        INTEGER NOT NULL,
    uplifting_score  REAL,
    published_at     DATETIME,
    crawled_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(id)
);

CREATE TABLE IF NOT EXISTS crawl_metrics (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    crawl_start              DATETIME,
    crawl_end                DATETIME,
    articles_fetched         INTEGER,
    articles_classified      INTEGER,
    articles_stored          INTEGER,
    avg_classification_score REAL,
    errors                   TEXT
);

CREATE TABLE IF NOT EXISTS bookmarks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id   INTEGER NOT NULL,
    session_uuid TEXT NOT NULL,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (article_id) REFERENCES articles(id),
    UNIQUE(article_id, session_uuid)
);

CREATE INDEX IF NOT EXISTS idx_articles_source       ON articles(source_id);
CREATE INDEX IF NOT EXISTS idx_articles_published    ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_score        ON articles(uplifting_score);
CREATE INDEX IF NOT EXISTS idx_articles_category     ON articles(category);
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_bookmarks_session_uuid ON bookmarks(session_uuid);
"""

# Default categories matching upnews-api seed data
_CATEGORIES_SEED = [
    ("Science & Tech",  "science-tech",  "Scientific discoveries, technology breakthroughs, and innovation", "🔬", "#3498db"),
    ("Health",          "health",        "Medical advances, wellness, and health recovery stories",           "💊", "#2ecc71"),
    ("Environment",     "environment",   "Environmental conservation, wildlife protection, and sustainability","🌍", "#27ae60"),
    ("Community",       "community",     "Human kindness, volunteering, charity, and community support",      "🤝", "#e74c3c"),
    ("Education",       "education",     "Educational achievements, scholarships, and learning opportunities", "📚", "#f39c12"),
    ("Sports",          "sports",        "Athletic achievements, inspiring sports stories, and competitions",  "⚽", "#9b59b6"),
    ("Arts & Culture",  "arts-culture",  "Creative expression, cultural celebrations, and artistic achievements","🎨", "#1abc9c"),
]

# Slug lookup used by _article_to_row — covers both new and legacy category names
_CATEGORY_SLUG_MAP: Dict[str, str] = {
    # Current labels (matching DEFAULT_CATEGORIES in enrichment/categorizer.py)
    "Science & Tech":         "science-tech",
    "Health":                 "health",
    "Environment":            "environment",
    "Community":              "community",
    "Education":              "education",
    "Sports":                 "sports",
    "Arts & Culture":         "arts-culture",
    # Legacy pipeline labels (backwards compatibility)
    "Health & Wellness":      "health",
    "Environment & Nature":   "environment",
    "Community & Social Good":"community",
    "Technology & Science":   "science-tech",
    "Business & Economics":   "science-tech",
    "Culture & Arts":         "arts-culture",
    "Human Interest":         "community",
}

# ---------------------------------------------------------------------------
# SQLiteDB
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = "data/upnews.db"


class SQLiteDB:
    """Thin wrapper around a SQLite connection for the upnews pipeline.

    Usage::

        db = SQLiteDB("data/upnews.db")
        db.init()
        db.upsert_articles([{"url": "...", "title": "...", "source_id": "BBCNews"}])
        db.close()

    Or as a context manager::

        with SQLiteDB("data/upnews.db") as db:
            db.upsert_articles(articles)
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """Open (or return existing) database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self) -> None:
        """Close the database connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SQLiteDB":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Create schema if it does not already exist, then seed categories."""
        conn = self.connect()
        conn.executescript(_DDL)
        conn.commit()
        self._seed_categories()
        logger.info("Database initialised at %s", self.db_path)

    def _seed_categories(self) -> None:
        """Populate default categories if they don't already exist."""
        conn = self.connect()
        conn.executemany(
            """
            INSERT OR IGNORE INTO categories (name, slug, description, icon, color)
            VALUES (?, ?, ?, ?, ?)
            """,
            _CATEGORIES_SEED,
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Context manager for transactions
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield the connection inside a transaction block."""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    def upsert_source(self, source: Dict) -> int:
        """Insert or update a single source record.

        Returns:
            The integer ``id`` of the inserted or existing row.
        """
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO sources (source_id, name, domain, rss_url, url, active, category)
            VALUES (:source_id, :name, :domain, :rss_url, :url, :active, :category)
            ON CONFLICT(source_id) DO UPDATE SET
                name     = excluded.name,
                domain   = excluded.domain,
                rss_url  = excluded.rss_url,
                url      = excluded.url,
                active   = excluded.active,
                category = excluded.category
            """,
            {
                "source_id": source.get("source_id", ""),
                "name":      source.get("name", ""),
                "domain":    source.get("domain"),
                "rss_url":   source.get("rss_url"),
                "url":       source.get("url"),
                "active":    int(bool(source.get("active", True))),
                "category":  source.get("category"),
            },
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM sources WHERE source_id = ?", (source.get("source_id", ""),)
        ).fetchone()
        return row["id"] if row else -1

    def upsert_sources(self, sources: Sequence[Dict]) -> int:
        """Bulk upsert a list of source dicts. Returns count of rows affected."""
        with self.transaction() as conn:
            cursor = conn.executemany(
                """
                INSERT INTO sources (source_id, name, domain, rss_url, url, active, category)
                VALUES (:source_id, :name, :domain, :rss_url, :url, :active, :category)
                ON CONFLICT(source_id) DO UPDATE SET
                    name     = excluded.name,
                    domain   = excluded.domain,
                    rss_url  = excluded.rss_url,
                    url      = excluded.url,
                    active   = excluded.active,
                    category = excluded.category
                """,
                [
                    {
                        "source_id": s.get("source_id", ""),
                        "name":      s.get("name", ""),
                        "domain":    s.get("domain"),
                        "rss_url":   s.get("rss_url"),
                        "url":       s.get("url"),
                        "active":    int(bool(s.get("active", True))),
                        "category":  s.get("category"),
                    }
                    for s in sources
                ],
            )
        return cursor.rowcount

    def get_source_int_id(self, source_key: str) -> Optional[int]:
        """Return the integer ``id`` for a source identified by its text key."""
        conn = self.connect()
        row = conn.execute(
            "SELECT id FROM sources WHERE source_id = ? LIMIT 1", (source_key,)
        ).fetchone()
        return row["id"] if row else None

    def get_source_int_id_map(self, source_keys: List[str]) -> Dict[str, int]:
        """Bulk-resolve string source keys to integer ids."""
        if not source_keys:
            return {}
        placeholders = ",".join("?" * len(source_keys))
        rows = self.connect().execute(
            f"SELECT id, source_id FROM sources WHERE source_id IN ({placeholders})",
            list(source_keys),
        ).fetchall()
        return {row["source_id"]: row["id"] for row in rows}

    # ------------------------------------------------------------------
    # Articles
    # ------------------------------------------------------------------

    def upsert_article(self, article: Dict) -> None:
        """Insert or update a single article.

        The string ``source_id`` is resolved to an integer FK automatically.
        """
        str_key = article.get("source_id", "")
        int_id = self.get_source_int_id(str_key)
        if int_id is None:
            raise ValueError(
                f"source_id {str_key!r} not found in sources table — "
                "call upsert_source() first."
            )
        row = _article_to_row(article)
        row["source_id"] = int_id
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO articles
                (title, url, content, summary, thumbnail_url, category,
                 source_id, uplifting_score, published_at, crawled_at, updated_at)
            VALUES
                (:title, :url, :content, :summary, :thumbnail_url, :category,
                 :source_id, :uplifting_score, :published_at, :crawled_at, :updated_at)
            ON CONFLICT(url) DO UPDATE SET
                title           = excluded.title,
                content         = excluded.content,
                summary         = excluded.summary,
                thumbnail_url   = excluded.thumbnail_url,
                category        = excluded.category,
                uplifting_score = excluded.uplifting_score,
                published_at    = excluded.published_at,
                crawled_at      = excluded.crawled_at,
                updated_at      = excluded.updated_at
            """,
            row,
        )
        conn.commit()

    def upsert_articles(self, articles: Sequence[Dict]) -> int:
        """Bulk upsert a list of article dicts inside a single transaction.

        String ``source_id`` values are resolved to integer FKs before the
        insert.  Raises ``ValueError`` if any source_id is not found in the
        sources table.

        Returns:
            Number of rows inserted or updated.

        Raises:
            ValueError: If a source_id string cannot be resolved to an integer id.
            sqlite3.Error: On any database error; the transaction is rolled back.
        """
        if not articles:
            return 0

        # Bulk-resolve string source keys → integer ids
        source_keys = list({a.get("source_id", "") for a in articles if a.get("source_id")})
        source_id_map = self.get_source_int_id_map(source_keys)

        rows = []
        for a in articles:
            str_key = a.get("source_id", "")
            int_id = source_id_map.get(str_key)
            if int_id is None:
                raise ValueError(
                    f"source_id {str_key!r} not found in sources table — "
                    "call upsert_source() first."
                )
            row = _article_to_row(a)
            row["source_id"] = int_id
            rows.append(row)

        with self.transaction() as conn:
            cursor = conn.executemany(
                """
                INSERT INTO articles
                    (title, url, content, summary, thumbnail_url, category,
                     source_id, uplifting_score, published_at, crawled_at, updated_at)
                VALUES
                    (:title, :url, :content, :summary, :thumbnail_url, :category,
                     :source_id, :uplifting_score, :published_at, :crawled_at, :updated_at)
                ON CONFLICT(url) DO UPDATE SET
                    title           = excluded.title,
                    content         = excluded.content,
                    summary         = excluded.summary,
                    thumbnail_url   = excluded.thumbnail_url,
                    category        = excluded.category,
                    uplifting_score = excluded.uplifting_score,
                    published_at    = excluded.published_at,
                    crawled_at      = excluded.crawled_at,
                    updated_at      = excluded.updated_at
                """,
                rows,
            )
        count = cursor.rowcount
        logger.info("Upserted %d articles into %s", count, self.db_path)
        return count

    def article_exists(self, url: str) -> bool:
        """Return True if an article with the given URL is in the DB."""
        conn = self.connect()
        row = conn.execute(
            "SELECT 1 FROM articles WHERE url = ? LIMIT 1", (url,)
        ).fetchone()
        return row is not None

    def get_existing_urls(self, urls: Sequence[str]) -> set:
        """Return the subset of *urls* that already exist in the DB."""
        if not urls:
            return set()
        conn = self.connect()
        placeholders = ",".join("?" * len(urls))
        rows = conn.execute(
            f"SELECT url FROM articles WHERE url IN ({placeholders})",
            list(urls),
        ).fetchall()
        return {row["url"] for row in rows}

    # ------------------------------------------------------------------
    # Crawl metrics
    # ------------------------------------------------------------------

    def record_crawl_metrics(
        self,
        crawl_start: datetime,
        crawl_end: datetime,
        articles_fetched: int,
        articles_classified: int,
        articles_stored: int,
        avg_classification_score: Optional[float] = None,
        errors: Optional[str] = None,
    ) -> int:
        """Insert a crawl_metrics row and return its new id."""
        conn = self.connect()
        cursor = conn.execute(
            """
            INSERT INTO crawl_metrics
                (crawl_start, crawl_end, articles_fetched,
                 articles_classified, articles_stored,
                 avg_classification_score, errors)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _fmt_dt(crawl_start),
                _fmt_dt(crawl_end),
                articles_fetched,
                articles_classified,
                articles_stored,
                avg_classification_score,
                errors,
            ),
        )
        conn.commit()
        return cursor.lastrowid


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------


def init_db(db_path: str = _DEFAULT_DB_PATH) -> SQLiteDB:
    """Create (or open) the database at *db_path*, apply schema, return instance."""
    db = SQLiteDB(db_path)
    db.init()
    return db


def write_articles(
    articles: Sequence[Dict],
    db_path: str = _DEFAULT_DB_PATH,
    db: Optional[SQLiteDB] = None,
) -> int:
    """Write *articles* to the database, creating it if necessary."""
    if db is not None:
        return db.upsert_articles(articles)

    with SQLiteDB(db_path) as _db:
        _db.init()
        return _db.upsert_articles(articles)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _article_to_row(article: Dict) -> Dict:
    """Convert an article dict to the param dict expected by the INSERT.

    ``source_id`` is left as-is here (string); callers must replace it with
    the resolved integer id before executing SQL.
    """
    return {
        "title":          article.get("title", ""),
        "url":            article.get("url") or article.get("original_url", ""),
        # Support both 'content' (new) and 'body' (legacy crawler field)
        "content":        article.get("content") or article.get("body"),
        "summary":        article.get("summary"),
        "thumbnail_url":  article.get("thumbnail_url"),
        "category":       _category_to_slug(article.get("category")),
        "source_id":      article.get("source_id", ""),  # replaced by caller
        "uplifting_score": article.get("uplifting_score"),
        "published_at":   _coerce_dt(article.get("published_at")),
        "crawled_at":     _fmt_dt(datetime.now(timezone.utc)),
        "updated_at":     _fmt_dt(datetime.now(timezone.utc)),
    }


def _category_to_slug(category: Optional[str]) -> Optional[str]:
    """Convert a human-readable category name to its URL slug."""
    if not category:
        return None
    return _CATEGORY_SLUG_MAP.get(
        category,
        category.lower().replace(" & ", "-").replace(" ", "-"),
    )


def _coerce_dt(value) -> Optional[str]:
    """Convert a datetime / ISO string / None to an ISO-format string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return _fmt_dt(value)
    return str(value)


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")
