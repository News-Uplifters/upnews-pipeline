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
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence

import yaml

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
    source_url       TEXT,
    external_url     TEXT,
    content          TEXT,
    summary          TEXT,
    thumbnail_url    TEXT,
    category         TEXT,
    category_confidence REAL,
    category_scores  TEXT,
    source_id        INTEGER NOT NULL,
    uplifting_score  REAL,
    published_at     DATETIME,
    crawled_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(id)
);

CREATE TABLE IF NOT EXISTS crawled_articles (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    title            TEXT NOT NULL,
    url              TEXT NOT NULL UNIQUE,
    source_url       TEXT,
    external_url     TEXT,
    content          TEXT,
    summary          TEXT,
    thumbnail_url    TEXT,
    category         TEXT,
    category_confidence REAL,
    category_scores  TEXT,
    source_id        INTEGER NOT NULL,
    uplifting_score  REAL,
    is_uplifting     INTEGER,
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
CREATE INDEX IF NOT EXISTS idx_crawled_articles_source       ON crawled_articles(source_id);
CREATE INDEX IF NOT EXISTS idx_crawled_articles_published    ON crawled_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_crawled_articles_score        ON crawled_articles(uplifting_score);
CREATE INDEX IF NOT EXISTS idx_crawled_articles_category     ON crawled_articles(category);
CREATE INDEX IF NOT EXISTS idx_crawled_articles_published_at ON crawled_articles(published_at);
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
        """Create schema if it does not already exist, then seed categories and sources."""
        conn = self.connect()
        conn.executescript(_DDL)
        conn.commit()
        self._migrate_schema()
        self._seed_categories()
        self._seed_sources()
        logger.info("Database initialised at %s", self.db_path)

    def _migrate_schema(self) -> None:
        """Add any newer article columns to existing databases."""
        conn = self.connect()
        for table_name in ("articles", "crawled_articles"):
            existing_columns = {
                row["name"]
                for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            }
            for column_name, column_type in (
                ("source_url", "TEXT"),
                ("external_url", "TEXT"),
                ("category_confidence", "REAL"),
                ("category_scores", "TEXT"),
            ):
                if column_name not in existing_columns:
                    conn.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
                    )
        conn.commit()

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

    def _seed_sources(self) -> None:
        """Populate sources table from config/sources.yaml if not already present."""
        config_path = Path(__file__).parent.parent / "config" / "sources.yaml"
        if not config_path.exists():
            return
        with open(config_path) as f:
            sources = yaml.safe_load(f).get("sources", [])
        conn = self.connect()
        conn.executemany(
            """
            INSERT OR IGNORE INTO sources (source_id, name, domain, rss_url, url, active, category)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    src.get("source_id"),
                    src["name"],
                    src.get("domain"),
                    src.get("rss_url"),
                    src.get("url"),
                    int(bool(src.get("active", True))),
                    src.get("category"),
                )
                for src in sources
                if src.get("source_id")
            ],
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
        self._upsert_articles_into_table("articles", [article], include_is_uplifting=False)

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
        return self._upsert_articles_into_table("articles", articles, include_is_uplifting=False)

    def upsert_crawled_articles(self, articles: Sequence[Dict]) -> int:
        """Bulk upsert raw crawled articles into ``crawled_articles``."""
        return self._upsert_articles_into_table("crawled_articles", articles, include_is_uplifting=True)

    def article_exists(self, url: str) -> bool:
        """Return True if an article with the given URL is in the DB."""
        conn = self.connect()
        row = conn.execute(
            """
            SELECT 1
            FROM (
                SELECT url FROM crawled_articles
                UNION
                SELECT url FROM articles
            )
            WHERE url = ?
            LIMIT 1
            """,
            (url,),
        ).fetchone()
        return row is not None

    def get_existing_urls(self, urls: Sequence[str]) -> set:
        """Return the subset of *urls* that already exist in the DB."""
        if not urls:
            return set()
        conn = self.connect()
        placeholders = ",".join("?" * len(urls))
        rows = conn.execute(
            f"""
            SELECT url FROM crawled_articles WHERE url IN ({placeholders})
            UNION
            SELECT url FROM articles WHERE url IN ({placeholders})
            """,
            list(urls) + list(urls),
        ).fetchall()
        return {row["url"] for row in rows}

    def get_existing_title_published_pairs(self, candidates: List[Dict]) -> set:
        """Return (title, published_at) pairs from *candidates* already in the DB."""
        if not candidates:
            return set()
        conn = self.connect()
        existing: set = set()
        for article in candidates:
            title = article["title"]
            published_at = str(article["published_at"])
            row = conn.execute(
                """
                SELECT 1 FROM (
                    SELECT title, published_at FROM crawled_articles
                    UNION
                    SELECT title, published_at FROM articles
                ) WHERE title = ? AND published_at = ?
                LIMIT 1
                """,
                (title, published_at),
            ).fetchone()
            if row is not None:
                existing.add((title, published_at))
        return existing

    def purge_non_uplifting_articles(self, threshold: float = 0.75) -> int:
        """Remove non-uplifting rows from the final ``articles`` table."""
        conn = self.connect()
        cursor = conn.execute(
            """
            DELETE FROM articles
            WHERE uplifting_score IS NULL OR uplifting_score < ?
            """,
            (threshold,),
        )
        conn.commit()
        return cursor.rowcount

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

    # ------------------------------------------------------------------
    # Internal bulk writer
    # ------------------------------------------------------------------

    def _upsert_articles_into_table(
        self,
        table_name: str,
        articles: Sequence[Dict],
        include_is_uplifting: bool,
    ) -> int:
        """Generic bulk upsert helper for article tables."""
        if not articles:
            return 0

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
            if include_is_uplifting:
                uplift_flag = row.get("is_uplifting")
                if uplift_flag is None:
                    uplift_flag = a.get("is_uplifting")
                if uplift_flag is None and row.get("uplifting_score") is not None:
                    uplift_flag = float(row["uplifting_score"]) >= 0.75
                row["is_uplifting"] = int(bool(uplift_flag))
            rows.append(row)

        if table_name == "crawled_articles":
            insert_sql = """
                INSERT INTO crawled_articles
                    (title, url, source_url, external_url, content, summary, thumbnail_url, category,
                     category_confidence, category_scores,
                     source_id, uplifting_score, is_uplifting, published_at,
                     crawled_at, updated_at)
                VALUES
                    (:title, :url, :source_url, :external_url, :content, :summary, :thumbnail_url, :category,
                     :category_confidence, :category_scores,
                     :source_id, :uplifting_score, :is_uplifting, :published_at,
                     :crawled_at, :updated_at)
                ON CONFLICT(url) DO UPDATE SET
                    title           = excluded.title,
                    source_url      = excluded.source_url,
                    external_url    = excluded.external_url,
                    content         = excluded.content,
                    summary         = excluded.summary,
                    thumbnail_url   = excluded.thumbnail_url,
                    category        = excluded.category,
                    category_confidence = excluded.category_confidence,
                    category_scores = excluded.category_scores,
                    uplifting_score = excluded.uplifting_score,
                    is_uplifting    = excluded.is_uplifting,
                    published_at    = excluded.published_at,
                    crawled_at      = excluded.crawled_at,
                    updated_at      = excluded.updated_at
            """
        else:
            insert_sql = """
                INSERT INTO articles
                    (title, url, source_url, external_url, content, summary, thumbnail_url, category,
                     category_confidence, category_scores,
                     source_id, uplifting_score, published_at, crawled_at, updated_at)
                VALUES
                    (:title, :url, :source_url, :external_url, :content, :summary, :thumbnail_url, :category,
                     :category_confidence, :category_scores,
                     :source_id, :uplifting_score, :published_at, :crawled_at, :updated_at)
                ON CONFLICT(url) DO UPDATE SET
                    title           = excluded.title,
                    source_url      = excluded.source_url,
                    external_url    = excluded.external_url,
                    content         = excluded.content,
                    summary         = excluded.summary,
                    thumbnail_url   = excluded.thumbnail_url,
                    category        = excluded.category,
                    category_confidence = excluded.category_confidence,
                    category_scores = excluded.category_scores,
                    uplifting_score = excluded.uplifting_score,
                    published_at    = excluded.published_at,
                    crawled_at      = excluded.crawled_at,
                    updated_at      = excluded.updated_at
            """

        with self.transaction() as conn:
            cursor = conn.executemany(insert_sql, rows)
        count = cursor.rowcount
        logger.info("Upserted %d articles into %s", count, table_name)
        return count


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------


def init_db(db_path: str = _DEFAULT_DB_PATH):
    """Create (or open) the database, apply schema, return instance.

    When the ``DATABASE_URL`` environment variable is set the pipeline writes
    to Postgres (used by the Rust stack).  Otherwise it falls back to SQLite at
    *db_path*.
    """
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        db = PostgresDB(database_url)
        db.init()
        return db
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
# PostgresDB — used when DATABASE_URL is set (Rust/Postgres stack)
# ---------------------------------------------------------------------------


class PostgresDB:
    """Postgres backend for the upnews pipeline (Rust stack).

    Implements the same interface as SQLiteDB so run_pipeline.py is
    backend-agnostic.  DDL is managed by the Rust API's sqlx migrations,
    so init() just verifies connectivity.
    """

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._conn = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self):
        if self._conn is None or self._conn.closed:
            import psycopg2  # noqa: import-outside-toplevel
            self._conn = psycopg2.connect(self.database_url)
            self._conn.autocommit = False
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "PostgresDB":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Verify connectivity and seed sources — DDL is owned by the Rust API's migrations."""
        self.connect()
        host = self.database_url.split("@")[-1] if "@" in self.database_url else self.database_url
        logger.info("Connected to Postgres at %s", host)
        self._seed_sources()

    def _seed_sources(self) -> None:
        """Upsert sources from config/sources.yaml into Postgres."""
        config_path = Path(__file__).parent.parent / "config" / "sources.yaml"
        if not config_path.exists():
            return
        with open(config_path) as f:
            sources = yaml.safe_load(f).get("sources", [])
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO sources (source_id, name, domain, rss_url, url, active, category)
                    VALUES (%(source_id)s, %(name)s, %(domain)s, %(rss_url)s, %(url)s, %(active)s, %(category)s)
                    ON CONFLICT(source_id) DO UPDATE SET
                        name=EXCLUDED.name, domain=EXCLUDED.domain, rss_url=EXCLUDED.rss_url,
                        url=EXCLUDED.url, active=EXCLUDED.active, category=EXCLUDED.category
                    """,
                    [
                        {
                            "source_id": src.get("source_id"),
                            "name": src["name"],
                            "domain": src.get("domain"),
                            "rss_url": src.get("rss_url"),
                            "url": src.get("url"),
                            "active": bool(src.get("active", True)),
                            "category": src.get("category"),
                        }
                        for src in sources
                        if src.get("source_id")
                    ],
                )
        logger.info("Seeded %d sources into Postgres", len(sources))

    # ------------------------------------------------------------------
    # Context manager for transactions
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(self):
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
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sources (source_id, name, domain, rss_url, url, active, category)
                VALUES (%(source_id)s, %(name)s, %(domain)s, %(rss_url)s, %(url)s, %(active)s, %(category)s)
                ON CONFLICT(source_id) DO UPDATE SET
                    name=EXCLUDED.name, domain=EXCLUDED.domain, rss_url=EXCLUDED.rss_url,
                    url=EXCLUDED.url, active=EXCLUDED.active, category=EXCLUDED.category
                """,
                {
                    "source_id": source.get("source_id", ""),
                    "name": source.get("name", ""),
                    "domain": source.get("domain"),
                    "rss_url": source.get("rss_url"),
                    "url": source.get("url"),
                    "active": bool(source.get("active", True)),
                    "category": source.get("category"),
                },
            )
            cur.execute("SELECT id FROM sources WHERE source_id = %s", (source.get("source_id", ""),))
            row = cur.fetchone()
            conn.commit()
        return row[0] if row else -1

    def upsert_sources(self, sources: Sequence[Dict]) -> int:
        data = [
            {
                "source_id": s.get("source_id", ""),
                "name": s.get("name", ""),
                "domain": s.get("domain"),
                "rss_url": s.get("rss_url"),
                "url": s.get("url"),
                "active": bool(s.get("active", True)),
                "category": s.get("category"),
            }
            for s in sources
        ]
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO sources (source_id, name, domain, rss_url, url, active, category)
                    VALUES (%(source_id)s, %(name)s, %(domain)s, %(rss_url)s, %(url)s, %(active)s, %(category)s)
                    ON CONFLICT(source_id) DO UPDATE SET
                        name=EXCLUDED.name, domain=EXCLUDED.domain, rss_url=EXCLUDED.rss_url,
                        url=EXCLUDED.url, active=EXCLUDED.active, category=EXCLUDED.category
                    """,
                    data,
                )
                return cur.rowcount

    def get_source_int_id(self, source_key: str) -> Optional[int]:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM sources WHERE source_id = %s LIMIT 1", (source_key,))
            row = cur.fetchone()
        return row[0] if row else None

    def get_source_int_id_map(self, source_keys: List[str]) -> Dict[str, int]:
        if not source_keys:
            return {}
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, source_id FROM sources WHERE source_id = ANY(%s)",
                (list(source_keys),),
            )
            return {row[1]: row[0] for row in cur.fetchall()}

    # ------------------------------------------------------------------
    # Articles
    # ------------------------------------------------------------------

    def upsert_articles(self, articles: Sequence[Dict]) -> int:
        if not articles:
            return 0
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

        insert_sql = """
            INSERT INTO articles
                (title, url, content, summary, thumbnail_url, category,
                 source_id, uplifting_score, published_at, crawled_at, updated_at)
            VALUES
                (%(title)s, %(url)s, %(content)s, %(summary)s, %(thumbnail_url)s, %(category)s,
                 %(source_id)s, %(uplifting_score)s, %(published_at)s, %(crawled_at)s, %(updated_at)s)
            ON CONFLICT(url) DO UPDATE SET
                title=EXCLUDED.title, content=EXCLUDED.content, summary=EXCLUDED.summary,
                thumbnail_url=EXCLUDED.thumbnail_url, category=EXCLUDED.category,
                uplifting_score=EXCLUDED.uplifting_score, published_at=EXCLUDED.published_at,
                crawled_at=EXCLUDED.crawled_at, updated_at=EXCLUDED.updated_at
        """
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.executemany(insert_sql, rows)
        count = len(rows)
        logger.info("Upserted %d articles into articles", count)
        return count

    def upsert_crawled_articles(self, articles: Sequence[Dict]) -> int:
        """No-op — crawled_articles is not in the Rust API's Postgres schema."""
        logger.debug(
            "upsert_crawled_articles: skipped for Postgres backend (%d articles)", len(articles)
        )
        return len(articles)

    def article_exists(self, url: str) -> bool:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM articles WHERE url = %s LIMIT 1", (url,))
            return cur.fetchone() is not None

    def get_existing_urls(self, urls: Sequence[str]) -> set:
        if not urls:
            return set()
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("SELECT url FROM articles WHERE url = ANY(%s)", (list(urls),))
            return {row[0] for row in cur.fetchall()}

    def get_existing_title_published_pairs(self, candidates: List[Dict]) -> set:
        """Return (title, published_at) pairs from *candidates* already in the DB."""
        if not candidates:
            return set()
        conn = self.connect()
        existing: set = set()
        with conn.cursor() as cur:
            for article in candidates:
                title = article["title"]
                published_at = str(article["published_at"])
                # Match on date prefix to handle timezone format differences between
                # the pipeline's string representation and Postgres TIMESTAMPTZ output.
                cur.execute(
                    "SELECT 1 FROM articles WHERE title = %s AND published_at::text LIKE %s LIMIT 1",
                    (title, published_at[:10] + "%"),
                )
                if cur.fetchone() is not None:
                    existing.add((title, published_at))
        return existing

    def purge_non_uplifting_articles(self, threshold: float = 0.75) -> int:
        """Remove non-uplifting rows from the articles table."""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM articles WHERE uplifting_score IS NULL OR uplifting_score < %s",
                (threshold,),
            )
            count = cur.rowcount
            conn.commit()
        return count

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
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO crawl_metrics
                    (crawl_start, crawl_end, articles_fetched, articles_classified,
                     articles_stored, avg_classification_score, errors)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    crawl_start,
                    crawl_end,
                    articles_fetched,
                    articles_classified,
                    articles_stored,
                    avg_classification_score,
                    errors,
                ),
            )
            row = cur.fetchone()
            conn.commit()
        return row[0] if row else -1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _article_to_row(article: Dict) -> Dict:
    """Convert an article dict to the param dict expected by the INSERT.

    ``source_id`` is left as-is here (string); callers must replace it with
    the resolved integer id before executing SQL.
    """
    external_url = article.get("external_url") if "external_url" in article else None
    source_url = external_url or article.get("source_url")
    if source_url is None:
        source_url = article.get("rss_link") or article.get("url")

    return {
        "title":          article.get("title", ""),
        "url":            article.get("url") or article.get("original_url", ""),
        "source_url":     source_url,
        "external_url":   external_url,
        # Support both 'content' (new) and 'body' (legacy crawler field)
        "content":        article.get("content") or article.get("body"),
        "summary":        article.get("summary"),
        "thumbnail_url":  article.get("thumbnail_url"),
        "category":       _category_to_slug(article.get("category")),
        "category_confidence": article.get("category_confidence"),
        "category_scores": json.dumps(article.get("category_scores")) if isinstance(article.get("category_scores"), dict) else article.get("category_scores"),
        "source_id":      article.get("source_id", ""),  # replaced by caller
        "uplifting_score": article.get("uplifting_score"),
        "is_uplifting":   article.get("is_uplifting"),
        "published_at":   _coerce_dt(article.get("published_at")),
        "crawled_at":     _fmt_dt(datetime.now(timezone.utc)),
        "updated_at":     _fmt_dt(datetime.now(timezone.utc)),
    }


def _category_to_slug(category: Optional[str]) -> Optional[str]:
    """Normalize a category to the canonical API/frontend display name.

    The helper name is kept for backwards compatibility, but the pipeline now
    stores the same human-readable labels that the API seeds and frontend show.
    """
    if not category:
        return None

    canonical_map = {
        # Current names
        "Science & Tech": "Science & Tech",
        "Health": "Health",
        "Environment": "Environment",
        "Community": "Community",
        "Education": "Education",
        "Sports": "Sports",
        "Arts & Culture": "Arts & Culture",
        # Slugs from older runs
        "science-tech": "Science & Tech",
        "health": "Health",
        "environment": "Environment",
        "community": "Community",
        "education": "Education",
        "sports": "Sports",
        "arts-culture": "Arts & Culture",
        # Legacy labels from earlier iterations
        "Health & Wellness": "Health",
        "Environment & Nature": "Environment",
        "Community & Social Good": "Community",
        "Technology & Science": "Science & Tech",
        "Business & Economics": "Science & Tech",
        "Culture & Arts": "Arts & Culture",
        "Human Interest": "Community",
    }

    if category in canonical_map:
        return canonical_map[category]
    return category


def _coerce_dt(value) -> Optional[str]:
    """Convert a datetime / ISO string / None to an ISO-format string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return _fmt_dt(value)
    return str(value)


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")
