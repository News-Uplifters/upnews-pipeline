"""SQLite database layer for the upnews pipeline (Task 6).

Provides:
- Schema creation / migration (articles, sources, crawl_metrics tables)
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
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS sources (
    source_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    rss_url     TEXT,
    active      BOOLEAN DEFAULT 1,
    category    TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS articles (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    url              TEXT UNIQUE NOT NULL,
    title            TEXT NOT NULL,
    summary          TEXT,
    body             TEXT,
    thumbnail_url    TEXT,
    category         TEXT,
    source_id        TEXT NOT NULL,
    uplifting_score  REAL,
    published_at     DATETIME,
    crawled_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
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

CREATE INDEX IF NOT EXISTS idx_articles_source    ON articles(source_id);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_score     ON articles(uplifting_score);
"""

# ---------------------------------------------------------------------------
# SQLiteDB
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = "data/articles.db"


class SQLiteDB:
    """Thin wrapper around a SQLite connection for the upnews pipeline.

    Usage::

        db = SQLiteDB("data/articles.db")
        db.init()
        db.upsert_articles([{"url": "...", "title": "...", "source_id": "..."}])
        db.close()

    Or as a context manager::

        with SQLiteDB("data/articles.db") as db:
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
            # Enable WAL mode for better concurrent read performance
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
        """Create schema if it does not already exist."""
        conn = self.connect()
        conn.executescript(_DDL)
        conn.commit()
        logger.info("Database initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Context manager for transactions
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield the connection inside a transaction block.

        Commits on success, rolls back on any exception.
        """
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

    def upsert_source(self, source: Dict) -> None:
        """Insert or update a single source record."""
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO sources (source_id, name, rss_url, active, category)
            VALUES (:source_id, :name, :rss_url, :active, :category)
            ON CONFLICT(source_id) DO UPDATE SET
                name     = excluded.name,
                rss_url  = excluded.rss_url,
                active   = excluded.active,
                category = excluded.category
            """,
            {
                "source_id": source.get("source_id", ""),
                "name": source.get("name", ""),
                "rss_url": source.get("rss_url"),
                "active": int(bool(source.get("active", True))),
                "category": source.get("category"),
            },
        )
        conn.commit()

    def upsert_sources(self, sources: Sequence[Dict]) -> int:
        """Bulk upsert a list of source dicts. Returns count of rows affected."""
        with self.transaction() as conn:
            cursor = conn.executemany(
                """
                INSERT INTO sources (source_id, name, rss_url, active, category)
                VALUES (:source_id, :name, :rss_url, :active, :category)
                ON CONFLICT(source_id) DO UPDATE SET
                    name     = excluded.name,
                    rss_url  = excluded.rss_url,
                    active   = excluded.active,
                    category = excluded.category
                """,
                [
                    {
                        "source_id": s.get("source_id", ""),
                        "name": s.get("name", ""),
                        "rss_url": s.get("rss_url"),
                        "active": int(bool(s.get("active", True))),
                        "category": s.get("category"),
                    }
                    for s in sources
                ],
            )
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Articles
    # ------------------------------------------------------------------

    def upsert_article(self, article: Dict) -> None:
        """Insert or update a single article.

        The URL is the unique key.  If the URL already exists the record is
        updated with the latest enrichment data (summary, category, etc.) but
        the original ``created_at`` timestamp is preserved.
        """
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO articles
                (url, title, summary, body, thumbnail_url, category,
                 source_id, uplifting_score, published_at, crawled_at)
            VALUES
                (:url, :title, :summary, :body, :thumbnail_url, :category,
                 :source_id, :uplifting_score, :published_at, :crawled_at)
            ON CONFLICT(url) DO UPDATE SET
                title           = excluded.title,
                summary         = excluded.summary,
                body            = excluded.body,
                thumbnail_url   = excluded.thumbnail_url,
                category        = excluded.category,
                uplifting_score = excluded.uplifting_score,
                published_at    = excluded.published_at,
                crawled_at      = excluded.crawled_at
            """,
            _article_to_row(article),
        )
        conn.commit()

    def upsert_articles(self, articles: Sequence[Dict]) -> int:
        """Bulk upsert a list of article dicts inside a single transaction.

        Returns:
            Number of rows inserted or updated.

        Raises:
            sqlite3.Error: On any database error; the transaction is rolled back.
        """
        if not articles:
            return 0

        rows = [_article_to_row(a) for a in articles]
        with self.transaction() as conn:
            cursor = conn.executemany(
                """
                INSERT INTO articles
                    (url, title, summary, body, thumbnail_url, category,
                     source_id, uplifting_score, published_at, crawled_at)
                VALUES
                    (:url, :title, :summary, :body, :thumbnail_url, :category,
                     :source_id, :uplifting_score, :published_at, :crawled_at)
                ON CONFLICT(url) DO UPDATE SET
                    title           = excluded.title,
                    summary         = excluded.summary,
                    body            = excluded.body,
                    thumbnail_url   = excluded.thumbnail_url,
                    category        = excluded.category,
                    uplifting_score = excluded.uplifting_score,
                    published_at    = excluded.published_at,
                    crawled_at      = excluded.crawled_at
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
    """Write *articles* to the database, creating it if necessary.

    Args:
        articles: Iterable of article dicts (must have at least ``url``,
            ``title``, and ``source_id``).
        db_path: Path to the SQLite file (ignored when *db* is supplied).
        db: An already-open :class:`SQLiteDB` instance.  When supplied,
            *db_path* is ignored and the caller owns the connection lifecycle.

    Returns:
        Number of rows written.
    """
    if db is not None:
        return db.upsert_articles(articles)

    with SQLiteDB(db_path) as _db:
        _db.init()
        return _db.upsert_articles(articles)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _article_to_row(article: Dict) -> Dict:
    """Convert an article dict to the flat param dict expected by the INSERT."""
    return {
        "url": article.get("url") or article.get("original_url", ""),
        "title": article.get("title", ""),
        "summary": article.get("summary"),
        "body": article.get("body"),
        "thumbnail_url": article.get("thumbnail_url"),
        "category": article.get("category"),
        "source_id": article.get("source_id", ""),
        "uplifting_score": article.get("uplifting_score"),
        "published_at": _coerce_dt(article.get("published_at")),
        "crawled_at": _fmt_dt(datetime.now(timezone.utc)),
    }


def _coerce_dt(value) -> Optional[str]:
    """Convert a datetime / ISO string / None to an ISO-format string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return _fmt_dt(value)
    return str(value)


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")
