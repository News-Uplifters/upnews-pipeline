"""Deduplication layer for the upnews pipeline (Task 7).

Filters out articles that are already stored in the database before they
reach the expensive classification and enrichment stages.

Primary dedup key  : URL  (unique constraint in DB)
Fallback dedup key : title + published_at  (for articles whose URL may
                     differ superficially, e.g. tracking params stripped)
"""

import logging
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "data/upnews.db"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def deduplicate_articles(
    articles: List[Dict],
    db_path: str = _DEFAULT_DB_PATH,
    db=None,
) -> List[Dict]:
    """Filter out articles already present in the database.

    Deduplication is performed in two passes:

    1. **URL pass** – any article whose ``url`` field matches a row already
       in the ``articles`` table is discarded.  This is the primary check and
       is done with a single bulk query for efficiency.

    2. **Title + published_at pass** – articles that survived the URL check
       but have *both* a ``title`` and a ``published_at`` value are checked
       against (title, published_at) pairs already in the DB.  This catches
       re-crawled articles where the URL has changed (e.g. redirect targets,
       UTM parameter stripping).

    Args:
        articles: List of article dicts as produced by the crawler.  Each
            dict should contain at least ``url`` (str) and optionally
            ``title`` (str) and ``published_at`` (str | datetime).
        db_path: Path to the SQLite database.  Ignored when *db* is provided.
        db: An already-open :class:`~pipeline.database.SQLiteDB` instance.
            When supplied, *db_path* is ignored and the caller owns the
            connection lifecycle.

    Returns:
        List containing only articles that are not yet in the database.
    """
    if not articles:
        return []

    _db, _owned = _resolve_db(db, db_path)

    try:
        new_articles = _url_dedup(articles, _db)
        new_articles = _title_published_dedup(new_articles, _db)
    finally:
        if _owned:
            _db.close()

    skipped = len(articles) - len(new_articles)
    if skipped:
        logger.info(
            "Deduplication: skipped %d/%d articles already in DB (%d new)",
            skipped,
            len(articles),
            len(new_articles),
        )
    else:
        logger.info(
            "Deduplication: all %d articles are new (0 duplicates found)",
            len(articles),
        )

    return new_articles


def article_exists(url: str, db_path: str = _DEFAULT_DB_PATH, db=None) -> bool:
    """Return True if an article with *url* is already in the database.

    Args:
        url: The article URL to check.
        db_path: Path to the SQLite database.  Ignored when *db* is provided.
        db: An already-open :class:`~pipeline.database.SQLiteDB` instance.

    Returns:
        ``True`` if the URL exists, ``False`` otherwise.
    """
    _db, _owned = _resolve_db(db, db_path)
    try:
        return _db.article_exists(url)
    finally:
        if _owned:
            _db.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _url_dedup(articles: List[Dict], db) -> List[Dict]:
    """Remove articles whose URLs already exist in the DB."""
    urls = [a.get("url", "") for a in articles]
    existing_urls = db.get_existing_urls(urls)
    if not existing_urls:
        return articles
    filtered = [a for a in articles if a.get("url", "") not in existing_urls]
    logger.debug("URL dedup: removed %d duplicates", len(articles) - len(filtered))
    return filtered


def _title_published_dedup(articles: List[Dict], db) -> List[Dict]:
    """Remove articles matching (title, published_at) pairs already in DB.

    Only articles that have *both* a non-empty title and a non-empty
    published_at are eligible for this check.  Articles missing either field
    are passed through unchanged to avoid false positives.
    """
    # Collect candidates: articles with both title and published_at
    candidates = [
        a for a in articles
        if a.get("title") and a.get("published_at")
    ]
    if not candidates:
        return articles

    existing_pairs = db.get_existing_title_published_pairs(candidates)

    if not existing_pairs:
        return articles

    filtered = []
    for a in articles:
        key = (a.get("title"), str(a.get("published_at", "")))
        if key in existing_pairs:
            logger.debug(
                "Title+published_at dedup: skipping '%s' (%s)",
                a.get("title", "")[:60],
                a.get("published_at"),
            )
        else:
            filtered.append(a)

    logger.debug(
        "Title+published_at dedup: removed %d duplicates",
        len(articles) - len(filtered),
    )
    return filtered


def _resolve_db(db, db_path: str):
    """Return (db_instance, owned) where *owned* means we should close it."""
    if db is not None:
        return db, False

    from pipeline.database import SQLiteDB  # noqa: import-outside-toplevel

    instance = SQLiteDB(db_path)
    instance.init()
    return instance, True
