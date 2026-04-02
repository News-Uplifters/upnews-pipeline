"""Keyword-based article categorization (Task 4).

Assigns articles to one of the predefined categories using keyword matching
against the title and body text.  No ML models are downloaded or loaded.
Results are cached in a SQLite table so the same title+body pair is never
classified twice.
"""

import hashlib
import json
import logging
import re
import sqlite3
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default category taxonomy
# ---------------------------------------------------------------------------

DEFAULT_CATEGORIES: List[str] = [
    "Health",
    "Environment",
    "Community",
    "Science & Tech",
    "Education",
    "Sports",
    "Arts & Culture",
]

# ---------------------------------------------------------------------------
# Keyword lists per category (lowercase; substring match)
# ---------------------------------------------------------------------------

_KEYWORDS: Dict[str, List[str]] = {
    "Health": [
        "health", "hospital", "doctor", "medical", "mental", "wellness",
        "vaccine", "treatment", "disease", "cancer", "therapy", "medicine",
        "nurse", "patient", "clinic", "surgery", "drug", "pharma",
    ],
    "Environment": [
        "climate", "environment", "carbon", "renewable", "solar", "wind",
        "ocean", "species", "conservation", "pollution", "green", "forest",
        "nature", "emission", "biodiversity", "wildlife", "ecosystem",
        "recycle", "sustainability", "wildfire", "flood",
    ],
    "Community": [
        "community", "volunteer", "charity", "local", "neighbourhood",
        "neighborhood", "donation", "fundrais", "nonprofit", "non-profit",
        "support", "neighbour", "neighbor", "grassroots", "initiative",
        "residents", "civic", "outreach",
    ],
    "Science & Tech": [
        "science", "research", "technology", "artificial intelligence", "ai",
        "robot", "space", "discovery", "innovation", "engineer", "software",
        "quantum", "breakthrough", "study", "scientist", "lab", "experiment",
        "data", "computing", "satellite",
    ],
    "Education": [
        "school", "education", "student", "teacher", "university", "college",
        "learning", "scholarship", "literacy", "classroom", "curriculum",
        "academic", "graduate", "professor", "tuition", "library",
    ],
    "Sports": [
        "sport", "team", "champion", "athlete", "game", "tournament",
        "olympic", "win", "league", "coach", "player", "match", "race",
        "marathon", "medal", "trophy", "stadium", "fitness",
    ],
    "Arts & Culture": [
        "art", "music", "film", "book", "museum", "culture", "performance",
        "festival", "theatre", "theater", "creative", "author", "concert",
        "gallery", "exhibition", "dance", "literature", "poetry", "cinema",
        "sculpture",
    ],
}

_MAX_BODY_CHARS = 500


# ---------------------------------------------------------------------------
# SQLite-backed result cache
# ---------------------------------------------------------------------------

_MISS = object()  # sentinel: key not in cache
_DEFAULT_CACHE_PATH = ":memory:"


def _get_cache_connection(cache_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(cache_path, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS category_cache (
            cache_key  TEXT PRIMARY KEY,
            category   TEXT NOT NULL,
            scores     TEXT NOT NULL,
            confidence REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


_cache_conn: Optional[sqlite3.Connection] = None


def _default_cache() -> sqlite3.Connection:
    global _cache_conn
    if _cache_conn is None:
        _cache_conn = _get_cache_connection(_DEFAULT_CACHE_PATH)
    return _cache_conn


def _make_cache_key(title: str, body: str, categories: List[str]) -> str:
    payload = json.dumps(
        {"title": title, "body": body, "categories": sorted(categories)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_get(key: str, conn: sqlite3.Connection):
    row = conn.execute(
        "SELECT category, scores, confidence FROM category_cache WHERE cache_key = ?",
        (key,),
    ).fetchone()
    if row is None:
        return _MISS
    return {"category": row[0], "scores": json.loads(row[1]), "confidence": row[2]}


def _cache_set(key: str, result: Dict, conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO category_cache (cache_key, category, scores, confidence)
        VALUES (?, ?, ?, ?)
        """,
        (key, result["category"], json.dumps(result["scores"]), result["confidence"]),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------


def _build_input_text(title: str, body: str) -> str:
    """Combine title and (optionally) body into a single classification input."""
    if body:
        return f"{title}. {body[:_MAX_BODY_CHARS]}"
    return title


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------


def _score_text(text: str, categories: List[str]) -> Dict[str, float]:
    """Return a per-category keyword hit count, normalised to [0, 1].

    Each keyword that appears (case-insensitive substring match) contributes
    one hit to its category.  Scores are normalised so the top category = 1.0
    when at least one keyword is found; equal weights (1/n) are returned when
    no keywords match.
    """
    lowered = text.lower()
    hits: Dict[str, int] = {}
    for cat in categories:
        keywords = _KEYWORDS.get(cat, [])
        count = sum(1 for kw in keywords if kw in lowered)
        hits[cat] = count

    total_hits = sum(hits.values())
    if total_hits == 0:
        equal = 1.0 / len(categories) if categories else 0.0
        return {cat: equal for cat in categories}

    max_hits = max(hits.values())
    return {cat: hits[cat] / max_hits for cat in categories}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def categorize_article(
    title: str,
    body: str = "",
    categories: Optional[List[str]] = None,
    _cache: Optional[sqlite3.Connection] = None,
) -> Dict:
    """Categorize an article into predefined or custom categories.

    Uses keyword matching — no ML model is required.

    Args:
        title: Article headline.
        body: Article body text (optional; first 500 chars are used).
        categories: Candidate category labels.  Defaults to DEFAULT_CATEGORIES.
        _cache: SQLite connection for result caching.

    Returns:
        Dict with keys:
        - ``category`` (str): Highest-scoring label.
        - ``scores`` (dict): ``{label: score}`` for every candidate.
        - ``confidence`` (float): Score of the top category.
    """
    if not title:
        raise ValueError("title must not be empty")

    if categories is None:
        categories = DEFAULT_CATEGORIES

    cache = _cache if _cache is not None else _default_cache()
    key = _make_cache_key(title, body, categories)

    cached = _cache_get(key, cache)
    if cached is not _MISS:
        logger.debug("Cache hit for title=%r", title[:60])
        return cached

    text = _build_input_text(title, body)
    scores = _score_text(text, categories)
    top_category = max(scores, key=lambda c: scores[c])
    confidence = float(scores[top_category])

    result: Dict = {
        "category": top_category,
        "scores": scores,
        "confidence": confidence,
    }

    _cache_set(key, result, cache)
    return result


def categorize_batch(
    articles: List[Dict],
    categories: Optional[List[str]] = None,
    _cache: Optional[sqlite3.Connection] = None,
) -> List[Dict]:
    """Categorize multiple articles, enriching each with category metadata.

    Each article dict must contain at least a ``title`` key.  Adds:
    - ``category`` (str): Top predicted category.
    - ``category_scores`` (dict): Per-label scores.
    - ``category_confidence`` (float): Score of the top category.

    Args:
        articles: List of article dicts (must have ``title``; may have ``body``).
        categories: Candidate category labels.  Defaults to DEFAULT_CATEGORIES.
        _cache: SQLite connection for result caching.

    Returns:
        New list of article dicts with category fields added.
    """
    if not articles:
        return []

    if categories is None:
        categories = DEFAULT_CATEGORIES

    cache = _cache if _cache is not None else _default_cache()
    enriched_articles: List[Dict] = []

    for article in articles:
        title = article.get("title", "")
        body = article.get("body", "")

        if not title:
            logger.warning("Skipping article with empty title: %r", article)
            enriched = dict(article)
            enriched["category"] = None
            enriched["category_scores"] = {}
            enriched["category_confidence"] = 0.0
            enriched_articles.append(enriched)
            continue

        cat_result = categorize_article(
            title=title,
            body=body,
            categories=categories,
            _cache=cache,
        )

        enriched = dict(article)
        enriched["category"] = cat_result["category"]
        enriched["category_scores"] = cat_result["scores"]
        enriched["category_confidence"] = cat_result["confidence"]
        enriched_articles.append(enriched)

    logger.info("Categorized %d articles.", len(enriched_articles))
    return enriched_articles
