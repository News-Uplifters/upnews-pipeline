"""LLM-based article categorization (Task 4).

Uses a zero-shot classification model (facebook/bart-large-mnli) to assign
articles to one or more predefined categories with confidence scores.
The model is lazy-loaded on first use and cached for the lifetime of the
process.  Results are additionally cached in a SQLite table so the same
title+body pair is never classified twice.
"""

import hashlib
import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default category taxonomy
# ---------------------------------------------------------------------------

DEFAULT_CATEGORIES: List[str] = [
    "Health & Wellness",
    "Environment & Nature",
    "Community & Social Good",
    "Technology & Science",
    "Business & Economics",
    "Culture & Arts",
    "Human Interest",
]

# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------

_classifier: Optional[Any] = None


def _get_classifier() -> Any:
    """Load and cache the zero-shot classification pipeline."""
    global _classifier
    if _classifier is None:
        from transformers import pipeline  # noqa: import-outside-toplevel
        logger.info(
            "Loading zero-shot classification model (facebook/bart-large-mnli)…"
        )
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
        logger.info("Model loaded.")
    return _classifier


def set_classifier(classifier: Any) -> None:
    """Override the global classifier instance (useful for testing)."""
    global _classifier
    _classifier = classifier


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

_MAX_BODY_CHARS = 500  # keep inputs short to stay within BART's token limit


def _build_input_text(title: str, body: str) -> str:
    """Combine title and (optionally) body into a single classification input."""
    if body:
        return f"{title}. {body[:_MAX_BODY_CHARS]}"
    return title


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def categorize_article(
    title: str,
    body: str = "",
    categories: Optional[List[str]] = None,
    _cache: Optional[sqlite3.Connection] = None,
    _classifier_override: Optional[Any] = None,
) -> Dict:
    """Categorize an article into predefined or custom categories.

    Uses multi-label zero-shot classification so an article can belong to
    more than one category.  Scores are independent probabilities per label
    (not a softmax distribution), allowing multiple high-confidence hits.

    Args:
        title: Article headline.
        body: Article body text (optional; first 500 chars are used).
        categories: Candidate category labels.  Defaults to DEFAULT_CATEGORIES.
        _cache: SQLite connection for result caching (tests may pass their own).
        _classifier_override: Replace the global model (for unit tests).

    Returns:
        Dict with keys:
        - ``category`` (str): Highest-scoring label.
        - ``scores`` (dict): ``{label: confidence}`` for every candidate.
        - ``confidence`` (float): Confidence of the top category.

    Example::

        >>> result = categorize_article("Scientists unveil carbon-capture plant")
        >>> result["category"]
        'Environment & Nature'
        >>> result["confidence"] > 0.5
        True
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

    classifier = (
        _classifier_override if _classifier_override is not None else _get_classifier()
    )
    text = _build_input_text(title, body)

    output = classifier(text, candidate_labels=categories, multi_label=True)

    scores = dict(zip(output["labels"], output["scores"]))
    top_category = output["labels"][0]
    confidence = float(output["scores"][0])

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
    _classifier_override: Optional[Any] = None,
) -> List[Dict]:
    """Categorize multiple articles, enriching each with category metadata.

    Each article dict must contain at least a ``title`` key.  The function
    adds three new keys to each article:

    - ``category`` (str): Top predicted category.
    - ``category_scores`` (dict): Per-label confidence scores.
    - ``category_confidence`` (float): Confidence of the top category.

    Args:
        articles: List of article dicts (must have ``title``; may have ``body``).
        categories: Candidate category labels.  Defaults to DEFAULT_CATEGORIES.
        _cache: SQLite connection for result caching.
        _classifier_override: Replace the global model (for unit tests).

    Returns:
        New list of article dicts with category fields added.

    Example::

        >>> articles = [{"title": "Local charity feeds 500 families"}]
        >>> enriched = categorize_batch(articles)
        >>> "category" in enriched[0]
        True
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
            _classifier_override=_classifier_override,
        )

        enriched = dict(article)
        enriched["category"] = cat_result["category"]
        enriched["category_scores"] = cat_result["scores"]
        enriched["category_confidence"] = cat_result["confidence"]
        enriched_articles.append(enriched)

    logger.info("Categorized %d articles.", len(enriched_articles))
    return enriched_articles
