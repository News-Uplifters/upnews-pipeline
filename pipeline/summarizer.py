"""Extractive article summarization (Task 5).

Returns the first 2–3 sentences of article text up to a character cap.
No ML models are downloaded or loaded.  Results are cached in a SQLite
table so the same text is never processed twice.
"""

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_INPUT_CHARS = 3800   # kept for API compatibility with existing tests
_MAX_SUMMARY_CHARS = 500  # extractive cap

_DEFAULT_CACHE_PATH = ":memory:"

# ---------------------------------------------------------------------------
# SQLite-backed result cache
# ---------------------------------------------------------------------------

_MISS = object()  # sentinel: key not in cache


def _get_cache_connection(cache_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(cache_path, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS summary_cache (
            cache_key TEXT PRIMARY KEY,
            summary   TEXT NOT NULL
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


def _make_cache_key(text: str, max_length: int, min_length: int) -> str:
    payload = json.dumps(
        {"text": text, "max_length": max_length, "min_length": min_length},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_get(key: str, conn: sqlite3.Connection):
    row = conn.execute(
        "SELECT summary FROM summary_cache WHERE cache_key = ?", (key,)
    ).fetchone()
    if row is None:
        return _MISS
    return row[0]


def _cache_set(key: str, summary: str, conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO summary_cache (cache_key, summary) VALUES (?, ?)",
        (key, summary),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Text helpers (kept for API compatibility)
# ---------------------------------------------------------------------------


def _truncate_text(text: str) -> str:
    """Truncate text to _MAX_INPUT_CHARS characters."""
    if len(text) > _MAX_INPUT_CHARS:
        logger.debug("Truncating input from %d to %d chars.", len(text), _MAX_INPUT_CHARS)
        return text[:_MAX_INPUT_CHARS]
    return text


def _is_too_short(text: str, min_length: int) -> bool:
    """Return True if word count is below min_length."""
    return len(text.split()) < min_length


# ---------------------------------------------------------------------------
# Extractive summarization
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _extract_summary(text: str, max_chars: int = _MAX_SUMMARY_CHARS) -> str:
    """Return the first sentences of *text* up to *max_chars* characters."""
    sentences = _SENTENCE_RE.split(text.strip())
    summary_parts: List[str] = []
    total = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if total + len(sentence) > max_chars and summary_parts:
            break
        summary_parts.append(sentence)
        total += len(sentence) + 1  # +1 for the space
    return " ".join(summary_parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize(
    text: str,
    max_length: int = 130,
    min_length: int = 30,
    _cache: Optional[sqlite3.Connection] = None,
) -> str:
    """Return an extractive summary of *text*.

    Takes the first 2-3 sentences up to ~500 characters.  Returns the
    original text unchanged if it is too short to summarize meaningfully.

    Args:
        text: Article body (or title + body combined).
        max_length: Ignored (kept for API compatibility).
        min_length: Word count below which the text is returned as-is.
        _cache: SQLite connection for result caching.

    Returns:
        Summary string.

    Raises:
        ValueError: If text is empty.
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    text = text.strip()
    cache = _cache if _cache is not None else _default_cache()
    key = _make_cache_key(text, max_length, min_length)

    cached = _cache_get(key, cache)
    if cached is not _MISS:
        logger.debug("Cache hit for text hash=%s", key[:12])
        return cached

    if _is_too_short(text, min_length):
        logger.debug("Text too short (%d words); returning as-is.", len(text.split()))
        _cache_set(key, text, cache)
        return text

    summary = _extract_summary(text)
    _cache_set(key, summary, cache)
    logger.debug("Extracted summary (%d chars) from text (%d chars).", len(summary), len(text))
    return summary


async def summarize_batch(
    texts: List[str],
    batch_size: int = 8,
    max_length: int = 130,
    _cache: Optional[sqlite3.Connection] = None,
) -> List[str]:
    """Generate extractive summaries for multiple texts.

    Args:
        texts: List of article bodies.
        batch_size: Ignored (kept for API compatibility).
        max_length: Ignored (kept for API compatibility).
        _cache: SQLite connection for result caching.

    Returns:
        List of summaries in the same order as the input.
    """
    if not texts:
        return []

    def _run_batch() -> List[str]:
        results: List[str] = []
        for text in texts:
            try:
                summary = summarize(text, max_length=max_length, _cache=_cache)
            except ValueError:
                logger.warning("Skipping empty text in batch.")
                summary = ""
            results.append(summary)
        return results

    return await asyncio.to_thread(_run_batch)
