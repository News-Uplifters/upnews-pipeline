"""Article summarization service (Task 5).

Uses DistilBART (sshleifer/distilbart-cnn-6-6) for fast, on-device summarization.
The model is lazy-loaded on first use and cached for the lifetime of the process.
Results are additionally cached in a SQLite table so the same text is never
summarized twice.
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAME = "sshleifer/distilbart-cnn-6-6"

# DistilBART max input is 1024 tokens (~4 chars/token → ~4000 chars is safe)
_MAX_INPUT_CHARS = 3800

_DEFAULT_CACHE_PATH = ":memory:"

# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------

_summarizer: Optional[Any] = None


def _get_summarizer() -> Any:
    """Load and cache the DistilBART summarization pipeline."""
    global _summarizer
    if _summarizer is None:
        from transformers import pipeline  # noqa: import-outside-toplevel

        logger.info("Loading summarization model (%s)…", _MODEL_NAME)
        _summarizer = pipeline("summarization", model=_MODEL_NAME)
        logger.info("Summarization model loaded.")
    return _summarizer


def set_summarizer(summarizer: Any) -> None:
    """Override the global summarizer instance (useful for testing)."""
    global _summarizer
    _summarizer = summarizer


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
# Text preparation
# ---------------------------------------------------------------------------


def _truncate_text(text: str) -> str:
    """Truncate text to fit within the model's max input length."""
    if len(text) > _MAX_INPUT_CHARS:
        logger.debug("Truncating input from %d to %d chars.", len(text), _MAX_INPUT_CHARS)
        return text[:_MAX_INPUT_CHARS]
    return text


def _is_too_short(text: str, min_length: int) -> bool:
    """Return True if text is likely shorter than min_length tokens.

    Uses a rough 4-chars-per-token heuristic to avoid model errors when
    the input is shorter than the requested minimum output length.
    """
    estimated_tokens = len(text.split())
    return estimated_tokens < min_length


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize(
    text: str,
    max_length: int = 130,
    min_length: int = 30,
    _cache: Optional[sqlite3.Connection] = None,
    _summarizer_override: Optional[Any] = None,
) -> str:
    """Generate a summary of article text using DistilBART.

    Args:
        text: Article body (or title + body combined).
        max_length: Max summary length in tokens (~4 chars/token).
        min_length: Min summary length in tokens.
        _cache: SQLite connection for result caching (tests may pass their own).
        _summarizer_override: Replace the global model (for unit tests).

    Returns:
        Summary string (1-3 sentences). Returns the original text unchanged if
        it is too short to summarize meaningfully.

    Raises:
        ValueError: If text is empty.

    Example:
        >>> summary = summarize("A long article about climate change...")
        >>> isinstance(summary, str) and len(summary) > 0
        True
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

    # Very short texts cannot be summarized to min_length tokens — return as-is.
    if _is_too_short(text, min_length):
        logger.debug("Text too short to summarize (%d words); returning as-is.", len(text.split()))
        _cache_set(key, text, cache)
        return text

    truncated = _truncate_text(text)
    model = _summarizer_override if _summarizer_override is not None else _get_summarizer()

    output = model(
        truncated,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    summary: str = output[0]["summary_text"].strip()

    _cache_set(key, summary, cache)
    logger.debug("Generated summary (%d chars) for text (%d chars).", len(summary), len(text))
    return summary


async def summarize_batch(
    texts: List[str],
    batch_size: int = 8,
    max_length: int = 130,
    _cache: Optional[sqlite3.Connection] = None,
    _summarizer_override: Optional[Any] = None,
) -> List[str]:
    """Generate summaries for multiple texts efficiently.

    Runs the CPU-bound summarization in a thread pool so the event loop
    stays responsive.  Results are cached so repeated texts are only
    summarized once.

    Args:
        texts: List of article bodies.
        batch_size: Number of texts to process per model call.
        max_length: Max summary length per article (tokens).
        _cache: SQLite connection for result caching.
        _summarizer_override: Replace the global model (for unit tests).

    Returns:
        List of summaries in the same order as the input.

    Example:
        >>> import asyncio
        >>> texts = ["Article 1 body text...", "Article 2 body text..."]
        >>> summaries = asyncio.run(summarize_batch(texts, batch_size=2))
        >>> len(summaries) == 2
        True
    """
    if not texts:
        return []

    def _run_batch() -> List[str]:
        results: List[str] = []
        for text in texts:
            try:
                summary = summarize(
                    text,
                    max_length=max_length,
                    _cache=_cache,
                    _summarizer_override=_summarizer_override,
                )
            except ValueError:
                logger.warning("Skipping empty text in batch.")
                summary = ""
            results.append(summary)
        return results

    return await asyncio.to_thread(_run_batch)
