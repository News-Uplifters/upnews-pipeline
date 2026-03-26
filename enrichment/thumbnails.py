"""Thumbnail extraction service for articles (Task 3)."""

import asyncio
import logging
import sqlite3
from typing import Dict, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache (SQLite-backed, in-memory by default for easy unit testing)
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_PATH = ":memory:"


def _get_cache_connection(cache_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(cache_path, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS thumbnail_cache (
            url TEXT PRIMARY KEY,
            thumbnail_url TEXT
        )"""
    )
    conn.commit()
    return conn


# Module-level connection used when callers don't supply their own.
# Initialised lazily so tests can supply a separate in-memory DB.
_cache_conn: Optional[sqlite3.Connection] = None


def _default_cache() -> sqlite3.Connection:
    global _cache_conn
    if _cache_conn is None:
        _cache_conn = _get_cache_connection(_DEFAULT_CACHE_PATH)
    return _cache_conn


def _cache_get(url: str, conn: sqlite3.Connection) -> Optional[str]:
    """Return cached thumbnail URL, or sentinel _MISS if not cached."""
    row = conn.execute(
        "SELECT thumbnail_url FROM thumbnail_cache WHERE url = ?", (url,)
    ).fetchone()
    if row is None:
        return _MISS
    return row[0]  # may be None (i.e. "no thumbnail found" was cached)


def _cache_set(url: str, thumbnail_url: Optional[str], conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO thumbnail_cache (url, thumbnail_url) VALUES (?, ?)",
        (url, thumbnail_url),
    )
    conn.commit()


# Sentinel so we can distinguish "cached as None" from "not in cache".
_MISS = object()


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

def _extract_from_meta(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    """Try og:image then twitter:image meta tags."""
    for prop in ("og:image", "twitter:image", "twitter:image:src"):
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if tag:
            content = tag.get("content", "").strip()
            if content:
                return _make_absolute(content, base_url)
    return None


def _extract_from_img(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    """Return the largest-looking <img> src in the page body."""
    for img in soup.find_all("img", src=True):
        src = img["src"].strip()
        if not src or src.startswith("data:"):
            continue
        return _make_absolute(src, base_url)
    return None


def _extract_favicon(base_url: str) -> Optional[str]:
    """Return the conventional favicon URL for a domain."""
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}/favicon.ico"


def _make_absolute(url: str, base_url: str) -> str:
    """Resolve a potentially relative URL against the page base URL."""
    if url.startswith("//"):
        parsed = urlparse(base_url)
        return f"{parsed.scheme}:{url}"
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return urljoin(base_url, url)


# ---------------------------------------------------------------------------
# Core async extraction
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; upnews-pipeline/1.0; "
        "+https://github.com/News-Uplifters/upnews-pipeline)"
    )
}


async def _fetch_html(
    url: str,
    session: aiohttp.ClientSession,
    timeout: int,
    attempt: int = 0,
    max_retries: int = 2,
) -> Optional[str]:
    """Fetch HTML for *url*, retrying on transient errors."""
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers=_HEADERS,
            allow_redirects=True,
        ) as resp:
            if resp.status >= 400:
                logger.debug("HTTP %s for %s", resp.status, url)
                return None
            return await resp.text(errors="replace")
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        if attempt < max_retries:
            logger.debug("Retry %d/%d for %s: %s", attempt + 1, max_retries, url, exc)
            await asyncio.sleep(0.5 * (attempt + 1))
            return await _fetch_html(url, session, timeout, attempt + 1, max_retries)
        logger.warning("Failed to fetch %s after %d retries: %s", url, max_retries, exc)
        return None


async def extract_thumbnail(
    url: str,
    timeout: int = 5,
    max_retries: int = 2,
    _cache: Optional[sqlite3.Connection] = None,
    _session: Optional[aiohttp.ClientSession] = None,
) -> Optional[str]:
    """Extract thumbnail URL from an article page.

    Attempts extraction in order:
    1. Open Graph ``og:image`` meta tag (preferred)
    2. Twitter Card ``twitter:image`` / ``twitter:image:src``
    3. Generic ``<img>`` tag in the page body
    4. Source favicon as last-resort fallback

    Args:
        url: Article URL to scrape.
        timeout: HTTP request timeout in seconds.
        max_retries: Number of retries on transient failures.

    Returns:
        Absolute thumbnail URL, or ``None`` if nothing found.
    """
    cache = _cache if _cache is not None else _default_cache()

    cached = _cache_get(url, cache)
    if cached is not _MISS:
        logger.debug("Cache hit for %s → %s", url, cached)
        return cached

    async def _do_extract(session: aiohttp.ClientSession) -> Optional[str]:
        html = await _fetch_html(url, session, timeout, max_retries=max_retries)
        if html is None:
            result = _extract_favicon(url)
            _cache_set(url, result, cache)
            return result

        soup = BeautifulSoup(html, "html.parser")

        # 1 & 2: OG / Twitter meta tags
        result = _extract_from_meta(soup, url)
        if result:
            _cache_set(url, result, cache)
            return result

        # 3: Generic <img> tag
        result = _extract_from_img(soup, url)
        if result:
            _cache_set(url, result, cache)
            return result

        # 4: Favicon fallback
        result = _extract_favicon(url)
        _cache_set(url, result, cache)
        return result

    if _session is not None:
        return await _do_extract(_session)

    async with aiohttp.ClientSession() as session:
        return await _do_extract(session)


async def extract_thumbnails_batch(
    urls: list,
    concurrency: int = 10,
    timeout: int = 5,
    max_retries: int = 2,
    _cache: Optional[sqlite3.Connection] = None,
) -> Dict[str, Optional[str]]:
    """Extract thumbnails for multiple URLs concurrently.

    Args:
        urls: List of article URLs.
        concurrency: Maximum number of simultaneous HTTP requests.
        timeout: Per-request timeout in seconds.
        max_retries: Retries per URL on transient failures.

    Returns:
        Dict mapping ``url → thumbnail_url`` (``None`` where not found).
    """
    if not urls:
        return {}

    cache = _cache if _cache is not None else _default_cache()
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(url: str, session: aiohttp.ClientSession) -> tuple:
        async with semaphore:
            result = await extract_thumbnail(
                url,
                timeout=timeout,
                max_retries=max_retries,
                _cache=cache,
                _session=session,
            )
            return url, result

    async with aiohttp.ClientSession() as session:
        tasks = [_bounded(url, session) for url in urls]
        pairs = await asyncio.gather(*tasks, return_exceptions=False)

    return dict(pairs)
