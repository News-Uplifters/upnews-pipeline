"""Thumbnail extraction service for articles (TASK 3 stub)."""

import asyncio
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

async def extract_thumbnail(
    url: str,
    timeout: int = 5,
    max_retries: int = 2
) -> Optional[str]:
    """Extract thumbnail URL from article page.

    Attempts to extract thumbnail in order:
    1. Open Graph og:image meta tag (preferred)
    2. Twitter Card twitter:image tag
    3. Generic <img> tags in article body
    4. Fallback: source favicon

    Args:
        url: Article URL to scrape
        timeout: HTTP request timeout (seconds)
        max_retries: Retry failed requests this many times

    Returns:
        Absolute URL of thumbnail image, or None if not found

    Example:
        >>> thumbnail = await extract_thumbnail("https://bbc.com/news/...")
        >>> print(thumbnail)
        "https://bbc.com/images/article-123.jpg"
    """
    # TODO: Implement with aiohttp and BeautifulSoup
    # 1. Fetch URL asynchronously
    # 2. Parse HTML with BeautifulSoup
    # 3. Extract og:image, twitter:image, or <img src>
    # 4. Return absolute URL (resolve relative paths)
    # 5. Add retry logic and timeout handling
    pass


async def extract_thumbnails_batch(
    urls: list[str],
    concurrency: int = 10
) -> Dict[str, Optional[str]]:
    """Extract thumbnails for multiple URLs concurrently.

    Respects rate limits and concurrency constraints to avoid
    hammering external servers.

    Args:
        urls: List of article URLs
        concurrency: Max concurrent requests (default 10)

    Returns:
        Dict mapping URL → thumbnail_url (or None if not found)

    Example:
        >>> urls = ["https://bbc.com/...", "https://reuters.com/..."]
        >>> thumbnails = await extract_thumbnails_batch(urls, concurrency=5)
        >>> print(thumbnails)
        {
            "https://bbc.com/...": "https://bbc.com/images/123.jpg",
            "https://reuters.com/...": None
        }
    """
    # TODO: Implement concurrent batch processing
    # 1. Create semaphore for concurrency control
    # 2. Schedule extract_thumbnail tasks with limit
    # 3. Gather results
    # 4. Add caching (Redis or SQLite) to avoid re-fetching
    pass
