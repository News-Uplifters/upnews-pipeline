"""Standard RSS/Atom feed adapter."""

from __future__ import annotations

import feedparser
import requests
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit
from typing import Optional

from . import BaseSource

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
}


def clean_url(url: str) -> str:
    """Strip query parameters and fragments from a URL."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


class RSSSource(BaseSource):
    """Standard RSS/Atom feed adapter using feedparser."""

    def fetch(self, limit: int = 50) -> list:
        """Fetch articles from an RSS/Atom feed.

        Args:
            limit: Maximum number of articles to return

        Returns:
            List of article dicts
        """
        feed_content = self._download_feed_content()
        url_feed = feedparser.parse(self.rss_url)
        content_feed = feedparser.parse(feed_content) if feed_content else None

        feed = url_feed
        if content_feed and len(content_feed.entries) > len(url_feed.entries):
            feed = content_feed

        articles = []
        for entry in feed.entries[:limit]:
            article = self._parse_entry(entry)
            if article:
                articles.append(article)

        return articles

    def _download_feed_content(self):
        """Download raw feed bytes with browser-like headers."""
        try:
            response = requests.get(
                self.rss_url,
                headers=DEFAULT_HEADERS,
                timeout=10,
                allow_redirects=True,
            )
            if response.ok and response.content:
                return response.content
        except Exception:
            pass
        return None

    def _parse_entry(self, entry) -> Optional[dict]:
        """Parse a single feedparser entry into an article dict."""
        title = entry.get("title", "").strip()
        rss_link = entry.get("link", "")
        if not title or not rss_link:
            return None

        published = None
        pub = getattr(entry, "published_parsed", None)
        upd = getattr(entry, "updated_parsed", None)
        if pub:
            published = datetime(*pub[:6])
        elif upd:
            published = datetime(*upd[:6])

        original_url = clean_url(rss_link)

        return {
            "title": title,
            "rss_link": rss_link,
            "original_url": original_url,
            "published": published,
            "source_id": self.source_id,
        }
