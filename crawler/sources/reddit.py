"""Reddit subreddit adapter — extracts external article links from Reddit posts."""

from __future__ import annotations

import logging
import requests
import feedparser
from bs4 import BeautifulSoup
from typing import Optional

from . import BaseSource
from .rss import RSSSource, clean_url

_REDDIT_DOMAINS = {"reddit.com", "redd.it", "i.redd.it", "v.redd.it"}
logger = logging.getLogger(__name__)


def _is_reddit_internal(url: str) -> bool:
    return any(domain in url for domain in _REDDIT_DOMAINS)


def extract_reddit_external(reddit_post_url: str) -> Optional[str]:
    """Scrape a Reddit post page and return the external article URL.

    Tries in order:
    1. ``shreddit-post[content-href]`` attribute (new Reddit UI)
    2. ``<a>`` tags inside ``[data-testid=post-content]``
    3. Any external ``<a>`` tag on the page

    Args:
        reddit_post_url: URL of the Reddit post (not the subreddit)

    Returns:
        Clean external URL, or None if only internal links found
    """
    try:
        r = requests.get(reddit_post_url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        # New Reddit UI: shreddit-post element
        tag = soup.find("shreddit-post")
        if tag and tag.get("content-href"):
            link = tag["content-href"]
            if not _is_reddit_internal(link) and "reddit.com/gallery" not in link:
                return clean_url(link)

        # Old Reddit UI: post-content div
        body = soup.find("div", {"data-testid": "post-content"})
        if body:
            for a in body.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http") and not _is_reddit_internal(href) and "reddit.com/gallery" not in href:
                    return clean_url(href)

        # Fallback: any external link on the page
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http") and not _is_reddit_internal(href) and "reddit.com/gallery" not in href:
                return clean_url(href)

        logger.debug("No external link found for Reddit post %s", reddit_post_url)
    except Exception as exc:
        logger.warning("Failed to extract external URL from Reddit post %s: %s", reddit_post_url, exc)

    return None


def _extract_external_from_entry(entry) -> Optional[str]:
    """Extract an external URL from a parsed RSS entry payload."""
    if not entry:
        return None

    candidates = []
    for content_block in entry.get("content", []) or []:
        if isinstance(content_block, dict):
            candidates.append(content_block.get("value", ""))
        else:
            candidates.append(getattr(content_block, "value", ""))

    for field in ("summary", "description"):
        value = entry.get(field)
        if value:
            candidates.append(value)

    for candidate in candidates:
        if not candidate:
            continue

        soup = BeautifulSoup(str(candidate), "html.parser")

        tag = soup.find("shreddit-post")
        if tag and tag.get("content-href"):
            link = tag["content-href"]
            if not _is_reddit_internal(link) and "reddit.com/gallery" not in link:
                return clean_url(link)

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http") and not _is_reddit_internal(href) and "reddit.com/gallery" not in href:
                return clean_url(href)

    return None


class RedditSource(RSSSource):
    """Reddit subreddit adapter.

    Parses the subreddit RSS feed and for each post resolves the external
    article URL that the post links to (skipping image/video posts and
    gallery links).
    """

    def fetch(self, limit: int = 50) -> list:
        """Fetch external articles linked from Reddit posts.

        Args:
            limit: Maximum number of articles to return

        Returns:
            List of article dicts (only posts that link to external articles)
        """
        raw_articles = super().fetch(limit=limit)
        feed_content = self._download_feed_content()
        url_feed = feedparser.parse(self.rss_url)
        content_feed = feedparser.parse(feed_content) if feed_content else None
        feed = url_feed
        if content_feed and len(content_feed.entries) > len(url_feed.entries):
            feed = content_feed

        entries_by_link = {}
        for entry in getattr(feed, "entries", []) or []:
            link = entry.get("link", "")
            if link:
                entries_by_link[clean_url(link)] = entry

        articles = []
        for article in raw_articles:
            rss_link = article["rss_link"]
            entry = entries_by_link.get(clean_url(rss_link))
            external_url = _extract_external_from_entry(entry)

            if external_url is None:
                external_url = extract_reddit_external(rss_link)

            if external_url is None:
                logger.info(
                    "No external URL found for Reddit post %s; keeping subreddit post URL",
                    rss_link,
                )
                article["source_url"] = rss_link
                article["external_url"] = None
                article["url"] = rss_link
                article["original_url"] = rss_link
            else:
                article["source_url"] = external_url
                article["external_url"] = external_url
                article["url"] = external_url
                article["original_url"] = rss_link
            articles.append(article)

        return articles
