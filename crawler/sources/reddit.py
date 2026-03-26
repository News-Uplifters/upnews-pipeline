"""Reddit subreddit adapter — extracts external article links from Reddit posts."""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from typing import Optional

from . import BaseSource
from .rss import RSSSource, clean_url

_REDDIT_DOMAINS = {"reddit.com", "redd.it", "i.redd.it", "v.redd.it"}


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

    except Exception:
        pass

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
        articles = []
        for article in raw_articles:
            rss_link = article["rss_link"]
            external_url = extract_reddit_external(rss_link)

            if external_url is None:
                # Post links only to Reddit itself (gallery, image, etc.) — skip
                continue

            article["original_url"] = external_url
            articles.append(article)

        return articles
