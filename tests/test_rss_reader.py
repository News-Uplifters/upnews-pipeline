"""Unit tests for crawler/rss_reader.py (Task 9).

Uses feedparser directly with inline XML strings and mocked HTTP requests
so no network calls are made during tests.
"""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import feedparser
import pytest

from crawler.rss_reader import (
    _download_feed_content,
    clean_url,
    extract_reddit_external,
    fetch_rss_headlines,
)

# ---------------------------------------------------------------------------
# Inline XML fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name: str) -> bytes:
    with open(os.path.join(FIXTURES_DIR, name), "rb") as f:
        return f.read()


SAMPLE_RSS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test News Feed</title>
    <link>https://example.com</link>
    <description>Test news feed</description>
    <item>
      <title>Scientists discover breakthrough cancer treatment</title>
      <link>https://example.com/health/cancer-breakthrough</link>
      <pubDate>Thu, 26 Mar 2026 10:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Community volunteers restore local park</title>
      <link>https://example.com/community/park-restored</link>
      <pubDate>Thu, 26 Mar 2026 09:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Teen wins international robotics competition</title>
      <link>https://example.com/tech/robotics-winner</link>
      <pubDate>Thu, 26 Mar 2026 08:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""

EMPTY_RSS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Empty Feed</title>
    <link>https://example.com</link>
    <description>No articles</description>
  </channel>
</rss>"""

RSS_WITH_NO_PUBDATE = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Feed Without Dates</title>
    <link>https://example.com</link>
    <item>
      <title>Article with no date</title>
      <link>https://example.com/no-date</link>
    </item>
  </channel>
</rss>"""

RSS_WITH_MISSING_FIELDS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Feed with incomplete items</title>
    <link>https://example.com</link>
    <item>
      <title>Valid Article</title>
      <link>https://example.com/valid</link>
    </item>
    <item>
      <title>Article without link</title>
    </item>
    <item>
      <link>https://example.com/no-title</link>
    </item>
  </channel>
</rss>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parsed_feed(xml_bytes: bytes):
    """Parse XML bytes with feedparser and return the result."""
    return feedparser.parse(xml_bytes)


def _mock_response(text: str, ok: bool = True):
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.ok = ok
    resp.text = text
    resp.content = text.encode()
    return resp


# ---------------------------------------------------------------------------
# clean_url tests
# ---------------------------------------------------------------------------


def test_clean_url_strips_query_params():
    assert clean_url("https://example.com/page?q=1&ref=feed") == "https://example.com/page"


def test_clean_url_strips_fragment():
    assert clean_url("https://example.com/page#section") == "https://example.com/page"


def test_clean_url_strips_query_and_fragment():
    assert clean_url("https://example.com/page?q=1#top") == "https://example.com/page"


def test_clean_url_preserves_scheme_host_path():
    url = "https://bbc.co.uk/news/world/article-123"
    assert clean_url(url) == url


def test_clean_url_no_modification_needed():
    url = "https://example.com/article"
    assert clean_url(url) == url


# ---------------------------------------------------------------------------
# fetch_rss_headlines tests
# ---------------------------------------------------------------------------


@patch("crawler.rss_reader._download_feed_content", return_value=None)
@patch("crawler.rss_reader.feedparser")
def test_fetch_rss_headlines_returns_articles(mock_fp, mock_download):
    mock_fp.parse.return_value = _parsed_feed(SAMPLE_RSS_XML)
    articles = fetch_rss_headlines("https://example.com/feed.xml")
    assert len(articles) == 3
    titles = [a["title"] for a in articles]
    assert "Scientists discover breakthrough cancer treatment" in titles
    assert "Community volunteers restore local park" in titles
    assert "Teen wins international robotics competition" in titles


@patch("crawler.rss_reader._download_feed_content", return_value=None)
@patch("crawler.rss_reader.feedparser")
def test_fetch_rss_headlines_article_has_required_keys(mock_fp, mock_download):
    mock_fp.parse.return_value = _parsed_feed(SAMPLE_RSS_XML)
    articles = fetch_rss_headlines("https://example.com/feed.xml")
    assert len(articles) > 0
    article = articles[0]
    assert "title" in article
    assert "rss_link" in article
    assert "original_url" in article
    assert "published" in article


@patch("crawler.rss_reader._download_feed_content", return_value=None)
@patch("crawler.rss_reader.feedparser")
def test_fetch_rss_headlines_empty_feed_returns_empty_list(mock_fp, mock_download):
    mock_fp.parse.return_value = _parsed_feed(EMPTY_RSS_XML)
    articles = fetch_rss_headlines("https://example.com/feed.xml")
    assert articles == []


@patch("crawler.rss_reader._download_feed_content", return_value=None)
@patch("crawler.rss_reader.feedparser")
def test_fetch_rss_headlines_respects_limit(mock_fp, mock_download):
    mock_fp.parse.return_value = _parsed_feed(SAMPLE_RSS_XML)
    articles = fetch_rss_headlines("https://example.com/feed.xml", limit=2)
    assert len(articles) == 2


@patch("crawler.rss_reader._download_feed_content", return_value=None)
@patch("crawler.rss_reader.feedparser")
def test_fetch_rss_headlines_parses_published_date(mock_fp, mock_download):
    mock_fp.parse.return_value = _parsed_feed(SAMPLE_RSS_XML)
    articles = fetch_rss_headlines("https://example.com/feed.xml")
    assert articles[0]["published"] is not None
    assert isinstance(articles[0]["published"], datetime)
    assert articles[0]["published"].year == 2026
    assert articles[0]["published"].month == 3
    assert articles[0]["published"].day == 26


@patch("crawler.rss_reader._download_feed_content", return_value=None)
@patch("crawler.rss_reader.feedparser")
def test_fetch_rss_headlines_no_date_is_none(mock_fp, mock_download):
    mock_fp.parse.return_value = _parsed_feed(RSS_WITH_NO_PUBDATE)
    articles = fetch_rss_headlines("https://example.com/feed.xml")
    assert len(articles) == 1
    assert articles[0]["published"] is None


@patch("crawler.rss_reader._download_feed_content", return_value=None)
@patch("crawler.rss_reader.feedparser")
def test_fetch_rss_headlines_skips_items_without_title(mock_fp, mock_download):
    mock_fp.parse.return_value = _parsed_feed(RSS_WITH_MISSING_FIELDS)
    articles = fetch_rss_headlines("https://example.com/feed.xml")
    assert len(articles) == 1
    assert articles[0]["title"] == "Valid Article"


@patch("crawler.rss_reader._download_feed_content", return_value=None)
@patch("crawler.rss_reader.feedparser")
def test_fetch_rss_headlines_strips_query_from_original_url(mock_fp, mock_download):
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>T</title><link>https://example.com</link>
  <item>
    <title>Article with tracking params</title>
    <link>https://example.com/article?utm_source=rss&amp;ref=feed</link>
  </item>
</channel></rss>"""
    mock_fp.parse.return_value = _parsed_feed(xml)
    articles = fetch_rss_headlines("https://example.com/feed.xml")
    assert len(articles) == 1
    assert articles[0]["original_url"] == "https://example.com/article"


@patch("crawler.rss_reader._download_feed_content", return_value=None)
@patch("crawler.rss_reader.feedparser")
def test_fetch_rss_headlines_skips_reddit_image_links(mock_fp, mock_download):
    """Articles whose resolved URL points to i.redd.it are skipped."""
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>T</title><link>https://reddit.com</link>
  <item>
    <title>Image post</title>
    <link>https://www.reddit.com/r/UpliftingNews/comments/img001/image_post/</link>
  </item>
</channel></rss>"""
    mock_fp.parse.return_value = _parsed_feed(xml)
    with patch("crawler.rss_reader.extract_reddit_external",
               return_value="https://i.redd.it/someimage.jpg"):
        articles = fetch_rss_headlines("https://reddit.com/feed.xml")
    assert articles == []


@patch("crawler.rss_reader._download_feed_content", return_value=None)
@patch("crawler.rss_reader.feedparser")
def test_fetch_rss_headlines_skips_reddit_video_links(mock_fp, mock_download):
    """Articles whose resolved URL points to v.redd.it are skipped."""
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>T</title><link>https://reddit.com</link>
  <item>
    <title>Video post</title>
    <link>https://www.reddit.com/r/UpliftingNews/comments/vid001/video_post/</link>
  </item>
</channel></rss>"""
    mock_fp.parse.return_value = _parsed_feed(xml)
    with patch("crawler.rss_reader.extract_reddit_external",
               return_value="https://v.redd.it/somevideo"):
        articles = fetch_rss_headlines("https://reddit.com/feed.xml")
    assert articles == []


@patch("crawler.rss_reader._download_feed_content", return_value=None)
@patch("crawler.rss_reader.feedparser")
def test_fetch_rss_headlines_skips_subreddit_urls(mock_fp, mock_download):
    """Articles whose resolved URL is a subreddit page (/r/) are skipped."""
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>T</title><link>https://reddit.com</link>
  <item>
    <title>Subreddit link post</title>
    <link>https://www.reddit.com/r/UpliftingNews/comments/sub001/subreddit/</link>
  </item>
</channel></rss>"""
    mock_fp.parse.return_value = _parsed_feed(xml)
    with patch("crawler.rss_reader.extract_reddit_external",
               return_value="https://www.reddit.com/r/SomeSubreddit/"):
        articles = fetch_rss_headlines("https://reddit.com/feed.xml")
    assert articles == []


@patch("crawler.rss_reader.feedparser")
def test_fetch_rss_headlines_prefers_larger_content_feed(mock_fp):
    """When content feed has more entries than URL feed, content feed is used."""
    small_feed = _parsed_feed(b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>T</title><link>https://example.com</link>
  <item><title>Article 1</title><link>https://example.com/1</link></item>
</channel></rss>""")
    big_feed = _parsed_feed(SAMPLE_RSS_XML)  # 3 articles

    with patch("crawler.rss_reader._download_feed_content", return_value=b"<some bytes>"):
        # First call: feedparser.parse(rss_url) → small feed
        # Second call: feedparser.parse(feed_content) → big feed
        mock_fp.parse.side_effect = [small_feed, big_feed]
        articles = fetch_rss_headlines("https://example.com/feed.xml")

    assert len(articles) == 3  # came from big_feed


@patch("crawler.rss_reader._download_feed_content", return_value=None)
@patch("crawler.rss_reader.feedparser")
def test_fetch_rss_headlines_malformed_feed_handled_gracefully(mock_fp, mock_download):
    """feedparser never raises on malformed XML — empty or partial list returned."""
    mock_fp.parse.return_value = _parsed_feed(_load_fixture("malformed_feed.xml"))
    articles = fetch_rss_headlines("https://example.com/broken.xml")
    assert isinstance(articles, list)


# ---------------------------------------------------------------------------
# _download_feed_content tests
# ---------------------------------------------------------------------------


@patch("crawler.rss_reader.requests")
def test_download_feed_content_returns_bytes_on_success(mock_requests):
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.content = b"<rss>...</rss>"
    mock_requests.get.return_value = mock_resp
    result = _download_feed_content("https://example.com/feed.xml")
    assert result == b"<rss>...</rss>"


@patch("crawler.rss_reader.requests")
def test_download_feed_content_returns_none_on_http_error(mock_requests):
    mock_resp = MagicMock()
    mock_resp.ok = False
    mock_resp.content = b""
    mock_requests.get.return_value = mock_resp
    result = _download_feed_content("https://example.com/feed.xml")
    assert result is None


@patch("crawler.rss_reader.requests")
def test_download_feed_content_returns_none_on_exception(mock_requests):
    mock_requests.get.side_effect = Exception("Connection refused")
    result = _download_feed_content("https://example.com/feed.xml")
    assert result is None


# ---------------------------------------------------------------------------
# extract_reddit_external tests
# ---------------------------------------------------------------------------


@patch("crawler.rss_reader.requests")
def test_extract_reddit_external_shreddit_post_element(mock_requests):
    """Extracts URL from shreddit-post[content-href] (new Reddit UI)."""
    html = """<html><body>
        <shreddit-post content-href="https://news.example.com/article/42"></shreddit-post>
    </body></html>"""
    mock_requests.get.return_value = _mock_response(html)
    result = extract_reddit_external("https://www.reddit.com/r/News/comments/abc123/post/")
    assert result == "https://news.example.com/article/42"


@patch("crawler.rss_reader.requests")
def test_extract_reddit_external_post_content_div(mock_requests):
    """Extracts URL from post-content div (old Reddit UI)."""
    html = """<html><body>
        <div data-testid="post-content">
            <a href="https://bbc.com/news/uk-12345678">BBC Article</a>
        </div>
    </body></html>"""
    mock_requests.get.return_value = _mock_response(html)
    result = extract_reddit_external("https://www.reddit.com/r/News/comments/def456/post/")
    assert result == "https://bbc.com/news/uk-12345678"


@patch("crawler.rss_reader.requests")
def test_extract_reddit_external_fallback_to_any_external_link(mock_requests):
    """Falls back to any non-Reddit external link on the page."""
    html = """<html><body>
        <a href="https://www.reddit.com/r/other">Internal</a>
        <a href="https://nytimes.com/science/discovery">NYT Article</a>
    </body></html>"""
    mock_requests.get.return_value = _mock_response(html)
    result = extract_reddit_external("https://www.reddit.com/r/News/comments/ghi789/post/")
    assert result == "https://nytimes.com/science/discovery"


@patch("crawler.rss_reader.requests")
def test_extract_reddit_external_ignores_reddit_internal_links(mock_requests):
    """Returns original URL when no external links found on page."""
    html = """<html><body>
        <a href="https://www.reddit.com/r/News">Subreddit</a>
        <a href="https://redd.it/abc123">Short link</a>
    </body></html>"""
    mock_requests.get.return_value = _mock_response(html)
    post_url = "https://www.reddit.com/r/News/comments/zzz/post/"
    result = extract_reddit_external(post_url)
    assert result == post_url


@patch("crawler.rss_reader.requests")
def test_extract_reddit_external_skips_gallery_links(mock_requests):
    """Links containing reddit.com/gallery are not returned; falls through to next link."""
    html = """<html><body>
        <shreddit-post content-href="https://www.reddit.com/gallery/abc123"></shreddit-post>
        <a href="https://nytimes.com/article">External</a>
    </body></html>"""
    mock_requests.get.return_value = _mock_response(html)
    result = extract_reddit_external("https://www.reddit.com/r/News/comments/gal1/post/")
    assert result == "https://nytimes.com/article"


@patch("crawler.rss_reader.requests")
def test_extract_reddit_external_handles_request_exception(mock_requests):
    """Returns clean original URL when request raises an exception."""
    mock_requests.get.side_effect = Exception("Network error")
    post_url = "https://www.reddit.com/r/News/comments/err1/post/"
    result = extract_reddit_external(post_url)
    assert result == post_url


# ---------------------------------------------------------------------------
# Fixture file integration tests
# ---------------------------------------------------------------------------


def test_sample_feed_fixture_parses_correctly():
    """Ensure the sample fixture file is valid and parseable."""
    feed = feedparser.parse(_load_fixture("sample_feed.xml"))
    assert len(feed.entries) == 3
    assert feed.entries[0].title == "Scientists discover breakthrough cancer treatment"


def test_empty_feed_fixture_has_no_entries():
    feed = feedparser.parse(_load_fixture("empty_feed.xml"))
    assert len(feed.entries) == 0


def test_reddit_feed_fixture_parses_correctly():
    feed = feedparser.parse(_load_fixture("reddit_feed.xml"))
    assert len(feed.entries) == 2
    assert "Dog saves child" in feed.entries[0].title
