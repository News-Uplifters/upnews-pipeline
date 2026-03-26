"""Unit tests for the pluggable source adapter pattern (Task 2)."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from crawler.sources import BaseSource, get_source_adapter
from crawler.sources.rss import RSSSource, clean_url
from crawler.sources.reddit import RedditSource, extract_reddit_external


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs):
    base = {
        "name": "Test Source",
        "source_id": "TestSource",
        "rss_url": "https://example.com/rss.xml",
        "threshold": 0.80,
        "active": True,
    }
    base.update(kwargs)
    return base


def _make_entry(title="Test Article", link="https://example.com/article-1", published_parsed=None):
    entry = MagicMock()
    entry.get = lambda k, default="": {
        "title": title,
        "link": link,
    }.get(k, default)
    entry.published_parsed = published_parsed
    entry.updated_parsed = None
    return entry


def _make_feed(entries):
    feed = MagicMock()
    feed.entries = entries
    return feed


# ---------------------------------------------------------------------------
# BaseSource
# ---------------------------------------------------------------------------

class TestBaseSource:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BaseSource(_make_config())

    def test_concrete_subclass_must_implement_fetch(self):
        class Incomplete(BaseSource):
            pass

        with pytest.raises(TypeError):
            Incomplete(_make_config())

    def test_concrete_subclass_works(self):
        class Concrete(BaseSource):
            def fetch(self, limit=50):
                return []

        src = Concrete(_make_config())
        assert src.name == "Test Source"
        assert src.source_id == "TestSource"
        assert src.rss_url == "https://example.com/rss.xml"
        assert src.threshold == 0.80
        assert src.active is True

    def test_defaults_applied(self):
        class Concrete(BaseSource):
            def fetch(self, limit=50):
                return []

        cfg = {"name": "X", "source_id": "X", "rss_url": "https://x.com/rss.xml"}
        src = Concrete(cfg)
        assert src.threshold == 0.75
        assert src.active is True

    def test_repr(self):
        class Concrete(BaseSource):
            def fetch(self, limit=50):
                return []

        src = Concrete(_make_config())
        assert "Concrete" in repr(src)
        assert "TestSource" in repr(src)


# ---------------------------------------------------------------------------
# clean_url
# ---------------------------------------------------------------------------

class TestCleanUrl:
    def test_removes_query_params(self):
        assert clean_url("https://example.com/article?utm_source=rss") == "https://example.com/article"

    def test_removes_fragment(self):
        assert clean_url("https://example.com/article#section") == "https://example.com/article"

    def test_preserves_path(self):
        assert clean_url("https://bbc.com/news/world/12345") == "https://bbc.com/news/world/12345"

    def test_no_change_when_clean(self):
        url = "https://example.com/article"
        assert clean_url(url) == url


# ---------------------------------------------------------------------------
# RSSSource
# ---------------------------------------------------------------------------

class TestRSSSource:
    def _source(self, **kwargs):
        return RSSSource(_make_config(**kwargs))

    def test_is_base_source(self):
        assert issubclass(RSSSource, BaseSource)

    def test_fetch_returns_articles(self):
        src = self._source()
        entries = [
            _make_entry("Article 1", "https://example.com/a1"),
            _make_entry("Article 2", "https://example.com/a2"),
        ]
        mock_feed = _make_feed(entries)

        with patch("crawler.sources.rss.feedparser.parse", return_value=mock_feed), \
             patch.object(src, "_download_feed_content", return_value=None):
            articles = src.fetch(limit=10)

        assert len(articles) == 2
        assert articles[0]["title"] == "Article 1"
        assert articles[0]["original_url"] == "https://example.com/a1"
        assert articles[0]["source_id"] == "TestSource"

    def test_fetch_respects_limit(self):
        src = self._source()
        entries = [_make_entry(f"Article {i}", f"https://example.com/{i}") for i in range(20)]
        mock_feed = _make_feed(entries)

        with patch("crawler.sources.rss.feedparser.parse", return_value=mock_feed), \
             patch.object(src, "_download_feed_content", return_value=None):
            articles = src.fetch(limit=5)

        assert len(articles) == 5

    def test_fetch_skips_entries_without_title(self):
        src = self._source()
        entries = [
            _make_entry("", "https://example.com/a1"),
            _make_entry("Good Article", "https://example.com/a2"),
        ]
        mock_feed = _make_feed(entries)

        with patch("crawler.sources.rss.feedparser.parse", return_value=mock_feed), \
             patch.object(src, "_download_feed_content", return_value=None):
            articles = src.fetch()

        assert len(articles) == 1
        assert articles[0]["title"] == "Good Article"

    def test_fetch_skips_entries_without_link(self):
        src = self._source()
        entries = [
            _make_entry("No Link", ""),
            _make_entry("Has Link", "https://example.com/a2"),
        ]
        mock_feed = _make_feed(entries)

        with patch("crawler.sources.rss.feedparser.parse", return_value=mock_feed), \
             patch.object(src, "_download_feed_content", return_value=None):
            articles = src.fetch()

        assert len(articles) == 1

    def test_fetch_parses_published_date(self):
        src = self._source()
        published_tuple = (2026, 3, 25, 10, 30, 0, 0, 0, 0)
        entry = _make_entry("Article", "https://example.com/a")
        entry.published_parsed = published_tuple
        mock_feed = _make_feed([entry])

        with patch("crawler.sources.rss.feedparser.parse", return_value=mock_feed), \
             patch.object(src, "_download_feed_content", return_value=None):
            articles = src.fetch()

        assert articles[0]["published"] == datetime(2026, 3, 25, 10, 30, 0)

    def test_fetch_falls_back_to_updated_parsed(self):
        src = self._source()
        updated_tuple = (2026, 3, 20, 8, 0, 0, 0, 0, 0)
        entry = _make_entry("Article", "https://example.com/a")
        entry.published_parsed = None
        entry.updated_parsed = updated_tuple
        mock_feed = _make_feed([entry])

        with patch("crawler.sources.rss.feedparser.parse", return_value=mock_feed), \
             patch.object(src, "_download_feed_content", return_value=None):
            articles = src.fetch()

        assert articles[0]["published"] == datetime(2026, 3, 20, 8, 0, 0)

    def test_fetch_published_none_when_missing(self):
        src = self._source()
        entry = _make_entry("Article", "https://example.com/a")
        entry.published_parsed = None
        entry.updated_parsed = None
        mock_feed = _make_feed([entry])

        with patch("crawler.sources.rss.feedparser.parse", return_value=mock_feed), \
             patch.object(src, "_download_feed_content", return_value=None):
            articles = src.fetch()

        assert articles[0]["published"] is None

    def test_fetch_uses_content_feed_when_richer(self):
        """Prefer the feed with more entries (content_feed vs url_feed)."""
        src = self._source()
        url_entries = [_make_entry(f"URL {i}", f"https://example.com/url{i}") for i in range(3)]
        content_entries = [_make_entry(f"Content {i}", f"https://example.com/c{i}") for i in range(8)]

        url_feed = _make_feed(url_entries)
        content_feed = _make_feed(content_entries)

        with patch("crawler.sources.rss.feedparser.parse", return_value=url_feed), \
             patch.object(src, "_download_feed_content", return_value=b"<rss/>"), \
             patch("crawler.sources.rss.feedparser.parse", side_effect=[url_feed, content_feed]):
            articles = src.fetch(limit=10)

        assert len(articles) == 8

    def test_download_feed_content_handles_network_error(self):
        src = self._source()
        with patch("crawler.sources.rss.requests.get", side_effect=Exception("timeout")):
            result = src._download_feed_content()
        assert result is None

    def test_download_feed_content_handles_non_ok_response(self):
        src = self._source()
        mock_resp = MagicMock()
        mock_resp.ok = False
        with patch("crawler.sources.rss.requests.get", return_value=mock_resp):
            result = src._download_feed_content()
        assert result is None

    def test_url_query_params_stripped(self):
        src = self._source()
        entry = _make_entry("Article", "https://example.com/a?utm_source=rss&ref=feed")
        mock_feed = _make_feed([entry])

        with patch("crawler.sources.rss.feedparser.parse", return_value=mock_feed), \
             patch.object(src, "_download_feed_content", return_value=None):
            articles = src.fetch()

        assert "utm_source" not in articles[0]["original_url"]
        assert articles[0]["original_url"] == "https://example.com/a"


# ---------------------------------------------------------------------------
# extract_reddit_external
# ---------------------------------------------------------------------------

class TestExtractRedditExternal:
    def _mock_html(self, content):
        mock_resp = MagicMock()
        mock_resp.text = content
        mock_resp.ok = True
        return mock_resp

    def test_extracts_shreddit_post_href(self):
        html = '<shreddit-post content-href="https://example.com/news"></shreddit-post>'
        with patch("crawler.sources.reddit.requests.get", return_value=self._mock_html(html)):
            url = extract_reddit_external("https://reddit.com/r/news/abc")
        assert url == "https://example.com/news"

    def test_skips_reddit_internal_shreddit(self):
        html = '<shreddit-post content-href="https://reddit.com/r/other/post"></shreddit-post><a href="https://external.com/article">link</a>'
        with patch("crawler.sources.reddit.requests.get", return_value=self._mock_html(html)):
            url = extract_reddit_external("https://reddit.com/r/news/abc")
        assert url == "https://external.com/article"

    def test_extracts_link_from_post_content_div(self):
        html = '<div data-testid="post-content"><a href="https://bbc.com/article">Read more</a></div>'
        with patch("crawler.sources.reddit.requests.get", return_value=self._mock_html(html)):
            url = extract_reddit_external("https://reddit.com/r/news/abc")
        assert url == "https://bbc.com/article"

    def test_fallback_to_any_external_link(self):
        html = '<a href="https://guardian.com/article">Guardian</a>'
        with patch("crawler.sources.reddit.requests.get", return_value=self._mock_html(html)):
            url = extract_reddit_external("https://reddit.com/r/news/abc")
        assert url == "https://guardian.com/article"

    def test_returns_none_when_only_reddit_links(self):
        html = '<a href="https://reddit.com/r/other">other</a>'
        with patch("crawler.sources.reddit.requests.get", return_value=self._mock_html(html)):
            url = extract_reddit_external("https://reddit.com/r/news/abc")
        assert url is None

    def test_skips_gallery_links(self):
        html = '<a href="https://reddit.com/gallery/abc123">gallery</a><a href="https://bbc.com/article">bbc</a>'
        with patch("crawler.sources.reddit.requests.get", return_value=self._mock_html(html)):
            url = extract_reddit_external("https://reddit.com/r/news/abc")
        assert url == "https://bbc.com/article"

    def test_skips_i_redd_it_links(self):
        html = '<a href="https://i.redd.it/image.jpg">img</a><a href="https://bbc.com/article">bbc</a>'
        with patch("crawler.sources.reddit.requests.get", return_value=self._mock_html(html)):
            url = extract_reddit_external("https://reddit.com/r/news/abc")
        assert url == "https://bbc.com/article"

    def test_returns_none_on_network_error(self):
        with patch("crawler.sources.reddit.requests.get", side_effect=Exception("network error")):
            url = extract_reddit_external("https://reddit.com/r/news/abc")
        assert url is None

    def test_strips_query_params_from_result(self):
        html = '<a href="https://bbc.com/article?ref=reddit">bbc</a>'
        with patch("crawler.sources.reddit.requests.get", return_value=self._mock_html(html)):
            url = extract_reddit_external("https://reddit.com/r/news/abc")
        assert url == "https://bbc.com/article"


# ---------------------------------------------------------------------------
# RedditSource
# ---------------------------------------------------------------------------

class TestRedditSource:
    def _source(self):
        return RedditSource(_make_config(
            source_id="RedditUpliftingNews",
            rss_url="https://www.reddit.com/r/UpliftingNews/.rss",
            adapter="reddit",
        ))

    def test_is_base_source(self):
        assert issubclass(RedditSource, BaseSource)

    def test_fetch_resolves_external_urls(self):
        src = self._source()
        rss_entry = _make_entry("Great News", "https://www.reddit.com/r/UpliftingNews/comments/abc123/")
        mock_feed = _make_feed([rss_entry])

        with patch("crawler.sources.rss.feedparser.parse", return_value=mock_feed), \
             patch.object(src, "_download_feed_content", return_value=None), \
             patch("crawler.sources.reddit.extract_reddit_external", return_value="https://bbc.com/article"):
            articles = src.fetch()

        assert len(articles) == 1
        assert articles[0]["original_url"] == "https://bbc.com/article"
        assert articles[0]["title"] == "Great News"

    def test_fetch_skips_posts_without_external_link(self):
        """Posts that only link to Reddit itself (gallery, image) should be excluded."""
        src = self._source()
        entries = [
            _make_entry("Gallery Post", "https://www.reddit.com/r/UpliftingNews/comments/gallery/"),
            _make_entry("External Article", "https://www.reddit.com/r/UpliftingNews/comments/xyz/"),
        ]
        mock_feed = _make_feed(entries)

        def mock_extract(url):
            if "gallery" in url:
                return None
            return "https://example.com/article"

        with patch("crawler.sources.rss.feedparser.parse", return_value=mock_feed), \
             patch.object(src, "_download_feed_content", return_value=None), \
             patch("crawler.sources.reddit.extract_reddit_external", side_effect=mock_extract):
            articles = src.fetch()

        assert len(articles) == 1
        assert articles[0]["original_url"] == "https://example.com/article"

    def test_fetch_respects_limit(self):
        src = self._source()
        entries = [_make_entry(f"Post {i}", f"https://reddit.com/r/n/comments/{i}/") for i in range(20)]
        mock_feed = _make_feed(entries)

        with patch("crawler.sources.rss.feedparser.parse", return_value=mock_feed), \
             patch.object(src, "_download_feed_content", return_value=None), \
             patch("crawler.sources.reddit.extract_reddit_external", return_value="https://bbc.com/a"):
            articles = src.fetch(limit=5)

        assert len(articles) <= 5

    def test_source_id_set_on_articles(self):
        src = self._source()
        entry = _make_entry("News", "https://reddit.com/r/n/comments/x/")
        mock_feed = _make_feed([entry])

        with patch("crawler.sources.rss.feedparser.parse", return_value=mock_feed), \
             patch.object(src, "_download_feed_content", return_value=None), \
             patch("crawler.sources.reddit.extract_reddit_external", return_value="https://bbc.com/a"):
            articles = src.fetch()

        assert articles[0]["source_id"] == "RedditUpliftingNews"


# ---------------------------------------------------------------------------
# Factory: get_source_adapter
# ---------------------------------------------------------------------------

class TestGetSourceAdapter:
    def test_returns_rss_source_by_default(self):
        config = _make_config()
        adapter = get_source_adapter(config)
        assert isinstance(adapter, RSSSource)

    def test_returns_rss_source_for_rss_adapter(self):
        config = _make_config(adapter="rss")
        adapter = get_source_adapter(config)
        assert isinstance(adapter, RSSSource)

    def test_returns_reddit_source_for_reddit_adapter(self):
        config = _make_config(adapter="reddit")
        adapter = get_source_adapter(config)
        assert isinstance(adapter, RedditSource)

    def test_returns_rss_source_for_none_adapter(self):
        config = _make_config(adapter=None)
        adapter = get_source_adapter(config)
        assert isinstance(adapter, RSSSource)

    def test_returns_rss_for_unknown_adapter(self):
        """Unknown adapter types fall back to RSSSource."""
        config = _make_config(adapter="twitter")
        adapter = get_source_adapter(config)
        assert isinstance(adapter, RSSSource)

    def test_config_propagated_to_adapter(self):
        config = _make_config(source_id="MySource", threshold=0.92)
        adapter = get_source_adapter(config)
        assert adapter.source_id == "MySource"
        assert adapter.threshold == 0.92


# ---------------------------------------------------------------------------
# Backward compatibility: rss_reader.py still works
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_fetch_rss_headlines_still_importable(self):
        from crawler.rss_reader import fetch_rss_headlines
        assert callable(fetch_rss_headlines)

    def test_crawl_package_exports(self):
        from crawler import fetch_rss_headlines, load_sources, get_source_adapter, BaseSource
        assert callable(fetch_rss_headlines)
        assert callable(load_sources)
        assert callable(get_source_adapter)
        assert BaseSource is not None

    def test_fetch_rss_headlines_returns_list(self):
        """Smoke test: existing function still works with mocked feedparser."""
        from crawler.rss_reader import fetch_rss_headlines
        from unittest.mock import patch, MagicMock

        entry = MagicMock()
        entry.get = lambda k, d="": {"title": "Old Article", "link": "https://bbc.com/a"}.get(k, d)
        entry.published_parsed = None
        entry.updated_parsed = None

        mock_feed = MagicMock()
        mock_feed.entries = [entry]

        with patch("crawler.rss_reader.feedparser.parse", return_value=mock_feed), \
             patch("crawler.rss_reader._download_feed_content", return_value=None):
            articles = fetch_rss_headlines("https://bbc.com/rss.xml", limit=10)

        assert isinstance(articles, list)


# ---------------------------------------------------------------------------
# crawl_all_sources integration (mocked)
# ---------------------------------------------------------------------------

class TestCrawlAllSources:
    def test_uses_adapter_factory(self):
        """crawl_all_sources should call get_source_adapter for each source."""
        import pandas as pd
        from crawler.crawl_all_sources import crawl_all_sources

        sources_df = pd.DataFrame([
            {"name": "BBC", "source_id": "BBC", "rss_url": "https://bbc.com/rss", "threshold": 0.9, "active": True, "adapter": "rss"},
            {"name": "Reddit", "source_id": "Reddit", "rss_url": "https://reddit.com/r/n/.rss", "threshold": 0.75, "active": True, "adapter": "reddit"},
        ])

        mock_rss_adapter = MagicMock(spec=RSSSource)
        mock_rss_adapter.rss_url = "https://bbc.com/rss"
        mock_rss_adapter.__class__ = RSSSource
        mock_rss_adapter.fetch.return_value = [{"title": "BBC article", "original_url": "https://bbc.com/a", "rss_link": "https://bbc.com/a", "published": None, "source_id": "BBC"}]

        mock_reddit_adapter = MagicMock(spec=RedditSource)
        mock_reddit_adapter.rss_url = "https://reddit.com/r/n/.rss"
        mock_reddit_adapter.__class__ = RedditSource
        mock_reddit_adapter.fetch.return_value = [{"title": "Reddit article", "original_url": "https://bbc.com/b", "rss_link": "https://reddit.com/x", "published": None, "source_id": "Reddit"}]

        adapters = [mock_rss_adapter, mock_reddit_adapter]

        with patch("crawler.crawl_all_sources.load_sources", return_value=sources_df), \
             patch("crawler.crawl_all_sources.get_source_adapter", side_effect=adapters):
            articles = crawl_all_sources(limit_per_source=10)

        assert len(articles) == 2
        titles = [a["title"] for a in articles]
        assert "BBC article" in titles
        assert "Reddit article" in titles

    def test_skips_sources_without_rss_url(self):
        import pandas as pd
        from crawler.crawl_all_sources import crawl_all_sources

        sources_df = pd.DataFrame([
            {"name": "NoURL", "source_id": "NoURL", "rss_url": None, "threshold": 0.8, "active": True, "adapter": "rss"},
        ])

        with patch("crawler.crawl_all_sources.load_sources", return_value=sources_df):
            articles = crawl_all_sources()

        assert articles == []

    def test_continues_after_source_error(self):
        import pandas as pd
        from crawler.crawl_all_sources import crawl_all_sources

        sources_df = pd.DataFrame([
            {"name": "Broken", "source_id": "Broken", "rss_url": "https://broken.com/rss", "threshold": 0.8, "active": True, "adapter": "rss"},
            {"name": "Good", "source_id": "Good", "rss_url": "https://good.com/rss", "threshold": 0.8, "active": True, "adapter": "rss"},
        ])

        good_adapter = MagicMock()
        good_adapter.rss_url = "https://good.com/rss"
        good_adapter.__class__ = RSSSource
        good_adapter.fetch.return_value = [{"title": "Good", "original_url": "https://good.com/a", "rss_link": "https://good.com/a", "published": None, "source_id": "Good"}]

        def side_effect(config):
            if config["source_id"] == "Broken":
                raise Exception("Connection refused")
            return good_adapter

        with patch("crawler.crawl_all_sources.load_sources", return_value=sources_df), \
             patch("crawler.crawl_all_sources.get_source_adapter", side_effect=side_effect):
            articles = crawl_all_sources()

        assert len(articles) == 1
        assert articles[0]["title"] == "Good"
