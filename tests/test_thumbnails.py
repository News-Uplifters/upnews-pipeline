"""Unit tests for the thumbnail extraction service (Task 3)."""

import asyncio
import sqlite3
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from enrichment.thumbnails import (
    _cache_get,
    _cache_set,
    _extract_favicon,
    _extract_from_img,
    _extract_from_meta,
    _get_cache_connection,
    _make_absolute,
    _MISS,
    extract_thumbnail,
    extract_thumbnails_batch,
)
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def _fresh_cache() -> sqlite3.Connection:
    """Return a clean in-memory cache DB for each test."""
    return _get_cache_connection(":memory:")


# ---------------------------------------------------------------------------
# _make_absolute
# ---------------------------------------------------------------------------

class TestMakeAbsolute:
    def test_absolute_https_unchanged(self):
        assert _make_absolute("https://cdn.example.com/img.jpg", "https://example.com") == "https://cdn.example.com/img.jpg"

    def test_absolute_http_unchanged(self):
        assert _make_absolute("http://cdn.example.com/img.jpg", "https://example.com") == "http://cdn.example.com/img.jpg"

    def test_protocol_relative_uses_base_scheme(self):
        result = _make_absolute("//cdn.example.com/img.jpg", "https://example.com/page")
        assert result == "https://cdn.example.com/img.jpg"

    def test_protocol_relative_http_base(self):
        result = _make_absolute("//cdn.example.com/img.jpg", "http://example.com/page")
        assert result == "http://cdn.example.com/img.jpg"

    def test_relative_path_resolved(self):
        result = _make_absolute("/images/photo.jpg", "https://example.com/article/1")
        assert result == "https://example.com/images/photo.jpg"

    def test_relative_subpath_resolved(self):
        result = _make_absolute("images/photo.jpg", "https://example.com/article/")
        assert result == "https://example.com/article/images/photo.jpg"


# ---------------------------------------------------------------------------
# _extract_from_meta
# ---------------------------------------------------------------------------

class TestExtractFromMeta:
    def test_og_image_extracted(self):
        html = '<meta property="og:image" content="https://example.com/og.jpg">'
        assert _extract_from_meta(_soup(html), "https://example.com") == "https://example.com/og.jpg"

    def test_twitter_image_extracted(self):
        html = '<meta name="twitter:image" content="https://example.com/tw.jpg">'
        assert _extract_from_meta(_soup(html), "https://example.com") == "https://example.com/tw.jpg"

    def test_twitter_image_src_extracted(self):
        html = '<meta name="twitter:image:src" content="https://example.com/tw2.jpg">'
        assert _extract_from_meta(_soup(html), "https://example.com") == "https://example.com/tw2.jpg"

    def test_og_takes_priority_over_twitter(self):
        html = (
            '<meta property="og:image" content="https://example.com/og.jpg">'
            '<meta name="twitter:image" content="https://example.com/tw.jpg">'
        )
        assert _extract_from_meta(_soup(html), "https://example.com") == "https://example.com/og.jpg"

    def test_relative_og_image_made_absolute(self):
        html = '<meta property="og:image" content="/images/photo.jpg">'
        result = _extract_from_meta(_soup(html), "https://example.com")
        assert result == "https://example.com/images/photo.jpg"

    def test_empty_content_skipped(self):
        html = '<meta property="og:image" content="">'
        assert _extract_from_meta(_soup(html), "https://example.com") is None

    def test_no_meta_tags_returns_none(self):
        html = "<html><body><p>No images here.</p></body></html>"
        assert _extract_from_meta(_soup(html), "https://example.com") is None


# ---------------------------------------------------------------------------
# _extract_from_img
# ---------------------------------------------------------------------------

class TestExtractFromImg:
    def test_first_img_returned(self):
        html = '<img src="https://example.com/first.jpg"><img src="https://example.com/second.jpg">'
        assert _extract_from_img(_soup(html), "https://example.com") == "https://example.com/first.jpg"

    def test_relative_img_made_absolute(self):
        html = '<img src="/images/banner.jpg">'
        result = _extract_from_img(_soup(html), "https://example.com")
        assert result == "https://example.com/images/banner.jpg"

    def test_data_uri_skipped(self):
        html = '<img src="data:image/png;base64,abc"><img src="https://example.com/real.jpg">'
        result = _extract_from_img(_soup(html), "https://example.com")
        assert result == "https://example.com/real.jpg"

    def test_empty_src_skipped(self):
        html = '<img src=""><img src="https://example.com/ok.jpg">'
        result = _extract_from_img(_soup(html), "https://example.com")
        assert result == "https://example.com/ok.jpg"

    def test_no_img_tags_returns_none(self):
        html = "<html><body><p>Text only.</p></body></html>"
        assert _extract_from_img(_soup(html), "https://example.com") is None


# ---------------------------------------------------------------------------
# _extract_favicon
# ---------------------------------------------------------------------------

class TestExtractFavicon:
    def test_favicon_url_constructed(self):
        assert _extract_favicon("https://bbc.com/news/article") == "https://bbc.com/favicon.ico"

    def test_favicon_http(self):
        assert _extract_favicon("http://example.com/page") == "http://example.com/favicon.ico"

    def test_invalid_url_returns_none(self):
        assert _extract_favicon("not-a-url") is None

    def test_empty_string_returns_none(self):
        assert _extract_favicon("") is None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

class TestCache:
    def test_miss_before_set(self):
        conn = _fresh_cache()
        assert _cache_get("https://example.com", conn) is _MISS

    def test_hit_after_set(self):
        conn = _fresh_cache()
        _cache_set("https://example.com", "https://cdn.example.com/img.jpg", conn)
        assert _cache_get("https://example.com", conn) == "https://cdn.example.com/img.jpg"

    def test_none_thumbnail_cached(self):
        """Caching None (no thumbnail) must be distinguishable from a cache miss."""
        conn = _fresh_cache()
        _cache_set("https://example.com", None, conn)
        result = _cache_get("https://example.com", conn)
        assert result is None
        assert result is not _MISS

    def test_overwrite_existing_entry(self):
        conn = _fresh_cache()
        _cache_set("https://example.com", "https://old.jpg", conn)
        _cache_set("https://example.com", "https://new.jpg", conn)
        assert _cache_get("https://example.com", conn) == "https://new.jpg"


# ---------------------------------------------------------------------------
# extract_thumbnail — using a fake aiohttp session
# ---------------------------------------------------------------------------

def _make_mock_session(html: str, status: int = 200) -> MagicMock:
    """Build a mock aiohttp.ClientSession that returns *html* for any GET."""
    resp = MagicMock()
    resp.status = status
    resp.text = AsyncMock(return_value=html)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.get = MagicMock(return_value=resp)
    return session


class TestExtractThumbnail:
    def test_og_image_returned(self):
        html = '<meta property="og:image" content="https://example.com/og.jpg">'
        session = _make_mock_session(html)
        cache = _fresh_cache()
        result = asyncio.run(
            extract_thumbnail("https://example.com/article", _cache=cache, _session=session)
        )
        assert result == "https://example.com/og.jpg"

    def test_twitter_image_fallback(self):
        html = '<meta name="twitter:image" content="https://example.com/tw.jpg">'
        session = _make_mock_session(html)
        cache = _fresh_cache()
        result = asyncio.run(
            extract_thumbnail("https://example.com/article", _cache=cache, _session=session)
        )
        assert result == "https://example.com/tw.jpg"

    def test_img_tag_fallback(self):
        html = "<body><img src='https://example.com/img.jpg'></body>"
        session = _make_mock_session(html)
        cache = _fresh_cache()
        result = asyncio.run(
            extract_thumbnail("https://example.com/article", _cache=cache, _session=session)
        )
        assert result == "https://example.com/img.jpg"

    def test_favicon_fallback_when_no_images(self):
        html = "<html><body><p>Text only.</p></body></html>"
        session = _make_mock_session(html)
        cache = _fresh_cache()
        result = asyncio.run(
            extract_thumbnail("https://example.com/article", _cache=cache, _session=session)
        )
        assert result == "https://example.com/favicon.ico"

    def test_favicon_fallback_on_http_error(self):
        session = _make_mock_session("", status=404)
        cache = _fresh_cache()
        result = asyncio.run(
            extract_thumbnail("https://example.com/article", _cache=cache, _session=session)
        )
        assert result == "https://example.com/favicon.ico"

    def test_cache_hit_skips_network(self):
        cache = _fresh_cache()
        _cache_set("https://example.com/article", "https://cached.jpg", cache)

        session = _make_mock_session("<html></html>")
        result = asyncio.run(
            extract_thumbnail("https://example.com/article", _cache=cache, _session=session)
        )
        assert result == "https://cached.jpg"
        session.get.assert_not_called()

    def test_result_written_to_cache(self):
        html = '<meta property="og:image" content="https://example.com/og.jpg">'
        session = _make_mock_session(html)
        cache = _fresh_cache()
        asyncio.run(
            extract_thumbnail("https://example.com/article", _cache=cache, _session=session)
        )
        assert _cache_get("https://example.com/article", cache) == "https://example.com/og.jpg"

    def test_network_failure_returns_favicon(self):
        """If the session raises a client error, fall back to favicon."""
        import aiohttp as _aiohttp

        session = MagicMock()
        resp = MagicMock()
        resp.__aenter__ = AsyncMock(side_effect=_aiohttp.ClientConnectionError("Network error"))
        resp.__aexit__ = AsyncMock(return_value=False)
        session.get = MagicMock(return_value=resp)

        cache = _fresh_cache()
        result = asyncio.run(
            extract_thumbnail(
                "https://example.com/article",
                max_retries=0,
                _cache=cache,
                _session=session,
            )
        )
        assert result == "https://example.com/favicon.ico"

    def test_none_cached_thumbnail_returned_without_network(self):
        """Cached None (no thumbnail ever found) is returned directly."""
        cache = _fresh_cache()
        _cache_set("https://example.com/article", None, cache)

        session = _make_mock_session("<html></html>")
        result = asyncio.run(
            extract_thumbnail("https://example.com/article", _cache=cache, _session=session)
        )
        assert result is None
        session.get.assert_not_called()


# ---------------------------------------------------------------------------
# extract_thumbnails_batch
# ---------------------------------------------------------------------------

class TestExtractThumbnailsBatch:
    def test_empty_list_returns_empty_dict(self):
        result = asyncio.run(
            extract_thumbnails_batch([])
        )
        assert result == {}

    def test_multiple_urls_returned(self):
        html_a = '<meta property="og:image" content="https://a.com/img.jpg">'
        html_b = '<meta property="og:image" content="https://b.com/img.jpg">'

        call_count = {"n": 0}
        pages = {"https://a.com/article": html_a, "https://b.com/article": html_b}

        def _make_resp(url):
            resp = MagicMock()
            resp.status = 200
            resp.text = AsyncMock(return_value=pages.get(url, ""))
            resp.__aenter__ = AsyncMock(return_value=resp)
            resp.__aexit__ = AsyncMock(return_value=False)
            return resp

        session = MagicMock()
        session.get = MagicMock(side_effect=lambda url, **kw: _make_resp(url))

        cache = _fresh_cache()

        with patch("enrichment.thumbnails.aiohttp.ClientSession") as MockSession:
            MockSession.return_value.__aenter__ = AsyncMock(return_value=session)
            MockSession.return_value.__aexit__ = AsyncMock(return_value=False)

            result = asyncio.run(
                extract_thumbnails_batch(
                    ["https://a.com/article", "https://b.com/article"],
                    concurrency=2,
                    _cache=cache,
                )
            )

        assert set(result.keys()) == {"https://a.com/article", "https://b.com/article"}

    def test_concurrency_limit_respected(self):
        """Semaphore should limit concurrent requests; just verify no errors."""
        urls = [f"https://example.com/article/{i}" for i in range(5)]
        cache = _fresh_cache()

        # Pre-populate cache so no real network calls happen
        for url in urls:
            _cache_set(url, f"https://cdn.example.com/{url[-1]}.jpg", cache)

        result = asyncio.run(
            extract_thumbnails_batch(urls, concurrency=2, _cache=cache)
        )
        assert len(result) == 5
        for url in urls:
            assert url in result

    def test_dict_keys_match_input_urls(self):
        urls = ["https://example.com/a", "https://example.com/b"]
        cache = _fresh_cache()
        for url in urls:
            _cache_set(url, None, cache)

        result = asyncio.run(
            extract_thumbnails_batch(urls, _cache=cache)
        )
        assert set(result.keys()) == set(urls)

    def test_partial_failures_dont_abort_batch(self):
        """If one URL fails, others should still be processed."""
        urls = ["https://good.com/article", "https://bad.com/article"]
        cache = _fresh_cache()
        _cache_set("https://good.com/article", "https://good.com/img.jpg", cache)
        _cache_set("https://bad.com/article", None, cache)

        result = asyncio.run(
            extract_thumbnails_batch(urls, _cache=cache)
        )
        assert result["https://good.com/article"] == "https://good.com/img.jpg"
        assert result["https://bad.com/article"] is None


# ---------------------------------------------------------------------------
# Fallback chain integration
# ---------------------------------------------------------------------------

class TestFallbackChain:
    """Verify the full OG → Twitter → img → favicon chain end-to-end."""

    def _run(self, html: str, url: str = "https://example.com/article") -> Optional[str]:
        session = _make_mock_session(html)
        cache = _fresh_cache()
        return asyncio.run(
            extract_thumbnail(url, _cache=cache, _session=session)
        )

    def test_og_wins_over_twitter_and_img(self):
        html = (
            '<meta property="og:image" content="https://example.com/og.jpg">'
            '<meta name="twitter:image" content="https://example.com/tw.jpg">'
            "<img src='https://example.com/img.jpg'>"
        )
        assert self._run(html) == "https://example.com/og.jpg"

    def test_twitter_wins_over_img_when_no_og(self):
        html = (
            '<meta name="twitter:image" content="https://example.com/tw.jpg">'
            "<img src='https://example.com/img.jpg'>"
        )
        assert self._run(html) == "https://example.com/tw.jpg"

    def test_img_wins_over_favicon_when_no_meta(self):
        html = "<body><img src='https://example.com/img.jpg'></body>"
        assert self._run(html) == "https://example.com/img.jpg"

    def test_favicon_when_nothing_else(self):
        html = "<html><body><p>No images.</p></body></html>"
        assert self._run(html) == "https://example.com/favicon.ico"


# Allow running directly
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
