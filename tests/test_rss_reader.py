"""Unit tests for RSS feed parsing (TASK 9 stub)."""

import pytest
from crawler.rss_reader import fetch_rss_headlines, clean_url, extract_reddit_external

def test_clean_url():
    """Test URL cleaning removes query parameters."""
    # TODO: Implement tests
    pass

def test_fetch_rss_headlines_with_mock():
    """Test RSS parsing with mock feed."""
    # TODO: Use mock RSS feed from tests/fixtures/sample_feed.xml
    pass

def test_fetch_rss_headlines_empty_feed():
    """Test handling of empty RSS feed."""
    # TODO: Test with empty feed
    pass

def test_fetch_rss_headlines_malformed():
    """Test handling of malformed XML."""
    # TODO: Test with invalid XML
    pass

def test_extract_reddit_external():
    """Test Reddit external link extraction."""
    # TODO: Mock Reddit page and test extraction
    pass

def test_reddit_gallery_skip():
    """Test that Reddit gallery links are skipped."""
    # TODO: Verify gallery links are filtered
    pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
