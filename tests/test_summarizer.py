"""Unit tests for the article summarization service (Task 5).

All tests inject a mock summarizer so the heavy DistilBART model is
never loaded during CI.
"""

import asyncio
import sqlite3
from unittest.mock import MagicMock

import pytest

from pipeline.summarizer import (
    _MISS,
    _cache_get,
    _cache_set,
    _get_cache_connection,
    _is_too_short,
    _make_cache_key,
    _truncate_text,
    _MAX_INPUT_CHARS,
    set_summarizer,
    summarize,
    summarize_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_cache() -> sqlite3.Connection:
    """Return a clean in-memory cache DB for each test."""
    return _get_cache_connection(":memory:")


def _make_mock_summarizer(summary_text: str = "This is a mock summary.") -> MagicMock:
    """Return a callable mock that mimics the transformers summarization pipeline."""
    mock = MagicMock(return_value=[{"summary_text": summary_text}])
    return mock


# Long text (>30 words) used in tests that actually call the model mock
_LONG_TEXT = (
    "Scientists at a leading research institute have announced a major breakthrough "
    "in renewable energy technology. The new solar panel design achieves efficiency "
    "rates of over 40 percent, nearly double the current commercial standard. This "
    "development could dramatically reduce the cost of clean energy and accelerate "
    "the global transition away from fossil fuels, experts say."
)


# ---------------------------------------------------------------------------
# _truncate_text
# ---------------------------------------------------------------------------


class TestTruncateText:
    def test_short_text_unchanged(self):
        text = "Short text."
        assert _truncate_text(text) == text

    def test_long_text_truncated_to_max_chars(self):
        text = "a" * (_MAX_INPUT_CHARS + 100)
        result = _truncate_text(text)
        assert len(result) == _MAX_INPUT_CHARS

    def test_exactly_at_limit_unchanged(self):
        text = "b" * _MAX_INPUT_CHARS
        assert _truncate_text(text) == text


# ---------------------------------------------------------------------------
# _is_too_short
# ---------------------------------------------------------------------------


class TestIsTooShort:
    def test_empty_string_is_too_short(self):
        assert _is_too_short("", 30) is True

    def test_one_word_is_too_short_for_min30(self):
        assert _is_too_short("Hello", 30) is True

    def test_long_text_not_too_short(self):
        text = " ".join(["word"] * 50)
        assert _is_too_short(text, 30) is False

    def test_exactly_at_min_length_not_too_short(self):
        text = " ".join(["word"] * 30)
        assert _is_too_short(text, 30) is False


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    def test_same_inputs_same_key(self):
        k1 = _make_cache_key("text", 130, 30)
        k2 = _make_cache_key("text", 130, 30)
        assert k1 == k2

    def test_different_text_different_key(self):
        assert _make_cache_key("text A", 130, 30) != _make_cache_key("text B", 130, 30)

    def test_different_max_length_different_key(self):
        assert _make_cache_key("text", 130, 30) != _make_cache_key("text", 100, 30)

    def test_different_min_length_different_key(self):
        assert _make_cache_key("text", 130, 30) != _make_cache_key("text", 130, 20)

    def test_returns_64_char_hex_string(self):
        key = _make_cache_key("hello", 130, 30)
        assert isinstance(key, str)
        assert len(key) == 64


class TestSummaryCache:
    def test_miss_before_set(self):
        conn = _fresh_cache()
        key = _make_cache_key("some text", 130, 30)
        assert _cache_get(key, conn) is _MISS

    def test_hit_after_set(self):
        conn = _fresh_cache()
        key = _make_cache_key("some text", 130, 30)
        _cache_set(key, "A great summary.", conn)
        assert _cache_get(key, conn) == "A great summary."

    def test_overwrite_existing_entry(self):
        conn = _fresh_cache()
        key = _make_cache_key("t", 130, 30)
        _cache_set(key, "Old summary.", conn)
        _cache_set(key, "New summary.", conn)
        assert _cache_get(key, conn) == "New summary."

    def test_different_keys_independent(self):
        conn = _fresh_cache()
        k1 = _make_cache_key("text A", 130, 30)
        k2 = _make_cache_key("text B", 130, 30)
        _cache_set(k1, "Summary A.", conn)
        assert _cache_get(k2, conn) is _MISS


# ---------------------------------------------------------------------------
# summarize()
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_empty_text_raises_value_error(self):
        cache = _fresh_cache()
        with pytest.raises(ValueError, match="text must not be empty"):
            summarize("", _cache=cache)

    def test_whitespace_only_raises_value_error(self):
        cache = _fresh_cache()
        with pytest.raises(ValueError, match="text must not be empty"):
            summarize("   ", _cache=cache)

    def test_returns_string(self):
        mock = _make_mock_summarizer("Solar energy is booming.")
        cache = _fresh_cache()
        result = summarize(_LONG_TEXT, _cache=cache, _summarizer_override=mock)
        assert isinstance(result, str)

    def test_model_called_with_truncated_text(self):
        mock = _make_mock_summarizer("Summary.")
        cache = _fresh_cache()
        summarize(_LONG_TEXT, _cache=cache, _summarizer_override=mock)
        call_args = mock.call_args
        passed_text = call_args[0][0]
        assert len(passed_text) <= _MAX_INPUT_CHARS

    def test_model_receives_max_and_min_length(self):
        mock = _make_mock_summarizer("Summary.")
        cache = _fresh_cache()
        summarize(_LONG_TEXT, max_length=80, min_length=20, _cache=cache, _summarizer_override=mock)
        _, kwargs = mock.call_args
        assert kwargs["max_length"] == 80
        assert kwargs["min_length"] == 20

    def test_result_cached_on_first_call(self):
        mock = _make_mock_summarizer("Cached summary.")
        cache = _fresh_cache()
        summarize(_LONG_TEXT, _cache=cache, _summarizer_override=mock)
        summarize(_LONG_TEXT, _cache=cache, _summarizer_override=mock)
        assert mock.call_count == 1  # second call hits cache

    def test_cache_hit_skips_model(self):
        mock = _make_mock_summarizer()
        cache = _fresh_cache()
        key = _make_cache_key(_LONG_TEXT.strip(), 130, 30)
        _cache_set(key, "Pre-cached summary.", cache)
        result = summarize(_LONG_TEXT, _cache=cache, _summarizer_override=mock)
        assert result == "Pre-cached summary."
        mock.assert_not_called()

    def test_short_text_returned_as_is(self):
        """Text shorter than min_length tokens should bypass the model."""
        mock = _make_mock_summarizer()
        cache = _fresh_cache()
        short = "Brief."
        result = summarize(short, _cache=cache, _summarizer_override=mock)
        assert result == short
        mock.assert_not_called()

    def test_short_text_is_cached(self):
        mock = _make_mock_summarizer()
        cache = _fresh_cache()
        short = "Short article."
        summarize(short, _cache=cache, _summarizer_override=mock)
        summarize(short, _cache=cache, _summarizer_override=mock)
        mock.assert_not_called()  # never called; cached from first attempt

    def test_summary_text_is_stripped(self):
        mock = _make_mock_summarizer("  Leading and trailing spaces.  ")
        cache = _fresh_cache()
        result = summarize(_LONG_TEXT, _cache=cache, _summarizer_override=mock)
        assert result == "Leading and trailing spaces."

    def test_do_sample_false(self):
        """Greedy decoding should be used for deterministic output."""
        mock = _make_mock_summarizer("Summary.")
        cache = _fresh_cache()
        summarize(_LONG_TEXT, _cache=cache, _summarizer_override=mock)
        _, kwargs = mock.call_args
        assert kwargs.get("do_sample") is False

    def test_set_summarizer_overrides_global(self):
        mock = _make_mock_summarizer("Global override summary.")
        set_summarizer(mock)
        cache = _fresh_cache()
        result = summarize(_LONG_TEXT, _cache=cache)
        assert result == "Global override summary."
        mock.assert_called_once()
        # Reset global so other tests are unaffected
        set_summarizer(None)

    def test_set_summarizer_none_resets(self):
        import pipeline.summarizer as mod

        set_summarizer(None)
        assert mod._summarizer is None

    def test_very_long_text_truncated_before_model_call(self):
        mock = _make_mock_summarizer("Summary of long text.")
        cache = _fresh_cache()
        very_long = "word " * 2000  # well above _MAX_INPUT_CHARS
        summarize(very_long, _cache=cache, _summarizer_override=mock)
        passed_text = mock.call_args[0][0]
        assert len(passed_text) <= _MAX_INPUT_CHARS

    def test_different_max_length_generates_separate_cache_entries(self):
        mock1 = _make_mock_summarizer("Short summary.")
        mock2 = _make_mock_summarizer("Longer, more detailed summary.")
        cache = _fresh_cache()
        r1 = summarize(_LONG_TEXT, max_length=50, _cache=cache, _summarizer_override=mock1)
        r2 = summarize(_LONG_TEXT, max_length=150, _cache=cache, _summarizer_override=mock2)
        assert r1 == "Short summary."
        assert r2 == "Longer, more detailed summary."
        assert mock1.call_count == 1
        assert mock2.call_count == 1


# ---------------------------------------------------------------------------
# summarize_batch()
# ---------------------------------------------------------------------------


class TestSummarizeBatch:
    def test_empty_list_returns_empty_list(self):
        result = asyncio.get_event_loop().run_until_complete(summarize_batch([]))
        assert result == []

    def test_returns_same_number_of_summaries(self):
        mock = _make_mock_summarizer("Summary.")
        cache = _fresh_cache()
        texts = [_LONG_TEXT, _LONG_TEXT + " extra.", _LONG_TEXT + " more."]
        result = asyncio.get_event_loop().run_until_complete(
            summarize_batch(texts, _cache=cache, _summarizer_override=mock)
        )
        assert len(result) == 3

    def test_each_result_is_string(self):
        mock = _make_mock_summarizer("A summary.")
        cache = _fresh_cache()
        result = asyncio.get_event_loop().run_until_complete(
            summarize_batch([_LONG_TEXT], _cache=cache, _summarizer_override=mock)
        )
        assert all(isinstance(s, str) for s in result)

    def test_order_preserved(self):
        texts = [_LONG_TEXT + str(i) for i in range(3)]
        summaries = [f"Summary {i}" for i in range(3)]
        call_count = [0]

        def fake_summarizer(text, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return [{"summary_text": summaries[idx]}]

        mock = MagicMock(side_effect=fake_summarizer)
        cache = _fresh_cache()
        result = asyncio.get_event_loop().run_until_complete(
            summarize_batch(texts, _cache=cache, _summarizer_override=mock)
        )
        assert result == summaries

    def test_empty_text_in_batch_returns_empty_string(self):
        mock = _make_mock_summarizer("Summary.")
        cache = _fresh_cache()
        texts = [_LONG_TEXT, ""]
        result = asyncio.get_event_loop().run_until_complete(
            summarize_batch(texts, _cache=cache, _summarizer_override=mock)
        )
        assert result[1] == ""

    def test_cached_texts_not_resummarized(self):
        mock = _make_mock_summarizer("Cached.")
        cache = _fresh_cache()
        texts = [_LONG_TEXT]
        asyncio.get_event_loop().run_until_complete(summarize_batch(texts, _cache=cache, _summarizer_override=mock))
        asyncio.get_event_loop().run_until_complete(summarize_batch(texts, _cache=cache, _summarizer_override=mock))
        assert mock.call_count == 1  # second batch hits cache

    def test_max_length_forwarded(self):
        mock = _make_mock_summarizer("Summary.")
        cache = _fresh_cache()
        asyncio.get_event_loop().run_until_complete(
            summarize_batch(
                [_LONG_TEXT], max_length=60, _cache=cache, _summarizer_override=mock
            )
        )
        _, kwargs = mock.call_args
        assert kwargs["max_length"] == 60

    def test_single_text_batch(self):
        mock = _make_mock_summarizer("Solo summary.")
        cache = _fresh_cache()
        result = asyncio.get_event_loop().run_until_complete(
            summarize_batch([_LONG_TEXT], _cache=cache, _summarizer_override=mock)
        )
        assert result == ["Solo summary."]


# ---------------------------------------------------------------------------
# Integration: BBC-style and Reddit-style article texts
# ---------------------------------------------------------------------------


class TestRealWorldArticles:
    """Smoke-tests with representative article text patterns."""

    BBC_ARTICLE = (
        "The United Kingdom's renewable energy capacity has reached a new record, "
        "with wind and solar now providing more than half of the country's electricity "
        "on some days. Government officials celebrated the milestone as a sign that "
        "the transition to clean energy is accelerating faster than expected. The "
        "energy minister said the country is on track to meet its 2035 targets for "
        "decarbonising the electricity grid entirely."
    )

    REDDIT_TITLE = "Local community builds free library for neighbourhood kids [uplifting]"

    def test_bbc_style_article_returns_summary(self):
        mock = _make_mock_summarizer("UK renewable energy hits new record milestone.")
        cache = _fresh_cache()
        result = summarize(self.BBC_ARTICLE, _cache=cache, _summarizer_override=mock)
        assert len(result) > 0

    def test_reddit_title_too_short_returned_as_is(self):
        mock = _make_mock_summarizer()
        cache = _fresh_cache()
        result = summarize(self.REDDIT_TITLE, _cache=cache, _summarizer_override=mock)
        assert result == self.REDDIT_TITLE
        mock.assert_not_called()

    def test_combined_title_and_body(self):
        mock = _make_mock_summarizer("Community builds library for local kids.")
        cache = _fresh_cache()
        combined = self.REDDIT_TITLE + ". " + self.BBC_ARTICLE
        result = summarize(combined, _cache=cache, _summarizer_override=mock)
        assert isinstance(result, str) and len(result) > 0


# Allow running directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
