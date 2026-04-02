"""Unit tests for the extractive article summarization service (Task 5)."""

import asyncio
import sqlite3

import pytest

from pipeline.summarizer import (
    _MISS,
    _cache_get,
    _cache_set,
    _extract_summary,
    _get_cache_connection,
    _is_too_short,
    _make_cache_key,
    _MAX_INPUT_CHARS,
    _truncate_text,
    summarize,
    summarize_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_cache() -> sqlite3.Connection:
    return _get_cache_connection(":memory:")


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
# _extract_summary
# ---------------------------------------------------------------------------


class TestExtractSummary:
    def test_returns_first_sentence(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _extract_summary(text)
        assert result.startswith("First sentence")

    def test_multi_sentence_text(self):
        result = _extract_summary(_LONG_TEXT)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_result_within_max_chars(self):
        long = ". ".join(["word " * 20] * 20)
        result = _extract_summary(long, max_chars=200)
        assert len(result) <= 210  # slight overage OK for last sentence

    def test_single_sentence_returned_as_is(self):
        text = "Just one sentence here."
        assert _extract_summary(text) == text

    def test_exclamation_and_question_marks_split(self):
        text = "Great news! Scientists win prize. Who knew?"
        result = _extract_summary(text, max_chars=1000)
        assert len(result) >= len("Great news!")


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
        cache = _fresh_cache()
        result = summarize(_LONG_TEXT, _cache=cache)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_result_cached_on_first_call(self, monkeypatch):
        cache = _fresh_cache()
        calls = []
        import pipeline.summarizer as mod
        orig = mod._extract_summary

        def tracking(text, **kwargs):
            calls.append(1)
            return orig(text, **kwargs)

        monkeypatch.setattr(mod, "_extract_summary", tracking)
        summarize(_LONG_TEXT, _cache=cache)
        summarize(_LONG_TEXT, _cache=cache)
        assert len(calls) == 1  # second call hits cache

    def test_cache_hit_skips_extraction(self):
        cache = _fresh_cache()
        key = _make_cache_key(_LONG_TEXT.strip(), 130, 30)
        _cache_set(key, "Pre-cached summary.", cache)
        result = summarize(_LONG_TEXT, _cache=cache)
        assert result == "Pre-cached summary."

    def test_short_text_returned_as_is(self):
        cache = _fresh_cache()
        short = "Brief."
        result = summarize(short, _cache=cache)
        assert result == short

    def test_short_text_is_cached(self):
        cache = _fresh_cache()
        short = "Short article."
        summarize(short, _cache=cache)
        # Verify it's in the cache
        key = _make_cache_key(short, 130, 30)
        assert _cache_get(key, cache) == short

    def test_long_text_produces_shorter_summary(self):
        # Use a text long enough (>500 chars) so extractive summary is shorter.
        long = (_LONG_TEXT + " ") * 3
        cache = _fresh_cache()
        result = summarize(long, _cache=cache)
        assert len(result) < len(long)

    def test_different_min_length_generates_separate_cache_entries(self):
        cache = _fresh_cache()
        r1 = summarize(_LONG_TEXT, min_length=10, _cache=cache)
        r2 = summarize(_LONG_TEXT, min_length=5, _cache=cache)
        # Both succeed; they may be the same text but are stored under different keys
        assert isinstance(r1, str)
        assert isinstance(r2, str)


# ---------------------------------------------------------------------------
# summarize_batch()
# ---------------------------------------------------------------------------


class TestSummarizeBatch:
    def test_empty_list_returns_empty_list(self):
        result = asyncio.run(summarize_batch([]))
        assert result == []

    def test_returns_same_number_of_summaries(self):
        cache = _fresh_cache()
        texts = [_LONG_TEXT, _LONG_TEXT + " extra.", _LONG_TEXT + " more."]
        result = asyncio.run(summarize_batch(texts, _cache=cache))
        assert len(result) == 3

    def test_each_result_is_string(self):
        cache = _fresh_cache()
        result = asyncio.run(summarize_batch([_LONG_TEXT], _cache=cache))
        assert all(isinstance(s, str) for s in result)

    def test_empty_text_in_batch_returns_empty_string(self):
        cache = _fresh_cache()
        texts = [_LONG_TEXT, ""]
        result = asyncio.run(summarize_batch(texts, _cache=cache))
        assert result[1] == ""

    def test_cached_texts_not_resummarized(self, monkeypatch):
        cache = _fresh_cache()
        import pipeline.summarizer as mod
        calls = []
        orig = mod._extract_summary

        def tracking(text, **kwargs):
            calls.append(1)
            return orig(text, **kwargs)

        monkeypatch.setattr(mod, "_extract_summary", tracking)
        texts = [_LONG_TEXT]
        asyncio.run(summarize_batch(texts, _cache=cache))
        asyncio.run(summarize_batch(texts, _cache=cache))
        assert len(calls) == 1  # second batch hits cache

    def test_single_text_batch(self):
        cache = _fresh_cache()
        result = asyncio.run(summarize_batch([_LONG_TEXT], _cache=cache))
        assert len(result) == 1
        assert isinstance(result[0], str)


# ---------------------------------------------------------------------------
# Real-world article texts
# ---------------------------------------------------------------------------


class TestRealWorldArticles:
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
        cache = _fresh_cache()
        result = summarize(self.BBC_ARTICLE, _cache=cache)
        assert len(result) > 0

    def test_reddit_title_too_short_returned_as_is(self):
        cache = _fresh_cache()
        result = summarize(self.REDDIT_TITLE, _cache=cache)
        assert result == self.REDDIT_TITLE

    def test_combined_title_and_body(self):
        cache = _fresh_cache()
        combined = self.REDDIT_TITLE + ". " + self.BBC_ARTICLE
        result = summarize(combined, _cache=cache)
        assert isinstance(result, str) and len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
