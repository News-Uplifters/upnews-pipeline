"""Unit tests for the keyword-based article categorization service (Task 4)."""

import sqlite3
from typing import List

import pytest

from enrichment.categorizer import (
    DEFAULT_CATEGORIES,
    _build_input_text,
    _cache_get,
    _cache_set,
    _get_cache_connection,
    _make_cache_key,
    _MISS,
    _score_text,
    categorize_article,
    categorize_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_cache() -> sqlite3.Connection:
    return _get_cache_connection(":memory:")


# ---------------------------------------------------------------------------
# _build_input_text
# ---------------------------------------------------------------------------


class TestBuildInputText:
    def test_title_only(self):
        assert _build_input_text("Hello world", "") == "Hello world"

    def test_title_and_body_combined(self):
        result = _build_input_text("Headline", "Body text")
        assert result == "Headline. Body text"

    def test_body_truncated_to_500_chars(self):
        long_body = "x" * 600
        result = _build_input_text("T", long_body)
        assert result == f"T. {'x' * 500}"

    def test_exact_500_char_body_not_truncated(self):
        body = "y" * 500
        result = _build_input_text("T", body)
        assert result == f"T. {'y' * 500}"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_same_inputs_same_key(self):
        k1 = _make_cache_key("title", "body", ["A", "B"])
        k2 = _make_cache_key("title", "body", ["A", "B"])
        assert k1 == k2

    def test_category_order_ignored(self):
        k1 = _make_cache_key("title", "", ["A", "B"])
        k2 = _make_cache_key("title", "", ["B", "A"])
        assert k1 == k2

    def test_different_title_different_key(self):
        assert _make_cache_key("T1", "", []) != _make_cache_key("T2", "", [])

    def test_different_body_different_key(self):
        assert _make_cache_key("T", "b1", []) != _make_cache_key("T", "b2", [])

    def test_returns_hex_string(self):
        key = _make_cache_key("title", "", DEFAULT_CATEGORIES)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex


class TestCache:
    def test_miss_before_set(self):
        conn = _fresh_cache()
        key = _make_cache_key("some title", "", DEFAULT_CATEGORIES)
        assert _cache_get(key, conn) is _MISS

    def test_hit_after_set(self):
        conn = _fresh_cache()
        key = _make_cache_key("some title", "", DEFAULT_CATEGORIES)
        result = {
            "category": "Science & Tech",
            "scores": {"Science & Tech": 0.9},
            "confidence": 0.9,
        }
        _cache_set(key, result, conn)
        cached = _cache_get(key, conn)
        assert cached["category"] == "Science & Tech"
        assert cached["confidence"] == pytest.approx(0.9)

    def test_scores_round_trip_as_dict(self):
        conn = _fresh_cache()
        key = _make_cache_key("t", "", DEFAULT_CATEGORIES)
        scores = {"A": 0.8, "B": 0.2}
        _cache_set(key, {"category": "A", "scores": scores, "confidence": 0.8}, conn)
        cached = _cache_get(key, conn)
        assert cached["scores"] == scores

    def test_overwrite_existing_entry(self):
        conn = _fresh_cache()
        key = _make_cache_key("t", "", DEFAULT_CATEGORIES)
        _cache_set(key, {"category": "Old", "scores": {}, "confidence": 0.1}, conn)
        _cache_set(key, {"category": "New", "scores": {}, "confidence": 0.9}, conn)
        assert _cache_get(key, conn)["category"] == "New"


# ---------------------------------------------------------------------------
# _score_text
# ---------------------------------------------------------------------------


class TestScoreText:
    def test_returns_score_for_every_category(self):
        scores = _score_text("Scientists discover new species", DEFAULT_CATEGORIES)
        assert set(scores.keys()) == set(DEFAULT_CATEGORIES)

    def test_scores_in_zero_one_range(self):
        scores = _score_text("local charity volunteers help community", DEFAULT_CATEGORIES)
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_no_keywords_gives_equal_scores(self):
        scores = _score_text("blah blah blah", DEFAULT_CATEGORIES)
        values = list(scores.values())
        assert all(v == pytest.approx(values[0]) for v in values)

    def test_strong_health_signal(self):
        scores = _score_text("hospital doctor medical cancer treatment", DEFAULT_CATEGORIES)
        assert scores["Health"] == pytest.approx(1.0)

    def test_strong_sports_signal(self):
        scores = _score_text("olympic champion athlete league tournament", DEFAULT_CATEGORIES)
        assert scores["Sports"] == pytest.approx(1.0)

    def test_custom_categories(self):
        custom = ["Alpha", "Beta"]
        scores = _score_text("some text", custom)
        assert set(scores.keys()) == {"Alpha", "Beta"}


# ---------------------------------------------------------------------------
# categorize_article
# ---------------------------------------------------------------------------


class TestCategorizeArticle:
    def test_returns_top_category(self):
        cache = _fresh_cache()
        result = categorize_article(
            "Scientists discover breakthrough in quantum computing research",
            _cache=cache,
        )
        assert result["category"] == "Science & Tech"

    def test_returns_confidence_float(self):
        cache = _fresh_cache()
        result = categorize_article("Local school wins education award", _cache=cache)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_scores_dict_contains_all_categories(self):
        cache = _fresh_cache()
        result = categorize_article(
            "title",
            categories=DEFAULT_CATEGORIES,
            _cache=cache,
        )
        for cat in DEFAULT_CATEGORIES:
            assert cat in result["scores"]

    def test_custom_categories_used(self):
        custom = ["Alpha", "Beta"]
        cache = _fresh_cache()
        result = categorize_article("some article title", categories=custom, _cache=cache)
        assert set(result["scores"].keys()) == set(custom)

    def test_result_cached_after_first_call(self, monkeypatch):
        cache = _fresh_cache()
        calls = []
        original_score = __import__("enrichment.categorizer", fromlist=["_score_text"])._score_text

        def counting_score(text, categories):
            calls.append(1)
            return original_score(text, categories)

        monkeypatch.setattr("enrichment.categorizer._score_text", counting_score)
        categorize_article("unique title abc123", _cache=cache)
        categorize_article("unique title abc123", _cache=cache)
        assert len(calls) == 1  # second call hits cache

    def test_cache_hit_skips_scoring(self):
        cache = _fresh_cache()
        key = _make_cache_key("cached title", "", DEFAULT_CATEGORIES)
        _cache_set(
            key,
            {"category": "Community", "scores": {}, "confidence": 0.88},
            cache,
        )
        result = categorize_article("cached title", _cache=cache)
        assert result["category"] == "Community"

    def test_empty_title_raises_value_error(self):
        cache = _fresh_cache()
        with pytest.raises(ValueError, match="title must not be empty"):
            categorize_article("", _cache=cache)

    def test_body_included_in_scoring(self):
        cache1, cache2 = _fresh_cache(), _fresh_cache()
        # Without sports keywords in body
        r1 = categorize_article("Breaking news today", _cache=cache1)
        # With sports keywords in body
        r2 = categorize_article(
            "Breaking news today",
            body="olympic champion athlete wins tournament league match",
            _cache=cache2,
        )
        assert r2["scores"]["Sports"] > r1["scores"]["Sports"]

    def test_health_article_categorized_correctly(self):
        cache = _fresh_cache()
        result = categorize_article(
            "Hospital opens new cancer treatment clinic for patients",
            _cache=cache,
        )
        assert result["category"] == "Health"

    def test_environment_article_categorized_correctly(self):
        cache = _fresh_cache()
        result = categorize_article(
            "Climate change drives new renewable solar energy conservation effort",
            _cache=cache,
        )
        assert result["category"] == "Environment"


# ---------------------------------------------------------------------------
# categorize_batch
# ---------------------------------------------------------------------------


class TestCategorizeBatch:
    def test_empty_list_returns_empty_list(self):
        assert categorize_batch([]) == []

    def test_enriches_each_article(self):
        cache = _fresh_cache()
        articles = [
            {"title": "Scientists announce research breakthrough"},
            {"title": "Local community volunteers fundraise for charity"},
        ]
        result = categorize_batch(articles, _cache=cache)
        assert len(result) == 2
        for art in result:
            assert "category" in art
            assert "category_scores" in art
            assert "category_confidence" in art

    def test_original_fields_preserved(self):
        cache = _fresh_cache()
        articles = [{"title": "Hello", "url": "https://example.com", "source_id": "BBC"}]
        result = categorize_batch(articles, _cache=cache)
        assert result[0]["url"] == "https://example.com"
        assert result[0]["source_id"] == "BBC"

    def test_article_without_body_ok(self):
        cache = _fresh_cache()
        result = categorize_batch([{"title": "No body here"}], _cache=cache)
        assert result[0]["category"] is not None

    def test_article_with_empty_title_handled_gracefully(self):
        cache = _fresh_cache()
        result = categorize_batch([{"title": ""}], _cache=cache)
        assert result[0]["category"] is None
        assert result[0]["category_confidence"] == 0.0

    def test_custom_categories_forwarded(self):
        custom = ["A", "B"]
        cache = _fresh_cache()
        result = categorize_batch(
            [{"title": "Test article"}],
            categories=custom,
            _cache=cache,
        )
        assert set(result[0]["category_scores"].keys()) == {"A", "B"}

    def test_input_dicts_not_mutated(self):
        cache = _fresh_cache()
        original = {"title": "Immutable article"}
        categorize_batch([original], _cache=cache)
        assert "category" not in original

    def test_cached_articles_not_rescored(self):
        cache = _fresh_cache()
        articles = [{"title": "Repeated article about science research"}]
        categorize_batch(articles, _cache=cache)
        # Pre-populate cache with different result for the same key
        key = _make_cache_key("Repeated article about science research", "", DEFAULT_CATEGORIES)
        _cache_set(key, {"category": "Sports", "scores": {}, "confidence": 0.5}, cache)
        result2 = categorize_batch(articles, _cache=cache)
        assert result2[0]["category"] == "Sports"  # came from cache

    def test_returns_new_list_not_same_reference(self):
        cache = _fresh_cache()
        articles = [{"title": "Article"}]
        result = categorize_batch(articles, _cache=cache)
        assert result is not articles


# ---------------------------------------------------------------------------
# DEFAULT_CATEGORIES content
# ---------------------------------------------------------------------------


class TestDefaultCategories:
    def test_has_seven_categories(self):
        assert len(DEFAULT_CATEGORIES) == 7

    def test_expected_labels_present(self):
        expected = {
            "Health",
            "Environment",
            "Community",
            "Science & Tech",
            "Education",
            "Sports",
            "Arts & Culture",
        }
        assert set(DEFAULT_CATEGORIES) == expected

    def test_no_duplicates(self):
        assert len(DEFAULT_CATEGORIES) == len(set(DEFAULT_CATEGORIES))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
