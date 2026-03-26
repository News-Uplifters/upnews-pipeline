"""Unit tests for the article categorization service (Task 4).

All tests mock the zero-shot classifier so the heavy
`facebook/bart-large-mnli` model is never loaded during CI.
"""

import sqlite3
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from enrichment.categorizer import (
    DEFAULT_CATEGORIES,
    _build_input_text,
    _cache_get,
    _cache_set,
    _get_cache_connection,
    _make_cache_key,
    _MISS,
    categorize_article,
    categorize_batch,
    set_classifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_cache() -> sqlite3.Connection:
    """Return a clean in-memory cache DB for each test."""
    return _get_cache_connection(":memory:")


def _make_mock_classifier(
    top_label: str = "Technology & Science",
    all_labels: List[str] = None,
    all_scores: List[float] = None,
) -> MagicMock:
    """Build a callable mock that mimics the transformers zero-shot pipeline output."""
    if all_labels is None:
        all_labels = [top_label] + [
            c for c in DEFAULT_CATEGORIES if c != top_label
        ]
    if all_scores is None:
        all_scores = [0.95] + [0.05] * (len(all_labels) - 1)

    output = {"labels": all_labels, "scores": all_scores}
    mock = MagicMock(return_value=output)
    return mock


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
            "category": "Technology & Science",
            "scores": {"Technology & Science": 0.9},
            "confidence": 0.9,
        }
        _cache_set(key, result, conn)
        cached = _cache_get(key, conn)
        assert cached["category"] == "Technology & Science"
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
# categorize_article
# ---------------------------------------------------------------------------


class TestCategorizeArticle:
    def test_returns_top_category(self):
        mock_clf = _make_mock_classifier("Technology & Science")
        cache = _fresh_cache()
        result = categorize_article(
            "Scientists invent new battery",
            _cache=cache,
            _classifier_override=mock_clf,
        )
        assert result["category"] == "Technology & Science"

    def test_returns_confidence_float(self):
        mock_clf = _make_mock_classifier()
        cache = _fresh_cache()
        result = categorize_article("title", _cache=cache, _classifier_override=mock_clf)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_scores_dict_contains_all_categories(self):
        mock_clf = _make_mock_classifier()
        cache = _fresh_cache()
        result = categorize_article(
            "title",
            categories=DEFAULT_CATEGORIES,
            _cache=cache,
            _classifier_override=mock_clf,
        )
        for cat in DEFAULT_CATEGORIES:
            assert cat in result["scores"]

    def test_custom_categories_used(self):
        custom = ["Sports", "Politics", "Entertainment"]
        labels = custom[:]
        scores = [0.7, 0.2, 0.1]
        mock_clf = MagicMock(return_value={"labels": labels, "scores": scores})
        cache = _fresh_cache()
        result = categorize_article(
            "Local team wins championship",
            categories=custom,
            _cache=cache,
            _classifier_override=mock_clf,
        )
        assert result["category"] == "Sports"
        assert set(result["scores"].keys()) == set(custom)

    def test_classifier_called_with_multi_label_true(self):
        mock_clf = _make_mock_classifier()
        cache = _fresh_cache()
        categorize_article("title", _cache=cache, _classifier_override=mock_clf)
        _, kwargs = mock_clf.call_args
        assert kwargs.get("multi_label") is True

    def test_body_included_in_classifier_input(self):
        mock_clf = _make_mock_classifier()
        cache = _fresh_cache()
        categorize_article(
            "title", body="some body text", _cache=cache, _classifier_override=mock_clf
        )
        positional_text = mock_clf.call_args[0][0]
        assert "some body text" in positional_text

    def test_result_cached_after_first_call(self):
        mock_clf = _make_mock_classifier()
        cache = _fresh_cache()
        categorize_article("title", _cache=cache, _classifier_override=mock_clf)
        categorize_article("title", _cache=cache, _classifier_override=mock_clf)
        # Classifier should be called exactly once (second call hits cache)
        assert mock_clf.call_count == 1

    def test_cache_hit_skips_classifier(self):
        mock_clf = _make_mock_classifier()
        cache = _fresh_cache()
        key = _make_cache_key("cached title", "", DEFAULT_CATEGORIES)
        _cache_set(
            key,
            {"category": "Human Interest", "scores": {}, "confidence": 0.88},
            cache,
        )
        result = categorize_article(
            "cached title", _cache=cache, _classifier_override=mock_clf
        )
        assert result["category"] == "Human Interest"
        mock_clf.assert_not_called()

    def test_empty_title_raises_value_error(self):
        cache = _fresh_cache()
        with pytest.raises(ValueError, match="title must not be empty"):
            categorize_article("", _cache=cache)

    def test_default_categories_used_when_none(self):
        mock_clf = _make_mock_classifier()
        cache = _fresh_cache()
        categorize_article("title", _cache=cache, _classifier_override=mock_clf)
        _, kwargs = mock_clf.call_args
        assert kwargs["candidate_labels"] == DEFAULT_CATEGORIES

    def test_multi_label_scores_are_independent(self):
        """With multi_label=True scores are not forced to sum to 1."""
        labels = DEFAULT_CATEGORIES
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        mock_clf = MagicMock(return_value={"labels": labels, "scores": scores})
        cache = _fresh_cache()
        result = categorize_article("title", _cache=cache, _classifier_override=mock_clf)
        # All scores should be preserved as given (not re-normalised)
        for cat, expected in zip(labels, scores):
            assert result["scores"][cat] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# categorize_batch
# ---------------------------------------------------------------------------


class TestCategorizeBatch:
    def test_empty_list_returns_empty_list(self):
        assert categorize_batch([]) == []

    def test_enriches_each_article(self):
        mock_clf = _make_mock_classifier("Human Interest")
        cache = _fresh_cache()
        articles = [
            {"title": "Article 1"},
            {"title": "Article 2"},
        ]
        result = categorize_batch(
            articles, _cache=cache, _classifier_override=mock_clf
        )
        assert len(result) == 2
        for art in result:
            assert "category" in art
            assert "category_scores" in art
            assert "category_confidence" in art

    def test_original_fields_preserved(self):
        mock_clf = _make_mock_classifier()
        cache = _fresh_cache()
        articles = [{"title": "Hello", "url": "https://example.com", "source_id": "BBC"}]
        result = categorize_batch(
            articles, _cache=cache, _classifier_override=mock_clf
        )
        assert result[0]["url"] == "https://example.com"
        assert result[0]["source_id"] == "BBC"

    def test_body_forwarded_to_categorize_article(self):
        mock_clf = _make_mock_classifier()
        cache = _fresh_cache()
        articles = [{"title": "Title", "body": "Some body text"}]
        categorize_batch(articles, _cache=cache, _classifier_override=mock_clf)
        positional_text = mock_clf.call_args[0][0]
        assert "Some body text" in positional_text

    def test_article_without_body_ok(self):
        mock_clf = _make_mock_classifier()
        cache = _fresh_cache()
        result = categorize_batch(
            [{"title": "No body here"}], _cache=cache, _classifier_override=mock_clf
        )
        assert result[0]["category"] is not None

    def test_article_with_empty_title_handled_gracefully(self):
        cache = _fresh_cache()
        articles = [{"title": ""}]
        result = categorize_batch(articles, _cache=cache)
        assert result[0]["category"] is None
        assert result[0]["category_confidence"] == 0.0

    def test_custom_categories_forwarded(self):
        custom = ["A", "B"]
        labels = ["A", "B"]
        scores = [0.6, 0.4]
        mock_clf = MagicMock(return_value={"labels": labels, "scores": scores})
        cache = _fresh_cache()
        result = categorize_batch(
            [{"title": "Test"}],
            categories=custom,
            _cache=cache,
            _classifier_override=mock_clf,
        )
        assert set(result[0]["category_scores"].keys()) == {"A", "B"}

    def test_input_dicts_not_mutated(self):
        mock_clf = _make_mock_classifier()
        cache = _fresh_cache()
        original = {"title": "Immutable article"}
        articles = [original]
        categorize_batch(articles, _cache=cache, _classifier_override=mock_clf)
        assert "category" not in original  # original dict unchanged

    def test_classifier_called_once_per_unique_article(self):
        mock_clf = _make_mock_classifier()
        cache = _fresh_cache()
        articles = [
            {"title": "Unique A"},
            {"title": "Unique B"},
        ]
        categorize_batch(articles, _cache=cache, _classifier_override=mock_clf)
        assert mock_clf.call_count == 2

    def test_cached_articles_not_reclassified(self):
        mock_clf = _make_mock_classifier()
        cache = _fresh_cache()
        articles = [{"title": "Repeated"}]
        # First batch
        categorize_batch(articles, _cache=cache, _classifier_override=mock_clf)
        # Second batch with same article
        categorize_batch(articles, _cache=cache, _classifier_override=mock_clf)
        assert mock_clf.call_count == 1  # cached on second run

    def test_returns_new_list_not_same_reference(self):
        mock_clf = _make_mock_classifier()
        cache = _fresh_cache()
        articles = [{"title": "Article"}]
        result = categorize_batch(articles, _cache=cache, _classifier_override=mock_clf)
        assert result is not articles

    def test_category_confidence_matches_top_score(self):
        labels = DEFAULT_CATEGORIES
        scores = [0.91] + [0.1] * (len(labels) - 1)
        mock_clf = MagicMock(return_value={"labels": labels, "scores": scores})
        cache = _fresh_cache()
        result = categorize_batch(
            [{"title": "T"}], _cache=cache, _classifier_override=mock_clf
        )
        assert result[0]["category_confidence"] == pytest.approx(0.91)


# ---------------------------------------------------------------------------
# set_classifier / global state
# ---------------------------------------------------------------------------


class TestSetClassifier:
    def test_set_classifier_overrides_global(self, monkeypatch):
        """set_classifier() replaces the module-level singleton."""
        import enrichment.categorizer as mod

        mock_clf = _make_mock_classifier("Culture & Arts")
        set_classifier(mock_clf)

        cache = _fresh_cache()
        result = categorize_article("Any title", _cache=cache)
        assert result["category"] == "Culture & Arts"
        mock_clf.assert_called_once()

        # Restore to None so other tests aren't affected
        set_classifier(None)

    def test_set_classifier_none_resets_to_lazy_load(self, monkeypatch):
        """After set_classifier(None), the next call attempts lazy model load."""
        import enrichment.categorizer as mod

        set_classifier(None)
        # The global should be None now (model will be lazy-loaded on demand)
        assert mod._classifier is None


# ---------------------------------------------------------------------------
# DEFAULT_CATEGORIES content
# ---------------------------------------------------------------------------


class TestDefaultCategories:
    def test_has_seven_categories(self):
        assert len(DEFAULT_CATEGORIES) == 7

    def test_expected_labels_present(self):
        expected = {
            "Health & Wellness",
            "Environment & Nature",
            "Community & Social Good",
            "Technology & Science",
            "Business & Economics",
            "Culture & Arts",
            "Human Interest",
        }
        assert set(DEFAULT_CATEGORIES) == expected

    def test_no_duplicates(self):
        assert len(DEFAULT_CATEGORIES) == len(set(DEFAULT_CATEGORIES))


# Allow running directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
