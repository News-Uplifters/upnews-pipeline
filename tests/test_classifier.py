"""Unit tests for classifier/classify_headlines.py (Task 9).

All ML model calls are mocked so no GPU/model download is required.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from classifier.classify_headlines import (
    STRICT_SOURCE_THRESHOLDS,
    UPLIFTING_HINTS,
    _has_uplifting_hint,
    filter_positive_news,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model(scores: list):
    """Create a mock SetFit model returning given positive-class probabilities.

    Args:
        scores: List of floats in [0, 1] representing P(uplifting) per article.
    """
    model = MagicMock()
    model.predict_proba.return_value = [[1.0 - s, s] for s in scores]
    return model


def _make_df(titles, source=None):
    """Build a minimal articles DataFrame."""
    data = {"title": titles}
    if source is not None:
        data["source"] = source if isinstance(source, list) else [source] * len(titles)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# _has_uplifting_hint tests
# ---------------------------------------------------------------------------


def test_has_uplifting_hint_with_matching_word():
    assert _has_uplifting_hint("Dog wins championship", UPLIFTING_HINTS) is True


def test_has_uplifting_hint_rescued():
    assert _has_uplifting_hint("Whale rescued from beach", UPLIFTING_HINTS) is True


def test_has_uplifting_hint_hope():
    assert _has_uplifting_hint("New hope for climate recovery", UPLIFTING_HINTS) is True


def test_has_uplifting_hint_no_match():
    assert _has_uplifting_hint("Market falls amid trade war", UPLIFTING_HINTS) is False


def test_has_uplifting_hint_case_insensitive():
    assert _has_uplifting_hint("Record HIGH temperatures SAVED wildlife", UPLIFTING_HINTS) is True


def test_has_uplifting_hint_word_boundary_prevents_false_positive():
    # "wins" should not match "windows" (word boundary check)
    assert _has_uplifting_hint("windows update released", UPLIFTING_HINTS) is False


def test_has_uplifting_hint_empty_string():
    assert _has_uplifting_hint("", UPLIFTING_HINTS) is False


def test_has_uplifting_hint_none_input():
    assert _has_uplifting_hint(None, UPLIFTING_HINTS) is False


def test_has_uplifting_hint_whitespace_only():
    assert _has_uplifting_hint("   ", UPLIFTING_HINTS) is False


# ---------------------------------------------------------------------------
# filter_positive_news tests
# ---------------------------------------------------------------------------


def test_filter_positive_news_empty_df():
    """Empty DataFrame returns empty DataFrame without error."""
    model = _make_mock_model([])
    df = pd.DataFrame({"title": []})
    result = filter_positive_news(df, model)
    assert result.empty


def test_filter_positive_news_all_above_threshold():
    """All articles with high scores are kept."""
    model = _make_mock_model([0.95, 0.90, 0.85])
    df = _make_df(["Dog saves drowning child", "Teen wins science fair", "Community helps elderly"])
    result = filter_positive_news(df, model, threshold=0.75)
    assert len(result) == 3


def test_filter_positive_news_all_below_threshold():
    """All articles with low scores are filtered out."""
    model = _make_mock_model([0.10, 0.20, 0.30])
    df = _make_df(["War breaks out", "Economy crashes", "Scandal rocks government"])
    result = filter_positive_news(df, model, threshold=0.75)
    assert result.empty


def test_filter_positive_news_mixed_scores():
    """Only articles above threshold survive filtering."""
    model = _make_mock_model([0.95, 0.40, 0.88])
    df = _make_df(["Dog saves child", "Disaster strikes", "Volunteer helps community"])
    result = filter_positive_news(df, model, threshold=0.75)
    assert len(result) == 2
    assert "Disaster strikes" not in result["title"].values


def test_filter_positive_news_adds_uplifting_score_column():
    """Result DataFrame contains an uplifting_score column."""
    model = _make_mock_model([0.80])
    df = _make_df(["Breakthrough in renewable energy"])
    result = filter_positive_news(df, model, threshold=0.75)
    assert "uplifting_score" in result.columns
    assert abs(result.iloc[0]["uplifting_score"] - 0.80) < 1e-6


def test_filter_positive_news_does_not_mutate_input():
    """Input DataFrame is not modified in place."""
    model = _make_mock_model([0.90])
    df = _make_df(["Happy news"])
    original_columns = list(df.columns)
    filter_positive_news(df, model, threshold=0.75)
    assert list(df.columns) == original_columns


def test_filter_positive_news_no_source_column_uses_default_threshold():
    """When there is no 'source' column, the default threshold is used."""
    # Score = 0.80, threshold = 0.85 → should be filtered out
    model = _make_mock_model([0.80])
    df = _make_df(["Positive but low score"])
    result = filter_positive_news(df, model, threshold=0.85)
    assert result.empty


def test_filter_positive_news_strict_source_threshold_applied():
    """Strict sources (e.g. BBCNews at 0.90) filter out scores below their threshold."""
    # Score = 0.85 which is above default 0.75 but below BBC's 0.90
    model = _make_mock_model([0.85])
    df = _make_df(["BBC article with moderate score"], source=["BBCNews"])
    result = filter_positive_news(df, model, threshold=0.75)
    assert result.empty


def test_filter_positive_news_strict_source_needs_uplifting_hint():
    """Strict source articles above threshold but without hint are removed."""
    # Score = 0.95 (above BBC threshold 0.90) but no UPLIFTING_HINTS keyword
    model = _make_mock_model([0.95])
    df = _make_df(["Geopolitical tensions rise"], source=["BBCNews"])
    result = filter_positive_news(df, model, threshold=0.75)
    assert result.empty


def test_filter_positive_news_strict_source_with_hint_passes():
    """Strict source articles above threshold WITH hint are kept."""
    # Score = 0.95, has "wins" hint → passes both checks
    model = _make_mock_model([0.95])
    df = _make_df(["Local charity wins prestigious award"], source=["BBCNews"])
    result = filter_positive_news(df, model, threshold=0.75)
    assert len(result) == 1


def test_filter_positive_news_non_strict_source_no_hint_required():
    """Non-strict source articles above threshold pass even without hint."""
    # Score = 0.80, no hint, source not in STRICT_SOURCE_THRESHOLDS
    model = _make_mock_model([0.80])
    df = _make_df(["New art exhibition opens downtown"], source=["LocalBlog"])
    result = filter_positive_news(df, model, threshold=0.75)
    assert len(result) == 1


def test_filter_positive_news_multiple_strict_sources():
    """Multiple strict sources each apply their own threshold correctly."""
    # APNews threshold = 0.93; both articles score 0.91
    model = _make_mock_model([0.91, 0.91])
    df = _make_df(
        ["AP: breakthrough in medicine", "AP: records broken"],
        source=["APNews", "APNews"],
    )
    result = filter_positive_news(df, model, threshold=0.75)
    # "breakthrough" contains no UPLIFTING_HINTS and score < 0.93 → both filtered
    assert result.empty


def test_strict_source_thresholds_dict_has_expected_sources():
    """STRICT_SOURCE_THRESHOLDS contains the expected high-stakes sources."""
    expected = {"BBCNews", "CBSNews", "APNews", "ReutersWorld", "NYTimesWorld",
                "NPRNews", "GuardianWorld", "AlJazeeraAll"}
    assert expected.issubset(STRICT_SOURCE_THRESHOLDS.keys())


def test_strict_source_thresholds_values_above_default():
    """Strict source thresholds should all be >= 0.75 (default)."""
    for source, threshold in STRICT_SOURCE_THRESHOLDS.items():
        assert threshold >= 0.75, f"{source} threshold {threshold} is below default"
