"""Unit tests for YAML-based source loading (Task 1)."""

import os
import tempfile
import textwrap

import pandas as pd
import pytest
import yaml

from crawler.fetch_sources import (
    SourceValidationError,
    load_sources,
    validate_sources,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_SOURCES = [
    {
        "name": "BBC News",
        "source_id": "BBCNews",
        "rss_url": "http://feeds.bbc.co.uk/news/world/rss.xml",
        "active": True,
        "category": "news",
        "threshold": 0.90,
    },
    {
        "name": "Reddit r/UpliftingNews",
        "source_id": "RedditUpliftingNews",
        "rss_url": "https://www.reddit.com/r/UpliftingNews/.rss",
        "active": True,
        "category": "reddit",
        "adapter": "reddit",
        "threshold": 0.75,
    },
    {
        "name": "Inactive Source",
        "source_id": "Inactive",
        "rss_url": "https://example.com/rss.xml",
        "active": False,
        "category": "news",
        "threshold": 0.85,
    },
]


def _write_yaml(tmp_path, sources):
    """Write a sources YAML file and return its path."""
    p = tmp_path / "sources.yaml"
    p.write_text(yaml.dump({"sources": sources}))
    return str(p)


# ---------------------------------------------------------------------------
# validate_sources() tests
# ---------------------------------------------------------------------------


class TestValidateSources:
    def test_valid_sources_returns_list(self):
        result = validate_sources(VALID_SOURCES)
        assert result is VALID_SOURCES

    def test_not_a_list_raises(self):
        with pytest.raises(SourceValidationError, match="must be a list"):
            validate_sources({"name": "bad"})

    def test_missing_required_name_raises(self):
        bad = [{"source_id": "X", "rss_url": "http://example.com"}]
        with pytest.raises(SourceValidationError, match="missing required fields"):
            validate_sources(bad)

    def test_missing_required_source_id_raises(self):
        bad = [{"name": "X", "rss_url": "http://example.com"}]
        with pytest.raises(SourceValidationError, match="missing required fields"):
            validate_sources(bad)

    def test_missing_required_rss_url_raises(self):
        bad = [{"name": "X", "source_id": "Y"}]
        with pytest.raises(SourceValidationError, match="missing required fields"):
            validate_sources(bad)

    def test_threshold_above_1_raises(self):
        bad = [{"name": "X", "source_id": "Y", "rss_url": "http://x.com", "threshold": 1.5}]
        with pytest.raises(SourceValidationError, match="threshold.*between 0 and 1"):
            validate_sources(bad)

    def test_threshold_below_0_raises(self):
        bad = [{"name": "X", "source_id": "Y", "rss_url": "http://x.com", "threshold": -0.1}]
        with pytest.raises(SourceValidationError, match="threshold.*between 0 and 1"):
            validate_sources(bad)

    def test_threshold_string_raises(self):
        bad = [{"name": "X", "source_id": "Y", "rss_url": "http://x.com", "threshold": "high"}]
        with pytest.raises(SourceValidationError, match="threshold.*must be a number"):
            validate_sources(bad)

    def test_threshold_at_boundaries_valid(self):
        sources = [
            {"name": "A", "source_id": "A", "rss_url": "http://a.com", "threshold": 0.0},
            {"name": "B", "source_id": "B", "rss_url": "http://b.com", "threshold": 1.0},
        ]
        assert validate_sources(sources) is sources

    def test_threshold_omitted_is_valid(self):
        sources = [{"name": "A", "source_id": "A", "rss_url": "http://a.com"}]
        assert validate_sources(sources) is sources

    def test_active_non_bool_raises(self):
        bad = [{"name": "X", "source_id": "Y", "rss_url": "http://x.com", "active": "yes"}]
        with pytest.raises(SourceValidationError, match="'active' must be a boolean"):
            validate_sources(bad)

    def test_invalid_adapter_raises(self):
        bad = [{"name": "X", "source_id": "Y", "rss_url": "http://x.com", "adapter": "twitter"}]
        with pytest.raises(SourceValidationError, match="adapter.*must be one of"):
            validate_sources(bad)

    def test_valid_adapter_reddit(self):
        sources = [{"name": "X", "source_id": "Y", "rss_url": "http://x.com", "adapter": "reddit"}]
        assert validate_sources(sources) is sources

    def test_valid_adapter_rss(self):
        sources = [{"name": "X", "source_id": "Y", "rss_url": "http://x.com", "adapter": "rss"}]
        assert validate_sources(sources) is sources

    def test_non_dict_source_raises(self):
        with pytest.raises(SourceValidationError, match="not a dict"):
            validate_sources(["not_a_dict"])

    def test_multiple_errors_reported_together(self):
        bad = [
            {"source_id": "Y", "rss_url": "http://x.com", "threshold": 2.0},  # missing name + bad threshold
        ]
        with pytest.raises(SourceValidationError) as exc_info:
            validate_sources(bad)
        msg = str(exc_info.value)
        assert "missing required fields" in msg
        assert "threshold" in msg

    def test_empty_list_is_valid(self):
        assert validate_sources([]) == []


# ---------------------------------------------------------------------------
# load_sources() — YAML path tests
# ---------------------------------------------------------------------------


class TestLoadSourcesFromYaml:
    def test_loads_active_sources_only(self, tmp_path):
        path = _write_yaml(tmp_path, VALID_SOURCES)
        df = load_sources(path=path)
        assert "Inactive Source" not in df["name"].values
        assert "BBC News" in df["name"].values
        assert "Reddit r/UpliftingNews" in df["name"].values

    def test_returns_dataframe(self, tmp_path):
        path = _write_yaml(tmp_path, VALID_SOURCES)
        df = load_sources(path=path)
        assert isinstance(df, pd.DataFrame)

    def test_preserves_thresholds(self, tmp_path):
        path = _write_yaml(tmp_path, VALID_SOURCES)
        df = load_sources(path=path)
        bbc_row = df[df["source_id"] == "BBCNews"].iloc[0]
        assert bbc_row["threshold"] == pytest.approx(0.90)

    def test_preserves_source_ids(self, tmp_path):
        path = _write_yaml(tmp_path, VALID_SOURCES)
        df = load_sources(path=path)
        assert set(df["source_id"]) == {"BBCNews", "RedditUpliftingNews"}

    def test_drops_sources_missing_rss_url(self, tmp_path):
        sources = [
            {"name": "Good", "source_id": "Good", "rss_url": "http://good.com", "active": True},
            {"name": "NoUrl", "source_id": "NoUrl", "rss_url": None, "active": True},
        ]
        path = _write_yaml(tmp_path, sources)
        df = load_sources(path=path)
        assert "NoUrl" not in df["source_id"].values
        assert "Good" in df["source_id"].values

    def test_preserves_category_and_adapter_fields(self, tmp_path):
        path = _write_yaml(tmp_path, VALID_SOURCES)
        df = load_sources(path=path)
        reddit_row = df[df["source_id"] == "RedditUpliftingNews"].iloc[0]
        assert reddit_row["category"] == "reddit"
        assert reddit_row["adapter"] == "reddit"

    def test_invalid_yaml_schema_raises(self, tmp_path):
        bad_sources = [{"source_id": "missing_name", "rss_url": "http://x.com"}]
        path = _write_yaml(tmp_path, bad_sources)
        with pytest.raises(SourceValidationError):
            load_sources(path=path)

    def test_yaml_with_no_sources_key_returns_empty(self, tmp_path):
        """YAML file without a 'sources' key should return an empty DataFrame."""
        p = tmp_path / "sources.yaml"
        p.write_text(yaml.dump({"other_key": []}))
        df = load_sources(path=str(p))
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_all_inactive_returns_empty_dataframe(self, tmp_path):
        sources = [
            {"name": "X", "source_id": "X", "rss_url": "http://x.com", "active": False},
        ]
        path = _write_yaml(tmp_path, sources)
        df = load_sources(path=path)
        assert df.empty

    def test_no_active_column_returns_all(self, tmp_path):
        """Sources without 'active' field should all be included."""
        sources = [
            {"name": "A", "source_id": "A", "rss_url": "http://a.com"},
            {"name": "B", "source_id": "B", "rss_url": "http://b.com"},
        ]
        path = _write_yaml(tmp_path, sources)
        df = load_sources(path=path)
        assert len(df) == 2

    def test_custom_yaml_path(self, tmp_path):
        """load_sources respects the 'path' argument."""
        path = _write_yaml(tmp_path, VALID_SOURCES[:1])
        df = load_sources(path=path)
        assert len(df) == 1
        assert df.iloc[0]["source_id"] == "BBCNews"


# ---------------------------------------------------------------------------
# load_sources() — missing file tests
# ---------------------------------------------------------------------------


class TestLoadSourcesMissingFile:
    def test_returns_empty_df_when_yaml_missing(self, tmp_path):
        df = load_sources(path=str(tmp_path / "missing.yaml"))
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ---------------------------------------------------------------------------
# Integration: load from the real sources.yaml
# ---------------------------------------------------------------------------


class TestRealSourcesYaml:
    YAML_PATH = "config/sources.yaml"

    def test_real_yaml_loads_without_error(self):
        if not os.path.exists(self.YAML_PATH):
            pytest.skip("config/sources.yaml not found")
        df = load_sources(path=self.YAML_PATH)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_real_yaml_has_required_columns(self):
        if not os.path.exists(self.YAML_PATH):
            pytest.skip("config/sources.yaml not found")
        df = load_sources(path=self.YAML_PATH)
        for col in ("name", "source_id", "rss_url", "threshold"):
            assert col in df.columns, f"Missing column: {col}"

    def test_real_yaml_thresholds_in_range(self):
        if not os.path.exists(self.YAML_PATH):
            pytest.skip("config/sources.yaml not found")
        df = load_sources(path=self.YAML_PATH)
        if "threshold" in df.columns:
            thresholds = df["threshold"].dropna()
            assert (thresholds >= 0.0).all()
            assert (thresholds <= 1.0).all()

    def test_real_yaml_no_duplicate_source_ids(self):
        if not os.path.exists(self.YAML_PATH):
            pytest.skip("config/sources.yaml not found")
        df = load_sources(path=self.YAML_PATH)
        dupes = df[df.duplicated("source_id", keep=False)]
        assert dupes.empty, f"Duplicate source_ids found: {dupes['source_id'].tolist()}"

    def test_real_yaml_all_active_have_rss_url(self):
        if not os.path.exists(self.YAML_PATH):
            pytest.skip("config/sources.yaml not found")
        df = load_sources(path=self.YAML_PATH)
        missing_url = df[df["rss_url"].isna()]
        assert missing_url.empty, f"Active sources missing rss_url: {missing_url['source_id'].tolist()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
