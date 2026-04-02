"""Pluggable source adapter pattern for news crawlers."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class BaseSource(ABC):
    """Abstract base class for news source adapters."""

    def __init__(self, config: dict):
        """Initialize source from config dict.

        Args:
            config: Source config from YAML (name, rss_url, threshold, etc.)
        """
        self.name = config.get("name")
        self.source_id = config.get("source_id")
        self.rss_url = config.get("rss_url")
        self.threshold = config.get("threshold", 0.75)
        self.active = config.get("active", True)

    @abstractmethod
    def fetch(self, limit: int = 50) -> list:
        """Fetch articles from this source.

        Args:
            limit: Maximum number of articles to return

        Returns:
            List of article dicts with keys:
            - title (str)
            - rss_link (str): link from RSS feed
            - original_url (str): canonical article URL
            - published (datetime or None)
            - source_id (str): source identifier
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(source_id={self.source_id!r}, url={self.rss_url!r})"


def get_source_adapter(config: dict) -> "BaseSource":
    """Factory: instantiate the right adapter based on config's 'adapter' field.

    Args:
        config: Source config dict from YAML

    Returns:
        Instantiated source adapter
    """
    from .rss import RSSSource
    from .reddit import RedditSource

    # pandas loads missing YAML fields as float NaN, not None.
    # NaN is truthy so `NaN or "rss"` returns NaN, causing .lower() to fail.
    adapter_val = config.get("adapter")
    if not adapter_val or (isinstance(adapter_val, float) and math.isnan(adapter_val)):
        adapter_val = "rss"
    adapter_type = str(adapter_val).lower()

    registry = {
        "rss": RSSSource,
        "reddit": RedditSource,
    }

    cls = registry.get(adapter_type, RSSSource)
    return cls(config)
