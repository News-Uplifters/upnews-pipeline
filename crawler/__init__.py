"""News article crawler package."""

from .rss_reader import fetch_rss_headlines
from .fetch_sources import load_sources
from .sources import get_source_adapter, BaseSource

__all__ = ["fetch_rss_headlines", "load_sources", "get_source_adapter", "BaseSource"]
