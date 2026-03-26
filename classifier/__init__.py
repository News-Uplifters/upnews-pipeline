"""Article classification package."""

from .classify_headlines import filter_positive_news, load_model

__all__ = ["filter_positive_news", "load_model"]
