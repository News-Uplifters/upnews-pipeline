"""Orchestrate crawling across all configured sources using pluggable adapters."""

import logging
from .fetch_sources import load_sources
from .sources import get_source_adapter

logger = logging.getLogger(__name__)


def crawl_all_sources(limit_per_source: int = 50) -> list:
    """Crawl all active sources and return aggregated articles.

    Each source is instantiated via the adapter factory based on its
    ``adapter`` field in ``config/sources.yaml`` (defaults to ``rss``).

    Args:
        limit_per_source: Max articles to fetch per source

    Returns:
        List of article dicts with ``source_id`` and ``threshold`` added
    """
    sources_df = load_sources()
    all_articles = []

    for _, row in sources_df.iterrows():
        config = row.to_dict()
        source_id = config.get("source_id", config.get("name"))

        if not config.get("rss_url"):
            logger.warning(f"Skipping source {source_id}: no rss_url")
            continue

        try:
            adapter = get_source_adapter(config)
            logger.info(f"Crawling {source_id} via {adapter.__class__.__name__}: {adapter.rss_url}")

            articles = adapter.fetch(limit=limit_per_source)

            for article in articles:
                article["source_id"] = source_id
                article["threshold"] = config.get("threshold", 0.75)

            all_articles.extend(articles)
            logger.info(f"  → {len(articles)} articles fetched from {source_id}")

        except Exception as e:
            logger.error(f"Error crawling {source_id}: {e}")
            continue

    logger.info(f"Crawl complete: {len(all_articles)} total articles from {len(sources_df)} sources")
    return all_articles
