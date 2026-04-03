"""Orchestrate crawling across all configured sources using pluggable adapters."""

import logging

from .fetch_sources import load_sources
from .sources import get_source_adapter

logger = logging.getLogger(__name__)


def crawl_all_sources(limit_per_source: int = 50, max_articles_total: int | None = None) -> list:
    """Crawl all active sources and return aggregated articles.

    Each source is instantiated via the adapter factory based on its
    ``adapter`` field in ``config/sources.yaml`` (defaults to ``rss``).

    Args:
        limit_per_source: Max articles to fetch per source
        max_articles_total: Optional global cap across all sources

    Returns:
        List of article dicts with ``source_id`` and ``threshold`` added
    """
    sources_df = load_sources()
    all_articles = []

    for _, row in sources_df.iterrows():
        if max_articles_total is not None and len(all_articles) >= max_articles_total:
            logger.info("Reached global article cap of %d; stopping crawl", max_articles_total)
            break

        config = row.to_dict()
        source_id = config.get("source_id", config.get("name"))

        if not config.get("rss_url"):
            logger.warning(f"Skipping source {source_id}: no rss_url")
            continue

        try:
            adapter = get_source_adapter(config)
            logger.info(f"Crawling {source_id} via {adapter.__class__.__name__}: {adapter.rss_url}")

            if max_articles_total is None:
                fetch_limit = limit_per_source
            else:
                remaining = max_articles_total - len(all_articles)
                fetch_limit = min(limit_per_source, max(0, remaining))
                if fetch_limit <= 0:
                    logger.info("Reached global article cap of %d; stopping crawl", max_articles_total)
                    break

            articles = adapter.fetch(limit=fetch_limit)

            if max_articles_total is not None:
                remaining = max_articles_total - len(all_articles)
                articles = articles[:remaining]

            for article in articles:
                article["source_id"] = source_id
                article["threshold"] = config.get("threshold", 0.75)

            all_articles.extend(articles)
            logger.info(f"  -> {len(articles)} articles fetched from {source_id}")

            if max_articles_total is not None and len(all_articles) >= max_articles_total:
                logger.info("Reached global article cap of %d; stopping crawl", max_articles_total)
                break

        except Exception as e:
            logger.error(f"Error crawling {source_id}: {e}")
            continue

    logger.info(f"Crawl complete: {len(all_articles)} total articles from {len(sources_df)} sources")
    return all_articles
