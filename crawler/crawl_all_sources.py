"""Orchestrate crawling across all configured RSS sources."""

import logging
from .rss_reader import fetch_rss_headlines
from .fetch_sources import load_sources

logger = logging.getLogger(__name__)

def crawl_all_sources(limit_per_source=50):
    """Crawl all active sources and return aggregated articles.

    Args:
        limit_per_source: Max articles per source

    Returns:
        List of article dicts with source_id added
    """
    sources_df = load_sources()
    all_articles = []

    for idx, source in sources_df.iterrows():
        source_id = source.get('source_id', source.get('name'))
        rss_url = source.get('rss_url')

        if not rss_url:
            logger.warning(f"Skipping source {source_id}: no rss_url")
            continue

        try:
            logger.info(f"Crawling {source_id}: {rss_url}")
            articles = fetch_rss_headlines(rss_url, limit=limit_per_source)

            # Add source_id to each article
            for article in articles:
                article['source_id'] = source_id
                article['threshold'] = source.get('threshold', 0.75)

            all_articles.extend(articles)
            logger.info(f"  → {len(articles)} articles fetched from {source_id}")
        except Exception as e:
            logger.error(f"Error crawling {source_id}: {e}")
            continue

    logger.info(f"Crawl complete: {len(all_articles)} total articles from {len(sources_df)} sources")
    return all_articles
