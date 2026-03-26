"""Main pipeline orchestrator."""

import logging
from crawler.crawl_all_sources import crawl_all_sources
from classifier.classify_headlines import filter_positive_news, load_model

logger = logging.getLogger(__name__)

def run_pipeline(limit_per_source=50, classification_threshold=0.75):
    """Run the complete pipeline: fetch → classify → enrich → store.

    Args:
        limit_per_source: Max articles to fetch per RSS source
        classification_threshold: Confidence threshold for positive classification

    Returns:
        Dict with pipeline metrics: articles_fetched, articles_classified, etc.
    """
    logger.info("="*60)
    logger.info("Starting upnews pipeline")
    logger.info("="*60)

    # Step 1: Crawl all sources
    logger.info("Step 1: Crawling all RSS sources...")
    articles = crawl_all_sources(limit_per_source=limit_per_source)
    logger.info(f"  → Fetched {len(articles)} articles")

    if not articles:
        logger.warning("No articles fetched. Pipeline complete.")
        return {"articles_fetched": 0, "articles_classified": 0}

    # TODO: Step 2: Deduplicate against DB (TASK 7)
    # from pipeline.deduplication import deduplicate_articles
    # articles = deduplicate_articles(articles)

    # Step 3: Classify articles
    logger.info("Step 2: Classifying articles...")
    try:
        model = load_model()
        # Convert to DataFrame for processing
        import pandas as pd
        df = pd.DataFrame(articles)
        classified_df = filter_positive_news(df, model, threshold=classification_threshold)
        logger.info(f"  → Classified {len(classified_df)}/{len(df)} articles as uplifting")
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return {"articles_fetched": len(articles), "articles_classified": 0}

    # TODO: Step 4: Enrich articles (TASK 3, 4, 5)
    # from enrichment.thumbnails import extract_thumbnails_batch
    # from enrichment.categorizer import categorize_batch
    # from pipeline.summarizer import summarize_batch
    # logger.info("Step 3: Enriching articles...")
    # classified_df['thumbnail_url'] = ...
    # classified_df['category'] = ...
    # classified_df['summary'] = ...

    # TODO: Step 5: Write to database (TASK 6)
    # from pipeline.database import write_articles
    # write_articles(classified_df)

    logger.info("="*60)
    logger.info("Pipeline complete")
    logger.info("="*60)

    return {
        "articles_fetched": len(articles),
        "articles_classified": len(classified_df),
    }

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    metrics = run_pipeline()
    print(f"\nPipeline metrics: {metrics}")
