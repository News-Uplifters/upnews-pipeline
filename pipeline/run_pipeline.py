"""Main pipeline orchestrator."""

import logging
from datetime import datetime, timezone

from crawler.crawl_all_sources import crawl_all_sources
from classifier.classify_headlines import filter_positive_news, load_model

logger = logging.getLogger(__name__)

def run_pipeline(limit_per_source=50, classification_threshold=0.75, db_path="data/articles.db"):
    """Run the complete pipeline: fetch → classify → enrich → store.

    Args:
        limit_per_source: Max articles to fetch per RSS source
        classification_threshold: Confidence threshold for positive classification
        db_path: Path to the SQLite database file

    Returns:
        Dict with pipeline metrics: articles_fetched, articles_classified, etc.
    """
    crawl_start = datetime.now(timezone.utc)

    logger.info("="*60)
    logger.info("Starting upnews pipeline")
    logger.info("="*60)

    # Initialise DB
    from pipeline.database import init_db  # noqa: import-outside-toplevel
    db = init_db(db_path)

    # Step 1: Crawl all sources
    logger.info("Step 1: Crawling all RSS sources...")
    articles = crawl_all_sources(limit_per_source=limit_per_source)
    logger.info(f"  → Fetched {len(articles)} articles")

    if not articles:
        logger.warning("No articles fetched. Pipeline complete.")
        return {"articles_fetched": 0, "articles_classified": 0}

    # Step 2: Deduplicate against DB (TASK 7)
    logger.info("Step 2: Deduplicating articles against DB...")
    from pipeline.deduplication import deduplicate_articles  # noqa: import-outside-toplevel
    articles_before_dedup = len(articles)
    articles = deduplicate_articles(articles, db=db)
    articles_deduplicated = articles_before_dedup - len(articles)
    logger.info(
        "  → %d duplicates skipped, %d new articles to process",
        articles_deduplicated,
        len(articles),
    )

    if not articles:
        logger.info("All fetched articles already in DB. Pipeline complete.")
        db.close()
        return {
            "articles_fetched": articles_before_dedup,
            "articles_deduplicated": articles_deduplicated,
            "articles_classified": 0,
        }

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

    # Step 4: Enrich articles with categories (TASK 4)
    logger.info("Step 3: Categorizing articles...")
    try:
        from enrichment.categorizer import categorize_batch  # noqa: import-outside-toplevel
        article_dicts = classified_df.to_dict(orient="records")
        categorized = categorize_batch(article_dicts)
        import pandas as pd  # noqa: import-outside-toplevel
        categorized_df = pd.DataFrame(categorized)
        logger.info(
            "  → Categories assigned: %s",
            categorized_df["category"].value_counts().to_dict() if "category" in categorized_df.columns else {},
        )
    except Exception as e:
        logger.error("Categorization failed: %s", e)
        categorized_df = classified_df

    # TODO: Step 5: Thumbnail extraction (TASK 3 integration)

    # Step 6: Generate summaries (TASK 5)
    logger.info("Step 4: Generating article summaries...")
    try:
        from pipeline.summarizer import summarize  # noqa: import-outside-toplevel
        article_dicts = categorized_df.to_dict(orient="records")
        summarized = []
        for art in article_dicts:
            text = art.get("body") or art.get("title", "")
            try:
                art["summary"] = summarize(text)
            except (ValueError, Exception) as e:
                logger.warning("Summarization failed for article '%s': %s", art.get("title", "")[:60], e)
                art["summary"] = None
            summarized.append(art)
        import pandas as pd  # noqa: import-outside-toplevel
        summarized_df = pd.DataFrame(summarized)
        summaries_generated = summarized_df["summary"].notna().sum()
        logger.info("  → Summaries generated: %d/%d", summaries_generated, len(summarized_df))
    except Exception as e:
        logger.error("Summary generation failed: %s", e)
        summarized_df = categorized_df
        summaries_generated = 0

    # Step 7: Write to database (TASK 6)
    logger.info("Step 5: Writing articles to database...")
    articles_stored = 0
    errors_text = None
    try:
        article_dicts = summarized_df.to_dict(orient="records")
        articles_stored = db.upsert_articles(article_dicts)
        logger.info("  → Stored %d articles in %s", articles_stored, db_path)
    except Exception as e:
        logger.error("Database write failed: %s", e)
        errors_text = str(e)

    crawl_end = datetime.now(timezone.utc)

    # Record crawl metrics
    try:
        import pandas as pd  # noqa: import-outside-toplevel
        avg_score = None
        if "uplifting_score" in summarized_df.columns:
            avg_score = float(summarized_df["uplifting_score"].mean())
        db.record_crawl_metrics(
            crawl_start=crawl_start,
            crawl_end=crawl_end,
            articles_fetched=articles_before_dedup,
            articles_classified=len(classified_df),
            articles_stored=articles_stored,
            avg_classification_score=avg_score,
            errors=errors_text,
        )
    except Exception as e:
        logger.warning("Failed to record crawl metrics: %s", e)

    db.close()

    logger.info("="*60)
    logger.info("Pipeline complete")
    logger.info("="*60)

    return {
        "articles_fetched": articles_before_dedup,
        "articles_deduplicated": articles_deduplicated,
        "articles_classified": len(classified_df),
        "articles_categorized": len(categorized_df),
        "articles_summarized": int(summaries_generated),
        "articles_stored": articles_stored,
    }

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    metrics = run_pipeline()
    print(f"\nPipeline metrics: {metrics}")
