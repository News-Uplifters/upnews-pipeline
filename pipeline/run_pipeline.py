"""Main pipeline orchestrator."""

import logging
from datetime import datetime, timezone

from crawler.crawl_all_sources import crawl_all_sources
from classifier.classify_headlines import filter_positive_news, load_model
from pipeline.logging_config import CrawlMetrics, get_pipeline_logger, log_article_event

logger = get_pipeline_logger(__name__)


def run_pipeline(limit_per_source=50, classification_threshold=0.75, db_path="data/articles.db"):
    """Run the complete pipeline: fetch → classify → enrich → store.

    Args:
        limit_per_source: Max articles to fetch per RSS source
        classification_threshold: Confidence threshold for positive classification
        db_path: Path to the SQLite database file

    Returns:
        Dict with pipeline metrics: articles_fetched, articles_classified, etc.
    """
    metrics = CrawlMetrics.start()

    logger.info("pipeline_start", extra={"event": "pipeline_start", "crawl_id": metrics.crawl_id})
    logger.info("="*60)
    logger.info("Starting upnews pipeline  [%s]", metrics.crawl_id)
    logger.info("="*60)

    # Initialise DB
    from pipeline.database import init_db  # noqa: import-outside-toplevel
    db = init_db(db_path)

    # Step 1: Crawl all sources
    logger.info("Step 1: Crawling all RSS sources...")
    metrics.start_stage("crawl")
    articles = crawl_all_sources(limit_per_source=limit_per_source)
    crawl_ms = metrics.end_stage("crawl")
    metrics.articles_fetched = len(articles)
    logger.info(
        "crawl_complete",
        extra={
            "event": "crawl_complete",
            "crawl_id": metrics.crawl_id,
            "articles_fetched": metrics.articles_fetched,
            "duration_ms": round(crawl_ms, 1),
        },
    )
    logger.info("  → Fetched %d articles in %.0fms", metrics.articles_fetched, crawl_ms)

    if not articles:
        logger.warning("No articles fetched. Pipeline complete.")
        metrics.finish()
        _record_and_close(db, metrics)
        return {"articles_fetched": 0, "articles_classified": 0}

    # Normalise field names: crawler uses original_url/published; DB layer uses url/published_at
    for article in articles:
        if "url" not in article or not article["url"]:
            article["url"] = article.get("original_url") or article.get("rss_link", "")
        if "published_at" not in article or article["published_at"] is None:
            article["published_at"] = article.get("published")

    # Step 2: Deduplicate against DB
    logger.info("Step 2: Deduplicating articles against DB...")
    metrics.start_stage("dedup")
    from pipeline.deduplication import deduplicate_articles  # noqa: import-outside-toplevel
    articles_before_dedup = len(articles)
    articles = deduplicate_articles(articles, db=db)
    dedup_ms = metrics.end_stage("dedup")
    metrics.articles_skipped = articles_before_dedup - len(articles)
    metrics.articles_new = len(articles)
    logger.info(
        "dedup_complete",
        extra={
            "event": "dedup_complete",
            "crawl_id": metrics.crawl_id,
            "articles_skipped": metrics.articles_skipped,
            "articles_new": metrics.articles_new,
            "duration_ms": round(dedup_ms, 1),
        },
    )
    logger.info(
        "  → %d duplicates skipped, %d new articles to process",
        metrics.articles_skipped,
        metrics.articles_new,
    )

    if not articles:
        logger.info("All fetched articles already in DB. Pipeline complete.")
        metrics.finish()
        _record_and_close(db, metrics)
        return {
            "articles_fetched": articles_before_dedup,
            "articles_deduplicated": metrics.articles_skipped,
            "articles_classified": 0,
        }

    # Step 3: Classify articles
    logger.info("Step 3: Classifying articles...")
    metrics.start_stage("classify")
    try:
        model = load_model()
        import pandas as pd
        df = pd.DataFrame(articles)
        classified_df = filter_positive_news(df, model, threshold=classification_threshold)
        classify_ms = metrics.end_stage("classify")
        metrics.articles_classified = len(classified_df)
        logger.info(
            "classify_complete",
            extra={
                "event": "classify_complete",
                "crawl_id": metrics.crawl_id,
                "articles_classified": metrics.articles_classified,
                "articles_total": len(df),
                "duration_ms": round(classify_ms, 1),
            },
        )
        logger.info("  → Classified %d/%d articles as uplifting", metrics.articles_classified, len(df))
    except Exception as e:
        metrics.end_stage("classify")
        metrics.record_error(f"classification_failed: {e}")
        logger.error(
            "classify_error",
            extra={"event": "classify_error", "crawl_id": metrics.crawl_id, "error": str(e)},
        )
        logger.error("Classification failed: %s", e)
        metrics.finish()
        _record_and_close(db, metrics)
        return {"articles_fetched": len(articles), "articles_classified": 0}

    # Step 4: Enrich articles with categories
    logger.info("Step 4: Categorizing articles...")
    metrics.start_stage("categorize")
    try:
        from enrichment.categorizer import categorize_batch  # noqa: import-outside-toplevel
        article_dicts = classified_df.to_dict(orient="records")
        categorized = categorize_batch(article_dicts)
        import pandas as pd  # noqa: import-outside-toplevel
        categorized_df = pd.DataFrame(categorized)
        cat_ms = metrics.end_stage("categorize")
        metrics.articles_categorized = len(categorized_df)
        logger.info(
            "categorize_complete",
            extra={
                "event": "categorize_complete",
                "crawl_id": metrics.crawl_id,
                "articles_categorized": metrics.articles_categorized,
                "category_counts": categorized_df["category"].value_counts().to_dict()
                    if "category" in categorized_df.columns else {},
                "duration_ms": round(cat_ms, 1),
            },
        )
        logger.info(
            "  → Categories assigned: %s",
            categorized_df["category"].value_counts().to_dict() if "category" in categorized_df.columns else {},
        )
    except Exception as e:
        metrics.end_stage("categorize")
        metrics.record_error(f"categorization_failed: {e}")
        logger.error("Categorization failed: %s", e)
        categorized_df = classified_df

    # Step 5: Thumbnail extraction
    logger.info("Step 5: Extracting article thumbnails...")
    metrics.start_stage("thumbnails")
    try:
        import asyncio  # noqa: import-outside-toplevel
        from enrichment.thumbnails import extract_thumbnails_batch  # noqa: import-outside-toplevel

        article_dicts_for_thumbs = categorized_df.to_dict(orient="records")
        urls = [a.get("url", "") for a in article_dicts_for_thumbs if a.get("url")]
        concurrency = int(__import__("os").environ.get("THUMBNAIL_CONCURRENCY", "10"))
        timeout = int(__import__("os").environ.get("THUMBNAIL_TIMEOUT", "5"))

        if urls:
            thumbnail_map = asyncio.run(
                extract_thumbnails_batch(urls, concurrency=concurrency, timeout=timeout)
            )
            for article in article_dicts_for_thumbs:
                if not article.get("thumbnail_url"):
                    article["thumbnail_url"] = thumbnail_map.get(article.get("url", ""))
        else:
            thumbnail_map = {}

        import pandas as pd  # noqa: import-outside-toplevel
        thumb_df = pd.DataFrame(article_dicts_for_thumbs)
        thumb_ms = metrics.end_stage("thumbnails")
        thumbnails_found = int(thumb_df["thumbnail_url"].notna().sum()) if "thumbnail_url" in thumb_df.columns else 0
        logger.info(
            "thumbnails_complete",
            extra={
                "event": "thumbnails_complete",
                "crawl_id": metrics.crawl_id,
                "thumbnails_found": thumbnails_found,
                "urls_attempted": len(urls),
                "duration_ms": round(thumb_ms, 1),
            },
        )
        logger.info("  → Thumbnails found: %d/%d", thumbnails_found, len(urls))
    except Exception as e:
        metrics.end_stage("thumbnails")
        metrics.record_error(f"thumbnails_failed: {e}")
        logger.error("Thumbnail extraction failed: %s", e)
        thumb_df = categorized_df

    # Step 6: Generate summaries
    logger.info("Step 6: Generating article summaries...")
    metrics.start_stage("summarize")
    try:
        from pipeline.summarizer import summarize  # noqa: import-outside-toplevel
        article_dicts = thumb_df.to_dict(orient="records")
        summarized = []
        for art in article_dicts:
            text = art.get("body") or art.get("title", "")
            t0 = __import__("time").monotonic()
            try:
                art["summary"] = summarize(text)
                dur_ms = (__import__("time").monotonic() - t0) * 1000
                log_article_event(
                    logger,
                    event="article_processed",
                    source_id=art.get("source_id", ""),
                    url=art.get("url", ""),
                    title=art.get("title", ""),
                    stage="summarized",
                    uplifting_score=art.get("uplifting_score"),
                    duration_ms=dur_ms,
                )
            except (ValueError, Exception) as e:
                logger.warning(
                    "summarize_error",
                    extra={
                        "event": "summarize_error",
                        "url": art.get("url", ""),
                        "error": str(e),
                    },
                )
                logger.warning("Summarization failed for article '%s': %s", art.get("title", "")[:60], e)
                art["summary"] = None
            summarized.append(art)
        import pandas as pd  # noqa: import-outside-toplevel
        summarized_df = pd.DataFrame(summarized)
        sum_ms = metrics.end_stage("summarize")
        summaries_generated = int(summarized_df["summary"].notna().sum())
        metrics.articles_summarized = summaries_generated
        logger.info(
            "summarize_complete",
            extra={
                "event": "summarize_complete",
                "crawl_id": metrics.crawl_id,
                "summaries_generated": summaries_generated,
                "total": len(summarized_df),
                "duration_ms": round(sum_ms, 1),
            },
        )
        logger.info("  → Summaries generated: %d/%d", summaries_generated, len(summarized_df))
    except Exception as e:
        metrics.end_stage("summarize")
        metrics.record_error(f"summarization_failed: {e}")
        logger.error("Summary generation failed: %s", e)
        summarized_df = thumb_df
        summaries_generated = 0

    # Step 7: Write to database
    logger.info("Step 7: Writing articles to database...")
    metrics.start_stage("db_write")
    try:
        article_dicts = summarized_df.to_dict(orient="records")

        if "uplifting_score" in summarized_df.columns and not summarized_df["uplifting_score"].isna().all():
            metrics.avg_uplifting_score = float(summarized_df["uplifting_score"].mean())

        articles_stored = db.upsert_articles(article_dicts)
        db_ms = metrics.end_stage("db_write")
        metrics.articles_stored = articles_stored
        logger.info(
            "db_write_complete",
            extra={
                "event": "db_write_complete",
                "crawl_id": metrics.crawl_id,
                "articles_stored": articles_stored,
                "duration_ms": round(db_ms, 1),
            },
        )
        logger.info("  → Stored %d articles in %s", articles_stored, db_path)
    except Exception as e:
        metrics.end_stage("db_write")
        metrics.record_error(f"db_write_failed: {e}")
        logger.error("Database write failed: %s", e)
        articles_stored = 0

    metrics.finish()

    # Persist crawl metrics to DB
    _record_and_close(db, metrics)

    # Emit final crawl-complete structured log
    log_dict = metrics.to_log_dict()
    logger.info(log_dict["event"], extra=log_dict)
    logger.info("="*60)
    logger.info("Pipeline complete  [%s]  duration=%.1fs", metrics.crawl_id, metrics.duration_sec or 0)
    logger.info("="*60)

    return {
        "articles_fetched": metrics.articles_fetched,
        "articles_deduplicated": metrics.articles_skipped,
        "articles_classified": metrics.articles_classified,
        "articles_categorized": metrics.articles_categorized,
        "articles_summarized": metrics.articles_summarized,
        "articles_stored": metrics.articles_stored,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _record_and_close(db, metrics: CrawlMetrics) -> None:
    """Persist crawl metrics then close the DB connection."""
    try:
        db.record_crawl_metrics(**metrics.to_db_kwargs())
    except Exception as e:
        logger.warning("Failed to record crawl metrics: %s", e)
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    from pipeline.logging_config import setup_logging
    setup_logging()
    result = run_pipeline()
    print(f"\nPipeline metrics: {result}")
