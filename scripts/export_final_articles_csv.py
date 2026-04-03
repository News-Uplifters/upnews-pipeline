"""Export all rows from the final articles table to CSV."""

from __future__ import annotations

import csv
import os
import sqlite3
from pathlib import Path


def main() -> None:
    db_path = Path(os.getenv("DATABASE_PATH", "/app/data/upnews.db"))
    output_path = Path(os.getenv("CSV_OUTPUT_PATH", "/app/logs/final_uplifting_articles.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
            a.id,
            a.title,
            a.url,
            a.source_url,
            a.external_url,
            a.content,
            a.summary,
            a.thumbnail_url,
            a.category,
            a.category_confidence,
            a.category_scores,
            s.source_id AS source,
            a.uplifting_score,
            a.published_at,
            a.crawled_at,
            a.created_at,
            a.updated_at
        FROM articles a
        JOIN sources s ON s.id = a.source_id
        ORDER BY a.crawled_at DESC, a.uplifting_score DESC, a.id DESC
        """
    ).fetchall()

    fields = [
        "id",
        "title",
        "url",
        "source_url",
        "external_url",
        "content",
        "summary",
        "thumbnail_url",
        "category",
        "category_confidence",
        "category_scores",
        "source",
        "uplifting_score",
        "published_at",
        "crawled_at",
        "created_at",
        "updated_at",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    print(f"rows={len(rows)}")
    print(f"csv={output_path}")


if __name__ == "__main__":
    main()
