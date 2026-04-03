"""Export the latest crawled_articles batch into uplifting/non-uplifting CSVs."""

from __future__ import annotations

import csv
import os
import sqlite3
from pathlib import Path


def main() -> None:
    db_path = Path(os.getenv("DATABASE_PATH", "/app/data/upnews.db"))
    output_dir = Path(os.getenv("CSV_OUTPUT_DIR", "/app/logs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    latest = conn.execute("SELECT MAX(crawled_at) AS ts FROM crawled_articles").fetchone()["ts"]
    rows = conn.execute(
        """
        SELECT ca.title,
               s.source_id AS source,
               ca.uplifting_score,
               CASE WHEN COALESCE(ca.is_uplifting, 0) = 1 THEN 'yes' ELSE 'no' END AS uplifting,
               ca.category,
               ca.category_confidence,
               ca.category_scores,
               ca.summary,
               ca.url,
               ca.source_url,
               ca.external_url,
               ca.crawled_at
        FROM crawled_articles ca
        JOIN sources s ON s.id = ca.source_id
        WHERE ca.crawled_at = ?
        ORDER BY ca.uplifting_score DESC, ca.title
        """,
        (latest,),
    ).fetchall()

    fields = [
        "title",
        "source",
        "uplifting_score",
        "uplifting",
        "category",
        "category_confidence",
        "category_scores",
        "summary",
        "url",
        "source_url",
        "external_url",
        "crawled_at",
    ]
    uplifting_path = output_dir / "uplifting_articles.csv"
    non_uplifting_path = output_dir / "non_uplifting_articles.csv"

    with uplifting_path.open("w", newline="", encoding="utf-8") as uplifting_fh, non_uplifting_path.open(
        "w", newline="", encoding="utf-8"
    ) as non_uplifting_fh:
        uplifting_writer = csv.DictWriter(uplifting_fh, fieldnames=fields)
        non_uplifting_writer = csv.DictWriter(non_uplifting_fh, fieldnames=fields)
        uplifting_writer.writeheader()
        non_uplifting_writer.writeheader()

        uplifting_count = 0
        non_uplifting_count = 0
        for row in rows:
            item = {field: row[field] for field in fields}
            if item["uplifting"] == "yes":
                uplifting_writer.writerow(item)
                uplifting_count += 1
            else:
                non_uplifting_writer.writerow(item)
                non_uplifting_count += 1

    print(f"latest_crawled_at={latest}")
    print(f"total_rows={len(rows)}")
    print(f"uplifting_rows={uplifting_count}")
    print(f"non_uplifting_rows={non_uplifting_count}")
    print(f"uplifting_csv={uplifting_path}")
    print(f"non_uplifting_csv={non_uplifting_path}")


if __name__ == "__main__":
    main()
