# upnews-pipeline

A scalable news crawling and ML classification pipeline that discovers uplifting, positive news from RSS feeds, enriches articles with metadata (thumbnails, summaries, categories), and stores them in a SQLite database for the upnews platform.

## Overview

The pipeline runs in a continuous loop:

**RSS Feeds → Fetch → Deduplicate → Classify → Enrich → Write to DB**

1. **Crawls** RSS feeds from curated sources defined in `config/sources.yaml`
2. **Deduplicates** against existing DB entries to skip already-processed articles
3. **Classifies** articles as uplifting/positive using a fine-tuned SetFit model with rule-based fallback
4. **Enriches** articles with thumbnails (OG image extraction), summaries (DistilBART), and categories (zero-shot classification)
5. **Stores** articles in SQLite with structured JSON logs and crawl health metrics

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  upnews-pipeline                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  config/sources.yaml                                │
│         │                                           │
│         ▼                                           │
│  crawler/crawl_all_sources.py                       │
│  ├─ crawler/sources/rss.py    (RSSSource adapter)   │
│  └─ crawler/sources/reddit.py (RedditSource adapter)│
│         │                                           │
│         ▼                                           │
│  pipeline/deduplication.py  ← checks articles.db   │
│         │                                           │
│         ▼                                           │
│  classifier/classify_headlines.py (SetFit + rules)  │
│         │                                           │
│         ▼                                           │
│  enrichment/                                        │
│  ├─ thumbnails.py   (OG image extraction, async)    │
│  ├─ summarizer.py   (DistilBART summarization)      │
│  └─ categorizer.py  (zero-shot classification)      │
│         │                                           │
│         ▼                                           │
│  pipeline/database.py → data/articles.db            │
│  pipeline/logging_config.py → logs/pipeline.log     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
upnews-pipeline/
├── config/
│   └── sources.yaml               # RSS feed source registry
│
├── crawler/
│   ├── rss_reader.py              # RSS feed parsing via feedparser
│   ├── fetch_sources.py           # Load and validate sources from YAML
│   ├── crawl_all_sources.py       # Orchestrate crawl across all sources
│   └── sources/
│       ├── base.py                # BaseSource abstract class
│       ├── rss.py                 # RSSSource adapter
│       └── reddit.py              # RedditSource adapter (extracts external links)
│
├── classifier/
│   └── classify_headlines.py      # SetFit + rule-based uplifting classifier
│
├── enrichment/
│   ├── thumbnails.py              # Async OG image extraction with batch support
│   ├── summarizer.py              # DistilBART article summarization
│   └── categorizer.py             # Zero-shot multi-category classification
│
├── pipeline/
│   ├── run_pipeline.py            # Main orchestrator
│   ├── database.py                # SQLite layer (schema, UPSERT, transactions)
│   ├── deduplication.py           # URL + title/published_at deduplication
│   ├── logging_config.py          # JSON formatter and CrawlMetrics tracker
│   └── summarizer.py              # Summary generation integration
│
├── tests/
│   ├── conftest.py                # Shared fixtures, ML dependency stubs
│   ├── fixtures/                  # Sample RSS XML files for testing
│   │   ├── sample_feed.xml        # Valid RSS feed (3 articles)
│   │   ├── empty_feed.xml         # RSS feed with no entries
│   │   ├── malformed_feed.xml     # Broken XML
│   │   └── reddit_feed.xml        # Reddit Atom feed
│   └── test_*.py                  # Unit and integration tests (95% coverage)
│
├── models/
│   └── setfit_uplifting_model/    # Pre-trained SetFit model weights
│
├── data/
│   └── articles.db                # SQLite database
│
├── logs/
│   └── pipeline.log               # Structured JSON crawl logs
│
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.9+
- SQLite3 (built-in with Python)

### Installation

```bash
git clone https://github.com/News-Uplifters/upnews-pipeline.git
cd upnews-pipeline
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt          # production deps
pip install -r requirements-dev.txt     # testing + linting (optional)
```

### ML Model Setup

The classifier requires a SetFit model in `models/setfit_uplifting_model/`. Either download a pre-trained model or train one:

```bash
# Copy pre-trained weights to the models directory
cp -r /path/to/model models/setfit_uplifting_model/

# Or download from HuggingFace Hub
# python -c "from setfit import SetFitModel; SetFitModel.from_pretrained('...').save_pretrained('models/setfit_uplifting_model')"
```

The summarizer (DistilBART) and categorizer (BART-large-MNLI) download their models automatically on first run via HuggingFace.

## Configuration

Sources are defined in `config/sources.yaml`:

```yaml
sources:
  - name: "BBC News"
    source_id: "BBCNews"
    rss_url: "http://feeds.bbc.co.uk/news/world/rss.xml"
    active: true
    category: "news"
    threshold: 0.90

  - name: "Reddit r/UpliftingNews"
    source_id: "RedditUpliftingNews"
    rss_url: "https://www.reddit.com/r/UpliftingNews/.rss"
    active: true
    category: "reddit"
    adapter: "reddit"
    threshold: 0.75
```

| Field | Description |
|-------|-------------|
| `source_id` | Unique identifier used in DB and logs |
| `rss_url` | RSS/Atom feed URL |
| `active` | Set `false` to skip this source without removing it |
| `category` | Source type (`news`, `reddit`, etc.) |
| `adapter` | Source adapter to use (`rss` default, `reddit` for Reddit feeds) |
| `threshold` | Minimum uplifting score to store an article (0.0–1.0) |

## Running the Pipeline

```bash
# Full crawl cycle
python -m pipeline.run_pipeline

# Debug logging
LOG_LEVEL=DEBUG python -m pipeline.run_pipeline
```

## Running with Docker

Docker is the recommended way to run the pipeline — no Python environment setup required.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac/Windows) or Docker Engine + Compose v2 (Linux)

### Environment files

The project ships with a set of pre-configured env files — pick the one that matches your target:

| File | Purpose | Log level | Articles/source | Classifier |
|------|---------|-----------|-----------------|------------|
| `.env.local` | Personal dev machine | `DEBUG` | 10 | rules |
| `.env.dev` | Dev / feature deploys | `DEBUG` | 25 | rules |
| `.env.pr` | PR / staging | `INFO` | 50 | rules |
| `.env.qa` | QA (pre-prod) | `INFO` | 50 | SetFit |
| `.env.prod` | Production | `WARNING` | 100 | SetFit |

`.env.local` is gitignored so you can edit it freely. The others are committed as safe configuration templates — inject secrets (e.g. `CLAUDE_API_KEY`) via CI/CD environment variables or a secrets manager, never by editing these files directly.

See `.env.example` for the full list of supported variables with descriptions.

### Quick start (local)

```bash
# 1. Clone and enter the repo
git clone https://github.com/News-Uplifters/upnews-pipeline.git
cd upnews-pipeline

# 2. Build and run a single pipeline cycle (exits when done)
docker compose --env-file .env.local run --rm pipeline

# 3. Inspect results
sqlite3 data/upnews.db "SELECT title, uplifting_score, category FROM articles ORDER BY crawled_at DESC LIMIT 10;"
```

Or use the Makefile shortcut:

```bash
make run-local
```

### Start / stop

```bash
# Build + start (stays up; Ctrl-C to stop)
make local          # or: docker compose --env-file .env.local up --build

# Run once and remove container
make run-local      # or: docker compose --env-file .env.local run --rm pipeline

# Stop and remove containers
make stop           # or: docker compose down
```

### Other environments

```bash
make dev            # dev
make pr             # PR / staging
make qa             # QA — requires SetFit model weights (see ML Model Setup)
make prod           # production (detached / background)
```

Or call Docker Compose directly with any env file:

```bash
docker compose --env-file .env.prod up --build -d
```

### Makefile reference

```
make local | dev | pr | qa | prod      — build + start for that environment
make run-local | run-dev | ...          — one-shot run, then remove container
make build-local | build-dev | ...      — build image without running
make logs                               — tail container logs
make shell                              — open bash inside the container
make db                                 — query last 20 articles from SQLite
make stop                               — stop and remove containers
make help                               — print this summary
```

### SetFit classifier in Docker

The `qa` and `prod` env files set `CLASSIFIER_MODE=setfit`. This requires pre-trained model weights to be present at `models/setfit_uplifting_model/` on the **host** machine — the volume mount (`./models:/app/models`) makes them available inside the container automatically.

```bash
# Copy or download weights first
cp -r /path/to/weights models/setfit_uplifting_model/

# Then run QA or prod
make run-qa
```

If the model directory is missing the pipeline falls back to the rules-based classifier and logs a warning.

### Persisted data

Docker volumes map three host directories into the container:

| Host path | Container path | Contents |
|-----------|---------------|---------|
| `./data/` | `/app/data/` | SQLite database (`upnews.db`) |
| `./logs/` | `/app/logs/` | Structured JSON logs |
| `./models/` | `/app/models/` | HuggingFace cache + SetFit weights |

These directories are created automatically on first run and persist across container restarts.

### Individual Components

```bash
# Test RSS feed parsing
python -c "from crawler.rss_reader import fetch_rss_headlines; import json; print(json.dumps(fetch_rss_headlines('http://feeds.bbc.co.uk/news/world/rss.xml')[:2], indent=2, default=str))"

# Inspect the database
sqlite3 data/articles.db "SELECT title, uplifting_score, category FROM articles ORDER BY crawled_at DESC LIMIT 10;"

# View crawl history
sqlite3 data/upnews.db "SELECT crawl_start, articles_fetched, articles_stored FROM crawl_metrics ORDER BY crawl_start DESC LIMIT 5;"
```

## Database Schema

```sql
CREATE TABLE articles (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    url            TEXT UNIQUE NOT NULL,
    title          TEXT NOT NULL,
    summary        TEXT,
    body           TEXT,
    thumbnail_url  TEXT,
    category       TEXT,
    source_id      TEXT NOT NULL,
    uplifting_score REAL,
    published_at   DATETIME,
    crawled_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
);

CREATE TABLE sources (
    source_id  TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    rss_url    TEXT,
    active     BOOLEAN DEFAULT 1,
    category   TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE crawl_metrics (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    crawl_start              DATETIME,
    crawl_end                DATETIME,
    articles_fetched         INTEGER,
    articles_classified      INTEGER,
    articles_stored          INTEGER,
    avg_classification_score REAL,
    errors                   TEXT
);
```

## Logging

Structured JSON logs are written to `logs/pipeline.log`:

```json
{"timestamp": "2026-03-25T10:30:15.123Z", "event": "article_processed", "source_id": "BBCNews", "url": "https://...", "uplifting_score": 0.87, "status": "success"}
{"timestamp": "2026-03-25T10:35:00.000Z", "event": "crawl_complete", "articles_fetched": 342, "articles_stored": 256, "articles_skipped": 31, "status": "success"}
```

## Testing

```bash
# Run all tests
pytest -v

# With coverage report
pytest --cov=. --cov-report=html

# Specific module
pytest tests/test_integration.py -v
```

Tests use fixture RSS XML files and stub out heavy ML dependencies (SetFit, torch, transformers) so they run without a GPU or model download. Current coverage: **95%** across 387 tests.

## Current News Sources

Configured in `config/sources.yaml`. Active sources include BBC News, CBS News, Associated Press, Reuters, The New York Times, NPR, The Guardian, Al Jazeera, Hacker News, and Reddit r/UpliftingNews.

## Code Quality

```bash
black .
isort .
flake8 .
mypy . --ignore-missing-imports
```

## License

MIT

## Contact

For questions or contributions, contact the News Uplifters team.
