# upnews-pipeline

A scalable news crawling and ML classification pipeline that discovers uplifting, positive news from RSS feeds, enriches articles with metadata (thumbnails, summaries), and stores them in a database for the upnews platform.

## Project Description

This pipeline:
1. **Crawls** RSS feeds from curated news sources
2. **Classifies** articles as uplifting/positive using a fine-tuned ML model
3. **Enriches** articles with thumbnails, summaries, and structured metadata
4. **Stores** articles in SQLite (eventually synced to upnews-api)
5. **Deduplicates** against existing entries to avoid crawling the same article twice

Target flow: **RSS Feed → Fetch Articles → Classify → Enrich → Write to DB**

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           upnews-pipeline                           │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐      ┌──────────────┐           │
│  │  Sources     │      │  Fetchers    │           │
│  │  YAML config │──→   │  RSS, Reddit,│           │
│  │              │      │  Media sites │           │
│  └──────────────┘      └──────────────┘           │
│         │                     │                    │
│         └─────────────┬───────┘                    │
│                       ↓                            │
│         ┌─────────────────────────┐               │
│         │   Article Raw Data      │               │
│         │  title, url, published  │               │
│         └─────────────────────────┘               │
│                       │                            │
│                       ↓                            │
│         ┌─────────────────────────┐               │
│         │   Deduplication         │               │
│         │   (Check DB for URL)    │               │
│         └─────────────────────────┘               │
│                       │                            │
│                       ↓                            │
│         ┌─────────────────────────┐               │
│         │ ML Classification       │               │
│         │ SetFit + Rule-based     │               │
│         │ → uplifting_score       │               │
│         └─────────────────────────┘               │
│                       │                            │
│                       ↓                            │
│         ┌─────────────────────────┐               │
│         │  Enrichment Services    │               │
│         │  ├─ Thumbnail Extract   │               │
│         │  ├─ Summary Generation  │               │
│         │  └─ Categorization      │               │
│         └─────────────────────────┘               │
│                       │                            │
│                       ↓                            │
│         ┌─────────────────────────┐               │
│         │   Database Write        │               │
│         │   SQLite (local)        │               │
│         │   + Health Metrics      │               │
│         └─────────────────────────┘               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Getting Started

### Prerequisites
- Python 3.9+
- `pip` or `poetry`
- SQLite3 (usually built-in)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/News-Uplifters/upnews-pipeline.git
cd upnews-pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure RSS sources (see below)

5. Download or train the ML model:
```bash
# Copy pre-trained model to models/setfit_uplifting_model/
# Or download from HuggingFace Hub
```

### Configuration

**Sources** are defined in `config/sources.yaml`. Example format:

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
    adapter: "reddit"  # Uses RedditSource adapter
    threshold: 0.75

  - name: "NPR News"
    source_id: "NPRNews"
    rss_url: "https://feeds.npr.org/1001/rss.xml"
    active: true
    category: "news"
    threshold: 0.93
```

### Running the Pipeline

Execute a full crawl cycle:
```bash
python -m pipeline.run_pipeline
```

Or run individual components:
```bash
# Fetch articles from RSS feeds
python -c "from crawler.rss_reader import fetch_rss_headlines; print(fetch_rss_headlines('http://feeds.bbc.co.uk/news/world/rss.xml'))"

# Classify articles
python -c "from classifier.classify_headlines import filter_positive_news; ..."

# Generate summaries
python -c "from pipeline.summarizer import summarize; ..."
```

### Project Structure

```
upnews-pipeline/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
│
├── config/
│   └── sources.yaml               # RSS feed source registry
│
├── crawler/
│   ├── __init__.py
│   ├── rss_reader.py              # RSS feed parsing (existing code)
│   ├── fetch_sources.py           # Source loading from YAML
│   └── crawl_all_sources.py       # Orchestrate crawl across all sources
│
├── classifier/
│   ├── __init__.py
│   └── classify_headlines.py      # SetFit + rule-based classification
│
├── enrichment/
│   ├── __init__.py
│   ├── thumbnails.py              # OG image extraction (TASK 3)
│   └── categorizer.py             # LLM-based categorization (TASK 4)
│
├── pipeline/
│   ├── __init__.py
│   ├── run_pipeline.py            # Main orchestrator
│   └── summarizer.py              # DistilBART summarization
│
├── tests/
│   ├── __init__.py
│   ├── test_fetch_sources.py      # YAML source loading (TASK 1) ✅
│   ├── test_rss_reader.py         # Mock RSS feeds (TASK 9)
│   ├── test_classifier.py
│   └── test_integration.py
│
├── models/
│   └── setfit_uplifting_model/    # Pre-trained SetFit model
│
├── data/
│   └── articles.db                # SQLite database (TASK 6)
│
└── logs/                           # Crawl logs and metrics (TASK 8)
```

## Current News Sources

Sources are now managed in `config/sources.yaml` (migrated from Excel). Current active sources include:

- BBC News (World)
- CBS News
- Associated Press
- Reuters (World)
- The New York Times (World)
- NPR News
- The Guardian (World)
- Al Jazeera
- Hacker News (via RSS)
- Reddit r/UpliftingNews
- Local news feeds (varies by geography)

See `config/sources.yaml` for the complete list with thresholds and metadata.

---

## TODO: Numbered Tasks for Contributors & Claude Agents

These tasks represent the core improvements needed to scale and enhance the pipeline. Each task is numbered, scoped, and ready to assign.

### TASK 1: Replace Excel source registry with YAML config

**Status:** ✅ Done
**Type:** Data migration + refactoring
**Effort:** ~2 hours

**Description:**
The crawler currently loads sources from `data/news_sources.xlsx` (via pandas). Migrate to `config/sources.yaml` for version control, readability, and easier dynamic source management.

**Target YAML Format** (already in this README under Configuration section):
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

**Changes Required:**
1. Create `config/sources.yaml` with all current sources from the Excel file
2. Update `crawler/fetch_sources.py` to load YAML instead of Excel
3. Add YAML validation schema
4. Deprecate Excel loading (keep fallback for compatibility)
5. Test with existing source list

**Acceptance Criteria:**
- [x] `config/sources.yaml` created with all current sources
- [x] `fetch_sources.py` loads from YAML
- [x] Thresholds and source IDs preserved
- [x] Fallback to Excel if YAML missing (for backward compatibility)
- [x] Unit tests pass

---

### TASK 2: Implement pluggable source adapter pattern

**Status:** ✅ Done
**Type:** Refactoring + feature
**Effort:** ~4 hours

**Description:**
Currently, all RSS feeds are treated uniformly. However, Reddit, Twitter, and other social media require special parsing (extract external links, etc.). Create a pluggable adapter pattern:
- `BaseSource` abstract base class
- `RSSSource` concrete adapter (existing feedparser logic)
- `RedditSource` concrete adapter (extract_reddit_external logic)
- `TwitterSource`, `MediaSource` stubs for future

**Target Class Interface:**
```python
from abc import ABC, abstractmethod

class BaseSource(ABC):
    """Abstract base class for news sources."""

    def __init__(self, config: dict):
        """Initialize source from config dict.

        Args:
            config: Source config from YAML (name, rss_url, threshold, etc.)
        """
        self.name = config.get("name")
        self.source_id = config.get("source_id")
        self.rss_url = config.get("rss_url")
        self.threshold = config.get("threshold", 0.75)
        self.active = config.get("active", True)

    @abstractmethod
    def fetch(self, limit: int = 50) -> list[dict]:
        """Fetch articles from this source.

        Returns:
            List of article dicts with keys:
            - title (str)
            - url (str)
            - published (datetime or None)
            - original_url (str): canonical article URL
            - source_id (str): source identifier
        """
        pass


class RSSSource(BaseSource):
    """Standard RSS feed adapter."""

    def fetch(self, limit: int = 50) -> list[dict]:
        # Use existing fetch_rss_headlines logic
        pass


class RedditSource(BaseSource):
    """Reddit subreddit adapter (extract external links)."""

    def fetch(self, limit: int = 50) -> list[dict]:
        # Use existing extract_reddit_external logic
        pass
```

**Changes Required:**
1. Create `crawler/sources/__init__.py` with `BaseSource` ABC
2. Implement `crawler/sources/rss.py` (RSSSource)
3. Implement `crawler/sources/reddit.py` (RedditSource)
4. Update `crawler/crawl_all_sources.py` to instantiate adapters via factory pattern
5. Add adapter selection logic in `fetch_sources.py`

**Acceptance Criteria:**
- [x] `BaseSource` ABC defined
- [x] `RSSSource` adapter passes all existing RSS tests
- [x] `RedditSource` adapter extracts external links correctly
- [x] `crawl_all_sources.py` uses factory to instantiate sources
- [x] Backward compatible with existing crawler code

---

### TASK 3: Add thumbnail extraction service

**Status:** Ready to start
**Type:** New feature
**Effort:** ~3 hours

**Description:**
Articles should have associated thumbnails (cover images). Extract from:
1. Open Graph `og:image` meta tag (preferred, 80% hit rate)
2. Twitter Card `twitter:image`
3. Generic `<img>` tags in article body
4. Fallback: source favicon

Use async requests with concurrency limits to avoid hammering external servers.

**Target Function Signature:**
```python
async def extract_thumbnail(url: str, timeout: int = 5, max_retries: int = 2) -> str | None:
    """Extract thumbnail URL from article page.

    Args:
        url: Article URL to scrape
        timeout: HTTP request timeout (seconds)
        max_retries: Retry failed requests this many times

    Returns:
        Absolute URL of thumbnail image, or None if not found
    """
    pass


async def extract_thumbnails_batch(urls: list[str], concurrency: int = 10) -> dict[str, str | None]:
    """Extract thumbnails for multiple URLs concurrently.

    Args:
        urls: List of article URLs
        concurrency: Max concurrent requests

    Returns:
        Dict mapping URL → thumbnail_url (or None)
    """
    pass
```

**Changes Required:**
1. Create `enrichment/thumbnails.py` with `extract_thumbnail()` and `extract_thumbnails_batch()`
2. Use `aiohttp` for async requests, `BeautifulSoup` for parsing
3. Respect rate limits (concurrency limit, delays)
4. Add caching (Redis or local SQLite) to avoid re-fetching
5. Add unit tests with mock HTTP responses

**Acceptance Criteria:**
- [ ] `extract_thumbnail()` retrieves OG images successfully
- [ ] Async batch processing works with concurrency limits
- [ ] Fallback chain: OG → Twitter → img → favicon
- [ ] Caching reduces redundant requests
- [ ] Rate limits respected (no server spam)
- [ ] Unit tests with mocked responses pass

---

### TASK 4: Upgrade categorization from keyword-matching to zero-shot classifier or LLM API

**Status:** Ready to start
**Type:** Enhancement + ML
**Effort:** ~5 hours

**Description:**
Currently, articles are classified as "uplifting" or not (binary). Expand to multi-category classification:
- Health & Wellness
- Environment & Nature
- Community & Social Good
- Technology & Science
- Business & Economics
- Culture & Arts
- Human Interest

Use either:
- **Zero-shot classifier** (HuggingFace `facebook/bart-large-mnli`, fast, on-device)
- **LLM API** (Claude Haiku, OpenAI, more flexible)

Recommend: Start with zero-shot for speed, add LLM option later.

**Target Function Signature:**
```python
def categorize_article(title: str, body: str = "", categories: list[str] = None) -> dict:
    """Categorize article into predefined or custom categories.

    Args:
        title: Article title
        body: Article body (optional, for context)
        categories: List of category names. If None, use defaults.

    Returns:
        Dict with:
        - category (str): Top category
        - scores (dict): {category: confidence_score}
        - confidence (float): Confidence in top category
    """
    pass


def categorize_batch(articles: list[dict], categories: list[str] = None) -> list[dict]:
    """Categorize multiple articles."""
    pass
```

**Changes Required:**
1. Create `enrichment/categorizer.py` with zero-shot classifier
2. Load model: `pipeline("zero-shot-classification", model="facebook/bart-large-mnli")`
3. Integrate with pipeline: enrich articles with `category` field
4. Add unit tests
5. Document category definitions and scoring

**Acceptance Criteria:**
- [ ] Zero-shot classifier integrated
- [ ] Articles assigned to 1+ categories with confidence scores
- [ ] Multi-label support (one article, multiple categories possible)
- [ ] Pipeline enriches articles with category field
- [ ] Inference time < 100ms per article (batch)
- [ ] Unit tests pass

---

### TASK 5: Add article summary generation

**Status:** Partially complete (DistilBART stub exists)
**Type:** Enhancement + NLP
**Effort:** ~3 hours

**Description:**
Articles should include a brief summary (2-3 sentences) for the UI. Existing code has a `summarizer.py` stub; implement with:
- **DistilBART** (on-device, ~100ms per article)
- Or **Claude Haiku API** (higher quality, ~2-5 seconds, requires credits)

Recommend: Start with DistilBART, add Claude option as fallback for short/complex articles.

**Target Function Signature:**
```python
def summarize(text: str, max_length: int = 130, min_length: int = 30) -> str:
    """Generate a summary of article text.

    Args:
        text: Article body (or title + body)
        max_length: Max summary length (tokens)
        min_length: Min summary length (tokens)

    Returns:
        Summary string (1-3 sentences)
    """
    pass


async def summarize_batch(texts: list[str], batch_size: int = 8) -> list[str]:
    """Generate summaries for multiple texts efficiently."""
    pass
```

**Changes Required:**
1. Implement `pipeline/summarizer.py` with DistilBART pipeline
2. Add summary to article enrichment in `run_pipeline.py`
3. Add caching to avoid re-summarizing
4. (Optional) Add Claude Haiku fallback for complex articles
5. Unit tests with real articles

**Acceptance Criteria:**
- [ ] DistilBART summarizer working
- [ ] Summaries 1-3 sentences, coherent
- [ ] Batch processing efficient
- [ ] Caching prevents redundant summarization
- [ ] Unit tests pass (test with BBC, Reddit articles)

---

### TASK 6: Replace CSV output with SQLite DB writes

**Status:** Ready to start
**Type:** Infrastructure + Data
**Effort:** ~4 hours

**Description:**
Currently, articles are written to CSV. Replace with SQLite database for:
- Relational querying
- Deduplication
- Sync with upnews-api
- Transaction support

Create a shared schema compatible with upnews-api.

**Target Schema:**
```sql
CREATE TABLE articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    body TEXT,
    thumbnail_url TEXT,
    category TEXT,
    source_id TEXT NOT NULL,
    uplifting_score REAL,
    published_at DATETIME,
    crawled_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
);

CREATE TABLE sources (
    source_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    rss_url TEXT,
    active BOOLEAN DEFAULT 1,
    category TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE crawl_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    crawl_start DATETIME,
    crawl_end DATETIME,
    articles_fetched INTEGER,
    articles_classified INTEGER,
    articles_stored INTEGER,
    avg_classification_score REAL,
    errors TEXT
);

CREATE INDEX idx_articles_source ON articles(source_id);
CREATE INDEX idx_articles_published ON articles(published_at);
CREATE INDEX idx_articles_score ON articles(uplifting_score);
```

**Changes Required:**
1. Create `pipeline/database.py` with SQLiteDB class (connection, schema, transactions)
2. Add `init_db()` function to create schema
3. Replace CSV writes in `run_pipeline.py` with DB inserts
4. Add UPSERT logic (update if URL already exists)
5. Add transaction support for atomicity
6. Unit tests with in-memory SQLite

**Acceptance Criteria:**
- [ ] SQLite schema created and migrated
- [ ] Articles written to DB instead of CSV
- [ ] Transactions atomic (all-or-nothing per crawl)
- [ ] Schema compatible with upnews-api
- [ ] Unit tests pass (test UPSERT, foreign keys)
- [ ] Performance acceptable (< 500ms for 100 articles)

---

### TASK 7: Add deduplication against existing DB entries

**Status:** Ready to start
**Type:** Feature
**Effort:** ~2 hours

**Description:**
Skip articles already in the DB to avoid duplicates and redundant processing. Check by:
1. URL (primary key, most reliable)
2. Title + published_at (fallback)

Short-circuit the pipeline: skip classification/enrichment for known URLs.

**Target Function Signature:**
```python
def deduplicate_articles(articles: list[dict], db_path: str = "data/articles.db") -> list[dict]:
    """Filter out articles already in the database.

    Args:
        articles: List of fetched articles
        db_path: Path to SQLite database

    Returns:
        Filtered list of new articles only
    """
    pass


def article_exists(url: str, db_path: str = "data/articles.db") -> bool:
    """Check if article URL already exists in DB."""
    pass
```

**Changes Required:**
1. Implement `pipeline/deduplication.py` with `deduplicate_articles()` and `article_exists()`
2. Call deduplication early in pipeline (after fetch, before classify)
3. Log skipped articles for metrics
4. Add tests with sample URLs

**Acceptance Criteria:**
- [ ] Duplicate URLs detected and skipped
- [ ] Fallback to title + published_at dedup works
- [ ] Pipeline logs deduplicated count
- [ ] Zero false positives (no real articles skipped)
- [ ] Unit tests pass

---

### TASK 8: Add structured logging and crawl health metrics

**Status:** Ready to start
**Type:** Operations + Observability
**Effort:** ~3 hours

**Description:**
Add comprehensive logging and metrics tracking:
- Structured JSON logs (one entry per article processed)
- Crawl-level metrics: start time, end time, counts, errors
- Health checks: feed availability, classifier uptime, DB writes
- Alerts: if crawl takes > 5min, if error rate > 5%, etc.

Use Python's `logging` module with JSON formatter.

**Target Logging:**
```python
# Example article-level log
{
    "timestamp": "2026-03-25T10:30:15.123Z",
    "event": "article_processed",
    "source_id": "BBCNews",
    "url": "https://bbc.com/news/...",
    "title": "...",
    "stage": "classified",
    "uplifting_score": 0.87,
    "duration_ms": 245,
    "status": "success"
}

# Example crawl-level metric
{
    "timestamp": "2026-03-25T10:35:00.000Z",
    "event": "crawl_complete",
    "crawl_id": "crawl_20260325_103015",
    "duration_sec": 285,
    "sources_processed": 12,
    "articles_fetched": 342,
    "articles_new": 287,
    "articles_stored": 256,
    "articles_skipped": 31,
    "classification_error_rate": 0.02,
    "avg_uplifting_score": 0.78,
    "status": "success"
}
```

**Changes Required:**
1. Create `pipeline/logging_config.py` with JSON formatter and handlers
2. Log to file (`logs/pipeline.log`) and stdout
3. Add metrics class: `CrawlMetrics` to track counts, timings, errors
4. Add health check endpoints in pipeline (optional, for monitoring)
5. Store crawl metrics in `crawl_metrics` table
6. Unit tests for logging output

**Acceptance Criteria:**
- [ ] JSON logs written to `logs/pipeline.log`
- [ ] Article-level events logged
- [ ] Crawl-level metrics recorded
- [ ] Metrics stored in DB (`crawl_metrics` table)
- [ ] Structured format valid JSON
- [ ] Unit tests pass

---

### TASK 9: Write integration tests with mock RSS feeds

**Status:** Ready to start
**Type:** Testing
**Effort:** ~3 hours

**Description:**
Ensure the full pipeline works end-to-end. Write integration tests using mock RSS feeds and a temporary SQLite DB.

**Target Test Structure:**
```python
# tests/test_integration.py

def test_end_to_end_crawl():
    """Full pipeline: fetch → classify → enrich → store."""
    pass

def test_rss_parsing():
    """Test RSS reader with mock feeds (valid, malformed, empty)."""
    pass

def test_classification_pipeline():
    """Test classification with known uplifting titles."""
    pass

def test_deduplication():
    """Test duplicate detection."""
    pass

def test_database_writes():
    """Test article storage and transactions."""
    pass

def test_error_handling():
    """Test graceful handling of missing feeds, bad URLs, etc."""
    pass
```

**Changes Required:**
1. Create `tests/fixtures/` with sample RSS XML files (BBC, Reddit, broken feeds)
2. Write `tests/test_rss_reader.py` (mock feedparser responses)
3. Write `tests/test_classifier.py` (known uplifting/non-uplifting titles)
4. Write `tests/test_integration.py` (end-to-end)
5. Use pytest with fixtures and temporary DB
6. Add test coverage report (aim for > 80%)

**Acceptance Criteria:**
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Mock RSS feeds cover: valid, empty, malformed, timeout
- [ ] Test DB cleaned up after each test
- [ ] Test coverage > 80%
- [ ] CI/CD integration ready (GitHub Actions compatible)

---

## Development Guide

### Running Tests

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_classifier.py -v
```

### Code Style

We use:
- `black` for formatting
- `isort` for imports
- `flake8` for linting
- `mypy` for type checking

```bash
black .
isort .
flake8 .
mypy . --ignore-missing-imports
```

### Building & Deploying

Deployment pipeline (planned):
1. Run tests locally
2. Push to GitHub
3. GitHub Actions runs tests + linting
4. Deploy to upnews-api database (if successful)
5. Trigger scheduled crawls (e.g., every 6 hours)

### Logging & Debugging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python -m pipeline.run_pipeline
```

View crawl metrics:
```bash
# Query crawl history
sqlite3 data/articles.db "SELECT * FROM crawl_metrics ORDER BY crawl_start DESC LIMIT 10;"
```

---

## Contributing

1. Pick a numbered task from the TODO list above
2. Create a feature branch: `git checkout -b task-X-brief-name`
3. Implement the feature
4. Write tests
5. Ensure code passes linting and tests
6. Submit a PR with description linking to the task

Example:
```bash
git checkout -b task-3-thumbnail-extraction
# ... implement feature ...
pytest -v
black . && isort . && flake8 .
git commit -m "Task 3: Add thumbnail extraction service

- Implement extract_thumbnail() with OG image scraping
- Add async batch processing with concurrency limits
- Add caching to avoid redundant requests
- Full test coverage

Closes #3"
git push origin task-3-thumbnail-extraction
# Create PR on GitHub
```

---

## License

MIT

## Contact

For questions or contributions, contact the News Uplifters team.
