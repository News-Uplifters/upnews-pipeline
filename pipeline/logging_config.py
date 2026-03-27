"""Structured logging configuration for the upnews pipeline (Task 8).

Provides:
- JSON formatter for structured log output
- setup_logging() to configure file + stdout handlers
- CrawlMetrics dataclass for tracking pipeline run statistics
- get_pipeline_logger() convenience helper
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Standard fields included in every record:
    - timestamp (ISO 8601 UTC)
    - level
    - logger
    - message
    - event (if present in the extra dict)

    Any extra key/value pairs passed via ``extra=`` are merged into the top
    level of the JSON object.
    """

    _RESERVED = frozenset(
        {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "message", "module",
            "msecs", "msg", "name", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3] + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }

        # Merge any extra fields (skip logging internals)
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_LOG_DIR = "logs"
_LOG_FILE = "pipeline.log"


def setup_logging(
    level: int = logging.INFO,
    log_dir: str = _LOG_DIR,
    log_file: str = _LOG_FILE,
    json_stdout: bool = False,
) -> None:
    """Configure root logger with a JSON file handler and a stdout handler.

    Args:
        level: Minimum log level (e.g. logging.INFO, logging.DEBUG).
        log_dir: Directory where log files are written (created if missing).
        log_file: Base filename for the rotating log file.
        json_stdout: If True, stdout also emits JSON; otherwise human-readable.
    """
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid adding duplicate handlers if called more than once
    root.handlers.clear()

    # --- File handler (always JSON) ---
    file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(JSONFormatter())
    root.addHandler(file_handler)

    # --- Stdout handler ---
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(level)
    if json_stdout:
        stdout_handler.setFormatter(JSONFormatter())
    else:
        stdout_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
    root.addHandler(stdout_handler)


def get_pipeline_logger(name: str = "upnews.pipeline") -> logging.Logger:
    """Return a named logger under the upnews namespace."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# CrawlMetrics
# ---------------------------------------------------------------------------

@dataclass
class CrawlMetrics:
    """Mutable container that accumulates stats for a single pipeline run.

    Usage::

        metrics = CrawlMetrics.start()
        # ... pipeline work ...
        metrics.articles_fetched = 300
        metrics.articles_new = 250
        metrics.finish()
        log_entry = metrics.to_log_dict()
    """

    crawl_id: str = field(default_factory=lambda: f"crawl_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Article counts
    articles_fetched: int = 0
    articles_new: int = 0
    articles_stored: int = 0
    articles_skipped: int = 0
    articles_classified: int = 0
    articles_categorized: int = 0
    articles_summarized: int = 0

    # Source counts
    sources_processed: int = 0

    # Quality metrics
    avg_uplifting_score: Optional[float] = None
    classification_error_rate: Optional[float] = None

    # Errors
    errors: List[str] = field(default_factory=list)

    # Internal timing helpers
    _stage_timers: Dict[str, float] = field(default_factory=dict, repr=False)

    @classmethod
    def start(cls) -> "CrawlMetrics":
        """Create a new CrawlMetrics instance with start_time set to now."""
        m = cls()
        m.start_time = datetime.now(timezone.utc)
        return m

    def finish(self) -> None:
        """Record end_time."""
        self.end_time = datetime.now(timezone.utc)

    @property
    def duration_sec(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    # --- Stage timers ---

    def start_stage(self, stage: str) -> None:
        """Start a wall-clock timer for *stage*."""
        self._stage_timers[stage] = time.monotonic()

    def end_stage(self, stage: str) -> float:
        """Stop timer for *stage* and return elapsed milliseconds."""
        started = self._stage_timers.pop(stage, None)
        if started is None:
            return 0.0
        return (time.monotonic() - started) * 1000

    # --- Error tracking ---

    def record_error(self, message: str) -> None:
        self.errors.append(message)

    # --- Serialisation ---

    def to_log_dict(self) -> Dict[str, Any]:
        """Return a dict suitable for structured logging (crawl_complete event)."""
        d: Dict[str, Any] = {
            "event": "crawl_complete",
            "crawl_id": self.crawl_id,
            "sources_processed": self.sources_processed,
            "articles_fetched": self.articles_fetched,
            "articles_new": self.articles_new,
            "articles_stored": self.articles_stored,
            "articles_skipped": self.articles_skipped,
            "articles_classified": self.articles_classified,
            "status": "success" if not self.errors else "partial_failure",
        }
        if self.duration_sec is not None:
            d["duration_sec"] = round(self.duration_sec, 3)
        if self.avg_uplifting_score is not None:
            d["avg_uplifting_score"] = round(self.avg_uplifting_score, 4)
        if self.classification_error_rate is not None:
            d["classification_error_rate"] = round(self.classification_error_rate, 4)
        if self.errors:
            d["errors"] = self.errors
        return d

    def to_db_kwargs(self) -> Dict[str, Any]:
        """Return kwargs suitable for ``SQLiteDB.record_crawl_metrics()``."""
        return {
            "crawl_start": self.start_time,
            "crawl_end": self.end_time or datetime.now(timezone.utc),
            "articles_fetched": self.articles_fetched,
            "articles_classified": self.articles_classified,
            "articles_stored": self.articles_stored,
            "avg_classification_score": self.avg_uplifting_score,
            "errors": "; ".join(self.errors) if self.errors else None,
        }


# ---------------------------------------------------------------------------
# Article-level event helper
# ---------------------------------------------------------------------------

def log_article_event(
    logger: logging.Logger,
    event: str,
    source_id: str,
    url: str,
    title: str = "",
    stage: str = "",
    uplifting_score: Optional[float] = None,
    duration_ms: Optional[float] = None,
    status: str = "success",
    **extra: Any,
) -> None:
    """Emit a structured log entry for a single article.

    Args:
        logger: The logger to use.
        event: Event name, e.g. ``"article_processed"``.
        source_id: Source identifier string.
        url: Article URL.
        title: Article title (optional).
        stage: Pipeline stage name (e.g. ``"classified"``, ``"stored"``).
        uplifting_score: Optional classification score.
        duration_ms: Optional processing duration in milliseconds.
        status: ``"success"`` or ``"error"``.
        **extra: Additional fields merged into the log record.
    """
    payload: Dict[str, Any] = {
        "event": event,
        "source_id": source_id,
        "url": url,
        "title": title[:120] if title else "",
        "stage": stage,
        "status": status,
    }
    if uplifting_score is not None:
        payload["uplifting_score"] = round(uplifting_score, 4)
    if duration_ms is not None:
        payload["duration_ms"] = round(duration_ms, 1)
    payload.update(extra)

    logger.info(event, extra=payload)
