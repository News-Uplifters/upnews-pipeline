"""Unit tests for pipeline/logging_config.py (Task 8)."""

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from io import StringIO
from unittest.mock import patch

import pytest

from pipeline.logging_config import (
    CrawlMetrics,
    JSONFormatter,
    get_pipeline_logger,
    log_article_event,
    setup_logging,
)


# ---------------------------------------------------------------------------
# JSONFormatter
# ---------------------------------------------------------------------------

class TestJSONFormatter:
    def _make_record(self, msg="test message", level=logging.INFO, **extra):
        record = logging.LogRecord(
            name="test.logger",
            level=level,
            pathname=__file__,
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )
        for k, v in extra.items():
            setattr(record, k, v)
        return record

    def test_output_is_valid_json(self):
        formatter = JSONFormatter()
        record = self._make_record("hello")
        line = formatter.format(record)
        parsed = json.loads(line)
        assert isinstance(parsed, dict)

    def test_required_fields_present(self):
        formatter = JSONFormatter()
        record = self._make_record("some message", level=logging.WARNING)
        parsed = json.loads(formatter.format(record))
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "message" in parsed
        assert parsed["level"] == "WARNING"
        assert parsed["message"] == "some message"

    def test_timestamp_is_utc_iso(self):
        formatter = JSONFormatter()
        record = self._make_record("ts test")
        parsed = json.loads(formatter.format(record))
        # Should end with 'Z'
        assert parsed["timestamp"].endswith("Z")
        # Should be parseable
        ts = parsed["timestamp"].rstrip("Z")
        datetime.fromisoformat(ts)  # raises if invalid

    def test_extra_fields_included(self):
        formatter = JSONFormatter()
        record = self._make_record("msg", event="article_processed", source_id="BBC")
        parsed = json.loads(formatter.format(record))
        assert parsed.get("event") == "article_processed"
        assert parsed.get("source_id") == "BBC"

    def test_exception_info_included(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="error occurred",
            args=(),
            exc_info=exc_info,
        )
        parsed = json.loads(formatter.format(record))
        assert "exc_info" in parsed
        assert "ValueError" in parsed["exc_info"]

    def test_single_line_output(self):
        """Each log record must be exactly one line (no embedded newlines)."""
        formatter = JSONFormatter()
        record = self._make_record("line\nbreak")
        line = formatter.format(record)
        # json.dumps should escape newlines inside strings
        assert "\n" not in line


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_creates_log_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "nested", "logs")
            setup_logging(log_dir=log_dir)
            assert os.path.isdir(log_dir)

    def test_creates_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(log_dir=tmpdir, log_file="test_pipeline.log")
            log = logging.getLogger()
            log.info("setup test")
            log_path = os.path.join(tmpdir, "test_pipeline.log")
            assert os.path.isfile(log_path)

    def test_log_file_contains_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(log_dir=tmpdir, log_file="json_test.log")
            log = logging.getLogger("test_json_write")
            log.info("json write test")
            log_path = os.path.join(tmpdir, "json_test.log")
            with open(log_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            # At least one line should be valid JSON with our message
            found = False
            for line in lines:
                try:
                    parsed = json.loads(line)
                    if parsed.get("message") == "json write test":
                        found = True
                        break
                except json.JSONDecodeError:
                    pass
            assert found, "Expected to find our log message in the JSON log file"

    def test_no_duplicate_handlers_on_repeated_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(log_dir=tmpdir)
            setup_logging(log_dir=tmpdir)
            root = logging.getLogger()
            assert len(root.handlers) == 2  # file + stdout

    def test_json_stdout_flag(self):
        """When json_stdout=True the stdout handler should also use JSONFormatter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(log_dir=tmpdir, json_stdout=True)
            root = logging.getLogger()
            stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)
                                and not isinstance(h, logging.FileHandler)]
            assert any(isinstance(h.formatter, JSONFormatter) for h in stream_handlers)


# ---------------------------------------------------------------------------
# CrawlMetrics
# ---------------------------------------------------------------------------

class TestCrawlMetrics:
    def test_start_sets_start_time(self):
        before = datetime.now(timezone.utc)
        m = CrawlMetrics.start()
        after = datetime.now(timezone.utc)
        assert m.start_time is not None
        assert before <= m.start_time <= after

    def test_finish_sets_end_time(self):
        m = CrawlMetrics.start()
        m.finish()
        assert m.end_time is not None

    def test_duration_sec_calculated(self):
        m = CrawlMetrics.start()
        time.sleep(0.05)
        m.finish()
        assert m.duration_sec is not None
        assert m.duration_sec >= 0.04

    def test_duration_sec_none_without_finish(self):
        m = CrawlMetrics.start()
        assert m.duration_sec is None

    def test_crawl_id_unique(self):
        ids = {CrawlMetrics.start().crawl_id for _ in range(5)}
        assert len(ids) == 5

    def test_record_error(self):
        m = CrawlMetrics.start()
        m.record_error("db_write_failed: timeout")
        m.record_error("classify_error: model not found")
        assert len(m.errors) == 2

    def test_stage_timers(self):
        m = CrawlMetrics.start()
        m.start_stage("crawl")
        time.sleep(0.02)
        elapsed = m.end_stage("crawl")
        assert elapsed >= 15  # at least 15ms

    def test_end_stage_unknown_returns_zero(self):
        m = CrawlMetrics.start()
        assert m.end_stage("nonexistent") == 0.0

    def test_to_log_dict_success(self):
        m = CrawlMetrics.start()
        m.articles_fetched = 100
        m.articles_new = 80
        m.articles_stored = 75
        m.articles_skipped = 20
        m.articles_classified = 80
        m.sources_processed = 5
        m.finish()
        d = m.to_log_dict()
        assert d["event"] == "crawl_complete"
        assert d["articles_fetched"] == 100
        assert d["articles_stored"] == 75
        assert d["status"] == "success"
        assert "duration_sec" in d

    def test_to_log_dict_partial_failure(self):
        m = CrawlMetrics.start()
        m.record_error("something went wrong")
        m.finish()
        d = m.to_log_dict()
        assert d["status"] == "partial_failure"
        assert "errors" in d

    def test_to_log_dict_valid_json_serialisable(self):
        m = CrawlMetrics.start()
        m.avg_uplifting_score = 0.823456
        m.finish()
        d = m.to_log_dict()
        serialised = json.dumps(d)
        parsed = json.loads(serialised)
        assert parsed["avg_uplifting_score"] == pytest.approx(0.8235, abs=0.001)

    def test_to_db_kwargs_keys(self):
        m = CrawlMetrics.start()
        m.articles_fetched = 50
        m.articles_classified = 40
        m.articles_stored = 38
        m.finish()
        kwargs = m.to_db_kwargs()
        expected_keys = {
            "crawl_start", "crawl_end", "articles_fetched",
            "articles_classified", "articles_stored",
            "avg_classification_score", "errors",
        }
        assert set(kwargs.keys()) == expected_keys

    def test_to_db_kwargs_errors_joined(self):
        m = CrawlMetrics.start()
        m.record_error("error_one")
        m.record_error("error_two")
        m.finish()
        kwargs = m.to_db_kwargs()
        assert "error_one" in kwargs["errors"]
        assert "error_two" in kwargs["errors"]

    def test_to_db_kwargs_no_errors_is_none(self):
        m = CrawlMetrics.start()
        m.finish()
        assert m.to_db_kwargs()["errors"] is None


# ---------------------------------------------------------------------------
# log_article_event
# ---------------------------------------------------------------------------

class TestLogArticleEvent:
    def _capture_logger(self, name="test_article_event"):
        """Return (logger, StringIO stream) with JSON formatter."""
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)
        log.handlers.clear()
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        log.addHandler(handler)
        log.propagate = False
        return log, stream

    def test_emits_json_with_required_fields(self):
        log, stream = self._capture_logger()
        log_article_event(
            log,
            event="article_processed",
            source_id="BBCNews",
            url="https://bbc.com/news/test",
            title="Test article",
            stage="classified",
            uplifting_score=0.87,
            duration_ms=245.3,
        )
        line = stream.getvalue().strip()
        parsed = json.loads(line)
        assert parsed["event"] == "article_processed"
        assert parsed["source_id"] == "BBCNews"
        assert parsed["url"] == "https://bbc.com/news/test"
        assert parsed["stage"] == "classified"
        assert parsed["uplifting_score"] == pytest.approx(0.87)
        assert parsed["duration_ms"] == pytest.approx(245.3, abs=0.1)
        assert parsed["status"] == "success"

    def test_title_truncated_to_120_chars(self):
        log, stream = self._capture_logger("test_truncate")
        long_title = "A" * 200
        log_article_event(log, event="article_processed", source_id="src", url="http://x.com", title=long_title)
        parsed = json.loads(stream.getvalue().strip())
        assert len(parsed["title"]) == 120

    def test_optional_fields_omitted_when_none(self):
        log, stream = self._capture_logger("test_optional")
        log_article_event(log, event="article_processed", source_id="src", url="http://x.com")
        parsed = json.loads(stream.getvalue().strip())
        assert "uplifting_score" not in parsed
        assert "duration_ms" not in parsed

    def test_extra_kwargs_included(self):
        log, stream = self._capture_logger("test_extra")
        log_article_event(
            log, event="article_processed", source_id="src", url="http://x.com",
            custom_field="custom_value"
        )
        parsed = json.loads(stream.getvalue().strip())
        assert parsed.get("custom_field") == "custom_value"

    def test_error_status(self):
        log, stream = self._capture_logger("test_error_status")
        log_article_event(
            log, event="article_error", source_id="src", url="http://x.com",
            status="error"
        )
        parsed = json.loads(stream.getvalue().strip())
        assert parsed["status"] == "error"


# ---------------------------------------------------------------------------
# get_pipeline_logger
# ---------------------------------------------------------------------------

class TestGetPipelineLogger:
    def test_returns_logger_instance(self):
        log = get_pipeline_logger()
        assert isinstance(log, logging.Logger)

    def test_default_namespace(self):
        log = get_pipeline_logger()
        assert log.name == "upnews.pipeline"

    def test_custom_name(self):
        log = get_pipeline_logger("upnews.crawler")
        assert log.name == "upnews.crawler"
