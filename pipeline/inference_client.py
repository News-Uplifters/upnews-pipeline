"""HTTP client for the upnews-inference microservice.

When INFERENCE_SERVICE_URL is set, all classify/categorize/summarize calls
are delegated to the running inference service. When it is absent or the
service is unreachable, the client falls back to the local implementations
(rule-based classifier, direct categorizer/summarizer imports).

Usage in run_pipeline.py:
    with InferenceClient.from_env() as client:
        results = client.classify(articles, threshold=0.75)
        categorized = client.categorize(uplifting_articles)
        summaries = client.summarize(texts)
"""

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_CLASSIFY_PATH = "/v1/classify"
_CATEGORIZE_PATH = "/v1/categorize"
_SUMMARIZE_PATH = "/v1/summarize"


class InferenceServiceError(RuntimeError):
    """Raised when the inference service returns an unexpected error."""


class InferenceClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/") if base_url else None
        self._timeout = timeout
        self._session: Optional[requests.Session] = None

    @classmethod
    def from_env(cls) -> "InferenceClient":
        url = os.environ.get("INFERENCE_SERVICE_URL", "").strip() or None
        timeout = float(os.environ.get("INFERENCE_TIMEOUT_SEC", "30"))
        return cls(base_url=url, timeout=timeout)

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "InferenceClient":
        if self._base_url:
            self._session = requests.Session()
        return self

    def __exit__(self, *args) -> None:
        if self._session:
            self._session.close()
            self._session = None

    # ── public interface ─────────────────────────────────────────────────────

    def classify(
        self,
        articles: list[dict],
        threshold: float = 0.75,
    ) -> list[dict]:
        """Score articles for upliftingness.

        Each dict in *articles* must have 'title'; may have 'source_id'.
        Returns the input list enriched with 'uplifting_score', 'is_uplifting',
        and 'min_threshold' — the same fields the pipeline already expects.
        """
        if not self._base_url:
            return self._local_classify(articles, threshold)

        payload = {
            "articles": [
                {"title": a.get("title", ""), "source_id": a.get("source_id")}
                for a in articles
            ],
            "threshold": threshold,
        }
        try:
            data = self._post(_CLASSIFY_PATH, payload)
        except InferenceServiceError as exc:
            logger.warning("Inference service unavailable (%s); falling back to local classifier.", exc)
            return self._local_classify(articles, threshold)

        enriched = []
        for article, result in zip(articles, data["results"]):
            enriched_article = dict(article)
            enriched_article["uplifting_score"] = result["uplifting_score"]
            enriched_article["is_uplifting"] = result["is_uplifting"]
            enriched_article["min_threshold"] = result["threshold_used"]
            enriched.append(enriched_article)
        return enriched

    def categorize(
        self,
        articles: list[dict],
        categories: Optional[list[str]] = None,
    ) -> list[dict]:
        """Assign topic categories to articles.

        Each dict must have 'title'; may have 'body'.
        Returns the input list enriched with 'category', 'category_scores',
        'category_confidence'.
        """
        if not self._base_url:
            return self._local_categorize(articles, categories)

        payload: dict = {
            "articles": [
                {"title": a.get("title", ""), "body": a.get("body", "")}
                for a in articles
            ]
        }
        if categories:
            payload["categories"] = categories

        try:
            data = self._post(_CATEGORIZE_PATH, payload)
        except InferenceServiceError as exc:
            logger.warning("Inference service unavailable (%s); falling back to local categorizer.", exc)
            return self._local_categorize(articles, categories)

        enriched = []
        for article, result in zip(articles, data["results"]):
            enriched_article = dict(article)
            enriched_article["category"] = result["category"]
            enriched_article["category_scores"] = result["scores"]
            enriched_article["category_confidence"] = result["confidence"]
            enriched.append(enriched_article)
        return enriched

    def summarize(
        self,
        articles: list[dict],
        max_length: int = 130,
        min_length: int = 30,
    ) -> list[str]:
        """Generate summaries for articles.

        Each dict must have 'body' or 'title' as the text source.
        Returns a list of summary strings in the same order as input.
        """
        texts = [a.get("body") or a.get("title", "") for a in articles]

        if not self._base_url:
            return self._local_summarize(texts, max_length, min_length)

        payload = {
            "articles": [{"text": t} for t in texts if t],
            "max_length": max_length,
            "min_length": min_length,
        }
        try:
            data = self._post(_SUMMARIZE_PATH, payload)
        except InferenceServiceError as exc:
            logger.warning("Inference service unavailable (%s); falling back to local summarizer.", exc)
            return self._local_summarize(texts, max_length, min_length)

        return [r["summary"] for r in data["results"]]

    # ── internal HTTP ─────────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        url = self._base_url + path
        try:
            session = self._session or requests
            response = session.post(url, json=payload, timeout=self._timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as exc:
            raise InferenceServiceError(f"connection error to {url}: {exc}") from exc
        except requests.exceptions.Timeout as exc:
            raise InferenceServiceError(f"timeout calling {url}") from exc
        except requests.exceptions.HTTPError as exc:
            raise InferenceServiceError(f"HTTP {exc.response.status_code} from {url}") from exc

    # ── local fallbacks ───────────────────────────────────────────────────────

    def _local_classify(self, articles: list[dict], threshold: float) -> list[dict]:
        """Fall back to rule-based or local SetFit model."""
        import pandas as pd
        from classifier.classify_headlines import load_model, score_news

        df = pd.DataFrame(articles)
        model = load_model()
        scored = score_news(df, model, threshold=threshold)
        return scored.to_dict(orient="records")

    def _local_categorize(
        self, articles: list[dict], categories: Optional[list[str]]
    ) -> list[dict]:
        """Fall back to local categorizer."""
        from enrichment.categorizer import categorize_batch

        categorized = categorize_batch(articles, categories=categories)
        # Rename keys to match what the service returns
        for a in categorized:
            if "category_scores" not in a and "scores" in a:
                a["category_scores"] = a.pop("scores")
            if "category_confidence" not in a and "confidence" in a:
                a["category_confidence"] = a.pop("confidence")
        return categorized

    def _local_summarize(
        self, texts: list[str], max_length: int, min_length: int
    ) -> list[str]:
        """Fall back to local DistilBART summarizer or title passthrough."""
        method = os.environ.get("SUMMARIZATION_METHOD", "distilbart").strip().lower()
        if method in {"none", "skip", "off"}:
            return ["" for _ in texts]
        if method in {"title", "headline"}:
            return list(texts)
        try:
            from pipeline.summarizer import summarize

            return [summarize(t, max_length=max_length, min_length=min_length) if t else "" for t in texts]
        except Exception as exc:
            logger.warning("Local summarizer failed: %s", exc)
            return list(texts)
