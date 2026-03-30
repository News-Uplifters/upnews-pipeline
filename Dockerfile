# syntax=docker/dockerfile:1
# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System build deps (needed for lxml, aiohttp, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime system libs (lxml needs libxml2/libxslt at runtime too)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create writable directories for runtime data
RUN mkdir -p data logs models && \
    chmod -R 777 data logs

# HuggingFace model cache goes inside the container's home dir by default;
# expose it as a volume so models persist across runs.
ENV TRANSFORMERS_CACHE=/app/models/hf_cache
ENV HF_HOME=/app/models/hf_cache

# Default environment — override with docker-compose or --env-file
ENV LOG_LEVEL=INFO \
    DATABASE_PATH=/app/data/articles.db \
    ARTICLES_LIMIT_PER_SOURCE=50 \
    CLASSIFICATION_THRESHOLD=0.75 \
    THUMBNAIL_CONCURRENCY=10 \
    THUMBNAIL_TIMEOUT=5 \
    SUMMARIZATION_METHOD=distilbart \
    CATEGORIZATION_METHOD=zero-shot \
    # Use rule-based classifier by default so Docker works without a
    # pre-trained SetFit model.  Set CLASSIFIER_MODE=setfit and mount
    # models/setfit_uplifting_model/ to use the ML model instead.
    CLASSIFIER_MODE=rules

# Expose the data directory as a volume so the SQLite DB is persisted
VOLUME ["/app/data", "/app/logs", "/app/models"]

ENTRYPOINT ["python", "-m", "pipeline.run_pipeline"]
