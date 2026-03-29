FROM python:3.11

WORKDIR /app

# System dependencies required for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY config/ config/
COPY crawler/ crawler/
COPY classifier/ classifier/
COPY enrichment/ enrichment/
COPY pipeline/ pipeline/

# Writable directories for DB and logs
RUN mkdir -p data logs

# No exposed ports — this is a batch job, not a server
CMD ["python", "-m", "pipeline.run_pipeline"]
