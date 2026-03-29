.PHONY: install run test lint format docker-build docker-run clean

VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

## Local dev (venv)
install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_sm

run:
	$(PYTHON) -m pipeline.run_pipeline

test:
	$(VENV)/bin/pytest -v --cov=. --cov-report=term-missing

lint:
	$(VENV)/bin/flake8 crawler/ classifier/ enrichment/ pipeline/
	$(VENV)/bin/mypy . --ignore-missing-imports

format:
	$(VENV)/bin/black .
	$(VENV)/bin/isort .

## Docker
docker-build:
	docker build -t upnews-pipeline:local .

docker-run:
	docker run --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/logs:/app/logs \
		--env-file .env \
		upnews-pipeline:local

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .mypy_cache htmlcov .coverage
