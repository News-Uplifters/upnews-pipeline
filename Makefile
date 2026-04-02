.PHONY: local dev pr qa prod run-local run-dev run-pr run-qa run-prod build-local build-dev build-pr build-qa build-prod logs shell db help

# ─── Environment shortcuts ───────────────────────────────────────────────────
# Build + start (keeps container running; use Ctrl-C to stop)
local:
	docker compose --env-file .env.local up --build

dev:
	docker compose --env-file .env.dev up --build

pr:
	docker compose --env-file .env.pr up --build

qa:
	docker compose --env-file .env.qa up --build

prod:
	docker compose --env-file .env.prod up --build -d

# ─── One-shot run (build → run → remove container) ───────────────────────────
run-local:
	docker compose --env-file .env.local run --rm pipeline

run-dev:
	docker compose --env-file .env.dev run --rm pipeline

run-pr:
	docker compose --env-file .env.pr run --rm pipeline

run-qa:
	docker compose --env-file .env.qa run --rm pipeline

run-prod:
	docker compose --env-file .env.prod run --rm pipeline

# ─── Build only (no run) ─────────────────────────────────────────────────────
build-local:
	docker compose --env-file .env.local build

build-dev:
	docker compose --env-file .env.dev build

build-pr:
	docker compose --env-file .env.pr build

build-qa:
	docker compose --env-file .env.qa build

build-prod:
	docker compose --env-file .env.prod build

# ─── Utilities ───────────────────────────────────────────────────────────────
logs:
	docker compose logs -f

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

shell:
	docker compose run --rm --entrypoint bash pipeline

db:
	sqlite3 data/articles.db "SELECT id, title, uplifting_score, category, crawled_at FROM articles ORDER BY crawled_at DESC LIMIT 20;"

stop:
	docker compose down

help:
	@echo ""
	@echo "upnews-pipeline  –  Makefile targets"
	@echo ""
	@echo "  Run environments (build + start):"
	@echo "    make local      — DEBUG logging, 10 articles/source, rules classifier"
	@echo "    make dev        — DEBUG logging, 25 articles/source, rules classifier"
	@echo "    make pr         — INFO  logging, 50 articles/source, rules classifier"
	@echo "    make qa         — INFO  logging, 50 articles/source, SetFit classifier"
	@echo "    make prod       — WARN  logging, 100 articles/source, SetFit classifier (detached)"
	@echo ""
	@echo "  One-shot runs (run then remove container):"
	@echo "    make run-local | run-dev | run-pr | run-qa | run-prod"
	@echo ""
	@echo "  Build only:"
	@echo "    make build-local | build-dev | build-pr | build-qa | build-prod"
	@echo ""
	@echo "  Utilities:"
	@echo "    make logs       — tail container logs"
	@echo "    make shell      — open bash inside container"
	@echo "    make db         — query last 20 articles from SQLite"
	@echo "    make stop       — stop and remove containers"
	@echo ""
