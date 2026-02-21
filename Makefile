.PHONY: help install dev test lint format type-check run run-dev docker-build docker-up docker-down clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt
	playwright install chromium

dev: ## Install all dependencies (production + dev tools)
	pip install -r requirements.txt
	pip install black ruff mypy pytest pytest-asyncio
	playwright install chromium

test: ## Run tests
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	pytest tests/ -v --tb=short --cov=. --cov-report=term-missing

lint: ## Run linter (Ruff)
	ruff check .

format: ## Format code (Black)
	black .

format-check: ## Check code formatting without changes
	black --check --diff .

type-check: ## Run type checker (mypy)
	mypy --ignore-missing-imports --no-strict-optional core/ detectors/ monitors/

run: ## Run the web dashboard (production)
	python app.py

run-dev: ## Run with auto-reload (development)
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

docker-build: ## Build Docker image
	docker build -t stake-downloader .

docker-up: ## Start with Docker Compose
	docker compose up -d

docker-down: ## Stop Docker Compose
	docker compose down

clean: ## Remove build artifacts and cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/
