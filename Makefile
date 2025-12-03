# Makefile for Santa Hat API
# Run tests and other development tasks

.PHONY: help test test-unit test-debug test-file test-coverage test-watch build-test clean lint install install-dev install-deps venv

define HELP_TEXT
Santa Hat API - Development Commands

Test Commands (run in Docker):
  make test          - Run all tests with coverage
  make test-unit     - Run unit tests only (no integration tests)
  make test-debug    - Run tests with verbose output, stop on first failure
  make test-file     - Run specific test file (TEST_FILE=tests/test_config.py)
  make test-watch    - Run tests in watch mode (re-run on file changes)
  make test-coverage - Run tests and generate HTML coverage report

Local Test Commands (no Docker):
  make test-local    - Run tests locally (requires dependencies installed)

Build Commands:
  make build-test    - Build test Docker image
  make build-app     - Build production Docker image

Development Commands:
  make install       - Install all dependencies (app + test)
  make install-dev   - Same as install (alias)
  make install-deps  - Install dependencies including mediapipe
  make venv          - Create virtual environment and install dependencies
  make run           - Run the application locally
  make run-docker    - Run the application in Docker
  make clean         - Remove build artifacts and coverage files
  make lint          - Run linting checks (if configured)
endef

# Default target
help:
	$(info $(HELP_TEXT))
	@:

# ============================================================================
# Docker Test Commands
# ============================================================================

# Build the test Docker image
build-test:
	docker-compose -f docker-compose.test.yml build test

# Run all tests with coverage report
test: build-test
	docker-compose -f docker-compose.test.yml run --rm test

# Run unit tests only
test-unit: build-test
	docker-compose -f docker-compose.test.yml run --rm test-unit

# Run tests with debug output (stop on first failure)
test-debug: build-test
	docker-compose -f docker-compose.test.yml run --rm test-debug

# Run a specific test file
# Usage: make test-file TEST_FILE=tests/test_config.py
test-file: build-test
	@if [ -z "$(TEST_FILE)" ]; then \
		echo "Usage: make test-file TEST_FILE=tests/test_config.py"; \
		exit 1; \
	fi
	TEST_FILE=$(TEST_FILE) docker-compose -f docker-compose.test.yml run --rm test-file

# Run tests and generate HTML coverage report
test-coverage: build-test
	docker-compose -f docker-compose.test.yml run --rm test
	@echo ""
	@echo "Coverage report generated at: coverage_html/index.html"

# Run tests in watch mode (re-run on file changes)
test-watch: build-test
	docker-compose -f docker-compose.test.yml run --rm test-watch

# ============================================================================
# Local Test Commands (without Docker)
# ============================================================================

# Run tests locally (requires virtual environment with dependencies)
test-local:
	PYTHONPATH=. pytest tests/ -v --cov=app --cov-report=term-missing

# Run specific test locally
test-local-file:
	@if [ -z "$(TEST_FILE)" ]; then \
		echo "Usage: make test-local-file TEST_FILE=tests/test_config.py"; \
		exit 1; \
	fi
	PYTHONPATH=. pytest $(TEST_FILE) -v --tb=short

# ============================================================================
# Build Commands
# ============================================================================

# Build production Docker image
build-app:
	docker-compose build santa-hat-api

# ============================================================================
# Development Commands
# ============================================================================

# Install all dependencies (app + test)
install:
	pip install -r requirements.txt -r requirements-test.txt

# Alias for install
install-dev: install

# Install dependencies including mediapipe
install-deps:
	python -m pip install -r requirements.txt -r requirements-test.txt
	python -m pip install mediapipe

# Create virtual environment and install all dependencies
venv:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt -r requirements-test.txt
	@echo ""
	@echo "Virtual environment created. Activate with:"
	@echo "  source .venv/bin/activate"

# Run application locally
run:
	PYTHONPATH=. uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run application in Docker
run-docker:
	docker-compose up --build

# Stop all Docker containers
stop:
	docker-compose down
	docker-compose -f docker-compose.test.yml down

# Clean up build artifacts, coverage files, and Docker images
clean:
	# Remove Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	# Remove coverage reports
	rm -rf coverage_html/ htmlcov/ 2>/dev/null || true
	# Remove Docker test images
	docker-compose -f docker-compose.test.yml down --rmi local 2>/dev/null || true
	@echo "Cleaned up build artifacts"

# Lint checks (placeholder - add your preferred linter)
lint:
	@echo "Running lint checks..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check app/ tests/; \
	elif command -v flake8 >/dev/null 2>&1; then \
		flake8 app/ tests/; \
	else \
		echo "No linter found. Install ruff or flake8 for linting."; \
	fi

# ============================================================================
# CI/CD Commands
# ============================================================================

# Run tests for CI (exit with proper code)
ci-test:
	docker-compose -f docker-compose.test.yml run --rm test pytest tests/ -v --cov=app --cov-report=xml --cov-fail-under=70

# Quick sanity check
smoke-test: build-test
	docker-compose -f docker-compose.test.yml run --rm test pytest tests/test_config.py tests/test_main.py::TestRootEndpoint -v
