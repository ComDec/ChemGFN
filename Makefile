# Makefile for ChemGFN Project
# Common commands for development workflow

.PHONY: help install install-hooks clean test lint format

# Default target
help:
	@echo "ChemGFN Development Commands"
	@echo "============================"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install project dependencies"
	@echo "  make install-hooks    Install pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make format           Format code with black and isort"
	@echo "  make lint             Run linters (flake8)"
	@echo "  make test             Run tests"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove generated files and caches"
	@echo ""

# Installation
install:
	pip install -e .
	pip install -r requirements.txt

install-hooks:
	pip install pre-commit
	pre-commit install
	@echo "Pre-commit hooks installed"

# Code quality
format:
	black chemgfn tests scripts --line-length 100
	isort chemgfn tests scripts --profile black

lint:
	flake8 chemgfn tests scripts --max-line-length 100

test:
	pytest tests/ -v

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	@echo "Cleaned up"

# Documentation
docs:
	@echo "Documentation:"
	@echo "  README.md - Project overview"
