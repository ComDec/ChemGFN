# Makefile for ChemGFN Project
# Common commands for development workflow

.PHONY: help install install-hooks clean test lint format generate-templates check-templates

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
	@echo "Config Templates:"
	@echo "  make generate-templates      Generate all config templates"
	@echo "  make check-templates         Check if templates are up-to-date"
	@echo "  make list-templates          List template generation status"
	@echo "  make generate-template MODEL=<name>  Generate specific template"
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
	@echo "✅ Pre-commit hooks installed"

# Code quality
format:
	black chemgfn tests scripts --line-length 100
	isort chemgfn tests scripts --profile black

lint:
	flake8 chemgfn tests scripts --max-line-length 100

test:
	pytest tests/ -v

# Config template generation
generate-templates:
	@echo "🔄 Generating config templates..."
	python scripts/generate_config_templates.py
	@echo "✅ Done"

check-templates:
	@echo "🔍 Checking if templates are up-to-date..."
	python scripts/generate_config_templates.py --dry-run

list-templates:
	@echo "📋 Template Generation Status:"
	python scripts/generate_config_templates.py --list

generate-template:
	@if [ -z "$(MODEL)" ]; then \
		echo "❌ Error: MODEL variable not set"; \
		echo "Usage: make generate-template MODEL=llama3_expr24"; \
		exit 1; \
	fi
	@echo "🔄 Generating template for $(MODEL)..."
	python scripts/generate_config_templates.py --model $(MODEL)

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	@echo "✅ Cleaned up"

# Training shortcuts
train-expr24:
	python chemgfn/train.py experiment=expr24_split_loss

train-debug:
	python chemgfn/train.py experiment=debug_expr24

# Documentation
docs:
	@echo "📚 Documentation:"
	@echo "  CONFIG_TEMPLATE_HOOK.md - Template generation guide"
	@echo "  CONFIG_GUIDE.md - Hydra configuration guide"
	@echo "  README.md - Project overview"
