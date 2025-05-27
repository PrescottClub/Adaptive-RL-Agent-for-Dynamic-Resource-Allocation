.PHONY: help install install-dev test test-cov lint format clean train evaluate

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install .

install-dev:  ## Install the package in development mode with all dependencies
	pip install -e ".[dev,notebooks]"

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=src --cov-report=html --cov-report=term

lint:  ## Run linting
	flake8 src tests
	mypy src

format:  ## Format code
	black src tests main_train.py main_evaluate.py
	isort src tests main_train.py main_evaluate.py

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

train:  ## Run training with default parameters
	python main_train.py --agent dqn --episodes 1000

evaluate:  ## Run evaluation
	python main_evaluate.py --mode single --agent dqn

setup-env:  ## Set up development environment
	python -m venv venv
	@echo "Activate the environment with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)" 