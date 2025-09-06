.PHONY: help install install-dev test test-cov lint format clean demo

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install with development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=gaze_lab --cov-report=html --cov-report=term

lint:  ## Run linting
	flake8 gaze_lab tests
	mypy gaze_lab

format:  ## Format code
	black gaze_lab tests scripts
	isort gaze_lab tests scripts

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

demo:  ## Generate synthetic demo data
	python -m scripts.make_synthetic_demo

demo-full: demo  ## Run full demo pipeline
	@echo "Running full demo pipeline..."
	gaze-overlay --world data/world.mp4 --gaze data/gaze.csv --out outputs/overlay.mp4 --dot-radius 10 --trail 12 --show-fixations
	gaze-aoi --gaze data/gaze.csv --aoi examples/aoi_example.json --report outputs/aoi_report.csv --timeline outputs/aoi_timeline.csv
	gaze-heatmap --gaze data/gaze.csv --out outputs/heatmap.png --mode world --background-frame 0
	@echo "Demo complete! Check outputs/ directory for results."

setup-outputs:  ## Create outputs directory
	mkdir -p outputs
