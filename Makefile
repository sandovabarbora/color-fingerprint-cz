.PHONY: help install fetch extract color metrics render all test lint format clean
.DEFAULT_GOAL := help

PYTHON := uv run python

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Install dependencies (uv sync)
	uv sync

fetch:  ## Download spots from YouTube into data/raw/
	$(PYTHON) -m src.fetch

extract:  ## Extract frames at 1 fps into data/frames/
	$(PYTHON) -m src.extract

color:  ## Run K-means in LAB per frame → data/processed/frames.parquet
	$(PYTHON) -m src.color

metrics:  ## Aggregate per-brand metrics → data/processed/campaigns.parquet
	$(PYTHON) -m src.metrics

render:  ## Render the HTML report → outputs/index.html
	$(PYTHON) -m src.render

all: fetch extract color metrics render  ## Run the whole pipeline

test:  ## Run pytest
	uv run pytest

lint:  ## Lint with ruff
	uv run ruff check src tests

format:  ## Format code with ruff
	uv run ruff format src tests

clean:  ## Remove derived data and outputs (keeps data/raw/)
	rm -rf data/frames/* data/processed/* outputs/* pipeline.log
