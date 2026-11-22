.PHONY: install test benchmark run docker-build docker-up docker-down clean lint

PYTHON ?= python

install:
	pip install --upgrade pip
	pip install -e ".[dev]"

test:
	pytest -v

benchmark:
	$(PYTHON) benchmarks/eval_hallucination_guard.py

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
