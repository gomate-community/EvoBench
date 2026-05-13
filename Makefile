.PHONY: install init sample eval api test docker

install:
	pip install -e ".[dev]"

init:
	python -m benchmark.cli init-db

sample:
	python -m benchmark.cli ingest-sample --topic AI --limit 5

eval:
	python -m benchmark.cli run-eval --model-id demo_model

api:
	uvicorn benchmark.api.main:app --reload

test:
	python -m pytest -q

docker:
	docker compose up --build
