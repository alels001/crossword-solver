.PHONY: install lint format typecheck test eval serve ui mcp docker

install:        ## dev install with all extras
	pip install -e ".[all,dev]"

lint:
	ruff check src tests eval tools

format:
	ruff format src tests eval tools && ruff check --fix src tests eval tools

typecheck:
	mypy

test:
	pytest --cov

eval:           ## accuracy / cost / latency report → eval/reports/
	python -m eval.run_eval

serve:
	crossword serve --reload

ui:
	crossword ui

mcp:
	crossword mcp

docker:
	docker compose up --build
