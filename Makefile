.PHONY: setup lint format test test-features test-lookahead coverage clean db-reset

setup:
	pip install -e ".[dev]"

lint:
	ruff check meridian/ tests/
	mypy meridian/

format:
	ruff check --fix meridian/ tests/
	ruff format meridian/ tests/

test:
	pytest -v

test-features:
	pytest tests/test_features/ -v

test-lookahead:
	pytest tests/test_features/test_lookahead.py -v --tb=long

coverage:
	pytest --cov=meridian --cov-report=term-missing --cov-report=html

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage dist/ build/ *.egg-info

db-reset:
	@echo "WARNING: This will delete the database and all ingested data."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -f data/meridian.duckdb data/meridian.duckdb.wal
	python scripts/setup_db.py
