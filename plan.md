# Day 1: Project Scaffolding + Market Data Pipeline

## Context
Sprint 1, Day 1 of a 6-month build targeting YC W27 (Oct 2026). Building the data foundation that feeds every future component: LLM agents, 4 ML experts, backtesting, live paper trading, and monitoring.

## Implementation Phases

### Phase A: Project Infrastructure
- `.gitignore` — Python bytecache, venv, `.env`, `*.duckdb`, `data/`, IDE, OS files
- `pyproject.toml` — hatchling build, Python >=3.11, all deps, ruff/mypy/pytest config
- `.env.example` — Template with `MERIDIAN_` prefixed vars
- `Makefile` — setup, lint, format, test, coverage, clean, db-reset targets

### Phase B: Core Module (`meridian/core/`)
- `exceptions.py` — `MeridianError(message, **context)` hierarchy: DataProviderError, DataValidationError, StorageError, ConfigurationError, IngestionError
- `logging.py` — structlog config with dev (ConsoleRenderer) / prod (JSONRenderer) modes
- `types.py` — Ticker alias, DateRange, DataFrequency, MarketRegime, Serializable protocol

### Phase C: Configuration (`meridian/config/`)
- `settings.py` — Pydantic BaseSettings with nested models (Database, DataProvider, Ingestion, Logging), singleton pattern
- `constants.py` — Named constants with financial rationale comments
- `universe.py` — Hardcoded S&P 500 ticker->sector mapping (~485 entries), TEST_UNIVERSE (22 tickers), helper functions

### Phase D: Data Models (`meridian/data/storage/models.py`)
- `OHLCVBar` — Pydantic model with OHLCV validators (high>=low, volume>=0, etc.)
- `DataQualityReport` — Validation results with is_clean computed property
- `TickerMetadata` — Ticker info (sector, date range, bar count, provider)

### Phase E: Data Providers (`meridian/data/providers/`)
- `base.py` — Abstract DataProvider with rate limiting (time.monotonic) and retry (tenacity)
- `yahoo.py` — YahooFinanceProvider using yfinance.download() bulk fetch, multi-level column handling
- `alpha_vantage.py` — Stub implementing DataProvider interface

### Phase F: DuckDB Storage (`meridian/data/storage/database.py`)
- `MeridianDatabase` — Context manager, 4 tables (ohlcv, data_quality, ticker_metadata, ingestion_log)
- Upsert bars, store quality reports, update metadata, log ingestion
- Query helpers: get_data, get_universe_data, get_latest_date, get_summary, get_completed_tickers

### Phase G: Data Validation (`meridian/data/validation/quality.py`)
- NYSE calendar checks via exchange_calendars
- Price outlier detection (>50% single-day return)
- Volume anomalies (zero volume), split detection (>40% change), stale data (5+ same close), OHLC consistency

### Phase H: Ingestion Orchestrator (`meridian/data/ingest.py`)
- `ingest_historical` — Batch fetch -> validate -> store with resume support, fault tolerance, rich progress
- `ingest_incremental` — From latest date + 1 to today

### Phase I: Scripts
- `scripts/setup_db.py` — Create data dir, init DB, create schema, verify tables (idempotent)
- `scripts/ingest_historical.py` — CLI with argparse (--universe, --provider, --batch-size, --resume), exit codes 0/1/2

### Phase J: Tests (50 tests)
- `fixtures/sample_data.py` — GBM random walk generators, inject_gap/split/outlier
- `conftest.py` — Fixtures for bars, temp DB, mock provider
- `test_exceptions.py` — Context carrying, hierarchy, str representation
- `test_providers.py` — Mock yfinance, bulk parsing, empty DF, rate limiting, health check
- `test_validation.py` — Clean data, missing days, outliers, splits, stale, volume, edge cases
- `test_storage.py` — Schema, roundtrip, upsert dedup, queries, metadata, context manager
- `test_ingest.py` — Full pipeline, resume, partial failure, incremental

### Phase K: Documentation
- `README.md` — Architecture diagram, quickstart, tech stack, project structure
- `ARCHITECTURE.md` — Layer descriptions, data flow, design principles, current state
- `docs/decisions/001_database_choice.md` — ADR: DuckDB vs PostgreSQL vs SQLite

### Phase L: Notebook
- `notebooks/01_data_exploration.ipynb` — Template with DB connection, sample queries, viz placeholders

## Key Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Database | DuckDB | Columnar, embedded, zero-config, fast analytics |
| Retry logic | tenacity | Battle-tested backoff/jitter |
| Data models | Pydantic v2 | Validation at boundaries, fast serialization |
| Logging | structlog | Structured JSON for prod, pretty for dev |
| Provider pattern | Abstract base class | Swap providers without changing consumers |
| S&P 500 list | Hardcoded | More reliable than scraping, bias documented |
| Trading calendar | exchange-calendars | Authoritative NYSE schedule |
| Ticker type | Plain alias | Reduces friction with external libraries |
| Build backend | hatchling | Modern, minimal config |
