# Meridian

**AI-native hedge fund: LLM agents orchestrate ML experts for systematic trading decisions.**

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  LLM Orchestrator               │
│          (Reasoning, Decision-Making)           │
├──────────┬──────────┬──────────┬────────────────┤
│ Momentum │ Value    │ Sentiment│ Risk           │
│ Expert   │ Expert   │ Expert   │ Expert         │
│ (ML)     │ (ML)     │ (ML)     │ (ML)           │
├──────────┴──────────┴──────────┴────────────────┤
│              Feature Engineering                │
├─────────────────────────────────────────────────┤
│              Market Data Pipeline               │  ← You are here
│         (Providers → Validation → DuckDB)       │
└─────────────────────────────────────────────────┘
```

## Quickstart

```bash
make setup                          # Install dependencies
python scripts/setup_db.py          # Initialize database
python scripts/ingest_historical.py --universe test --start-date 2024-01-01 --end-date 2024-12-31
```

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.11+ |
| Database | DuckDB (columnar, embedded) |
| Data Models | Pydantic v2 |
| Logging | structlog (JSON/pretty) |
| Market Data | yfinance, Alpha Vantage |
| Trading Calendar | exchange-calendars |
| Build | hatchling |

## Project Structure

```
meridian/
├── core/          # Exceptions, logging, types
├── config/        # Settings, constants, S&P 500 universe
├── data/
│   ├── providers/ # Market data APIs (Yahoo, Alpha Vantage)
│   ├── storage/   # DuckDB models and database layer
│   └── validation/# Data quality checks
├── agents/        # LLM orchestration (Sprint 2+)
├── experts/       # ML prediction models (Sprint 2+)
├── features/      # Feature engineering (Sprint 2+)
├── backtest/      # Backtesting engine (Sprint 3+)
├── strategy/      # Portfolio construction (Sprint 3+)
├── risk/          # Risk management (Sprint 3+)
├── gating/        # Paper → live trading gates (Sprint 5+)
├── monitoring/    # System observability (Sprint 4+)
└── dashboard/     # UI (Sprint 5+)
```

## Development

```bash
make lint       # ruff + mypy
make format     # Auto-fix lint issues
make test       # Run test suite
make coverage   # Tests + coverage report
make clean      # Remove caches
make db-reset   # Recreate database (prompts for confirmation)
```

## Sprint Roadmap

See [six-month-roadmap.md](six-month-roadmap.md) for the full plan targeting YC W27 (Oct 2026).
