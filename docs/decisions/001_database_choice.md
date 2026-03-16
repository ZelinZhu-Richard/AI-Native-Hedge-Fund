# ADR 001: Database Choice

## Status
Accepted

## Context
Meridian needs a database for storing time-series OHLCV market data. Requirements:
- Fast columnar analytics (OHLCV queries across tickers and date ranges)
- Embedded (no server setup for development or CI)
- Good pandas/DataFrame integration
- Handles millions of rows (500 tickers × 20 years × 252 days ≈ 2.5M rows)

## Options Considered

### PostgreSQL
- Pros: Mature, full SQL, concurrent access, extensions (TimescaleDB)
- Cons: Requires server setup, overkill for single-user analytics, slower columnar queries without TimescaleDB

### SQLite
- Pros: Embedded, zero-config, battle-tested, single-file
- Cons: Row-oriented (slow for analytics), poor DataFrame integration, no columnar compression

### DuckDB
- Pros: Columnar (fast analytics), embedded, zero-config, native pandas integration, SQL support, excellent compression
- Cons: Single-writer limitation, smaller ecosystem, newer project

## Decision
**DuckDB** — columnar storage is ideal for time-series analytics. Embedded deployment eliminates infrastructure overhead. Native pandas integration simplifies the data pipeline.

## Consequences
- **Single-writer constraint**: Only one process can write at a time. Analysis notebooks must use `read_only=True`.
- **Smaller ecosystem**: Fewer ORMs, migration tools, and community resources compared to PostgreSQL.
- **Future migration**: If we need concurrent writes (e.g., multiple ingestion workers), we'll need to either serialize writes or migrate to PostgreSQL/TimescaleDB.
