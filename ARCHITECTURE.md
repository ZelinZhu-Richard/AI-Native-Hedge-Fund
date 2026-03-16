# Meridian Architecture

## Layers

### 1. Market Data Pipeline (Sprint 1)
Ingests, validates, and stores OHLCV price data from multiple providers into DuckDB. Handles rate limiting, retries, resume, and data quality validation against the NYSE trading calendar.

### 2. Feature Engineering (Sprint 2)
Transforms raw OHLCV data into signals: technical indicators, fundamental ratios, sentiment scores, regime features.

### 3. ML Expert Models (Sprint 2-3)
Four specialized models — Momentum, Value, Sentiment, Risk — each trained on domain-specific features. Experts predict, they don't decide.

### 4. LLM Agent Orchestrator (Sprint 2-3)
Claude-based agents reason about expert outputs, market context, and risk constraints to make portfolio decisions. Agents reason, they don't predict.

### 5. Backtesting Engine (Sprint 3)
Event-driven backtester with realistic transaction costs, slippage, and position sizing.

### 6. Live Trading (Sprint 4-5)
Paper trading first, then live with circuit breakers and position limits.

## Data Flow

```
Yahoo/AV API → Provider → Validator → DuckDB
                                         ↓
                              Feature Engineering
                                         ↓
                              Expert ML Models
                                         ↓
                              LLM Agent Reasoning
                                         ↓
                              Portfolio Decisions
                                         ↓
                              Execution / Paper Trade
```

## Design Principles

1. **Single responsibility**: LLM agents reason about strategy; ML models predict market variables. Never mix these roles.
2. **Circuit breakers are hardcoded**: Position limits, max drawdown, and exposure caps are code constants, not LLM-adjustable parameters.
3. **Every decision is logged**: Every trade signal, expert prediction, and agent decision is persisted with full context for auditability.
4. **Graceful degradation**: If a data provider fails, use cached data. If an expert model errors, exclude it and proceed with remaining experts. Never halt on a single component failure.
5. **Validation at boundaries**: Pydantic models validate all data entering and leaving the system. Internal code trusts validated types.

## Current State

Sprint 1, Day 1: Market data pipeline operational. DuckDB storage, Yahoo Finance provider, NYSE calendar validation, data quality checks. ~20 test-universe tickers, expandable to full S&P 500.
