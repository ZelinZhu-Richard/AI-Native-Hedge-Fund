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

### 1b. Feature Engineering (Sprint 1, Day 2)
49 features across 4 categories: technical (25), volatility (10), cross-sectional (8), macro (6). All features have anti-lookahead guarantees enforced via truncation-invariance tests.

### 1c. Regime Detection (Sprint 1, Day 3)
Rolling PCA compresses 49 features into principal components. HMM and KMeans cluster components into market regimes (e.g., crisis, steady bull). Anti-lookahead enforced: StandardScaler + PCA fit within each rolling window, never on future data.

## Data Flow

```
Yahoo/AV API → Provider → Validator → DuckDB
                                         ↓
                              Feature Engineering (49 features)
                                         ↓
                              Rolling PCA (anti-lookahead)
                                         ↓
                              Clustering (HMM / KMeans)
                                         ↓
                              RegimeResult → Gating Network
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

## Regime Detection Pipeline

The regime detection pipeline is the foundation for the Mixture-of-Experts gating network (Day 11). The fund thesis — "markets misprice regime transitions" — requires detecting regimes before exploiting them.

**Key components:**
- `RollingPCA`: Compresses 49 features into ~10 principal components using a 252-day rolling window. Refits monthly (21 trading days). Anti-lookahead enforced via per-window StandardScaler + PCA fitting.
- `MarketPCA`: Cross-sectional PCA on ticker returns to extract statistical factors.
- `HMMRegimeDetector`: Gaussian HMM learns hidden regime states from PCA components. Captures temporal dynamics (transition probabilities).
- `KMeansRegimeDetector`: Simpler clustering fallback. No temporal dynamics but more robust.
- `RegimeDetector`: Unified pipeline orchestrating PCA → clustering → `RegimeResult`.
- `RegimeResult`: Pydantic model with labels, probabilities, transitions, stats. JSON-serializable.

**Design decisions:**
| Decision | Choice | Rationale |
|---|---|---|
| Rolling vs expanding PCA | Rolling (252d) | Adapts to structural breaks |
| Scaler scope | Per rolling window | Anti-lookahead guarantee |
| HMM + KMeans | Both | HMM for temporal dynamics, KMeans as stable fallback |
| Regime storage | In-memory only | Derived from features in DuckDB; re-deriving is cheap |

### 1d. Backtesting Framework (Sprint 1, Day 4)
Walk-forward backtesting engine with strict anti-lookahead guarantees. 504-day rolling train window, 63-day test window, 63-day step. SignalGenerator protocol decouples model implementation from backtesting infrastructure.

**Key components:**
- `WalkForwardEngine`: Orchestrates train/test windowing, signal generation, and execution with 1-day delay.
- `SignalGenerator`: Protocol class — any model with `fit()` and `predict()` works.
- `Portfolio`: Tracks positions, cash, equity curve, and executed trades.
- `RealisticCostModel`: Almgren-Chriss square-root market impact + commission + spread.
- `PerformanceMetrics`: Sharpe, Sortino, Calmar, drawdown, VaR/CVaR, rolling metrics.
- `BacktestValidator`: Post-hoc checks for signal timing, data snooping, turnover, plausibility.
- `BacktestReport`: Text summaries, strategy comparison tables, equity curve and monthly return plots.
- 5 benchmarks: BuyAndHold, Momentum 12-1, MeanReversion, SPY, 60/40.

**Anti-lookahead rules:**
1. `fit()` receives ONLY training window data
2. `predict()` receives ONLY test window data
3. Signals at time t executed at time t+1 (NOT same day)
4. Transaction costs applied at execution time
5. Validators run after every backtest to double-check

**Design decisions:**
| Decision | Choice | Rationale |
|---|---|---|
| Walk-forward vs simple split | Walk-forward (504d/63d/63d) | Multiple OOS periods across different regimes |
| Signal execution timing | Signal at t, execute at t+1 open | Prevents same-day lookahead |
| Cost model default | RealisticCostModel (Almgren-Chriss) | Industry standard, conservative |
| SignalGenerator interface | Python Protocol class | Any model with fit()/predict() works |
| BacktestResult | Pydantic BaseModel | JSON-serializable, includes validation |

## Current State

Sprint 1, Day 4: Market data pipeline, feature engineering (49 features), regime detection pipeline, and backtesting framework operational. DuckDB storage, Yahoo Finance provider, NYSE calendar validation, data quality checks. Rolling PCA + HMM/KMeans clustering with anti-lookahead guarantees. Walk-forward backtesting with 5 benchmarks, realistic cost models, and post-hoc validators. ~195 tests total (49 backtest + 146 prior).
