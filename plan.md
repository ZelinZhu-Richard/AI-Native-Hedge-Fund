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

---

# Day 2: Feature Engineering Pipeline

## Context

Day 2 of Sprint 1. Day 1 built the data foundation: DuckDB storage (4 tables), Yahoo Finance provider, NYSE-calendar validation, ingestion orchestrator, 50 passing tests at 81% coverage. ~22 test-universe tickers of OHLCV data are ingested.

Today we build the feature engineering pipeline that transforms raw OHLCV data into 49 model-ready features. Every MoE expert (TFT, MAML, N-BEATS, Adversarial Transformer) and the gating network will consume features from this pipeline. Anti-lookahead bias protection is the #1 priority — a single leak invalidates every backtest.

## Implementation Phases

### Phase 1: Foundation
- `meridian/core/exceptions.py` — Added `FeatureComputationError`
- `meridian/config/constants.py` — Added MAX_FEATURE_LOOKBACK_DAYS (200), LOOKBACK_CALENDAR_BUFFER_MULTIPLIER (1.5), DEFAULT_FEATURE_VERSION (1), TRADING_DAYS_PER_YEAR (252)
- `meridian/features/registry.py` — FeatureConfig dataclass + FeatureRegistry singleton (register, get, list, max_lookback, version_hash)
- `meridian/features/base.py` — Abstract FeatureComputer with auto-registration, SingleTickerFeature and CrossSectionalFeature marker subclasses
- `meridian/features/__init__.py` — Public exports

### Phase 2: Feature Computers (49 features)
- `meridian/features/technical.py` — **25 features**: returns (1d/5d/21d/63d), RSI (14/28), MACD signal+histogram, momentum 12-1, mean reversion (5d/21d), Bollinger bands, z-score, SMA ratios, EMA ratio, price-to-SMA (50/200), golden/death cross, volume ratios, OBV slope, VWAP distance
- `meridian/features/volatility.py` — **10 features**: realized vol (5d/21d/63d), vol ratios, vol-of-vol, Garman-Klass, Parkinson, vol regime z-score, HL range ratio
- `meridian/features/cross_sectional.py` — **8 features**: rank returns (21d/63d), sector relative returns, sector momentum, market breadth, dispersion, rank volume ratio
- `meridian/features/macro.py` — **6 features**: SPY returns (5d/21d), SPY vol, SPY drawdown, SPY above 200SMA, market return dispersion

### Phase 3: Storage + Pipeline
- `meridian/features/store.py` — DuckDB feature store: long-format storage (ticker, date, feature_name, value), wide-format retrieval via get_feature_matrix(), feature metadata tracking
- `meridian/features/pipeline.py` — Three-phase orchestrator: (1) single-ticker technical+volatility, (2) cross-sectional with universe data, (3) macro with SPY + dispersion. Lookback buffer, fault tolerance, Rich progress bars.

### Phase 4: Tests (64 new tests, 114 total)
- `test_technical.py` — 16 tests: column counts, return accuracy, RSI bounds, MACD, golden/death cross exclusivity, volume ratios, NaN warmup, determinism, edge cases
- `test_volatility.py` — 11 tests: column counts, non-negative vol, vol ratios, Garman-Klass/Parkinson consistency, NaN warmup, edge cases
- `test_cross_sectional.py` — 9 tests: rank bounds, sector-relative sums, market breadth bounds, dispersion non-negative, missing sector_map error
- `test_macro.py` — 9 tests: SPY returns, drawdown non-positive, drawdown zero at highs, above-200SMA binary, dispersion handling, edge cases
- `test_store.py` — 12 tests: schema idempotent, store/retrieve roundtrip, wide-format matrix, metadata, date filtering, upsert, coverage stats
- `test_z_pipeline.py` — 1 comprehensive end-to-end test (consolidated to avoid DuckDB Python 3.13 segfault)
- `test_lookahead.py` — 6 anti-lookahead tests: truncation invariance (technical, volatility, macro), cross-sectional no-retroactive-effect, NaN warmup, expanding causality

### Phase 5: Integration
- `Makefile` — Added test-features, test-lookahead targets
- `scripts/setup_db.py` — Added FeatureStore.create_schema() call
- `tests/conftest.py` — Added ohlcv_dataframe, long_ohlcv_dataframe, multi_ticker_ohlcv, feature_store fixtures

## Key Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Feature storage format | Long format (EAV) | Flexible schema, easy to add features without ALTER TABLE |
| Anti-lookahead | min_periods on all rolling ops | Prevents partial-window leakage; verified by truncation tests |
| adj_close for returns | Split-adjusted prices | Consistent return calculations across stock splits |
| Log returns for vol | ln(P_t/P_{t-1}) | Theoretically correct for annualization |
| Registry singleton | Auto-register on init | Features self-document; pipeline discovers them automatically |
| Pipeline phases | 3-phase (ticker→cross→macro) | Each phase depends on prior: cross-sectional needs returns, macro needs dispersion |
| DuckDB executemany | Parameterized INSERT | Avoids Python 3.13 segfault with DataFrame SQL operations |

---

# Day 3: PCA + Regime Clustering + Infrastructure

## Context

Day 3 of Sprint 1. Days 1-2 built data foundation (DuckDB, Yahoo Finance, validation, ingestion) and feature engineering (49 features across technical/volatility/cross-sectional/macro with anti-lookahead guarantees). 114 tests passing at 84% coverage.

Today we build the regime detection pipeline that the gating network (Day 11) will consume to route between MoE experts. The fund thesis is "markets misprice regime transitions" — we need to detect regimes and transitions before we can exploit them.

## Implementation Phases

### Phase 1: Dependencies + Foundation
- `pyproject.toml` — scikit-learn>=1.3, hmmlearn>=0.3, matplotlib moved to core deps
- `meridian/config/constants.py` — PCA_DEFAULT_WINDOW_DAYS (252), PCA_DEFAULT_REFIT_FREQUENCY (21), PCA_DEFAULT_N_COMPONENTS (10), DEFAULT_N_REGIMES (4), MIN_PCA_OBSERVATIONS (50)
- `meridian/core/exceptions.py` — RegimeDetectionError

### Phase 2: Rolling PCA (anti-lookahead critical)
- `meridian/regimes/pca.py` — RollingPCA (per-window StandardScaler+PCA, never global fit) + MarketPCA (cross-sectional factor extraction)

### Phase 3: Clustering
- `meridian/regimes/clustering.py` — HMMRegimeDetector (GaussianHMM with regime relabeling) + KMeansRegimeDetector (distance-based with pseudo-probabilities)

### Phase 4: Detector + Result
- `meridian/regimes/detector.py` — RegimeDetector (unified pipeline: features→PCA→clustering) + RegimeResult (Pydantic BaseModel, JSON-serializable)

### Phase 5: Analysis + Visualization
- `meridian/regimes/analysis.py` — RegimeAnalyzer (characterize, transitions, conditional performance)
- `meridian/regimes/visualization.py` — Timeline, scatter, heatmap, performance plots

### Phase 6: Tests (32 new)
- `test_pca.py` — 12 tests including truncation invariance + scaler window verification
- `test_clustering.py` — 10 tests including synthetic regime detection >90%
- `test_detector.py` — 6 tests including serialization roundtrip
- `test_lookahead_regimes.py` — 4 critical anti-lookahead tests

### Phase 7: Infrastructure
- `TODO.md`, `Dockerfile`, `docker-compose.yml`, Makefile additions, ARCHITECTURE.md update

## Key Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Rolling vs expanding PCA | Rolling (252d window) | Adapts to structural breaks; expanding dominated by old regimes |
| PCA refit frequency | Monthly (21 trading days) | Daily is noisy, quarterly misses transitions |
| StandardScaler scope | Per rolling window | Anti-lookahead: global scaling leaks future distribution |
| HMM + KMeans | Both implemented | HMM captures temporal dynamics, KMeans is stable fallback |
| Regime label stability | Sort by mean component value | Ensures semantic consistency across refits |
| Regime storage | In-memory only | Derived from features (in DuckDB); re-deriving is cheap |
| Docker base | python:3.11-slim | Avoids DuckDB + Python 3.13 segfault |

---

# Day 4: Backtesting Framework

## Context

Day 4 of Sprint 1. Days 1-3 built data pipeline, 50+ features, and regime detection. The backtesting framework validates whether the fund works. Every model decision from Day 5+ depends on backtest results being trustworthy. Must prevent: lookahead bias, overfitting, unrealistic cost assumptions.

## Implementation Phases

### Phase 1: Foundation models (costs.py, portfolio.py, metrics.py) — from planning
- `meridian/backtest/costs.py` — TradeCost, RealisticCostModel (Almgren-Chriss), ZeroCostModel, HighCostModel
- `meridian/backtest/portfolio.py` — Portfolio, Position, Trade, EquityPoint
- `meridian/backtest/metrics.py` — PerformanceMetrics (Sharpe, Sortino, drawdown, Calmar, VaR/CVaR, rolling)

### Phase 2: SignalGenerator protocol + benchmarks
- `meridian/backtest/benchmarks.py` — SignalGenerator Protocol, BuyAndHold, MomentumBaseline (12-1), MeanReversionBaseline, SPYBenchmark, SixtyFortyBenchmark

### Phase 3: Walk-forward engine
- `meridian/backtest/engine.py` — WalkForwardConfig, WindowResult, BacktestResult, WalkForwardEngine

### Phase 4: Validators
- `meridian/backtest/validators.py` — ValidationResult, BacktestValidator (signal timing, turnover, window consistency, data snooping, return plausibility)

### Phase 5: Report generation
- `meridian/backtest/report.py` — BacktestReport (text summary, comparison table, equity curve plot, monthly returns heatmap)

### Phase 6: Public exports
- `meridian/backtest/__init__.py` — All public exports

### Phase 7: Tests (49 tests across 7 files)
- `tests/test_backtest/test_costs.py` — 7 tests
- `tests/test_backtest/test_portfolio.py` — 7 tests
- `tests/test_backtest/test_metrics.py` — 8 tests
- `tests/test_backtest/test_benchmarks.py` — 11 tests
- `tests/test_backtest/test_engine.py` — 7 tests
- `tests/test_backtest/test_validators.py` — 5 tests
- `tests/test_backtest/test_integration.py` — 3 tests

### Phase 8: Documentation + infrastructure
- `Makefile` — Added test-backtest target
- `TODO.md` — Marked Day 4 complete, updated competitive intel + decision log
- `ARCHITECTURE.md` — Added backtesting framework section
- `PLAN.md` — Appended Day 4 plan
- `summary.md` — Appended Day 4 summary

## Key Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Walk-forward vs simple split | Walk-forward (504d train, 63d test, 63d step) | Multiple OOS periods across different regimes |
| Signal execution timing | Signal at t, execute at t+1 open | Prevents same-day lookahead |
| Cost model default | RealisticCostModel (Almgren-Chriss sqrt impact) | Industry standard, conservative |
| SignalGenerator protocol | Python Protocol class | Any model with fit()/predict() works |
| BacktestResult | Pydantic BaseModel | JSON-serializable, includes validation results |
