"""Backtesting framework: walk-forward engine, cost models, benchmarks, validation.

Public API:
    - WalkForwardEngine, BacktestResult, WalkForwardConfig
    - SignalGenerator protocol
    - BuyAndHold, MomentumBaseline, MeanReversionBaseline
    - SPYBenchmark, SixtyFortyBenchmark
    - RealisticCostModel, ZeroCostModel, HighCostModel, TradeCost
    - Portfolio, Position, Trade, EquityPoint
    - PerformanceMetrics
    - BacktestValidator
    - BacktestReport
"""

from meridian.backtest.benchmarks import (
    BuyAndHold,
    MeanReversionBaseline,
    MomentumBaseline,
    SignalGenerator,
    SixtyFortyBenchmark,
    SPYBenchmark,
)
from meridian.backtest.costs import (
    HighCostModel,
    RealisticCostModel,
    TradeCost,
    TransactionCostModel,
    ZeroCostModel,
)
from meridian.backtest.engine import (
    BacktestResult,
    WalkForwardConfig,
    WalkForwardEngine,
    WindowResult,
)
from meridian.backtest.metrics import PerformanceMetrics
from meridian.backtest.portfolio import EquityPoint, Portfolio, Position, Trade
from meridian.backtest.report import BacktestReport
from meridian.backtest.validators import BacktestValidator, ValidationResult

__all__ = [
    # Engine
    "WalkForwardEngine",
    "WalkForwardConfig",
    "BacktestResult",
    "WindowResult",
    # Protocol
    "SignalGenerator",
    # Benchmarks
    "BuyAndHold",
    "MomentumBaseline",
    "MeanReversionBaseline",
    "SPYBenchmark",
    "SixtyFortyBenchmark",
    # Cost models
    "TransactionCostModel",
    "RealisticCostModel",
    "ZeroCostModel",
    "HighCostModel",
    "TradeCost",
    # Portfolio
    "Portfolio",
    "Position",
    "Trade",
    "EquityPoint",
    # Metrics
    "PerformanceMetrics",
    # Validation
    "BacktestValidator",
    "ValidationResult",
    # Report
    "BacktestReport",
]
