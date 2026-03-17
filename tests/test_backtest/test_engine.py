"""Tests for walk-forward backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.backtest.benchmarks import BuyAndHold
from meridian.backtest.costs import ZeroCostModel
from meridian.backtest.engine import (
    BacktestResult,
    WalkForwardConfig,
    WalkForwardEngine,
)


def _make_backtest_data(tickers: list[str], n_days: int = 700) -> pd.DataFrame:
    """Create synthetic multi-ticker data for backtesting.

    700 days is enough for 504 train + 63 test + buffer.
    """
    records = []
    dates = pd.bdate_range(start="2021-01-04", periods=n_days)
    rng = np.random.default_rng(42)

    for ticker in tickers:
        price = 100.0
        for dt in dates:
            ret = rng.normal(0.0003, 0.015)
            price *= 1 + ret
            records.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "open": price * 0.999,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "adj_close": price,
                    "volume": float(rng.integers(1_000_000, 50_000_000)),
                }
            )

    df = pd.DataFrame(records)
    df = df.set_index("date")
    return df


class TestWalkForwardEngine:
    def test_runs_with_buy_and_hold(self):
        data = _make_backtest_data(["AAPL", "MSFT"], n_days=700)
        engine = WalkForwardEngine(
            config=WalkForwardConfig(train_days=504, test_days=63, step_days=63),
            cost_model=ZeroCostModel(),
        )
        result = engine.run(BuyAndHold(), data)
        assert isinstance(result, BacktestResult)
        assert result.total_windows >= 1
        assert result.strategy_name == "BuyAndHold"

    def test_insufficient_data_returns_empty(self):
        data = _make_backtest_data(["AAPL"], n_days=100)
        engine = WalkForwardEngine(
            config=WalkForwardConfig(train_days=504, test_days=63)
        )
        result = engine.run(BuyAndHold(), data)
        assert result.total_windows == 0
        assert "error" in result.metrics

    def test_multiple_windows(self):
        data = _make_backtest_data(["AAPL", "MSFT"], n_days=900)
        engine = WalkForwardEngine(
            config=WalkForwardConfig(train_days=400, test_days=63, step_days=63),
            cost_model=ZeroCostModel(),
        )
        result = engine.run(BuyAndHold(), data)
        assert result.total_windows >= 2

    def test_no_train_test_overlap(self):
        data = _make_backtest_data(["AAPL"], n_days=800)
        engine = WalkForwardEngine(
            config=WalkForwardConfig(train_days=400, test_days=63, step_days=63),
            cost_model=ZeroCostModel(),
        )
        result = engine.run(BuyAndHold(), data)
        for w in result.windows:
            assert w.test_start > w.train_end, (
                f"Window {w.window_idx}: test_start {w.test_start} "
                f"<= train_end {w.train_end}"
            )

    def test_equity_curve_populated(self):
        data = _make_backtest_data(["AAPL"], n_days=700)
        engine = WalkForwardEngine(
            config=WalkForwardConfig(train_days=504, test_days=63, step_days=63),
            cost_model=ZeroCostModel(),
        )
        result = engine.run(BuyAndHold(), data)
        assert len(result.equity_curve) > 0

    def test_result_json_serializable(self):
        data = _make_backtest_data(["AAPL"], n_days=700)
        engine = WalkForwardEngine(
            config=WalkForwardConfig(train_days=504, test_days=63, step_days=63),
            cost_model=ZeroCostModel(),
        )
        result = engine.run(BuyAndHold(), data)
        # Should not raise
        json_str = result.model_dump_json()
        assert len(json_str) > 0

    def test_zero_cost_no_cost_in_trades(self):
        data = _make_backtest_data(["AAPL"], n_days=700)
        engine = WalkForwardEngine(
            config=WalkForwardConfig(train_days=504, test_days=63, step_days=63),
            cost_model=ZeroCostModel(),
        )
        result = engine.run(BuyAndHold(), data)
        for trade in result.trades:
            assert trade.get("total_cost", 0.0) == 0.0
