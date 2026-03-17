"""Integration tests: full pipeline from data through backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.backtest.benchmarks import BuyAndHold, MomentumBaseline
from meridian.backtest.costs import RealisticCostModel, ZeroCostModel
from meridian.backtest.engine import WalkForwardConfig, WalkForwardEngine
from meridian.backtest.report import BacktestReport
from meridian.backtest.validators import BacktestValidator


def _make_full_data(tickers: list[str], n_days: int = 800) -> pd.DataFrame:
    """Create realistic multi-ticker data for integration testing."""
    records = []
    dates = pd.bdate_range(start="2020-01-02", periods=n_days)
    rng = np.random.default_rng(42)

    for ticker in tickers:
        price = 100.0 + rng.uniform(-20, 50)
        for dt in dates:
            ret = rng.normal(0.0003, 0.018)
            price *= 1 + ret
            price = max(price, 1.0)  # floor at $1
            records.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "open": price * (1 + rng.normal(0, 0.002)),
                    "high": price * (1 + abs(rng.normal(0, 0.01))),
                    "low": price * (1 - abs(rng.normal(0, 0.01))),
                    "close": price,
                    "adj_close": price,
                    "volume": float(rng.integers(500_000, 80_000_000)),
                }
            )

    df = pd.DataFrame(records)
    df = df.set_index("date")
    return df


class TestFullBacktestPipeline:
    def test_buy_and_hold_end_to_end(self):
        """Full pipeline: data -> signal -> backtest -> validate -> report."""
        data = _make_full_data(["AAPL", "MSFT", "GOOG"], n_days=800)

        engine = WalkForwardEngine(
            config=WalkForwardConfig(
                train_days=504,
                test_days=63,
                step_days=63,
                initial_capital=1_000_000.0,
            ),
            cost_model=RealisticCostModel(),
        )

        result = engine.run(BuyAndHold(max_positions=3), data)

        # Should complete with at least 1 window
        assert result.total_windows >= 1
        assert result.strategy_name == "BuyAndHold"

        # Validate
        validator = BacktestValidator()
        validation = validator.validate(result)
        result.validation_results = validation

        # Report generation should not raise
        report = BacktestReport(result)
        text = report.text_summary()
        assert "BACKTEST REPORT" in text
        assert "BuyAndHold" in text

    def test_zero_vs_realistic_costs(self):
        """Strategy with costs should underperform strategy without."""
        data = _make_full_data(["AAPL", "MSFT"], n_days=700)
        config = WalkForwardConfig(train_days=400, test_days=63, step_days=63)

        result_free = WalkForwardEngine(config=config, cost_model=ZeroCostModel()).run(
            BuyAndHold(), data
        )

        result_cost = WalkForwardEngine(
            config=config, cost_model=RealisticCostModel()
        ).run(BuyAndHold(), data)

        # Both should complete
        assert result_free.total_windows >= 1
        assert result_cost.total_windows >= 1

        # With costs, total return should be <= no-cost version
        # (or very close if few trades)
        free_return = result_free.metrics.get("total_return", 0)
        cost_return = result_cost.metrics.get("total_return", 0)
        # Allow small tolerance for rounding
        assert cost_return <= free_return + 0.01

    def test_comparison_report(self):
        """Compare multiple strategies in a single report."""
        data = _make_full_data(["AAPL", "MSFT", "GOOG"], n_days=700)
        config = WalkForwardConfig(train_days=400, test_days=63, step_days=63)
        engine = WalkForwardEngine(config=config, cost_model=ZeroCostModel())

        result_bh = engine.run(BuyAndHold(), data)
        result_mom = engine.run(MomentumBaseline(top_n=2), data)

        report = BacktestReport(result_bh)
        table = report.comparison_table([result_mom])
        assert "BuyAndHold" in table
        assert "Momentum" in table
