"""Tests for performance metrics computation."""

from __future__ import annotations

import pandas as pd

from meridian.backtest.metrics import PerformanceMetrics


def _make_returns(values: list[float], start: str = "2024-01-01") -> pd.Series:
    """Helper to create a returns series."""
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=dates, name="returns")


class TestPerformanceMetrics:
    def test_positive_returns_positive_sharpe(self):
        # Steady positive returns
        returns = _make_returns([0.001] * 252)
        metrics = PerformanceMetrics.compute_all(returns, risk_free_rate=0.0)
        assert metrics["sharpe_ratio"] > 0
        assert metrics["total_return"] > 0

    def test_negative_returns_negative_sharpe(self):
        returns = _make_returns([-0.001] * 252)
        metrics = PerformanceMetrics.compute_all(returns, risk_free_rate=0.0)
        assert metrics["sharpe_ratio"] < 0
        assert metrics["total_return"] < 0

    def test_empty_returns_error(self):
        returns = pd.Series(dtype=float, name="returns")
        metrics = PerformanceMetrics.compute_all(returns)
        assert "error" in metrics

    def test_all_nan_returns_error(self):
        returns = _make_returns([float("nan")] * 10)
        metrics = PerformanceMetrics.compute_all(returns)
        assert "error" in metrics

    def test_max_drawdown_negative(self):
        # Create a series that goes up then down
        up = [0.01] * 50
        down = [-0.02] * 50
        flat = [0.0] * 50
        returns = _make_returns(up + down + flat)
        metrics = PerformanceMetrics.compute_all(returns)
        assert metrics["max_drawdown"] < 0

    def test_drawdown_series_shape(self):
        returns = _make_returns([0.01, -0.02, 0.01, -0.01, 0.005] * 50)
        dd = PerformanceMetrics.compute_drawdown_series(returns)
        assert "drawdown" in dd.columns
        assert "running_max" in dd.columns
        assert "in_drawdown" in dd.columns
        assert len(dd) == len(returns)

    def test_win_rate_bounds(self):
        returns = _make_returns([0.01, -0.01, 0.02, -0.005, 0.0] * 50)
        metrics = PerformanceMetrics.compute_all(returns)
        assert 0.0 <= metrics["win_rate"] <= 1.0

    def test_rolling_metrics_shape(self):
        returns = _make_returns([0.001] * 200)
        rolling = PerformanceMetrics.compute_rolling_metrics(returns, window=63)
        assert "rolling_return" in rolling.columns
        assert "rolling_volatility" in rolling.columns
        assert "rolling_sharpe" in rolling.columns
        assert len(rolling) == 200
