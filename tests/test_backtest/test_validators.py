"""Tests for backtest validators."""

from __future__ import annotations

from datetime import date

from meridian.backtest.engine import BacktestResult, WalkForwardConfig, WindowResult
from meridian.backtest.validators import BacktestValidator


def _make_result(
    windows: list[WindowResult] | None = None,
    metrics: dict | None = None,
    trades: list[dict] | None = None,
    equity_curve: list[dict] | None = None,
) -> BacktestResult:
    """Helper to create a BacktestResult for testing."""
    if windows is None:
        windows = [
            WindowResult(
                window_idx=i,
                train_start=date(2022, 1, 1),
                train_end=date(2023, 12, 31),
                test_start=date(2024, 1, 2),
                test_end=date(2024, 3, 29),
                n_trades=10,
                window_return=0.02,
            )
            for i in range(5)
        ]
    if metrics is None:
        metrics = {
            "sharpe_ratio": 0.8,
            "annualized_return": 0.12,
            "max_drawdown": -0.08,
            "total_return": 0.10,
        }
    if trades is None:
        trades = []
    if equity_curve is None:
        equity_curve = [
            {"date": "2024-01-02", "portfolio_value": 1_000_000, "daily_return": 0.0}
        ] * 252

    return BacktestResult(
        strategy_name="TestStrategy",
        config=WalkForwardConfig(),
        windows=windows,
        metrics=metrics,
        equity_curve=equity_curve,
        trades=trades,
        total_windows=len(windows),
        total_trades=len(trades),
        backtest_start=date(2024, 1, 2),
        backtest_end=date(2024, 12, 31),
    )


class TestBacktestValidator:
    def test_valid_result_passes(self):
        result = _make_result()
        validator = BacktestValidator()
        validation = validator.validate(result)
        assert validation["passed"]

    def test_too_few_windows_fails(self):
        result = _make_result(
            windows=[
                WindowResult(
                    window_idx=0,
                    train_start=date(2022, 1, 1),
                    train_end=date(2023, 12, 31),
                    test_start=date(2024, 1, 2),
                    test_end=date(2024, 3, 29),
                    n_trades=5,
                    window_return=0.02,
                )
            ]
        )
        validator = BacktestValidator(min_windows=3)
        validation = validator.validate(result)
        # window_consistency check should fail
        checks = {c["name"]: c["passed"] for c in validation["checks"]}
        assert not checks["window_consistency"]

    def test_implausible_sharpe_fails(self):
        result = _make_result(
            metrics={
                "sharpe_ratio": 5.0,
                "annualized_return": 0.15,
                "max_drawdown": -0.05,
            }
        )
        validator = BacktestValidator()
        validation = validator.validate(result)
        checks = {c["name"]: c["passed"] for c in validation["checks"]}
        assert not checks["return_plausibility"]

    def test_excessive_turnover_fails(self):
        # Create trades worth way more than portfolio
        trades = [
            {"date": "2024-01-15", "gross_value": 5_000_000, "total_cost": 100}
            for _ in range(100)
        ]
        equity_curve = [
            {"date": "2024-01-15", "portfolio_value": 1_000_000, "daily_return": 0.0}
        ] * 252
        result = _make_result(trades=trades, equity_curve=equity_curve)
        validator = BacktestValidator(max_annual_turnover=10.0)
        validation = validator.validate(result)
        checks = {c["name"]: c["passed"] for c in validation["checks"]}
        assert not checks["turnover"]

    def test_signal_timing_no_trades_passes(self):
        result = _make_result(trades=[])
        validator = BacktestValidator()
        validation = validator.validate(result)
        checks = {c["name"]: c["passed"] for c in validation["checks"]}
        assert checks["signal_timing"]
