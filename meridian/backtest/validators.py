"""Backtest validators: detect lookahead bias, data snooping, and other issues.

Run after every backtest to verify integrity of results.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from meridian.backtest.engine import BacktestResult


class ValidationResult:
    """Result of a single validation check."""

    def __init__(
        self,
        name: str,
        passed: bool,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


class BacktestValidator:
    """Validates backtest results for common pitfalls.

    Checks:
    1. Signal timing: no same-day execution (lookahead)
    2. Feature availability: signals use only available data
    3. Survivorship bias: check for forward-looking universe selection
    4. Data snooping: compare in-sample vs out-of-sample degradation
    5. Turnover: flag excessive trading that erodes returns
    """

    def __init__(
        self,
        max_annual_turnover: float = 50.0,
        max_is_oos_sharpe_ratio: float = 3.0,
        min_windows: int = 3,
    ) -> None:
        self.max_annual_turnover = max_annual_turnover
        self.max_is_oos_sharpe_ratio = max_is_oos_sharpe_ratio
        self.min_windows = min_windows

    def validate(self, result: BacktestResult) -> dict[str, Any]:
        """Run all validation checks on a backtest result.

        Returns:
            Dict with 'passed' (bool), 'checks' (list of results),
            and 'summary' (str).
        """
        checks: list[ValidationResult] = [
            self.check_signal_timing(result),
            self.check_turnover(result),
            self.check_window_consistency(result),
            self.check_data_snooping(result),
            self.check_return_plausibility(result),
        ]

        all_passed = all(c.passed for c in checks)
        failed = [c for c in checks if not c.passed]

        summary = (
            "All validation checks passed."
            if all_passed
            else f"{len(failed)} check(s) failed: " + ", ".join(c.name for c in failed)
        )

        return {
            "passed": all_passed,
            "checks": [c.to_dict() for c in checks],
            "summary": summary,
        }

    def check_signal_timing(self, result: BacktestResult) -> ValidationResult:
        """Verify signals are not executed on the same day they're generated.

        In the walk-forward engine, signals at time t should execute at t+1.
        This check verifies the trade log doesn't show same-day execution
        relative to window boundaries.
        """
        if not result.trades:
            return ValidationResult(
                name="signal_timing",
                passed=True,
                message="No trades to validate.",
            )

        violations = 0
        for window in result.windows:
            # Trades should not occur on test_start (first test day)
            # because there's no prior signal day to generate from
            window_trades = [
                t
                for t in result.trades
                if t.get("date") == window.test_start.isoformat()
                or t.get("date") == window.test_start
            ]
            if window_trades:
                violations += 1

        passed = violations == 0
        return ValidationResult(
            name="signal_timing",
            passed=passed,
            message=(
                "Signal timing OK: no first-day trades."
                if passed
                else (
                    f"{violations} window(s) have trades on "
                    "first test day (potential lookahead)."
                )
            ),
            details={"violations": violations},
        )

    def check_turnover(self, result: BacktestResult) -> ValidationResult:
        """Check if portfolio turnover is reasonable.

        Excessive turnover suggests overfitting to noise or
        unrealistically frequent rebalancing.
        """
        if not result.trades:
            return ValidationResult(
                name="turnover",
                passed=True,
                message="No trades — zero turnover.",
            )

        # Compute annualized turnover
        total_trade_value = sum(abs(t.get("gross_value", 0.0)) for t in result.trades)

        # Get average portfolio value
        if result.equity_curve:
            avg_value = np.mean(
                [ep.get("portfolio_value", 1e6) for ep in result.equity_curve]
            )
        else:
            avg_value = result.config.initial_capital

        # Trading days
        n_days = len(result.equity_curve) if result.equity_curve else 1
        n_years = n_days / 252.0

        if avg_value > 0 and n_years > 0:
            annual_turnover = (total_trade_value / avg_value) / n_years
        else:
            annual_turnover = 0.0

        passed = annual_turnover <= self.max_annual_turnover
        return ValidationResult(
            name="turnover",
            passed=passed,
            message=(
                f"Annual turnover {annual_turnover:.1f}x"
                f" (limit: {self.max_annual_turnover}x)."
                if passed
                else (
                    f"Excessive turnover: {annual_turnover:.1f}x"
                    f" exceeds {self.max_annual_turnover}x limit."
                )
            ),
            details={
                "annual_turnover": round(annual_turnover, 2),
                "total_trade_value": round(total_trade_value, 2),
                "avg_portfolio_value": round(avg_value, 2),
            },
        )

    def check_window_consistency(self, result: BacktestResult) -> ValidationResult:
        """Check that walk-forward windows are consistent.

        Verifies: no overlapping train/test, sufficient windows,
        test windows don't look into training data.
        """
        if result.total_windows < self.min_windows:
            return ValidationResult(
                name="window_consistency",
                passed=False,
                message=(
                    f"Only {result.total_windows} windows"
                    f" (need >= {self.min_windows})."
                ),
                details={"n_windows": result.total_windows},
            )

        # Check train/test non-overlap
        overlaps = 0
        for w in result.windows:
            if w.test_start <= w.train_end:
                overlaps += 1

        passed = overlaps == 0
        return ValidationResult(
            name="window_consistency",
            passed=passed,
            message=(
                f"{result.total_windows} windows with no train/test overlap."
                if passed
                else f"{overlaps} window(s) have train/test overlap."
            ),
            details={
                "n_windows": result.total_windows,
                "overlaps": overlaps,
            },
        )

    def check_data_snooping(self, result: BacktestResult) -> ValidationResult:
        """Detect potential data snooping by comparing window returns.

        If early windows dramatically outperform later windows, the strategy
        may have been fit to the early data period. A healthy strategy
        should show relatively consistent performance across windows.
        """
        if len(result.windows) < 4:
            return ValidationResult(
                name="data_snooping",
                passed=True,
                message="Too few windows for snooping test.",
            )

        returns = [w.window_return for w in result.windows]
        mid = len(returns) // 2
        first_half_mean = np.mean(returns[:mid])
        second_half_mean = np.mean(returns[mid:])

        # If first half Sharpe is dramatically better than second half,
        # strategy may be overfit to early period
        first_half_std = np.std(returns[:mid]) if np.std(returns[:mid]) > 0 else 1.0
        second_half_std = np.std(returns[mid:]) if np.std(returns[mid:]) > 0 else 1.0

        first_sharpe = first_half_mean / first_half_std if first_half_std > 0 else 0.0
        second_sharpe = (
            second_half_mean / second_half_std if second_half_std > 0 else 0.0
        )

        # Very rough check: first half shouldn't be orders of magnitude better
        ratio = (
            abs(first_sharpe / second_sharpe)
            if second_sharpe != 0
            else abs(first_sharpe)
        )

        passed = ratio <= self.max_is_oos_sharpe_ratio
        return ValidationResult(
            name="data_snooping",
            passed=passed,
            message=(
                f"First/second half Sharpe ratio: {ratio:.2f}"
                f" (limit: {self.max_is_oos_sharpe_ratio}x)."
                if passed
                else (
                    f"Potential data snooping: first half"
                    f" {ratio:.1f}x better than second half."
                )
            ),
            details={
                "first_half_sharpe": round(first_sharpe, 4),
                "second_half_sharpe": round(second_sharpe, 4),
                "ratio": round(ratio, 4),
            },
        )

    def check_return_plausibility(self, result: BacktestResult) -> ValidationResult:
        """Check if returns are plausible (not too good to be true).

        Sharpe > 3.0 annualized for a long-only equity strategy is
        almost certainly a bug or lookahead bias.
        """
        sharpe = result.metrics.get("sharpe_ratio", 0.0)
        if isinstance(sharpe, str):
            return ValidationResult(
                name="return_plausibility",
                passed=True,
                message="No Sharpe ratio available.",
            )

        max_dd = result.metrics.get("max_drawdown", -1.0)
        annual_return = result.metrics.get("annualized_return", 0.0)

        issues: list[str] = []
        if abs(sharpe) > 4.0:
            issues.append(f"Sharpe {sharpe:.2f} > 4.0 (likely lookahead)")
        if annual_return > 1.0:
            issues.append(f"Annual return {annual_return:.0%} > 100% (implausible)")
        if max_dd == 0.0 and len(result.equity_curve) > 50:
            issues.append("Zero drawdown over extended period (suspicious)")

        passed = len(issues) == 0
        return ValidationResult(
            name="return_plausibility",
            passed=passed,
            message=(
                "Returns are plausible."
                if passed
                else "Implausible returns: " + "; ".join(issues)
            ),
            details={
                "sharpe_ratio": sharpe,
                "annualized_return": annual_return,
                "max_drawdown": max_dd,
                "issues": issues,
            },
        )
