"""Walk-forward backtesting engine.

Implements expanding/rolling walk-forward validation with strict
anti-lookahead guarantees. Signal at time t executed at t+1.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from pydantic import BaseModel

from meridian.backtest.benchmarks import SignalGenerator
from meridian.backtest.costs import RealisticCostModel, TransactionCostModel
from meridian.backtest.metrics import PerformanceMetrics
from meridian.backtest.portfolio import Portfolio, Trade


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward validation."""

    train_days: int = 504  # ~2 years
    test_days: int = 63  # ~1 quarter
    step_days: int = 63  # non-overlapping test windows
    initial_capital: float = 1_000_000.0
    rebalance_frequency: int = 1  # days between rebalances (1=daily)


class WindowResult(BaseModel):
    """Result from a single walk-forward window."""

    window_idx: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    n_trades: int
    window_return: float
    window_sharpe: float | None = None

    model_config = {"arbitrary_types_allowed": True}


class BacktestResult(BaseModel):
    """Complete backtest result. JSON-serializable."""

    strategy_name: str
    config: WalkForwardConfig
    windows: list[WindowResult]
    metrics: dict[str, Any]
    equity_curve: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    total_windows: int
    total_trades: int
    backtest_start: date | None = None
    backtest_end: date | None = None
    validation_results: dict[str, Any] | None = None

    model_config = {"arbitrary_types_allowed": True}


class WalkForwardEngine:
    """Walk-forward backtesting engine with anti-lookahead protection.

    Walk-forward validation:
    1. Train on window [t, t+train_days)
    2. Generate signals on [t+train_days, t+train_days+test_days)
    3. Execute signals with 1-day delay (signal at t, trade at t+1)
    4. Step forward by step_days and repeat

    Anti-lookahead rules:
    - fit() receives ONLY training window data
    - predict() receives ONLY test window data
    - Signals at time t are executed at time t+1
    - Transaction costs applied at execution time
    """

    def __init__(
        self,
        config: WalkForwardConfig | None = None,
        cost_model: TransactionCostModel | None = None,
    ) -> None:
        self.config = config or WalkForwardConfig()
        self.cost_model = cost_model or RealisticCostModel()

    def run(
        self,
        signal_generator: SignalGenerator,
        data: pd.DataFrame,
    ) -> BacktestResult:
        """Run walk-forward backtest.

        Args:
            signal_generator: Must implement fit() and predict().
            data: Full dataset with columns including 'ticker', 'adj_close',
                  'open', 'volume'. DatetimeIndex or 'date' column.

        Returns:
            BacktestResult with metrics, equity curve, trades.
        """
        data = self._normalize_data(data)
        all_dates = sorted(data.index.unique())

        if len(all_dates) < self.config.train_days + self.config.test_days:
            return self._empty_result(signal_generator.name)

        portfolio = Portfolio(initial_capital=self.config.initial_capital)
        windows: list[WindowResult] = []
        window_idx = 0

        # Walk forward through the data
        start_idx = 0
        while start_idx + self.config.train_days + self.config.test_days <= len(
            all_dates
        ):
            train_end_idx = start_idx + self.config.train_days
            test_end_idx = min(train_end_idx + self.config.test_days, len(all_dates))

            train_dates = all_dates[start_idx:train_end_idx]
            test_dates = all_dates[train_end_idx:test_end_idx]

            if not train_dates or not test_dates:
                break

            # ANTI-LOOKAHEAD: slice data strictly by window
            train_data = data.loc[
                (data.index >= train_dates[0]) & (data.index <= train_dates[-1])
            ].copy()
            test_data = data.loc[
                (data.index >= test_dates[0]) & (data.index <= test_dates[-1])
            ].copy()

            # Fit on train ONLY, predict on test ONLY
            signal_generator.fit(train_data)
            signals = signal_generator.predict(test_data)

            # Execute signals with 1-day delay
            window_trades = self._execute_window(portfolio, signals, data, test_dates)

            # Compute window metrics
            window_equity_before = (
                portfolio.equity_curve[-len(test_dates) - 1].portfolio_value
                if len(portfolio.equity_curve) > len(test_dates)
                else self.config.initial_capital
            )
            window_equity_after = (
                portfolio.equity_curve[-1].portfolio_value
                if portfolio.equity_curve
                else self.config.initial_capital
            )
            window_return = (
                (window_equity_after / window_equity_before - 1.0)
                if window_equity_before > 0
                else 0.0
            )

            windows.append(
                WindowResult(
                    window_idx=window_idx,
                    train_start=train_dates[0].date()
                    if hasattr(train_dates[0], "date")
                    else train_dates[0],
                    train_end=train_dates[-1].date()
                    if hasattr(train_dates[-1], "date")
                    else train_dates[-1],
                    test_start=test_dates[0].date()
                    if hasattr(test_dates[0], "date")
                    else test_dates[0],
                    test_end=test_dates[-1].date()
                    if hasattr(test_dates[-1], "date")
                    else test_dates[-1],
                    n_trades=len(window_trades),
                    window_return=window_return,
                )
            )

            window_idx += 1
            start_idx += self.config.step_days

        # Compute overall metrics
        returns = portfolio.get_returns()
        metrics = (
            PerformanceMetrics.compute_all(returns)
            if len(returns) > 0
            else {"error": "No returns"}
        )

        # Build result
        equity_records = [
            {
                "date": ep.date.isoformat()
                if isinstance(ep.date, date)
                else str(ep.date),
                "portfolio_value": ep.portfolio_value,
                "cash": ep.cash,
                "positions_value": ep.positions_value,
                "daily_return": ep.daily_return,
                "cumulative_return": ep.cumulative_return,
            }
            for ep in portfolio.equity_curve
        ]

        trade_records = [t.model_dump() for t in portfolio.trades]
        # Convert date objects to strings for JSON serialization
        for tr in trade_records:
            if isinstance(tr.get("date"), date):
                tr["date"] = tr["date"].isoformat()

        backtest_start = (
            portfolio.equity_curve[0].date if portfolio.equity_curve else None
        )
        backtest_end = (
            portfolio.equity_curve[-1].date if portfolio.equity_curve else None
        )

        return BacktestResult(
            strategy_name=signal_generator.name,
            config=self.config,
            windows=windows,
            metrics=metrics,
            equity_curve=equity_records,
            trades=trade_records,
            total_windows=len(windows),
            total_trades=len(portfolio.trades),
            backtest_start=backtest_start,
            backtest_end=backtest_end,
        )

    def _execute_window(
        self,
        portfolio: Portfolio,
        signals: pd.DataFrame,
        full_data: pd.DataFrame,
        test_dates: list,
    ) -> list[Trade]:
        """Execute signals within a test window with 1-day delay.

        Signal at date t is executed at date t+1 using t+1 open prices.
        """
        all_trades: list[Trade] = []

        if signals.empty:
            # Still update equity for each test date
            for dt in test_dates:
                prices = self._get_prices(full_data, dt)
                portfolio.update_equity(
                    dt.date() if hasattr(dt, "date") else dt, prices
                )
            return all_trades

        rebalance_counter = 0
        for i, dt in enumerate(test_dates):
            execution_date = dt.date() if hasattr(dt, "date") else dt

            # Check if we should execute signals from the previous day
            if i > 0:
                prev_dt = test_dates[i - 1]
                # Get signals from previous day (1-day delay)
                day_signals = self._get_signals_for_date(signals, prev_dt)

                if (
                    day_signals is not None
                    and rebalance_counter % self.config.rebalance_frequency == 0
                ):
                    # Execute using today's prices (next-day execution)
                    prices = self._get_prices(full_data, dt)
                    volumes = self._get_volumes(full_data, dt)

                    if prices:
                        target_weights = {
                            row["ticker"]: row["weight"]
                            for _, row in day_signals.iterrows()
                            if row["ticker"] in prices
                        }
                        trades = portfolio.execute_trades(
                            target_weights=target_weights,
                            prices=prices,
                            current_date=execution_date,
                            cost_model=self.cost_model,
                            daily_volumes=volumes,
                        )
                        all_trades.extend(trades)

                rebalance_counter += 1

            # Mark-to-market
            prices = self._get_prices(full_data, dt)
            portfolio.update_equity(execution_date, prices)

        return all_trades

    def _get_signals_for_date(
        self, signals: pd.DataFrame, dt: Any
    ) -> pd.DataFrame | None:
        """Extract signals for a specific date."""
        if "date" in signals.columns:
            mask = signals["date"] == dt
            day_signals = signals[mask]
            if not day_signals.empty:
                return day_signals
        return None

    def _get_prices(self, data: pd.DataFrame, dt: Any) -> dict[str, float]:
        """Get execution prices (open or adj_close) for a date."""
        day_data = data.loc[data.index == dt]
        prices: dict[str, float] = {}
        for _, row in day_data.iterrows():
            ticker = row.get("ticker", "")
            if ticker:
                # Prefer open price for next-day execution, fall back to adj_close
                price = row.get("open", row.get("adj_close", 0.0))
                if price and price > 0:
                    prices[ticker] = float(price)
        return prices

    def _get_volumes(self, data: pd.DataFrame, dt: Any) -> dict[str, float]:
        """Get daily volumes for impact calculation."""
        day_data = data.loc[data.index == dt]
        volumes: dict[str, float] = {}
        for _, row in day_data.iterrows():
            ticker = row.get("ticker", "")
            if ticker and "volume" in row.index:
                volumes[ticker] = float(row["volume"])
        return volumes

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data has DatetimeIndex."""
        if not isinstance(data.index, pd.DatetimeIndex):
            if "date" in data.columns:
                data = data.copy()
                data["date"] = pd.to_datetime(data["date"])
                data = data.set_index("date")
            else:
                raise ValueError("Data must have DatetimeIndex or 'date' column")
        return data

    def _empty_result(self, name: str) -> BacktestResult:
        """Return empty result when insufficient data."""
        return BacktestResult(
            strategy_name=name,
            config=self.config,
            windows=[],
            metrics={"error": "Insufficient data for walk-forward"},
            equity_curve=[],
            trades=[],
            total_windows=0,
            total_trades=0,
        )
