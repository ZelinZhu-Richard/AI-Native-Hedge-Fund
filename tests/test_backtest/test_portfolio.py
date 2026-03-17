"""Tests for portfolio tracking."""

from __future__ import annotations

from datetime import date

import pandas as pd

from meridian.backtest.costs import RealisticCostModel, ZeroCostModel
from meridian.backtest.portfolio import Portfolio, Position


class TestPosition:
    def test_position_fields(self):
        pos = Position(
            ticker="AAPL",
            shares=100.0,
            entry_price=150.0,
            entry_date=date(2024, 1, 1),
        )
        assert pos.ticker == "AAPL"
        assert pos.shares == 100.0
        assert pos.entry_price == 150.0


class TestPortfolio:
    def test_initial_state(self):
        p = Portfolio(initial_capital=1_000_000.0)
        assert p.cash == 1_000_000.0
        assert len(p.positions) == 0
        assert len(p.trades) == 0

    def test_buy_reduces_cash(self):
        p = Portfolio(initial_capital=1_000_000.0)
        prices = {"AAPL": 150.0}
        p.execute_trades(
            target_weights={"AAPL": 0.5},
            prices=prices,
            current_date=date(2024, 1, 2),
            cost_model=ZeroCostModel(),
        )
        assert p.cash < 1_000_000.0
        assert "AAPL" in p.positions

    def test_sell_increases_cash(self):
        p = Portfolio(initial_capital=1_000_000.0)
        prices = {"AAPL": 150.0}
        # Buy first
        p.execute_trades(
            target_weights={"AAPL": 0.5},
            prices=prices,
            current_date=date(2024, 1, 2),
            cost_model=ZeroCostModel(),
        )
        cash_after_buy = p.cash
        # Sell
        p.execute_trades(
            target_weights={"AAPL": 0.0},
            prices=prices,
            current_date=date(2024, 1, 3),
            cost_model=ZeroCostModel(),
        )
        assert p.cash > cash_after_buy

    def test_equity_curve_records(self):
        p = Portfolio(initial_capital=1_000_000.0)
        p.update_equity(date(2024, 1, 1), {})
        assert len(p.equity_curve) == 1
        assert p.equity_curve[0].portfolio_value == 1_000_000.0
        assert p.equity_curve[0].daily_return == 0.0

    def test_get_returns_series(self):
        p = Portfolio(initial_capital=1_000_000.0)
        p.update_equity(date(2024, 1, 1), {})
        p.update_equity(date(2024, 1, 2), {})
        returns = p.get_returns()
        assert isinstance(returns, pd.Series)
        assert len(returns) == 2

    def test_transaction_costs_reduce_portfolio_value(self):
        p_free = Portfolio(initial_capital=1_000_000.0)
        p_cost = Portfolio(initial_capital=1_000_000.0)
        prices = {"AAPL": 150.0}
        volumes = {"AAPL": 10_000_000}

        p_free.execute_trades(
            target_weights={"AAPL": 0.5},
            prices=prices,
            current_date=date(2024, 1, 2),
            cost_model=ZeroCostModel(),
        )
        p_cost.execute_trades(
            target_weights={"AAPL": 0.5},
            prices=prices,
            current_date=date(2024, 1, 2),
            cost_model=RealisticCostModel(),
            daily_volumes=volumes,
        )
        # Portfolio with costs should have less cash
        assert p_cost.cash < p_free.cash

    def test_get_trade_log_dataframe(self):
        p = Portfolio(initial_capital=1_000_000.0)
        p.execute_trades(
            target_weights={"AAPL": 0.5},
            prices={"AAPL": 150.0},
            current_date=date(2024, 1, 2),
            cost_model=ZeroCostModel(),
        )
        log = p.get_trade_log()
        assert isinstance(log, pd.DataFrame)
        assert len(log) > 0
        assert "ticker" in log.columns
