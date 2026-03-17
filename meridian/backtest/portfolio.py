"""Portfolio state tracking: positions, cash, P&L, equity curve."""

from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel

from meridian.backtest.costs import TransactionCostModel, ZeroCostModel


class Position(BaseModel):
    """A single stock position."""

    ticker: str
    shares: float
    entry_price: float
    entry_date: date
    current_value: float = 0.0


class Trade(BaseModel):
    """Record of a single executed trade."""

    date: date
    ticker: str
    side: str  # "buy" or "sell"
    shares: float
    price: float
    gross_value: float
    commission: float
    spread_cost: float
    market_impact: float
    total_cost: float
    net_value: float


class EquityPoint(BaseModel):
    """Snapshot of portfolio value at end of day."""

    date: date
    portfolio_value: float
    cash: float
    positions_value: float
    daily_return: float
    cumulative_return: float


class Portfolio:
    """Tracks portfolio state through the backtest.

    Records every position change, every trade, and the full
    equity curve. Exact P&L computed from actual execution prices.
    """

    def __init__(self, initial_capital: float = 1_000_000.0) -> None:
        self.cash: float = initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[EquityPoint] = []
        self.initial_capital = initial_capital

    def execute_trades(
        self,
        target_weights: dict[str, float],
        prices: dict[str, float],
        current_date: date,
        cost_model: TransactionCostModel | None = None,
        daily_volumes: dict[str, float] | None = None,
    ) -> list[Trade]:
        """Rebalance portfolio to match target position weights.

        Args:
            target_weights: ticker -> target weight (fraction of portfolio).
                Positive = long, negative = short. Sum should be <= 1.0.
            prices: ticker -> execution price (next-day open).
            current_date: Execution date (must be > signal date).
            cost_model: Transaction cost model.
            daily_volumes: ticker -> avg daily volume for impact calc.

        Returns:
            List of executed trades.
        """
        if cost_model is None:
            cost_model = ZeroCostModel()
        if daily_volumes is None:
            daily_volumes = {}

        # Current portfolio value
        portfolio_value = self._compute_portfolio_value(prices)

        executed_trades: list[Trade] = []

        # Compute current weights
        current_weights: dict[str, float] = {}
        for ticker, pos in self.positions.items():
            if ticker in prices and portfolio_value > 0:
                current_weights[ticker] = pos.shares * prices[ticker] / portfolio_value

        # Determine all tickers involved
        all_tickers = set(target_weights.keys()) | set(current_weights.keys())

        # First pass: sells (to free up cash)
        for ticker in all_tickers:
            target_w = target_weights.get(ticker, 0.0)
            current_w = current_weights.get(ticker, 0.0)
            delta_w = target_w - current_w

            if delta_w < 0 and ticker in prices:
                price = prices[ticker]
                shares_to_sell = abs(delta_w * portfolio_value / price)
                if shares_to_sell < 0.01:
                    continue

                cost = cost_model.compute_cost(
                    ticker,
                    shares_to_sell,
                    price,
                    daily_volumes.get(ticker, 1_000_000),
                )
                gross = shares_to_sell * price
                net = gross - cost.total

                trade = Trade(
                    date=current_date,
                    ticker=ticker,
                    side="sell",
                    shares=shares_to_sell,
                    price=price,
                    gross_value=gross,
                    commission=cost.commission,
                    spread_cost=cost.spread_cost,
                    market_impact=cost.market_impact,
                    total_cost=cost.total,
                    net_value=net,
                )
                executed_trades.append(trade)

                # Update state
                self.cash += net
                if ticker in self.positions:
                    self.positions[ticker].shares -= shares_to_sell
                    if self.positions[ticker].shares < 0.01:
                        del self.positions[ticker]

        # Second pass: buys
        for ticker in all_tickers:
            target_w = target_weights.get(ticker, 0.0)
            current_w = current_weights.get(ticker, 0.0)
            delta_w = target_w - current_w

            if delta_w > 0 and ticker in prices:
                price = prices[ticker]
                target_value = delta_w * portfolio_value
                # Don't spend more than available cash
                max_spend = self.cash * 0.999  # small buffer
                buy_value = min(target_value, max_spend)
                if buy_value < 1.0:
                    continue

                shares_to_buy = buy_value / price
                cost = cost_model.compute_cost(
                    ticker,
                    shares_to_buy,
                    price,
                    daily_volumes.get(ticker, 1_000_000),
                )
                gross = shares_to_buy * price
                total_outflow = gross + cost.total

                # Adjust if total outflow exceeds cash
                if total_outflow > self.cash:
                    shares_to_buy = (
                        self.cash * 0.99 / (price + cost.total / max(shares_to_buy, 1))
                    )
                    cost = cost_model.compute_cost(
                        ticker,
                        shares_to_buy,
                        price,
                        daily_volumes.get(ticker, 1_000_000),
                    )
                    gross = shares_to_buy * price
                    total_outflow = gross + cost.total

                if shares_to_buy < 0.01:
                    continue

                trade = Trade(
                    date=current_date,
                    ticker=ticker,
                    side="buy",
                    shares=shares_to_buy,
                    price=price,
                    gross_value=gross,
                    commission=cost.commission,
                    spread_cost=cost.spread_cost,
                    market_impact=cost.market_impact,
                    total_cost=cost.total,
                    net_value=total_outflow,
                )
                executed_trades.append(trade)

                # Update state
                self.cash -= total_outflow
                if ticker in self.positions:
                    # Average into existing position
                    existing = self.positions[ticker]
                    total_shares = existing.shares + shares_to_buy
                    avg_price = (
                        existing.shares * existing.entry_price + shares_to_buy * price
                    ) / total_shares
                    existing.shares = total_shares
                    existing.entry_price = avg_price
                else:
                    self.positions[ticker] = Position(
                        ticker=ticker,
                        shares=shares_to_buy,
                        entry_price=price,
                        entry_date=current_date,
                    )

        self.trades.extend(executed_trades)
        return executed_trades

    def update_equity(self, current_date: date, prices: dict[str, float]) -> float:
        """Mark-to-market all positions and record equity point.

        Returns total portfolio value.
        """
        positions_value = 0.0
        for ticker, pos in self.positions.items():
            if ticker in prices:
                pos.current_value = pos.shares * prices[ticker]
                positions_value += pos.current_value

        portfolio_value = self.cash + positions_value

        # Compute returns
        if self.equity_curve:
            prev_value = self.equity_curve[-1].portfolio_value
            daily_return = (
                (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
            )
        else:
            daily_return = 0.0

        cumulative_return = portfolio_value / self.initial_capital - 1.0

        self.equity_curve.append(
            EquityPoint(
                date=current_date,
                portfolio_value=portfolio_value,
                cash=self.cash,
                positions_value=positions_value,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
            )
        )
        return portfolio_value

    def _compute_portfolio_value(self, prices: dict[str, float]) -> float:
        """Current portfolio value at given prices."""
        positions_value = sum(
            pos.shares * prices.get(ticker, pos.entry_price)
            for ticker, pos in self.positions.items()
        )
        return self.cash + positions_value

    def get_returns(self) -> pd.Series:
        """Daily portfolio returns from equity curve."""
        if not self.equity_curve:
            return pd.Series(dtype=float, name="returns")
        dates = [ep.date for ep in self.equity_curve]
        returns = [ep.daily_return for ep in self.equity_curve]
        return pd.Series(returns, index=pd.DatetimeIndex(dates), name="returns")

    def get_trade_log(self) -> pd.DataFrame:
        """All executed trades."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.model_dump() for t in self.trades])

    def get_exposure(self) -> pd.DataFrame:
        """Daily gross and net exposure is computed from equity curve."""
        # Simplified: exposure tracked at trade time
        records = []
        for ep in self.equity_curve:
            if ep.portfolio_value > 0:
                gross = ep.positions_value / ep.portfolio_value
            else:
                gross = 0.0
            records.append(
                {
                    "date": ep.date,
                    "gross_exposure": gross,
                    "net_exposure": gross,  # long-only for now
                    "cash_pct": ep.cash / ep.portfolio_value
                    if ep.portfolio_value > 0
                    else 1.0,
                }
            )
        return pd.DataFrame(records)
