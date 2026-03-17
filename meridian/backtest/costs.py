"""Transaction cost models for backtesting.

Three models: realistic (default), zero (signal isolation), high (stress test).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from pydantic import BaseModel


class TradeCost(BaseModel):
    """Breakdown of transaction costs for a single trade."""

    commission: float
    spread_cost: float
    market_impact: float
    total: float


class TransactionCostModel(ABC):
    """Abstract base for transaction cost models."""

    @abstractmethod
    def compute_cost(
        self,
        ticker: str,
        shares: float,
        price: float,
        daily_volume: float,
    ) -> TradeCost:
        """Compute total transaction cost for a trade.

        Args:
            ticker: Stock ticker.
            shares: Number of shares to trade (absolute value).
            price: Execution price per share.
            daily_volume: Average daily volume for market impact.

        Returns:
            TradeCost breakdown.
        """
        ...


class RealisticCostModel(TransactionCostModel):
    """Realistic transaction cost model for US equities.

    Components:
    1. Commission: $0.005/share (institutional rate)
    2. Spread: half-spread based on price level
    3. Market impact: Almgren-Chriss square-root model
    """

    def __init__(
        self,
        commission_per_share: float = 0.005,
        participation_rate: float = 0.01,
    ) -> None:
        self.commission_per_share = commission_per_share
        self.participation_rate = participation_rate

    def compute_cost(
        self,
        ticker: str,
        shares: float,
        price: float,
        daily_volume: float,
    ) -> TradeCost:
        shares = abs(shares)
        if shares == 0 or price == 0:
            return TradeCost(
                commission=0.0,
                spread_cost=0.0,
                market_impact=0.0,
                total=0.0,
            )

        trade_value = shares * price

        # Commission
        commission = shares * self.commission_per_share

        # Half-spread based on price level
        if price > 50:
            half_spread_pct = 0.0002  # 2 bps
        elif price > 10:
            half_spread_pct = 0.0005  # 5 bps
        else:
            half_spread_pct = 0.0010  # 10 bps
        spread_cost = trade_value * half_spread_pct

        # Market impact: sqrt model
        if daily_volume > 0 and price > 0:
            daily_dollar_vol = daily_volume * price
            participation = trade_value / (daily_dollar_vol * self.participation_rate)
            market_impact = spread_cost * math.sqrt(max(participation, 0))
        else:
            market_impact = spread_cost * 0.1  # fallback

        total = commission + spread_cost + market_impact
        return TradeCost(
            commission=commission,
            spread_cost=spread_cost,
            market_impact=market_impact,
            total=total,
        )


class ZeroCostModel(TransactionCostModel):
    """Zero transaction costs. For isolating signal quality only."""

    def compute_cost(
        self,
        ticker: str,
        shares: float,
        price: float,
        daily_volume: float,
    ) -> TradeCost:
        return TradeCost(
            commission=0.0,
            spread_cost=0.0,
            market_impact=0.0,
            total=0.0,
        )


class HighCostModel(TransactionCostModel):
    """Pessimistic cost model (2x realistic). For stress testing."""

    def __init__(self) -> None:
        self._realistic = RealisticCostModel(
            commission_per_share=0.01,
            participation_rate=0.005,
        )

    def compute_cost(
        self,
        ticker: str,
        shares: float,
        price: float,
        daily_volume: float,
    ) -> TradeCost:
        base = self._realistic.compute_cost(ticker, shares, price, daily_volume)
        return TradeCost(
            commission=base.commission * 2,
            spread_cost=base.spread_cost * 2,
            market_impact=base.market_impact * 2,
            total=base.total * 2,
        )
