"""Tests for transaction cost models."""

from __future__ import annotations

from meridian.backtest.costs import (
    HighCostModel,
    RealisticCostModel,
    TradeCost,
    ZeroCostModel,
)


class TestTradeCost:
    def test_trade_cost_fields(self):
        tc = TradeCost(commission=1.0, spread_cost=2.0, market_impact=3.0, total=6.0)
        assert tc.commission == 1.0
        assert tc.spread_cost == 2.0
        assert tc.market_impact == 3.0
        assert tc.total == 6.0


class TestRealisticCostModel:
    def test_nonzero_cost_for_normal_trade(self):
        model = RealisticCostModel()
        cost = model.compute_cost("AAPL", 100, 150.0, 50_000_000)
        assert cost.total > 0
        assert cost.commission > 0
        assert cost.spread_cost > 0
        assert cost.market_impact > 0

    def test_zero_shares_returns_zero(self):
        model = RealisticCostModel()
        cost = model.compute_cost("AAPL", 0, 150.0, 50_000_000)
        assert cost.total == 0.0

    def test_higher_price_lower_spread_pct(self):
        model = RealisticCostModel()
        high_price = model.compute_cost("AAPL", 100, 200.0, 50_000_000)
        low_price = model.compute_cost("PENNY", 100, 5.0, 50_000_000)
        # Spread as % of trade value should be higher for cheap stocks
        high_spread_pct = high_price.spread_cost / (100 * 200)
        low_spread_pct = low_price.spread_cost / (100 * 5)
        assert low_spread_pct > high_spread_pct

    def test_larger_trade_higher_impact(self):
        model = RealisticCostModel()
        small = model.compute_cost("AAPL", 100, 150.0, 1_000_000)
        large = model.compute_cost("AAPL", 10_000, 150.0, 1_000_000)
        assert large.market_impact > small.market_impact


class TestZeroCostModel:
    def test_always_zero(self):
        model = ZeroCostModel()
        cost = model.compute_cost("AAPL", 1000, 150.0, 50_000_000)
        assert cost.total == 0.0
        assert cost.commission == 0.0
        assert cost.spread_cost == 0.0
        assert cost.market_impact == 0.0


class TestHighCostModel:
    def test_higher_than_realistic(self):
        realistic = RealisticCostModel()
        high = HighCostModel()
        r_cost = realistic.compute_cost("AAPL", 100, 150.0, 50_000_000)
        h_cost = high.compute_cost("AAPL", 100, 150.0, 50_000_000)
        assert h_cost.total > r_cost.total
