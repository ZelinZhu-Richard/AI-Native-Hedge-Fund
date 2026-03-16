"""Anti-lookahead tests — MOST IMPORTANT test file.

Verifies that feature computations are strictly causal: future data
never leaks into present-day features. A single leak invalidates
every backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.features.cross_sectional import CrossSectionalFeatureComputer
from meridian.features.macro import MacroFeatureComputer
from meridian.features.registry import FeatureRegistry
from meridian.features.technical import TechnicalFeatureComputer
from meridian.features.volatility import VolatilityFeatureComputer


@pytest.fixture(autouse=True)
def reset_registry():
    FeatureRegistry.reset()
    yield
    FeatureRegistry.reset()


@pytest.fixture
def full_df():
    """300-day DataFrame for truncation tests."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=300)
    returns = np.random.normal(0.0005, 0.02, 300)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "adj_close": prices,
            "high": prices * (1 + np.abs(np.random.normal(0, 0.01, 300))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.01, 300))),
            "close": prices,
            "open": prices * (1 + np.random.normal(0, 0.005, 300)),
            "volume": np.random.randint(500_000, 2_000_000, 300).astype(float),
        },
        index=dates,
    )


class TestTechnicalNoLookahead:
    def test_truncation_invariance(self, full_df):
        """Features at date T computed on full data must match features
        computed on data truncated at date T."""
        computer = TechnicalFeatureComputer()
        cutoff = 200  # Compute on first 200 days vs full 300

        full_result = computer.compute(full_df)
        truncated_result = computer.compute(full_df.iloc[:cutoff])

        # Values at overlapping dates must match exactly
        overlap = full_result.iloc[:cutoff]
        pd.testing.assert_frame_equal(
            overlap, truncated_result,
            check_names=False,
            obj="technical features",
        )


class TestVolatilityNoLookahead:
    def test_truncation_invariance(self, full_df):
        """Volatility features must not change when future data is added."""
        computer = VolatilityFeatureComputer()
        cutoff = 200

        full_result = computer.compute(full_df)
        truncated_result = computer.compute(full_df.iloc[:cutoff])

        overlap = full_result.iloc[:cutoff]
        pd.testing.assert_frame_equal(
            overlap, truncated_result,
            check_names=False,
            obj="volatility features",
        )


class TestCrossSectionalNoLookahead:
    def test_new_ticker_no_retroactive_effect(self):
        """Adding a new ticker at date X must not change existing tickers'
        features before date X."""
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-02", periods=150)
        midpoint = 100

        # Base: two tickers for full period
        base_data = []
        for ticker in ["AAPL", "MSFT"]:
            prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 150)))
            for i, d in enumerate(dates):
                base_data.append({
                    "date": d, "ticker": ticker,
                    "adj_close": prices[i],
                    "volume": float(np.random.randint(500000, 2000000)),
                })

        df_without = pd.DataFrame(base_data)

        # With new ticker starting at midpoint
        new_ticker_data = base_data.copy()
        prices = 50.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.03, 150 - midpoint)))
        for i, d in enumerate(dates[midpoint:]):
            new_ticker_data.append({
                "date": d, "ticker": "NEWCO",
                "adj_close": prices[i],
                "volume": float(np.random.randint(100000, 500000)),
            })

        df_with = pd.DataFrame(new_ticker_data)

        sector_map = {"AAPL": "Tech", "MSFT": "Tech", "NEWCO": "Tech"}
        computer = CrossSectionalFeatureComputer()

        result_without = computer.compute(df_without, sector_map=sector_map)
        result_with = computer.compute(df_with, sector_map=sector_map)

        # For AAPL before midpoint, features should be identical
        aapl_before_without = result_without[
            (result_without["ticker"] == "AAPL") &
            (result_without.index < dates[midpoint])
        ].drop(columns=["ticker"])

        aapl_before_with = result_with[
            (result_with["ticker"] == "AAPL") &
            (result_with.index < dates[midpoint])
        ].drop(columns=["ticker"])

        pd.testing.assert_frame_equal(
            aapl_before_without, aapl_before_with,
            check_names=False,
            obj="cross-sectional features before new ticker",
        )


class TestMacroNoLookahead:
    def test_truncation_invariance(self, full_df):
        """Macro features must not change when future data is added."""
        computer = MacroFeatureComputer()
        cutoff = 200

        full_result = computer.compute(full_df)
        truncated_result = computer.compute(full_df.iloc[:cutoff])

        overlap = full_result.iloc[:cutoff]
        pd.testing.assert_frame_equal(
            overlap, truncated_result,
            check_names=False,
            obj="macro features",
        )


class TestNaNWarmup:
    def test_every_feature_has_nan_warmup(self, full_df):
        """Every feature with lookback > 0 should have NaN for at least
        its first lookback_days rows."""
        registry = FeatureRegistry.instance()

        tech = TechnicalFeatureComputer()
        vol = VolatilityFeatureComputer()

        tech_result = tech.compute(full_df)
        vol_result = vol.compute(full_df)

        for config in registry.list_features():
            if config.lookback_days == 0:
                continue
            if config.category in ("cross_sectional",):
                continue  # Cross-sectional tested separately

            if config.name in tech_result.columns:
                series = tech_result[config.name]
            elif config.name in vol_result.columns:
                series = vol_result[config.name]
            else:
                continue  # macro features tested with SPY data

            # At least some of the first lookback_days should be NaN
            warmup = series.iloc[:config.lookback_days]
            assert warmup.isna().any(), (
                f"Feature {config.name} (lookback={config.lookback_days}) "
                f"should have NaN in warmup period"
            )


class TestExpandingOperationsCausal:
    def test_spy_drawdown_is_causal(self, full_df):
        """spy_drawdown uses expanding().max() which is inherently causal."""
        computer = MacroFeatureComputer()
        cutoff = 200

        full_result = computer.compute(full_df)
        truncated_result = computer.compute(full_df.iloc[:cutoff])

        # Drawdown values must match exactly at overlapping dates
        full_dd = full_result["spy_drawdown"].iloc[:cutoff]
        trunc_dd = truncated_result["spy_drawdown"]
        pd.testing.assert_series_equal(full_dd, trunc_dd, check_names=False)
