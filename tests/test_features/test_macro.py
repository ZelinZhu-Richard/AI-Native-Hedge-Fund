"""Tests for MacroFeatureComputer — 6 features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.features.macro import MacroFeatureComputer
from meridian.features.registry import FeatureRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    FeatureRegistry.reset()
    yield
    FeatureRegistry.reset()


@pytest.fixture
def computer():
    return MacroFeatureComputer()


@pytest.fixture
def spy_df():
    """300 days of SPY-like data."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=300)
    returns = np.random.normal(0.0004, 0.01, 300)
    prices = 400.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "adj_close": prices,
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": 50_000_000,
        },
        index=dates,
    )


class TestMacroOutput:
    def test_output_has_6_columns(self, computer, spy_df):
        result = computer.compute(spy_df)
        assert result.shape[1] == 6

    def test_spy_returns_correct(self, computer, spy_df):
        result = computer.compute(spy_df)
        expected_5d = spy_df["adj_close"].pct_change(5)
        pd.testing.assert_series_equal(
            result["spy_returns_5d"], expected_5d, check_names=False
        )


class TestDrawdown:
    def test_spy_drawdown_always_non_positive(self, computer, spy_df):
        result = computer.compute(spy_df)
        dd = result["spy_drawdown"].dropna()
        assert (dd <= 0).all()

    def test_spy_drawdown_zero_at_highs(self, computer):
        """Drawdown should be 0 at all-time highs."""
        dates = pd.bdate_range("2023-01-02", periods=10)
        # Monotonically increasing prices
        prices = np.arange(100, 110, dtype=float)
        df = pd.DataFrame({"adj_close": prices}, index=dates)
        result = computer.compute(df)
        np.testing.assert_allclose(result["spy_drawdown"].values, 0.0)


class TestAbove200SMA:
    def test_binary_values(self, computer, spy_df):
        result = computer.compute(spy_df)
        vals = result["spy_above_200sma"].dropna()
        assert set(vals.unique()).issubset({0.0, 1.0})

    def test_nan_during_warmup(self, computer, spy_df):
        result = computer.compute(spy_df)
        # SMA with min_periods=200 produces first valid at index 199
        assert result["spy_above_200sma"].iloc[:199].isna().all()


class TestDispersion:
    def test_dispersion_from_kwarg(self, computer, spy_df):
        dispersion = pd.Series(0.05, index=spy_df.index)
        result = computer.compute(spy_df, dispersion=dispersion)
        assert result["market_return_dispersion"].notna().any()

    def test_dispersion_nan_without_kwarg(self, computer, spy_df):
        result = computer.compute(spy_df)
        assert result["market_return_dispersion"].isna().all()


class TestEdgeCases:
    def test_empty_dataframe(self, computer):
        result = computer.compute(pd.DataFrame())
        assert result.empty
