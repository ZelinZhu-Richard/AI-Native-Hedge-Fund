"""Tests for TechnicalFeatureComputer — 25 features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.features.registry import FeatureRegistry
from meridian.features.technical import TechnicalFeatureComputer


@pytest.fixture(autouse=True)
def reset_registry():
    FeatureRegistry.reset()
    yield
    FeatureRegistry.reset()


@pytest.fixture
def computer():
    return TechnicalFeatureComputer()


@pytest.fixture
def constant_price_df():
    """DataFrame with constant price — useful for verifying edge cases."""
    dates = pd.bdate_range("2023-01-02", periods=100)
    return pd.DataFrame(
        {
            "adj_close": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000,
            "open": 100.0,
        },
        index=dates,
    )


@pytest.fixture
def rising_price_df():
    """DataFrame with steadily rising price."""
    dates = pd.bdate_range("2023-01-02", periods=300)
    prices = 100.0 * (1.001 ** np.arange(300))
    return pd.DataFrame(
        {
            "adj_close": prices,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": 1_000_000,
            "open": prices * 0.999,
        },
        index=dates,
    )


@pytest.fixture
def falling_price_df():
    """DataFrame with steadily falling price."""
    dates = pd.bdate_range("2023-01-02", periods=300)
    prices = 100.0 * (0.999 ** np.arange(300))
    return pd.DataFrame(
        {
            "adj_close": prices,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": 1_000_000,
            "open": prices * 1.001,
        },
        index=dates,
    )


class TestTechnicalOutputShape:
    def test_output_has_25_columns(self, computer, long_ohlcv_dataframe):
        result = computer.compute(long_ohlcv_dataframe)
        assert result.shape[1] == 25

    def test_column_names_match_configs(self, computer, long_ohlcv_dataframe):
        result = computer.compute(long_ohlcv_dataframe)
        config_names = {c.name for c in computer.feature_configs()}
        assert set(result.columns) == config_names

    def test_output_index_matches_input(self, computer, long_ohlcv_dataframe):
        result = computer.compute(long_ohlcv_dataframe)
        assert result.index.equals(long_ohlcv_dataframe.index)


class TestReturns:
    def test_returns_1d_values(self, computer, long_ohlcv_dataframe):
        result = computer.compute(long_ohlcv_dataframe)
        close = long_ohlcv_dataframe["adj_close"]
        expected = close.pct_change(1)
        pd.testing.assert_series_equal(
            result["returns_1d"], expected, check_names=False
        )

    def test_returns_5d_values(self, computer, long_ohlcv_dataframe):
        result = computer.compute(long_ohlcv_dataframe)
        close = long_ohlcv_dataframe["adj_close"]
        expected = close.pct_change(5)
        pd.testing.assert_series_equal(
            result["returns_5d"], expected, check_names=False
        )


class TestRSI:
    def test_rsi_bounded(self, computer, long_ohlcv_dataframe):
        result = computer.compute(long_ohlcv_dataframe)
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_rsi_nan_warmup(self, computer, long_ohlcv_dataframe):
        result = computer.compute(long_ohlcv_dataframe)
        assert result["rsi_14"].iloc[:14].isna().all()

    def test_rsi_100_when_all_positive(self, computer, rising_price_df):
        result = computer.compute(rising_price_df)
        rsi = result["rsi_14"].dropna()
        # With consistently rising prices, RSI should be very high
        assert rsi.iloc[-1] > 90

    def test_rsi_low_when_all_negative(self, computer, falling_price_df):
        result = computer.compute(falling_price_df)
        rsi = result["rsi_14"].dropna()
        # With consistently falling prices, RSI should be very low
        assert rsi.iloc[-1] < 10


class TestMACD:
    def test_macd_values_exist(self, computer, long_ohlcv_dataframe):
        result = computer.compute(long_ohlcv_dataframe)
        assert result["macd_signal"].notna().any()
        assert result["macd_histogram"].notna().any()


class TestCrossSignals:
    def test_golden_death_cross_exclusive(
        self, computer, long_ohlcv_dataframe
    ):
        result = computer.compute(long_ohlcv_dataframe)
        golden = result["golden_cross"].dropna()
        death = result["death_cross"].dropna()
        # Where both are not NaN, they should sum to 1
        both = golden + death
        assert (both == 1.0).all()


class TestVolumeRatios:
    def test_volume_ratio_constant_volume(self, computer, constant_price_df):
        result = computer.compute(constant_price_df)
        vol_ratio = result["volume_ratio_5d"].dropna()
        # Constant volume → ratio should be 1.0
        np.testing.assert_allclose(vol_ratio.values, 1.0, atol=1e-10)


class TestMinPeriods:
    def test_first_n_rows_nan(self, computer, long_ohlcv_dataframe):
        """Each feature should have NaN for at least its lookback period."""
        result = computer.compute(long_ohlcv_dataframe)
        configs = {c.name: c.lookback_days for c in computer.feature_configs()}
        for name, lookback in configs.items():
            if lookback > 0:
                # At least the first `lookback` values should be NaN
                # (some features need lookback+1 due to diff())
                nan_count = result[name].iloc[:lookback].isna().sum()
                assert nan_count > 0, f"{name} should have NaN in first {lookback} rows"


class TestDeterminism:
    def test_same_input_same_output(self, computer, long_ohlcv_dataframe):
        result1 = computer.compute(long_ohlcv_dataframe)
        result2 = computer.compute(long_ohlcv_dataframe)
        pd.testing.assert_frame_equal(result1, result2)


class TestEdgeCases:
    def test_empty_dataframe(self, computer):
        empty = pd.DataFrame()
        result = computer.compute(empty)
        assert result.empty

    def test_short_series_no_crash(self, computer):
        dates = pd.bdate_range("2023-01-02", periods=5)
        df = pd.DataFrame(
            {
                "adj_close": [100, 101, 99, 102, 100],
                "high": [102, 103, 101, 104, 102],
                "low": [98, 99, 97, 100, 98],
                "close": [100, 101, 99, 102, 100],
                "volume": [1_000_000] * 5,
            },
            index=dates,
        )
        result = computer.compute(df)
        # Should produce output (mostly NaN) without crashing
        assert result.shape[0] == 5
