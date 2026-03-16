"""Tests for VolatilityFeatureComputer — 10 features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.features.registry import FeatureRegistry
from meridian.features.volatility import VolatilityFeatureComputer


@pytest.fixture(autouse=True)
def reset_registry():
    FeatureRegistry.reset()
    yield
    FeatureRegistry.reset()


@pytest.fixture
def computer():
    return VolatilityFeatureComputer()


@pytest.fixture
def constant_vol_df():
    """DataFrame with nearly constant volatility for ratio testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=300)
    # Very small, constant daily moves
    returns = np.random.normal(0.0005, 0.01, 300)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "adj_close": prices,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "open": prices * 1.0,
            "close": prices,
        },
        index=dates,
    )


class TestVolatilityOutputShape:
    def test_output_has_10_columns(self, computer, long_ohlcv_dataframe):
        result = computer.compute(long_ohlcv_dataframe)
        assert result.shape[1] == 10

    def test_column_names_match_configs(self, computer, long_ohlcv_dataframe):
        result = computer.compute(long_ohlcv_dataframe)
        config_names = {c.name for c in computer.feature_configs()}
        assert set(result.columns) == config_names


class TestVolValues:
    def test_all_vol_non_negative(self, computer, long_ohlcv_dataframe):
        result = computer.compute(long_ohlcv_dataframe)
        for col in ["realized_vol_5d", "realized_vol_21d", "realized_vol_63d",
                     "garman_klass_vol_21d", "parkinson_vol_21d"]:
            vals = result[col].dropna()
            assert (vals >= 0).all(), f"{col} should be non-negative"

    def test_vol_ratio_near_one_constant_vol(self, computer, constant_vol_df):
        result = computer.compute(constant_vol_df)
        vol_ratio = result["vol_ratio_5_21"].dropna()
        # With constant vol, ratio should be approximately 1
        assert abs(vol_ratio.median() - 1.0) < 0.5

    def test_garman_klass_reasonable(self, computer, long_ohlcv_dataframe):
        result = computer.compute(long_ohlcv_dataframe)
        gk = result["garman_klass_vol_21d"].dropna()
        rv = result["realized_vol_21d"].dropna()
        # GK and realized vol should be in same ballpark (both annualized)
        if len(gk) > 0 and len(rv) > 0:
            assert gk.median() > 0
            assert rv.median() > 0

    def test_parkinson_non_negative(self, computer, long_ohlcv_dataframe):
        result = computer.compute(long_ohlcv_dataframe)
        parkinson = result["parkinson_vol_21d"].dropna()
        assert (parkinson >= 0).all()


class TestMinPeriods:
    def test_warmup_enforced(self, computer, long_ohlcv_dataframe):
        result = computer.compute(long_ohlcv_dataframe)
        # realized_vol_5d needs 5 returns (so 6 prices minimum)
        assert result["realized_vol_5d"].iloc[:5].isna().all()
        assert result["realized_vol_21d"].iloc[:21].isna().all()


class TestVolEdgeCases:
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
                "open": [100, 100, 101, 99, 102],
                "close": [100, 101, 99, 102, 100],
            },
            index=dates,
        )
        result = computer.compute(df)
        assert result.shape[0] == 5

    def test_high_low_range_ratio_positive(self, computer, long_ohlcv_dataframe):
        result = computer.compute(long_ohlcv_dataframe)
        hlr = result["high_low_range_ratio"].dropna()
        assert (hlr >= 0).all()

    def test_deterministic(self, computer, long_ohlcv_dataframe):
        r1 = computer.compute(long_ohlcv_dataframe)
        r2 = computer.compute(long_ohlcv_dataframe)
        pd.testing.assert_frame_equal(r1, r2)
