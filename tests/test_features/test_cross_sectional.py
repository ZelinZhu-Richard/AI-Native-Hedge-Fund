"""Tests for CrossSectionalFeatureComputer — 8 features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.core.exceptions import FeatureComputationError
from meridian.features.cross_sectional import CrossSectionalFeatureComputer
from meridian.features.registry import FeatureRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    FeatureRegistry.reset()
    yield
    FeatureRegistry.reset()


@pytest.fixture
def computer():
    return CrossSectionalFeatureComputer()


@pytest.fixture
def sector_map():
    return {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOG": "Technology",
        "AMZN": "Consumer",
        "META": "Technology",
    }


@pytest.fixture
def stacked_df(multi_ticker_ohlcv):
    """Stacked DataFrame with all tickers for cross-sectional computation."""
    frames = []
    for ticker, df in multi_ticker_ohlcv.items():
        frame = df[["adj_close", "volume"]].copy()
        frame["ticker"] = ticker
        frames.append(frame.reset_index())
    return pd.concat(frames, ignore_index=True)


class TestCrossSectionalOutput:
    def test_output_has_correct_columns(self, computer, stacked_df, sector_map):
        result = computer.compute(stacked_df, sector_map=sector_map)
        expected_cols = {c.name for c in computer.feature_configs()} | {"ticker"}
        assert expected_cols == set(result.columns)

    def test_rank_returns_bounded(self, computer, stacked_df, sector_map):
        result = computer.compute(stacked_df, sector_map=sector_map)
        rank_21 = result["rank_returns_21d"].dropna()
        assert (rank_21 >= 0).all()
        assert (rank_21 <= 1).all()

    def test_rank_returns_63d_bounded(self, computer, stacked_df, sector_map):
        result = computer.compute(stacked_df, sector_map=sector_map)
        rank_63 = result["rank_returns_63d"].dropna()
        assert (rank_63 >= 0).all()
        assert (rank_63 <= 1).all()

    def test_market_breadth_bounded(self, computer, stacked_df, sector_map):
        result = computer.compute(stacked_df, sector_map=sector_map)
        breadth = result["market_breadth"].dropna()
        assert (breadth >= 0).all()
        assert (breadth <= 1).all()

    def test_dispersion_non_negative(self, computer, stacked_df, sector_map):
        result = computer.compute(stacked_df, sector_map=sector_map)
        disp = result["dispersion_21d"].dropna()
        assert (disp >= 0).all()


class TestSectorRelative:
    def test_sector_relative_sums_near_zero(self, computer, stacked_df, sector_map):
        result = computer.compute(stacked_df, sector_map=sector_map)
        # Within a sector, relative returns should approximately cancel out
        tech_tickers = [t for t, s in sector_map.items() if s == "Technology"]
        for date in result.index.unique()[:5]:  # Check a few dates
            date_mask = result.index == date
            date_data = result.loc[date_mask]
            tech_data = date_data[date_data["ticker"].isin(tech_tickers)]
            rel = tech_data["sector_relative_return_21d"].dropna()
            if len(rel) > 1:
                assert abs(rel.sum()) < 1e-10


class TestMissingSectorMap:
    def test_raises_without_sector_map(self, computer, stacked_df):
        with pytest.raises(FeatureComputationError):
            computer.compute(stacked_df)


class TestSingleTicker:
    def test_single_ticker_no_crash(self, computer, sector_map):
        dates = pd.bdate_range("2023-01-02", periods=100)
        df = pd.DataFrame(
            {
                "date": dates,
                "adj_close": np.random.RandomState(42).random(100) * 100 + 50,
                "volume": np.random.RandomState(42).randint(100000, 1000000, 100),
                "ticker": "AAPL",
            }
        )
        result = computer.compute(df, sector_map=sector_map)
        assert len(result) > 0


class TestEdgeCases:
    def test_empty_dataframe(self, computer):
        result = computer.compute(pd.DataFrame(), sector_map={})
        assert result.empty
