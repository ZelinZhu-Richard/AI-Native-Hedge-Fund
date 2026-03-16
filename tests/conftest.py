"""Shared test fixtures."""

from __future__ import annotations

import pytest

from meridian.data.storage.database import MeridianDatabase
from meridian.data.storage.models import OHLCVBar
from tests.fixtures.sample_data import (
    generate_random_walk,
    inject_gap,
    inject_outlier,
    inject_split,
)


@pytest.fixture
def sample_ohlcv_bars() -> list[OHLCVBar]:
    """100 days of realistic OHLCV data."""
    return generate_random_walk(ticker="TEST", num_days=100, seed=42)


@pytest.fixture
def sample_bars_with_gap() -> list[OHLCVBar]:
    """Bars with 3 missing trading days."""
    bars = generate_random_walk(ticker="GAP", num_days=100, seed=42)
    return inject_gap(bars, gap_start_idx=10, gap_days=3)


@pytest.fixture
def sample_bars_with_split() -> list[OHLCVBar]:
    """Bars with a simulated 2:1 stock split."""
    bars = generate_random_walk(ticker="SPLIT", num_days=100, seed=42)
    return inject_split(bars, split_idx=50, split_ratio=0.5)


@pytest.fixture
def sample_bars_with_outlier() -> list[OHLCVBar]:
    """Bars with a 3x price spike."""
    bars = generate_random_walk(ticker="OUTLIER", num_days=100, seed=42)
    return inject_outlier(bars, outlier_idx=60, multiplier=3.0)


@pytest.fixture
def temp_database(tmp_path) -> MeridianDatabase:
    """In-memory DuckDB database with schema created."""
    db_path = tmp_path / "test.duckdb"
    db = MeridianDatabase(db_path)
    db.__enter__()
    db.create_schema()
    yield db  # type: ignore[misc]
    db.__exit__(None, None, None)


@pytest.fixture
def mock_provider():
    """A mock data provider that returns deterministic data."""
    from unittest.mock import MagicMock

    from meridian.data.providers.base import DataProvider

    provider = MagicMock(spec=DataProvider)
    provider.provider_name = "mock"

    def fake_fetch_bulk(tickers, start_date, end_date):
        result = {}
        for ticker in tickers:
            result[ticker] = generate_random_walk(
                ticker=ticker,
                start_date=start_date,
                num_days=60,
                seed=hash(ticker) % 10000,
            )
        return result

    provider.fetch_bulk.side_effect = fake_fetch_bulk

    def fake_fetch_historical(ticker, start_date, end_date):
        return generate_random_walk(
            ticker=ticker,
            start_date=start_date,
            num_days=60,
            seed=hash(ticker) % 10000,
        )

    provider.fetch_historical.side_effect = fake_fetch_historical
    provider.health_check.return_value = True

    return provider
