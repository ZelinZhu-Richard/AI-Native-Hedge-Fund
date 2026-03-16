"""Tests for FeatureStore — DuckDB feature storage."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from meridian.features.registry import FeatureConfig, FeatureRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    FeatureRegistry.reset()
    yield
    FeatureRegistry.reset()


@pytest.fixture
def sample_features():
    """Sample wide-format feature DataFrame."""
    dates = pd.bdate_range("2023-06-01", periods=20)
    np.random.seed(42)
    return pd.DataFrame(
        {
            "returns_1d": np.random.randn(20) * 0.02,
            "rsi_14": np.random.uniform(30, 70, 20),
            "realized_vol_21d": np.random.uniform(0.1, 0.3, 20),
        },
        index=dates,
    )


class TestSchemaCreation:
    def test_schema_idempotent(self, feature_store):
        # Schema already created by fixture; calling again should not raise
        feature_store.create_schema()

    def test_tables_exist(self, feature_store):
        tables = feature_store.db.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "feature_metadata" in table_names
        assert "feature_values" in table_names


class TestStoreAndRetrieve:
    def test_roundtrip(self, feature_store, sample_features):
        feature_store.store_features("AAPL", sample_features)
        result = feature_store.get_features("AAPL")
        assert len(result) > 0
        assert set(result["feature_name"].unique()) == set(sample_features.columns)

    def test_store_returns_row_count(self, feature_store, sample_features):
        count = feature_store.store_features("AAPL", sample_features)
        assert count > 0
        # 20 dates × 3 features = 60 max (fewer if NaN exists)
        assert count <= 60

    def test_get_feature_matrix(self, feature_store, sample_features):
        feature_store.store_features("AAPL", sample_features)
        feature_store.store_features("MSFT", sample_features)
        matrix = feature_store.get_feature_matrix(["AAPL", "MSFT"])
        assert not matrix.empty
        assert isinstance(matrix.index, pd.MultiIndex)
        # Should have 3 feature columns
        assert matrix.shape[1] == 3


class TestMetadata:
    def test_store_metadata(self, feature_store):
        configs = [
            FeatureConfig("returns_1d", "technical", 1, description="1-day return"),
            FeatureConfig("rsi_14", "technical", 14, description="14-day RSI"),
        ]
        feature_store.store_feature_metadata(configs)
        result = feature_store.db.conn.execute(
            "SELECT * FROM feature_metadata"
        ).fetchdf()
        assert len(result) == 2


class TestDateFiltering:
    def test_date_range(self, feature_store, sample_features):
        feature_store.store_features("AAPL", sample_features)
        start = datetime.date(2023, 6, 5)
        end = datetime.date(2023, 6, 15)
        result = feature_store.get_features("AAPL", start_date=start, end_date=end)
        if not result.empty:
            dates = pd.to_datetime(result["date"])
            assert dates.min().date() >= start
            assert dates.max().date() <= end


class TestUpsert:
    def test_upsert_overwrites(self, feature_store):
        dates = pd.bdate_range("2023-06-01", periods=5)
        df1 = pd.DataFrame({"returns_1d": [0.01, 0.02, 0.03, 0.04, 0.05]}, index=dates)
        df2 = pd.DataFrame({"returns_1d": [0.10, 0.20, 0.30, 0.40, 0.50]}, index=dates)
        feature_store.store_features("AAPL", df1)
        feature_store.store_features("AAPL", df2)
        result = feature_store.get_features("AAPL")
        # Values should be from df2
        values = result["value"].tolist()
        assert all(v >= 0.10 for v in values)


class TestCoverage:
    def test_get_feature_coverage(self, feature_store, sample_features):
        feature_store.store_features("AAPL", sample_features)
        coverage = feature_store.get_feature_coverage()
        assert coverage["tickers"] == 1
        assert coverage["features"] == 3
        assert coverage["total_rows"] > 0

    def test_get_latest_feature_date(self, feature_store, sample_features):
        feature_store.store_features("AAPL", sample_features)
        latest = feature_store.get_latest_feature_date("AAPL")
        assert latest is not None

    def test_get_latest_feature_date_no_data(self, feature_store):
        latest = feature_store.get_latest_feature_date("NONEXISTENT")
        assert latest is None

    def test_get_available_features(self, feature_store, sample_features):
        feature_store.store_features("AAPL", sample_features)
        available = feature_store.get_available_features()
        assert set(available) == set(sample_features.columns)
