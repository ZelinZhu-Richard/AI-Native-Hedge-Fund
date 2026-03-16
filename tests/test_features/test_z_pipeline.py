"""Tests for FeaturePipeline — end-to-end orchestration.

These tests use a single pipeline instance to avoid DuckDB connection
lifecycle issues. The pipeline is created once and run once, then all
assertions are made on the results.
"""

from __future__ import annotations

import datetime

from meridian.data.storage.database import MeridianDatabase
from meridian.features.pipeline import FeaturePipeline
from meridian.features.registry import FeatureRegistry
from meridian.features.store import FeatureStore
from tests.fixtures.sample_data import generate_random_walk

SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOG": "Technology",
    "AMZN": "Consumer",
    "META": "Technology",
    "SPY": "Index",
}


class TestPipeline:
    """All pipeline tests in a single class to share state."""

    def test_pipeline_end_to_end(self, tmp_path):
        """Single comprehensive test covering pipeline functionality."""
        FeatureRegistry.reset()

        # Setup: create DB with data
        db_path = tmp_path / "pipeline.duckdb"
        db = MeridianDatabase(db_path)
        db.__enter__()
        try:
            db.create_schema()

            tickers = [
                "AAPL",
                "MSFT",
                "GOOG",
                "AMZN",
                "META",
                "SPY",
            ]
            start_date = datetime.date(2023, 1, 2)
            for i, ticker in enumerate(tickers):
                bars = generate_random_walk(
                    ticker=ticker,
                    start_date=start_date,
                    num_days=300,
                    seed=42 + i,
                )
                db.upsert_bars(bars)

            store = FeatureStore(db)
            store.create_schema()
            pipeline = FeaturePipeline(db, store)

            # Verify 49 features registered
            registry = FeatureRegistry.instance()
            features = registry.list_features()
            assert len(features) == 49

            # Run full pipeline
            result = pipeline.run(
                tickers=["AAPL", "MSFT", "GOOG", "AMZN", "META"],
                start_date=datetime.date(2023, 6, 1),
                end_date=datetime.date(2023, 12, 31),
                sector_map=SECTOR_MAP,
            )

            # Check summary dict structure
            assert "tickers_processed" in result
            assert "tickers_failed" in result
            assert "features_stored" in result
            assert "feature_count" in result
            assert "failures" in result

            # Check counts
            assert result["tickers_processed"] == 5
            assert result["features_stored"] > 0
            assert result["feature_count"] == 49

            # Check features stored
            available = store.get_available_features()
            assert len(available) > 20

            # Check feature matrix
            matrix = store.get_feature_matrix(["AAPL", "MSFT"])
            assert not matrix.empty
            assert matrix.shape[1] > 20

            # Run with missing ticker — should not crash
            result2 = pipeline.run(
                tickers=["AAPL", "NONEXISTENT"],
                start_date=datetime.date(2023, 6, 1),
                end_date=datetime.date(2023, 12, 31),
                sector_map=SECTOR_MAP,
            )
            assert result2["tickers_processed"] >= 1
            assert result2["tickers_failed"] >= 1

        finally:
            db.__exit__(None, None, None)
            FeatureRegistry.reset()
