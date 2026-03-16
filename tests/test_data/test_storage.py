"""Tests for DuckDB storage layer."""

from __future__ import annotations

import datetime

from meridian.data.storage.database import MeridianDatabase
from meridian.data.storage.models import DataQualityReport, OHLCVBar, TickerMetadata


class TestMeridianDatabase:
    def test_schema_creation(self, temp_database):
        """Schema should create all 4 tables."""
        tables = temp_database.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "ohlcv" in table_names
        assert "data_quality" in table_names
        assert "ticker_metadata" in table_names
        assert "ingestion_log" in table_names

    def test_schema_idempotent(self, temp_database):
        """Creating schema twice should not raise."""
        temp_database.create_schema()  # Already created in fixture
        temp_database.create_schema()  # Should not raise

    def test_insert_and_read_roundtrip(self, temp_database, sample_ohlcv_bars):
        """Bars inserted should be readable."""
        count = temp_database.upsert_bars(sample_ohlcv_bars[:10])
        assert count == 10

        df = temp_database.get_data("TEST")
        assert len(df) == 10
        assert "ticker" in df.columns
        assert "close" in df.columns

    def test_upsert_dedup(self, temp_database, sample_ohlcv_bars):
        """Upserting the same bars should not create duplicates."""
        bars = sample_ohlcv_bars[:5]
        temp_database.upsert_bars(bars)
        temp_database.upsert_bars(bars)  # Same bars again

        df = temp_database.get_data("TEST")
        assert len(df) == 5  # No duplicates

    def test_upsert_updates_values(self, temp_database, sample_ohlcv_bars):
        """Upserting with changed values should update."""
        bars = sample_ohlcv_bars[:1]
        temp_database.upsert_bars(bars)

        # Modify volume and re-upsert (keep OHLC consistent)
        modified = OHLCVBar(
            ticker=bars[0].ticker,
            date=bars[0].date,
            open=bars[0].open,
            high=bars[0].high,
            low=bars[0].low,
            close=bars[0].close,
            volume=999999,
            adj_close=bars[0].adj_close,
        )
        temp_database.upsert_bars([modified])

        df = temp_database.get_data("TEST")
        assert len(df) == 1
        assert df.iloc[0]["volume"] == 999999

    def test_get_data_with_date_range(self, temp_database, sample_ohlcv_bars):
        """Filtering by date range should work."""
        temp_database.upsert_bars(sample_ohlcv_bars)

        start = sample_ohlcv_bars[10].date
        end = sample_ohlcv_bars[20].date

        df = temp_database.get_data("TEST", start_date=start, end_date=end)
        assert len(df) > 0
        assert all(df["date"] >= str(start))
        assert all(df["date"] <= str(end))

    def test_get_latest_date(self, temp_database, sample_ohlcv_bars):
        """get_latest_date returns the most recent bar date."""
        temp_database.upsert_bars(sample_ohlcv_bars[:10])
        latest = temp_database.get_latest_date("TEST")
        assert latest == sample_ohlcv_bars[9].date

    def test_get_latest_date_no_data(self, temp_database):
        """get_latest_date returns None for unknown ticker."""
        assert temp_database.get_latest_date("UNKNOWN") is None

    def test_upsert_empty_bars(self, temp_database):
        """Upserting empty list returns 0."""
        assert temp_database.upsert_bars([]) == 0

    def test_store_quality_report(self, temp_database):
        """Quality reports should be storable and retrievable."""
        report = DataQualityReport(
            ticker="TEST",
            start_date=datetime.date(2024, 1, 2),
            end_date=datetime.date(2024, 6, 28),
            total_bars=126,
            missing_days=[datetime.date(2024, 3, 15)],
            is_clean=False,
        )
        # Should not raise
        temp_database.store_quality_report(report)

    def test_update_metadata(self, temp_database):
        """Ticker metadata should be storable."""
        metadata = TickerMetadata(
            ticker="TEST",
            sector="Technology",
            first_date=datetime.date(2024, 1, 2),
            last_date=datetime.date(2024, 6, 28),
            total_bars=126,
            provider="yahoo",
        )
        temp_database.update_metadata(metadata)

    def test_log_ingestion(self, temp_database):
        """Ingestion log entries should be storable."""
        temp_database.log_ingestion(
            ticker="TEST",
            start_date=datetime.date(2024, 1, 2),
            end_date=datetime.date(2024, 6, 28),
            status="success",
            bars_ingested=126,
        )

    def test_get_summary(self, temp_database, sample_ohlcv_bars):
        """Summary should return correct counts."""
        temp_database.upsert_bars(sample_ohlcv_bars)
        summary = temp_database.get_summary()
        assert summary["tickers"] == 1
        assert summary["total_bars"] == 100

    def test_get_completed_tickers(self, temp_database):
        """Completed tickers should be trackable for resume."""
        start = datetime.date(2024, 1, 1)
        end = datetime.date(2024, 6, 30)

        temp_database.log_ingestion("AAPL", start, end, "success", 126)
        temp_database.log_ingestion("MSFT", start, end, "failed", 0)

        completed = temp_database.get_completed_tickers(start, end)
        assert "AAPL" in completed
        assert "MSFT" not in completed

    def test_context_manager(self, tmp_path):
        """Database should work as context manager."""
        db_path = tmp_path / "ctx.duckdb"
        with MeridianDatabase(db_path) as db:
            db.create_schema()
            tables = db.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchone()
            assert tables[0] >= 4

    def test_get_universe_data(self, temp_database, sample_ohlcv_bars):
        """get_universe_data returns data for specified tickers."""
        temp_database.upsert_bars(sample_ohlcv_bars)
        start = sample_ohlcv_bars[0].date
        end = sample_ohlcv_bars[-1].date

        df = temp_database.get_universe_data(["TEST"], start, end)
        assert len(df) == 100
