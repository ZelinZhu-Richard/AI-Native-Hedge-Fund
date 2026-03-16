"""Tests for the ingestion orchestrator."""

from __future__ import annotations

import datetime

from meridian.data.ingest import ingest_historical, ingest_incremental


class TestIngestHistorical:
    def test_full_pipeline(self, temp_database, mock_provider):
        """End-to-end: fetch -> validate -> store."""
        result = ingest_historical(
            tickers=["AAPL", "MSFT"],
            start_date=datetime.date(2024, 1, 2),
            end_date=datetime.date(2024, 3, 29),
            provider=mock_provider,
            db=temp_database,
            batch_size=10,
        )

        assert result["total"] == 2
        assert result["succeeded"] == 2
        assert result["failed"] == 0
        assert result["skipped"] == 0

        # Verify data was stored
        df = temp_database.get_data("AAPL")
        assert len(df) > 0

        # Verify summary
        summary = temp_database.get_summary()
        assert summary["tickers"] == 2

    def test_resume_after_interruption(self, temp_database, mock_provider):
        """Already-completed tickers should be skipped on resume."""
        start = datetime.date(2024, 1, 2)
        end = datetime.date(2024, 3, 29)

        # Simulate AAPL already done
        temp_database.log_ingestion("AAPL", start, end, "success", 60)

        result = ingest_historical(
            tickers=["AAPL", "MSFT"],
            start_date=start,
            end_date=end,
            provider=mock_provider,
            db=temp_database,
        )

        assert result["skipped"] == 1
        assert result["succeeded"] == 1

    def test_partial_failure(self, temp_database, mock_provider):
        """Individual ticker failure should not kill the batch."""

        def failing_fetch(tickers, start_date, end_date):
            from tests.fixtures.sample_data import generate_random_walk

            result = {}
            for ticker in tickers:
                if ticker == "FAIL":
                    continue  # Return no data
                result[ticker] = generate_random_walk(
                    ticker=ticker,
                    start_date=start_date,
                    num_days=60,
                    seed=42,
                )
            return result

        mock_provider.fetch_bulk.side_effect = failing_fetch

        result = ingest_historical(
            tickers=["AAPL", "FAIL"],
            start_date=datetime.date(2024, 1, 2),
            end_date=datetime.date(2024, 3, 29),
            provider=mock_provider,
            db=temp_database,
        )

        assert result["succeeded"] == 1
        assert result["failed"] == 1

    def test_all_tickers_already_done(self, temp_database, mock_provider):
        """If all tickers are done, return immediately."""
        start = datetime.date(2024, 1, 2)
        end = datetime.date(2024, 3, 29)

        temp_database.log_ingestion("AAPL", start, end, "success", 60)
        temp_database.log_ingestion("MSFT", start, end, "success", 60)

        result = ingest_historical(
            tickers=["AAPL", "MSFT"],
            start_date=start,
            end_date=end,
            provider=mock_provider,
            db=temp_database,
        )

        assert result["skipped"] == 2
        assert result["succeeded"] == 0
        mock_provider.fetch_bulk.assert_not_called()


class TestIngestIncremental:
    def test_incremental_update(self, temp_database, mock_provider):
        """Incremental should fetch from latest date + 1."""
        from tests.fixtures.sample_data import generate_random_walk

        # First, insert some historical data
        bars = generate_random_walk(
            ticker="AAPL",
            start_date=datetime.date(2024, 1, 2),
            num_days=60,
        )
        temp_database.upsert_bars(bars)

        result = ingest_incremental(
            tickers=["AAPL"],
            provider=mock_provider,
            db=temp_database,
        )

        assert result["total"] == 1

    def test_incremental_no_existing_data(self, temp_database, mock_provider):
        """Incremental with no existing data should skip."""
        result = ingest_incremental(
            tickers=["NEW"],
            provider=mock_provider,
            db=temp_database,
        )

        assert result["failed"] == 1  # No existing data to update from
