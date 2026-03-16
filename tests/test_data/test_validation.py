"""Tests for data quality validation."""

from __future__ import annotations

from meridian.data.storage.models import OHLCVBar
from meridian.data.validation.quality import validate_ticker
from tests.fixtures.sample_data import (
    generate_random_walk,
)


class TestValidateTicker:
    def test_clean_data(self, sample_ohlcv_bars):
        """Clean data should produce a clean report."""
        report = validate_ticker(sample_ohlcv_bars, "TEST")
        assert report.ticker == "TEST"
        assert report.total_bars == 100
        # Random walk data may have minor issues but should be mostly clean
        assert len(report.ohlc_violations) == 0

    def test_empty_bars(self):
        """Empty bar list returns a report with zero bars."""
        report = validate_ticker([], "EMPTY")
        assert report.total_bars == 0
        assert report.is_clean

    def test_single_bar(self):
        """Single bar should not crash validation."""
        bars = generate_random_walk(num_days=1)
        report = validate_ticker(bars, "SINGLE")
        assert report.total_bars == 1

    def test_missing_days_detected(self, sample_bars_with_gap):
        """Gaps in trading days should be detected."""
        report = validate_ticker(sample_bars_with_gap, "GAP")
        # Should detect at least some missing days
        assert len(report.missing_days) > 0

    def test_price_outlier_detected(self, sample_bars_with_outlier):
        """Extreme price spikes should be flagged."""
        report = validate_ticker(sample_bars_with_outlier, "OUTLIER")
        # 3x price spike should be detected as outlier
        assert len(report.price_outliers) > 0 or len(report.suspected_splits) > 0

    def test_split_detected(self, sample_bars_with_split):
        """Unadjusted stock splits should be flagged."""
        report = validate_ticker(sample_bars_with_split, "SPLIT")
        assert len(report.suspected_splits) > 0

    def test_volume_anomaly_detected(self):
        """Zero volume on a trading day should be flagged."""
        bars = generate_random_walk(num_days=50)
        # Inject zero volume
        bars[25] = OHLCVBar(
            ticker=bars[25].ticker,
            date=bars[25].date,
            open=bars[25].open,
            high=bars[25].high,
            low=bars[25].low,
            close=bars[25].close,
            volume=0,
            adj_close=bars[25].adj_close,
        )
        report = validate_ticker(bars, "ZERVOL")
        assert len(report.volume_anomalies) >= 1

    def test_stale_data_detected(self):
        """Same close price for 5+ days should be flagged."""
        bars = generate_random_walk(num_days=50)
        stale_price = 100.0
        # Make 6 consecutive bars have the same close
        for i in range(10, 16):
            bars[i] = OHLCVBar(
                ticker=bars[i].ticker,
                date=bars[i].date,
                open=stale_price,
                high=stale_price,
                low=stale_price,
                close=stale_price,
                volume=bars[i].volume,
                adj_close=stale_price,
            )
        report = validate_ticker(bars, "STALE")
        assert len(report.stale_periods) > 0

    def test_insufficient_data_skips_outlier_check(self):
        """With < MIN_DATA_POINTS bars, outlier check is skipped."""
        bars = generate_random_walk(num_days=10)
        report = validate_ticker(bars, "SHORT")
        # Should not flag outliers on such a short series
        assert len(report.price_outliers) == 0

    def test_report_summary(self, sample_ohlcv_bars):
        """Summary dict should have all expected keys."""
        report = validate_ticker(sample_ohlcv_bars, "TEST")
        summary = report.summary()
        assert "ticker" in summary
        assert "total_bars" in summary
        assert "missing_days" in summary
        assert "is_clean" in summary
        assert isinstance(summary["is_clean"], bool)
