"""Tests for data providers (all mocked, no real API calls)."""

from __future__ import annotations

import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd

from meridian.data.providers.yahoo import YahooFinanceProvider


class TestYahooFinanceProvider:
    def setup_method(self):
        self.provider = YahooFinanceProvider(
            rate_limit_per_second=100.0,  # Fast for tests
            max_retries=2,
            retry_base_delay=0.01,
        )

    def _make_single_ticker_df(
        self,
        ticker: str = "AAPL",
        num_days: int = 5,
        start_date: datetime.date = datetime.date(2024, 1, 2),
    ) -> pd.DataFrame:
        """Create a DataFrame mimicking single-ticker yfinance output."""
        dates = pd.bdate_range(start=start_date, periods=num_days)
        data = {
            "Open": [150.0 + i for i in range(num_days)],
            "High": [155.0 + i for i in range(num_days)],
            "Low": [148.0 + i for i in range(num_days)],
            "Close": [152.0 + i for i in range(num_days)],
            "Adj Close": [152.0 + i for i in range(num_days)],
            "Volume": [1000000 + i * 100000 for i in range(num_days)],
        }
        return pd.DataFrame(data, index=dates)

    def _make_multi_ticker_df(
        self,
        tickers: list[str],
        num_days: int = 5,
    ) -> pd.DataFrame:
        """Create a multi-level column DataFrame mimicking bulk yfinance output."""
        dates = pd.bdate_range(start="2024-01-02", periods=num_days)
        columns = pd.MultiIndex.from_product(
            [tickers, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        )
        data = np.random.RandomState(42).rand(num_days, len(tickers) * 6) * 100 + 50
        df = pd.DataFrame(data, index=dates, columns=columns)

        # Ensure OHLC consistency per ticker
        for ticker in tickers:
            for i in range(num_days):
                o = df.loc[df.index[i], (ticker, "Open")]
                c = df.loc[df.index[i], (ticker, "Close")]
                df.loc[df.index[i], (ticker, "High")] = max(o, c) * 1.02
                df.loc[df.index[i], (ticker, "Low")] = min(o, c) * 0.98

        return df

    @patch("meridian.data.providers.yahoo.yf.download")
    def test_fetch_single_ticker(self, mock_download):
        """Single ticker returns a flat DataFrame."""
        mock_download.return_value = self._make_single_ticker_df()

        bars = self.provider.fetch_historical(
            "AAPL",
            datetime.date(2024, 1, 2),
            datetime.date(2024, 1, 8),
        )

        assert len(bars) == 5
        assert all(b.ticker == "AAPL" for b in bars)
        assert bars[0].open == 150.0
        assert bars[0].volume > 0

    @patch("meridian.data.providers.yahoo.yf.download")
    def test_fetch_bulk_multiple_tickers(self, mock_download):
        """Bulk fetch with multi-level column index."""
        tickers = ["AAPL", "MSFT"]
        mock_download.return_value = self._make_multi_ticker_df(tickers)

        result = self.provider.fetch_bulk(
            tickers,
            datetime.date(2024, 1, 2),
            datetime.date(2024, 1, 8),
        )

        assert "AAPL" in result
        assert "MSFT" in result
        assert len(result["AAPL"]) == 5
        assert len(result["MSFT"]) == 5

    @patch("meridian.data.providers.yahoo.yf.download")
    def test_empty_dataframe_handling(self, mock_download):
        """Empty DataFrame (delisted ticker) returns empty dict."""
        mock_download.return_value = pd.DataFrame()

        result = self.provider.fetch_bulk(
            ["DELISTED"],
            datetime.date(2024, 1, 2),
            datetime.date(2024, 1, 8),
        )

        assert result == {}

    @patch("meridian.data.providers.yahoo.yf.download")
    def test_nan_values_dropped(self, mock_download):
        """Rows with all NaN values are dropped."""
        df = self._make_single_ticker_df(num_days=5)
        df.iloc[2] = np.nan  # Make one row all NaN

        mock_download.return_value = df

        bars = self.provider.fetch_historical(
            "AAPL",
            datetime.date(2024, 1, 2),
            datetime.date(2024, 1, 8),
        )

        assert len(bars) == 4  # 5 - 1 NaN row

    @patch("meridian.data.providers.yahoo.yf.download")
    def test_invalid_ticker_in_bulk(self, mock_download):
        """Invalid ticker not in response is handled gracefully."""
        tickers = ["AAPL", "INVALID"]
        # Only return AAPL data
        df = self._make_multi_ticker_df(["AAPL"])
        mock_download.return_value = df

        result = self.provider.fetch_bulk(
            tickers,
            datetime.date(2024, 1, 2),
            datetime.date(2024, 1, 8),
        )

        assert "AAPL" in result
        assert "INVALID" not in result

    def test_rate_limiting(self):
        """Rate limiter waits between requests."""
        import time

        provider = YahooFinanceProvider(rate_limit_per_second=10.0)
        provider._last_request_time = time.monotonic()

        start = time.monotonic()
        provider._wait_for_rate_limit()
        elapsed = time.monotonic() - start

        # Should wait ~0.1s (1/10 per second)
        assert elapsed >= 0.05  # Allow some tolerance

    @patch("meridian.data.providers.yahoo.yf.download")
    def test_health_check_success(self, mock_download):
        """Health check succeeds when data is returned."""
        mock_download.return_value = self._make_single_ticker_df(num_days=1)
        assert self.provider.health_check() is True

    @patch("meridian.data.providers.yahoo.yf.download")
    def test_health_check_failure(self, mock_download):
        """Health check fails when download raises."""
        mock_download.side_effect = Exception("Network error")
        assert self.provider.health_check() is False
