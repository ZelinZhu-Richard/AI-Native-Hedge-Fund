"""Benchmark signal generators for backtest comparison.

SignalGenerator protocol: any model with fit() + predict() works.
Five benchmarks: buy-and-hold, momentum, mean-reversion, SPY, 60/40.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class SignalGenerator(Protocol):
    """Protocol for all signal generators (models, benchmarks, strategies).

    Any class implementing fit() and predict() with these signatures
    can be used in the walk-forward engine.
    """

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit model on training window data ONLY.

        Args:
            train_data: OHLCV + features for the training window.
                Must NOT contain any data from the test window.
        """
        ...

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for test window.

        Args:
            test_data: OHLCV + features for the test window.

        Returns:
            DataFrame with columns: date, ticker, weight.
            Weights are target portfolio fractions (positive=long).
            Signal at row t is executed at t+1 (engine handles this).
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable name for reports."""
        ...


class BuyAndHold:
    """Equal-weight buy-and-hold across all tickers.

    Simplest benchmark: allocate equally at start, never rebalance.
    """

    def __init__(self, max_positions: int = 20) -> None:
        self._tickers: list[str] = []
        self._max_positions = max_positions

    def fit(self, train_data: pd.DataFrame) -> None:
        tickers = self._extract_tickers(train_data)
        self._tickers = tickers[: self._max_positions]

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if not self._tickers:
            return pd.DataFrame(columns=["date", "ticker", "weight"])

        weight = 1.0 / len(self._tickers)
        dates = self._extract_dates(test_data)

        records: list[dict[str, Any]] = []
        for dt in dates:
            for ticker in self._tickers:
                records.append({"date": dt, "ticker": ticker, "weight": weight})
        return pd.DataFrame(records)

    @property
    def name(self) -> str:
        return "BuyAndHold"

    @staticmethod
    def _extract_tickers(data: pd.DataFrame) -> list[str]:
        if "ticker" in data.columns:
            return sorted(data["ticker"].unique().tolist())
        return []

    @staticmethod
    def _extract_dates(data: pd.DataFrame) -> list:
        if isinstance(data.index, pd.DatetimeIndex):
            return sorted(data.index.unique().tolist())
        if "date" in data.columns:
            return sorted(data["date"].unique().tolist())
        return []


class MomentumBaseline:
    """12-1 month momentum: long top N, skip most recent month.

    Classic momentum factor: sort by 12-month return excluding
    last month (to avoid short-term reversal), go long top quintile.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        skip_days: int = 21,
        top_n: int = 10,
    ) -> None:
        self._lookback = lookback_days
        self._skip = skip_days
        self._top_n = top_n
        self._rankings: dict[str, float] = {}

    def fit(self, train_data: pd.DataFrame) -> None:
        self._rankings = {}
        tickers = self._extract_tickers(train_data)

        for ticker in tickers:
            ticker_data = (
                train_data[train_data["ticker"] == ticker]
                if "ticker" in train_data.columns
                else train_data
            )
            if (
                "adj_close" in ticker_data.columns
                and len(ticker_data) > self._skip + 21
            ):
                prices = ticker_data["adj_close"].values
                if len(prices) > self._skip:
                    # 12-1 momentum: total return excluding last month
                    start_price = (
                        prices[0] if len(prices) >= self._lookback else prices[0]
                    )
                    end_price = prices[-self._skip] if self._skip > 0 else prices[-1]
                    if start_price > 0:
                        self._rankings[ticker] = (end_price / start_price) - 1.0

        # Sort descending, take top N
        sorted_tickers = sorted(
            self._rankings.items(), key=lambda x: x[1], reverse=True
        )
        top = sorted_tickers[: self._top_n]
        if top:
            weight = 1.0 / len(top)
            self._rankings = {t: weight for t, _ in top}
        else:
            self._rankings = {}

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if not self._rankings:
            return pd.DataFrame(columns=["date", "ticker", "weight"])

        dates = self._extract_dates(test_data)
        records: list[dict[str, Any]] = []
        for dt in dates:
            for ticker, weight in self._rankings.items():
                records.append({"date": dt, "ticker": ticker, "weight": weight})
        return pd.DataFrame(records)

    @property
    def name(self) -> str:
        return "Momentum12-1"

    @staticmethod
    def _extract_tickers(data: pd.DataFrame) -> list[str]:
        if "ticker" in data.columns:
            return sorted(data["ticker"].unique().tolist())
        return []

    @staticmethod
    def _extract_dates(data: pd.DataFrame) -> list:
        if isinstance(data.index, pd.DatetimeIndex):
            return sorted(data.index.unique().tolist())
        if "date" in data.columns:
            return sorted(data["date"].unique().tolist())
        return []


class MeanReversionBaseline:
    """Short-term mean reversion: long oversold, short overbought.

    Buy stocks that dropped most over lookback, sell those that rose most.
    Simple contrarian strategy.
    """

    def __init__(
        self,
        lookback_days: int = 5,
        top_n: int = 10,
    ) -> None:
        self._lookback = lookback_days
        self._top_n = top_n
        self._weights: dict[str, float] = {}

    def fit(self, train_data: pd.DataFrame) -> None:
        self._weights = {}
        tickers = self._extract_tickers(train_data)
        returns: dict[str, float] = {}

        for ticker in tickers:
            ticker_data = (
                train_data[train_data["ticker"] == ticker]
                if "ticker" in train_data.columns
                else train_data
            )
            if "adj_close" in ticker_data.columns and len(ticker_data) > self._lookback:
                prices = ticker_data["adj_close"].values
                recent_return = (prices[-1] / prices[-self._lookback]) - 1.0
                returns[ticker] = recent_return

        if not returns:
            return

        # Long the biggest losers (mean reversion)
        sorted_tickers = sorted(returns.items(), key=lambda x: x[1])
        losers = sorted_tickers[: self._top_n]
        if losers:
            weight = 1.0 / len(losers)
            self._weights = {t: weight for t, _ in losers}

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if not self._weights:
            return pd.DataFrame(columns=["date", "ticker", "weight"])

        dates = self._extract_dates(test_data)
        records: list[dict[str, Any]] = []
        for dt in dates:
            for ticker, weight in self._weights.items():
                records.append({"date": dt, "ticker": ticker, "weight": weight})
        return pd.DataFrame(records)

    @property
    def name(self) -> str:
        return "MeanReversion"

    @staticmethod
    def _extract_tickers(data: pd.DataFrame) -> list[str]:
        if "ticker" in data.columns:
            return sorted(data["ticker"].unique().tolist())
        return []

    @staticmethod
    def _extract_dates(data: pd.DataFrame) -> list:
        if isinstance(data.index, pd.DatetimeIndex):
            return sorted(data.index.unique().tolist())
        if "date" in data.columns:
            return sorted(data["date"].unique().tolist())
        return []


class SPYBenchmark:
    """100% SPY (S&P 500 ETF). The benchmark every fund must beat."""

    def __init__(self) -> None:
        self._has_spy = False

    def fit(self, train_data: pd.DataFrame) -> None:
        tickers = set()
        if "ticker" in train_data.columns:
            tickers = set(train_data["ticker"].unique())
        self._has_spy = "SPY" in tickers

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if not self._has_spy:
            return pd.DataFrame(columns=["date", "ticker", "weight"])

        dates = self._extract_dates(test_data)
        records = [{"date": dt, "ticker": "SPY", "weight": 1.0} for dt in dates]
        return pd.DataFrame(records)

    @property
    def name(self) -> str:
        return "SPY"

    @staticmethod
    def _extract_dates(data: pd.DataFrame) -> list:
        if isinstance(data.index, pd.DatetimeIndex):
            return sorted(data.index.unique().tolist())
        if "date" in data.columns:
            return sorted(data["date"].unique().tolist())
        return []


class SixtyFortyBenchmark:
    """60% SPY / 40% TLT. Classic balanced portfolio."""

    def __init__(self) -> None:
        self._available: dict[str, float] = {}

    def fit(self, train_data: pd.DataFrame) -> None:
        tickers = set()
        if "ticker" in train_data.columns:
            tickers = set(train_data["ticker"].unique())

        self._available = {}
        if "SPY" in tickers:
            self._available["SPY"] = 0.6
        if "TLT" in tickers:
            self._available["TLT"] = 0.4

        # Normalize if only one is available
        if self._available:
            total = sum(self._available.values())
            self._available = {t: w / total for t, w in self._available.items()}

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if not self._available:
            return pd.DataFrame(columns=["date", "ticker", "weight"])

        dates = self._extract_dates(test_data)
        records: list[dict[str, Any]] = []
        for dt in dates:
            for ticker, weight in self._available.items():
                records.append({"date": dt, "ticker": ticker, "weight": weight})
        return pd.DataFrame(records)

    @property
    def name(self) -> str:
        return "60/40"

    @staticmethod
    def _extract_dates(data: pd.DataFrame) -> list:
        if isinstance(data.index, pd.DatetimeIndex):
            return sorted(data.index.unique().tolist())
        if "date" in data.columns:
            return sorted(data["date"].unique().tolist())
        return []
