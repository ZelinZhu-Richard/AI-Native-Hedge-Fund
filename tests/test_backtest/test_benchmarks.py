"""Tests for benchmark signal generators."""

from __future__ import annotations

import pandas as pd

from meridian.backtest.benchmarks import (
    BuyAndHold,
    MeanReversionBaseline,
    MomentumBaseline,
    SignalGenerator,
    SixtyFortyBenchmark,
    SPYBenchmark,
)


def _make_multi_ticker_data(tickers: list[str], n_days: int = 300) -> pd.DataFrame:
    """Create synthetic multi-ticker OHLCV DataFrame."""
    import numpy as np

    records = []
    dates = pd.bdate_range(start="2022-01-03", periods=n_days)
    rng = np.random.default_rng(42)

    for ticker in tickers:
        price = 100.0
        for dt in dates:
            ret = rng.normal(0.0005, 0.02)
            price *= 1 + ret
            records.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "open": price * 0.999,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "adj_close": price,
                    "volume": rng.integers(1_000_000, 50_000_000),
                }
            )

    df = pd.DataFrame(records)
    df = df.set_index("date")
    return df


class TestSignalGeneratorProtocol:
    def test_buy_and_hold_is_signal_generator(self):
        assert isinstance(BuyAndHold(), SignalGenerator)

    def test_momentum_is_signal_generator(self):
        assert isinstance(MomentumBaseline(), SignalGenerator)

    def test_mean_reversion_is_signal_generator(self):
        assert isinstance(MeanReversionBaseline(), SignalGenerator)

    def test_spy_is_signal_generator(self):
        assert isinstance(SPYBenchmark(), SignalGenerator)

    def test_sixty_forty_is_signal_generator(self):
        assert isinstance(SixtyFortyBenchmark(), SignalGenerator)


class TestBuyAndHold:
    def test_equal_weights(self):
        data = _make_multi_ticker_data(["AAPL", "MSFT", "GOOG"])
        bh = BuyAndHold()
        bh.fit(data)
        signals = bh.predict(data)
        assert not signals.empty
        # All weights should be 1/3
        weights = signals["weight"].unique()
        assert len(weights) == 1
        assert abs(weights[0] - 1.0 / 3) < 1e-10

    def test_name(self):
        assert BuyAndHold().name == "BuyAndHold"


class TestMomentumBaseline:
    def test_selects_top_n(self):
        data = _make_multi_ticker_data(["AAPL", "MSFT", "GOOG", "AMZN", "META"])
        mom = MomentumBaseline(top_n=3)
        mom.fit(data)
        signals = mom.predict(data)
        # Should have at most 3 tickers
        assert signals["ticker"].nunique() <= 3

    def test_name(self):
        assert MomentumBaseline().name == "Momentum12-1"


class TestSPYBenchmark:
    def test_100pct_spy(self):
        data = _make_multi_ticker_data(["SPY", "AAPL"])
        spy = SPYBenchmark()
        spy.fit(data)
        signals = spy.predict(data)
        assert all(signals["ticker"] == "SPY")
        assert all(signals["weight"] == 1.0)

    def test_no_spy_empty_signals(self):
        data = _make_multi_ticker_data(["AAPL", "MSFT"])
        spy = SPYBenchmark()
        spy.fit(data)
        signals = spy.predict(data)
        assert signals.empty
