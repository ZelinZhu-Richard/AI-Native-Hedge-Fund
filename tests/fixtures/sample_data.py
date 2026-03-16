"""Sample data generators using geometric Brownian motion.

Used in tests to create realistic but deterministic market data.
"""

from __future__ import annotations

import datetime
import math
import random

from meridian.data.storage.models import OHLCVBar


def generate_random_walk(
    ticker: str = "TEST",
    start_date: datetime.date = datetime.date(2024, 1, 2),
    num_days: int = 100,
    start_price: float = 100.0,
    seed: int = 42,
) -> list[OHLCVBar]:
    """Generate realistic OHLCV data using geometric Brownian motion.

    Uses a fixed seed for deterministic test output.
    """
    rng = random.Random(seed)
    bars: list[OHLCVBar] = []
    price = start_price
    current_date = start_date

    for _ in range(num_days):
        # Skip weekends
        while current_date.weekday() >= 5:
            current_date += datetime.timedelta(days=1)

        # GBM daily return: mu=0.0005 (~12% annual), sigma=0.02 (~32% annual vol)
        daily_return = math.exp(0.0005 + 0.02 * rng.gauss(0, 1))
        close = round(price * daily_return, 2)

        # Intraday range
        intraday_vol = abs(rng.gauss(0, 0.01))
        high = round(max(price, close) * (1 + intraday_vol), 2)
        low = round(min(price, close) * (1 - intraday_vol), 2)

        # Ensure OHLC consistency
        open_price = round(price, 2)
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        volume = max(0, int(rng.gauss(1_000_000, 200_000)))

        bars.append(
            OHLCVBar(
                ticker=ticker,
                date=current_date,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                adj_close=close,
            )
        )

        price = close
        current_date += datetime.timedelta(days=1)

    return bars


def inject_gap(
    bars: list[OHLCVBar],
    gap_start_idx: int = 10,
    gap_days: int = 3,
) -> list[OHLCVBar]:
    """Remove bars to simulate missing trading days."""
    result = bars.copy()
    end_idx = min(gap_start_idx + gap_days, len(result))
    del result[gap_start_idx:end_idx]
    return result


def inject_split(
    bars: list[OHLCVBar],
    split_idx: int = 50,
    split_ratio: float = 0.5,
) -> list[OHLCVBar]:
    """Simulate an unadjusted stock split at the given index."""
    result = []
    for i, bar in enumerate(bars):
        if i >= split_idx:
            result.append(
                OHLCVBar(
                    ticker=bar.ticker,
                    date=bar.date,
                    open=round(bar.open * split_ratio, 2),
                    high=round(bar.high * split_ratio, 2),
                    low=round(bar.low * split_ratio, 2),
                    close=round(bar.close * split_ratio, 2),
                    volume=int(bar.volume / split_ratio),
                    adj_close=round(bar.adj_close * split_ratio, 2),
                )
            )
        else:
            result.append(bar)
    return result


def inject_outlier(
    bars: list[OHLCVBar],
    outlier_idx: int = 60,
    multiplier: float = 3.0,
) -> list[OHLCVBar]:
    """Inject an extreme price outlier at the given index."""
    result = bars.copy()
    bar = result[outlier_idx]
    result[outlier_idx] = OHLCVBar(
        ticker=bar.ticker,
        date=bar.date,
        open=round(bar.open * multiplier, 2),
        high=round(bar.high * multiplier, 2),
        low=round(bar.low * multiplier, 2),
        close=round(bar.close * multiplier, 2),
        volume=bar.volume,
        adj_close=round(bar.adj_close * multiplier, 2),
    )
    return result
