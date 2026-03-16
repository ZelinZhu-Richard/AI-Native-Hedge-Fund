"""Core type definitions used across Meridian."""

from __future__ import annotations

import datetime
from enum import StrEnum
from typing import Any, Protocol

# Plain alias -- avoids friction with yfinance/pandas/duckdb
Ticker = str

# Inclusive date range: (start, end)
DateRange = tuple[datetime.date, datetime.date]


class DataFrequency(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class MarketRegime(StrEnum):
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    CRISIS = "crisis"
    TRANSITION = "transition"
    UNKNOWN = "unknown"


class Serializable(Protocol):
    def model_dump(self) -> dict[str, Any]: ...
