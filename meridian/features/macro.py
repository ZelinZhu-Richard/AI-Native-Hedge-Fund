"""Macro feature computer — 6 market-level features.

Receives SPY data and outputs date-indexed features that get broadcast
to all tickers. Sprint 2 adds VIX, yield curve, credit spreads, Fed
funds rate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from meridian.config.constants import TRADING_DAYS_PER_YEAR
from meridian.features.base import SingleTickerFeature
from meridian.features.registry import FeatureConfig


class MacroFeatureComputer(SingleTickerFeature):
    """Compute 6 macro/market features from SPY data."""

    @property
    def category(self) -> str:
        return "macro"

    @property
    def required_columns(self) -> list[str]:
        return ["adj_close"]

    def feature_configs(self) -> list[FeatureConfig]:
        c = "macro"
        return [
            FeatureConfig("spy_returns_5d", c, 5),
            FeatureConfig("spy_returns_21d", c, 21),
            FeatureConfig("spy_vol_21d", c, 21),
            FeatureConfig("spy_drawdown", c, 0),
            FeatureConfig("spy_above_200sma", c, 200),
            FeatureConfig("market_return_dispersion", c, 0),
        ]

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(index=df.index)

        close = df["adj_close"]
        log_returns = np.log(close / close.shift(1))

        result = pd.DataFrame(index=df.index)

        # SPY returns
        result["spy_returns_5d"] = close.pct_change(periods=5)
        result["spy_returns_21d"] = close.pct_change(periods=21)

        # SPY volatility
        annualize = np.sqrt(TRADING_DAYS_PER_YEAR)
        spy_vol = log_returns.rolling(
            window=21, min_periods=21
        ).std() * annualize
        result["spy_vol_21d"] = spy_vol

        # SPY drawdown (expanding max is inherently causal — no lookahead)
        running_max = close.expanding(min_periods=1).max()
        result["spy_drawdown"] = (close - running_max) / running_max

        # SPY above 200 SMA
        sma_200 = close.rolling(window=200, min_periods=200).mean()
        result["spy_above_200sma"] = (close > sma_200).astype(float)
        result.loc[sma_200.isna(), "spy_above_200sma"] = np.nan

        # Market return dispersion (passed from cross-sectional step)
        dispersion: pd.Series | None = kwargs.get("dispersion")
        if dispersion is not None:
            result["market_return_dispersion"] = dispersion.reindex(df.index)
        else:
            result["market_return_dispersion"] = np.nan

        return result
