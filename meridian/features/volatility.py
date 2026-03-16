"""Volatility feature computer — 10 volatility regime features.

Uses log returns for volatility calculations (more theoretically correct
for annualization). All rolling operations use min_periods to prevent
partial-window leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from meridian.config.constants import TRADING_DAYS_PER_YEAR
from meridian.features.base import SingleTickerFeature
from meridian.features.registry import FeatureConfig


class VolatilityFeatureComputer(SingleTickerFeature):
    """Compute 10 volatility features for a single ticker."""

    @property
    def category(self) -> str:
        return "volatility"

    @property
    def required_columns(self) -> list[str]:
        return ["adj_close", "high", "low", "open"]

    def feature_configs(self) -> list[FeatureConfig]:
        c = "volatility"
        return [
            FeatureConfig("realized_vol_5d", c, 5),
            FeatureConfig("realized_vol_21d", c, 21),
            FeatureConfig("realized_vol_63d", c, 63),
            FeatureConfig("vol_ratio_5_21", c, 21),
            FeatureConfig("vol_ratio_5_63", c, 63),
            FeatureConfig("vol_of_vol_21d", c, 42),
            FeatureConfig("garman_klass_vol_21d", c, 21),
            FeatureConfig("parkinson_vol_21d", c, 21),
            FeatureConfig("vol_regime_z_score", c, 315),
            FeatureConfig("high_low_range_ratio", c, 21),
        ]

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(index=df.index)

        close = df["adj_close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        log_returns = np.log(close / close.shift(1))
        annualize = np.sqrt(TRADING_DAYS_PER_YEAR)

        result = pd.DataFrame(index=df.index)

        # Realized volatility at different windows
        def _rv(w: int) -> pd.Series:
            return log_returns.rolling(
                window=w, min_periods=w
            ).std() * annualize

        result["realized_vol_5d"] = _rv(5)
        result["realized_vol_21d"] = _rv(21)
        result["realized_vol_63d"] = _rv(63)

        # Vol ratios (regime change detection)
        rv5 = result["realized_vol_5d"]
        result["vol_ratio_5_21"] = rv5 / result["realized_vol_21d"]
        result["vol_ratio_5_63"] = rv5 / result["realized_vol_63d"]

        # Vol of vol
        result["vol_of_vol_21d"] = rv5.rolling(
            window=21, min_periods=21
        ).std()

        # Garman-Klass volatility
        result["garman_klass_vol_21d"] = self._compute_garman_klass(
            open_, high, low, close, 21
        )

        # Parkinson volatility
        result["parkinson_vol_21d"] = self._compute_parkinson(high, low, 21)

        # Vol regime z-score: how unusual is current 63d vol relative to 252d history
        vol_63 = result["realized_vol_63d"]
        vol_mean_252 = vol_63.rolling(window=252, min_periods=252).mean()
        vol_std_252 = vol_63.rolling(window=252, min_periods=252).std()
        result["vol_regime_z_score"] = (vol_63 - vol_mean_252) / vol_std_252

        # High-low range ratio
        daily_range = (high - low) / low
        avg_range = daily_range.rolling(window=21, min_periods=21).mean()
        result["high_low_range_ratio"] = daily_range / avg_range

        return result

    @staticmethod
    def _compute_garman_klass(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int,
    ) -> pd.Series:
        """Garman-Klass volatility estimator using OHLC data."""
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_) ** 2
        gk_daily = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        gk_var = gk_daily.rolling(window=window, min_periods=window).mean()
        return np.sqrt(gk_var * TRADING_DAYS_PER_YEAR)

    @staticmethod
    def _compute_parkinson(
        high: pd.Series, low: pd.Series, window: int
    ) -> pd.Series:
        """Parkinson volatility estimator using high-low range."""
        log_hl_sq = np.log(high / low) ** 2
        factor = 1 / (4 * window * np.log(2))
        rolled = log_hl_sq.rolling(
            window=window, min_periods=window
        ).sum()
        parkinson_var = rolled * factor
        return np.sqrt(parkinson_var * TRADING_DAYS_PER_YEAR)
