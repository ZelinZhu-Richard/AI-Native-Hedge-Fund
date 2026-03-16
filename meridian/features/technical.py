"""Technical feature computer — 25 price/volume features.

Uses adj_close for all return/price-level calculations (split-adjusted).
Uses raw high/low where intraday range matters.
All rolling operations use min_periods to prevent partial-window leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from meridian.features.base import SingleTickerFeature
from meridian.features.registry import FeatureConfig


class TechnicalFeatureComputer(SingleTickerFeature):
    """Compute 25 technical analysis features for a single ticker."""

    @property
    def category(self) -> str:
        return "technical"

    @property
    def required_columns(self) -> list[str]:
        return ["adj_close", "high", "low", "close", "volume"]

    def feature_configs(self) -> list[FeatureConfig]:
        c = "technical"
        return [
            FeatureConfig("returns_1d", c, 1),
            FeatureConfig("returns_5d", c, 5),
            FeatureConfig("returns_21d", c, 21),
            FeatureConfig("returns_63d", c, 63),
            FeatureConfig("rsi_14", c, 14),
            FeatureConfig("rsi_28", c, 28),
            FeatureConfig("macd_signal", c, 35),
            FeatureConfig("macd_histogram", c, 35),
            FeatureConfig("momentum_12_1", c, 252),
            FeatureConfig("mean_reversion_5d", c, 5),
            FeatureConfig("mean_reversion_21d", c, 21),
            FeatureConfig("bollinger_upper_dist", c, 20),
            FeatureConfig("bollinger_lower_dist", c, 20),
            FeatureConfig("z_score_21d", c, 21),
            FeatureConfig("sma_ratio_5_21", c, 21),
            FeatureConfig("sma_ratio_21_63", c, 63),
            FeatureConfig("ema_ratio_12_26", c, 26),
            FeatureConfig("price_to_sma_50", c, 50),
            FeatureConfig("price_to_sma_200", c, 200),
            FeatureConfig("golden_cross", c, 200),
            FeatureConfig("death_cross", c, 200),
            FeatureConfig("volume_ratio_5d", c, 5),
            FeatureConfig("volume_ratio_21d", c, 21),
            FeatureConfig("obv_slope_14d", c, 14),
            FeatureConfig("vwap_distance", c, 21),
        ]

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(index=df.index)

        close = df["adj_close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].astype(float)

        result = pd.DataFrame(index=df.index)

        # Returns
        result["returns_1d"] = close.pct_change(periods=1)
        result["returns_5d"] = close.pct_change(periods=5)
        result["returns_21d"] = close.pct_change(periods=21)
        result["returns_63d"] = close.pct_change(periods=63)

        # RSI
        result["rsi_14"] = self._compute_rsi(close, 14)
        result["rsi_28"] = self._compute_rsi(close, 28)

        # MACD
        macd_line, signal_line = self._compute_macd(close)
        result["macd_signal"] = macd_line - signal_line
        result["macd_histogram"] = macd_line - signal_line

        # Momentum 12-1
        ret_252 = close.pct_change(periods=252)
        ret_21 = close.pct_change(periods=21)
        result["momentum_12_1"] = ret_252 - ret_21

        # Mean reversion
        result["mean_reversion_5d"] = self._compute_mean_reversion(close, 5)
        result["mean_reversion_21d"] = self._compute_mean_reversion(close, 21)

        # Bollinger
        upper_dist, lower_dist = self._compute_bollinger_distance(close, 20, 2)
        result["bollinger_upper_dist"] = upper_dist
        result["bollinger_lower_dist"] = lower_dist

        # Z-score
        result["z_score_21d"] = self._compute_z_score(close, 21)

        # SMA ratios
        result["sma_ratio_5_21"] = self._compute_sma_ratio(close, 5, 21)
        result["sma_ratio_21_63"] = self._compute_sma_ratio(close, 21, 63)

        # EMA ratio
        result["ema_ratio_12_26"] = self._compute_ema_ratio(close, 12, 26)

        # Price to SMA
        sma_50 = close.rolling(window=50, min_periods=50).mean()
        sma_200 = close.rolling(window=200, min_periods=200).mean()
        result["price_to_sma_50"] = close / sma_50
        result["price_to_sma_200"] = close / sma_200

        # Golden/death cross
        result["golden_cross"] = (sma_50 > sma_200).astype(float)
        result["death_cross"] = (sma_50 < sma_200).astype(float)
        # NaN where either SMA is NaN
        sma_mask = sma_50.isna() | sma_200.isna()
        result.loc[sma_mask, "golden_cross"] = np.nan
        result.loc[sma_mask, "death_cross"] = np.nan

        # Volume ratios
        vol_avg_5 = volume.rolling(window=5, min_periods=5).mean()
        vol_avg_21 = volume.rolling(window=21, min_periods=21).mean()
        result["volume_ratio_5d"] = volume / vol_avg_5
        result["volume_ratio_21d"] = volume / vol_avg_21

        # OBV slope
        result["obv_slope_14d"] = self._compute_obv_slope(close, volume, 14)

        # VWAP distance
        result["vwap_distance"] = self._compute_vwap_distance(
            close, volume, high, low, 21
        )

        return result

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
        """RSI via Wilder smoothing (ewm with alpha=1/period)."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        # Where avg_loss is 0, RSI should be 100
        rsi = rsi.where(avg_loss > 0, 100.0)
        # Enforce NaN for warmup
        rsi.iloc[:period] = np.nan
        return rsi

    @staticmethod
    def _compute_macd(
        close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series]:
        """MACD line and signal line."""
        ema_fast = close.ewm(span=fast, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, min_periods=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
        return macd_line, signal_line

    @staticmethod
    def _compute_mean_reversion(close: pd.Series, window: int) -> pd.Series:
        """(close - SMA) / SMA."""
        sma = close.rolling(window=window, min_periods=window).mean()
        return (close - sma) / sma

    @staticmethod
    def _compute_bollinger_distance(
        close: pd.Series, window: int = 20, num_std: int = 2
    ) -> tuple[pd.Series, pd.Series]:
        """Distance from upper/lower Bollinger Band in std devs."""
        sma = close.rolling(window=window, min_periods=window).mean()
        std = close.rolling(window=window, min_periods=window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        upper_dist = (close - upper) / std
        lower_dist = (close - lower) / std
        return upper_dist, lower_dist

    @staticmethod
    def _compute_z_score(close: pd.Series, window: int) -> pd.Series:
        """(close - rolling_mean) / rolling_std."""
        mean = close.rolling(window=window, min_periods=window).mean()
        std = close.rolling(window=window, min_periods=window).std()
        return (close - mean) / std

    @staticmethod
    def _compute_sma_ratio(close: pd.Series, fast: int, slow: int) -> pd.Series:
        """SMA_fast / SMA_slow."""
        sma_fast = close.rolling(window=fast, min_periods=fast).mean()
        sma_slow = close.rolling(window=slow, min_periods=slow).mean()
        return sma_fast / sma_slow

    @staticmethod
    def _compute_ema_ratio(close: pd.Series, fast: int, slow: int) -> pd.Series:
        """EMA_fast / EMA_slow."""
        ema_fast = close.ewm(span=fast, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, min_periods=slow).mean()
        return ema_fast / ema_slow

    @staticmethod
    def _compute_obv_slope(
        close: pd.Series, volume: pd.Series, window: int
    ) -> pd.Series:
        """Linear regression slope of OBV over rolling window."""
        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()

        def _lr_slope(vals: np.ndarray) -> float:
            if len(vals) < window:
                return np.nan
            x = np.arange(len(vals))
            x_mean = x.mean()
            y_mean = vals.mean()
            denom = ((x - x_mean) ** 2).sum()
            if denom == 0:
                return 0.0
            return float(((x - x_mean) * (vals - y_mean)).sum() / denom)

        return obv.rolling(window=window, min_periods=window).apply(_lr_slope, raw=True)

    @staticmethod
    def _compute_vwap_distance(
        close: pd.Series,
        volume: pd.Series,
        high: pd.Series,
        low: pd.Series,
        window: int,
    ) -> pd.Series:
        """(close - VWAP) / VWAP over rolling window."""
        typical_price = (high + low + close) / 3
        tp_vol = typical_price * volume
        vwap = (
            tp_vol.rolling(window=window, min_periods=window).sum()
            / volume.rolling(window=window, min_periods=window).sum()
        )
        return (close - vwap) / vwap
