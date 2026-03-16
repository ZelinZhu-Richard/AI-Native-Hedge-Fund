"""Cross-sectional feature computer — 8 universe-wide features.

Computes rank and relative metrics across the full ticker universe on each
date. A ticker that IPOs mid-period does not affect rankings before its
first date.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from meridian.core.exceptions import FeatureComputationError
from meridian.features.base import CrossSectionalFeature
from meridian.features.registry import FeatureConfig


class CrossSectionalFeatureComputer(CrossSectionalFeature):
    """Compute 8 cross-sectional features across the ticker universe."""

    @property
    def category(self) -> str:
        return "cross_sectional"

    @property
    def required_columns(self) -> list[str]:
        return ["adj_close", "volume", "ticker"]

    def feature_configs(self) -> list[FeatureConfig]:
        c = "cross_sectional"
        return [
            FeatureConfig("rank_returns_21d", c, 21),
            FeatureConfig("rank_returns_63d", c, 63),
            FeatureConfig("sector_relative_return_21d", c, 21),
            FeatureConfig("sector_relative_return_63d", c, 63),
            FeatureConfig("sector_momentum_21d", c, 21),
            FeatureConfig("market_breadth", c, 21),
            FeatureConfig("dispersion_21d", c, 21),
            FeatureConfig("rank_volume_ratio", c, 21),
        ]

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(index=df.index)

        sector_map: dict[str, str] | None = kwargs.get("sector_map")
        if sector_map is None:
            raise FeatureComputationError(
                "sector_map kwarg required for cross-sectional features"
            )

        # Work with a copy to avoid mutating input
        data = df.copy()
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
        else:
            data["date"] = data.index

        # Ensure we have a date column (could be index)
        if "date" not in data.columns:
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index(names=["date"])
            else:
                raise FeatureComputationError(
                    "DataFrame must have a 'date' column or DatetimeIndex"
                )

        # Pivot to wide format: rows=dates, columns=tickers
        close_wide = data.pivot_table(
            index="date", columns="ticker", values="adj_close", aggfunc="first"
        )
        volume_wide = data.pivot_table(
            index="date", columns="ticker", values="volume", aggfunc="first"
        )

        tickers = close_wide.columns.tolist()

        # Compute returns in wide format
        returns_21d = close_wide.pct_change(periods=21)
        returns_63d = close_wide.pct_change(periods=63)

        # Volume ratio in wide format
        vol_avg_21 = volume_wide.rolling(window=21, min_periods=21).mean()
        volume_ratio = volume_wide / vol_avg_21

        # Percentile ranks (0-1) across tickers on each date
        rank_ret_21 = returns_21d.rank(axis=1, pct=True)
        rank_ret_63 = returns_63d.rank(axis=1, pct=True)
        rank_vol = volume_ratio.rank(axis=1, pct=True)

        # Sector-relative returns
        sector_series = pd.Series(
            {t: sector_map.get(t, "Unknown") for t in tickers}
        )
        idx = close_wide.index
        sector_rel_21d = pd.DataFrame(
            index=idx, columns=tickers, dtype=float
        )
        sector_rel_63d = pd.DataFrame(
            index=idx, columns=tickers, dtype=float
        )
        sector_mom_21d = pd.DataFrame(
            index=idx, columns=tickers, dtype=float
        )

        for sector in sector_series.unique():
            sector_tickers = sector_series[sector_series == sector].index.tolist()
            if not sector_tickers:
                continue

            # Sector mean return (only from tickers with data on that date)
            sector_mean_21 = returns_21d[sector_tickers].mean(axis=1)
            sector_mean_63 = returns_63d[sector_tickers].mean(axis=1)

            for t in sector_tickers:
                sector_rel_21d[t] = returns_21d[t] - sector_mean_21
                sector_rel_63d[t] = returns_63d[t] - sector_mean_63
                sector_mom_21d[t] = sector_mean_21

        # Market breadth: fraction of tickers with positive 21d returns
        market_breadth = (returns_21d > 0).sum(axis=1) / returns_21d.notna().sum(axis=1)

        # Dispersion: cross-sectional std of 21d returns
        dispersion = returns_21d.std(axis=1)

        # Melt back to long format matching input structure
        result_frames = []
        for ticker in tickers:
            ticker_dates = close_wide.index[close_wide[ticker].notna()]
            if len(ticker_dates) == 0:
                continue

            ticker_result = pd.DataFrame(index=ticker_dates)
            ticker_result["ticker"] = ticker
            ticker_result["rank_returns_21d"] = rank_ret_21[ticker]
            ticker_result["rank_returns_63d"] = rank_ret_63[ticker]
            ticker_result["sector_relative_return_21d"] = sector_rel_21d[ticker]
            ticker_result["sector_relative_return_63d"] = sector_rel_63d[ticker]
            ticker_result["sector_momentum_21d"] = sector_mom_21d[ticker]
            ticker_result["market_breadth"] = market_breadth
            ticker_result["dispersion_21d"] = dispersion
            ticker_result["rank_volume_ratio"] = rank_vol[ticker]
            result_frames.append(ticker_result)

        if not result_frames:
            return pd.DataFrame()

        result = pd.concat(result_frames)
        result = result.loc[result.index.notna()]
        return result
