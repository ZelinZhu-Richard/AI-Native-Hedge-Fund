"""Feature pipeline — three-phase orchestration.

Phase 1: Single-ticker features (technical + volatility) per ticker
Phase 2: Cross-sectional features (needs all tickers' data + Phase 1 returns)
Phase 3: Macro features (needs SPY data + Phase 2 dispersion)

Anti-lookahead: features are trimmed to [start_date, end_date] before storage.
Data is fetched with a lookback buffer to warm up rolling windows.
"""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn

from meridian.config.constants import (
    LOOKBACK_CALENDAR_BUFFER_MULTIPLIER,
    MAX_FEATURE_LOOKBACK_DAYS,
)
from meridian.core.logging import get_logger
from meridian.data.storage.database import MeridianDatabase
from meridian.features.cross_sectional import CrossSectionalFeatureComputer
from meridian.features.macro import MacroFeatureComputer
from meridian.features.registry import FeatureRegistry
from meridian.features.store import FeatureStore
from meridian.features.technical import TechnicalFeatureComputer
from meridian.features.volatility import VolatilityFeatureComputer

logger = get_logger("features.pipeline")


class FeaturePipeline:
    """Orchestrates feature computation across all computers."""

    def __init__(self, db: MeridianDatabase, store: FeatureStore) -> None:
        self.db = db
        self.store = store
        self.technical = TechnicalFeatureComputer()
        self.volatility = VolatilityFeatureComputer()
        self.cross_sectional = CrossSectionalFeatureComputer()
        self.macro = MacroFeatureComputer()

    def run(
        self,
        tickers: list[str],
        start_date: datetime.date,
        end_date: datetime.date,
        spy_ticker: str = "SPY",
        sector_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run the full feature pipeline.

        Returns summary dict: features_computed, tickers_processed, failures, etc.
        """
        if sector_map is None:
            sector_map = {}

        # Calculate lookback buffer for warming up rolling windows
        buffer_days = int(
            MAX_FEATURE_LOOKBACK_DAYS * LOOKBACK_CALENDAR_BUFFER_MULTIPLIER
        )
        data_start = start_date - datetime.timedelta(days=buffer_days)

        failures: list[dict[str, str]] = []
        tickers_processed = 0
        total_features_stored = 0

        # Collect per-ticker features for cross-sectional step
        ticker_features: dict[str, pd.DataFrame] = {}
        ticker_data: dict[str, pd.DataFrame] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            # === Phase 1: Single-ticker features ===
            task = progress.add_task(
                "Phase 1: Computing single-ticker features...",
                total=len(tickers),
            )

            for ticker in tickers:
                try:
                    df = self.db.get_data(
                        ticker, start_date=data_start, end_date=end_date
                    )
                    if df.empty:
                        logger.warning("no data for ticker", ticker=ticker)
                        failures.append({"ticker": ticker, "error": "no data"})
                        progress.advance(task)
                        continue

                    df = df.sort_values("date").set_index("date")
                    ticker_data[ticker] = df

                    # Technical features
                    tech_features = self.technical.compute(df)
                    # Volatility features
                    vol_features = self.volatility.compute(df)

                    combined = pd.concat([tech_features, vol_features], axis=1)
                    ticker_features[ticker] = combined
                    tickers_processed += 1

                except Exception as e:
                    logger.error(
                        "failed to compute features",
                        ticker=ticker, error=str(e),
                    )
                    failures.append({"ticker": ticker, "error": str(e)})

                progress.advance(task)

            # === Phase 2: Cross-sectional features ===
            progress.add_task("Phase 2: Computing cross-sectional features...")

            cs_features: dict[str, pd.DataFrame] = {}
            dispersion_series: pd.Series | None = None

            if ticker_features:
                try:
                    # Build stacked DataFrame for cross-sectional computer
                    stacked_frames = []
                    for ticker, df in ticker_data.items():
                        frame = df[["adj_close", "volume"]].copy()
                        frame["ticker"] = ticker
                        stacked_frames.append(frame)

                    stacked = pd.concat(stacked_frames).reset_index()
                    cs_result = self.cross_sectional.compute(
                        stacked, sector_map=sector_map
                    )

                    if not cs_result.empty:
                        # Extract dispersion for macro step
                        if "dispersion_21d" in cs_result.columns:
                            # Get one dispersion value per date (same for all tickers)
                            first_ticker = cs_result["ticker"].iloc[0]
                            mask = cs_result["ticker"] == first_ticker
                            dispersion_series = cs_result.loc[mask, "dispersion_21d"]

                        # Split back to per-ticker DataFrames
                        for ticker in cs_result["ticker"].unique():
                            ticker_mask = cs_result["ticker"] == ticker
                            ticker_cs = cs_result.loc[ticker_mask].drop(
                                columns=["ticker"]
                            )
                            cs_features[ticker] = ticker_cs

                except Exception as e:
                    logger.error("cross-sectional computation failed", error=str(e))
                    failures.append({"ticker": "CROSS_SECTIONAL", "error": str(e)})

            # === Phase 3: Macro features ===
            progress.add_task("Phase 3: Computing macro features...")

            macro_features: pd.DataFrame = pd.DataFrame()
            try:
                spy_df = self.db.get_data(
                    spy_ticker, start_date=data_start, end_date=end_date
                )
                if not spy_df.empty:
                    spy_df = spy_df.sort_values("date").set_index("date")
                    macro_features = self.macro.compute(
                        spy_df, dispersion=dispersion_series
                    )
            except Exception as e:
                logger.error("macro computation failed", error=str(e))
                failures.append({"ticker": "MACRO", "error": str(e)})

            # === Store features ===
            progress.add_task("Storing features...")

            for ticker in ticker_features:
                try:
                    combined = ticker_features[ticker]

                    # Add cross-sectional features if available
                    if ticker in cs_features:
                        combined = pd.concat([combined, cs_features[ticker]], axis=1)

                    # Add macro features (broadcast to all tickers)
                    if not macro_features.empty:
                        combined = pd.concat(
                            [combined, macro_features.reindex(combined.index)], axis=1
                        )

                    # Trim to requested date range
                    start_ts = pd.Timestamp(start_date)
                    end_ts = pd.Timestamp(end_date)
                    combined = combined.loc[
                        (combined.index >= start_ts) & (combined.index <= end_ts)
                    ]

                    rows = self.store.store_features(ticker, combined)
                    total_features_stored += rows

                except Exception as e:
                    logger.error(
                        "failed to store features",
                        ticker=ticker, error=str(e),
                    )
                    failures.append({"ticker": ticker, "error": f"store: {e}"})

        # Store feature metadata
        registry = FeatureRegistry.instance()
        self.store.store_feature_metadata(registry.list_features())

        summary = {
            "tickers_processed": tickers_processed,
            "tickers_failed": len(failures),
            "features_stored": total_features_stored,
            "feature_count": len(registry.list_features()),
            "failures": failures,
            "start_date": str(start_date),
            "end_date": str(end_date),
        }

        log_summary = {
            k: v for k, v in summary.items() if k != "failures"
        }
        logger.info("pipeline complete", **log_summary)
        return summary

    def run_incremental(
        self,
        tickers: list[str],
        spy_ticker: str = "SPY",
        sector_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Compute from last stored date + 1 to today."""
        # Find the latest feature date across all tickers
        latest_dates = []
        for ticker in tickers:
            d = self.store.get_latest_feature_date(ticker)
            if d is not None:
                latest_dates.append(d)

        if latest_dates:
            start_date = max(latest_dates) + datetime.timedelta(days=1)
        else:
            # No features stored — need a start date from data
            start_date = datetime.date(2020, 1, 1)

        end_date = datetime.date.today()

        if start_date > end_date:
            logger.info("features already up to date")
            return {
                "tickers_processed": 0,
                "tickers_failed": 0,
                "features_stored": 0,
                "feature_count": 0,
                "failures": [],
                "start_date": str(start_date),
                "end_date": str(end_date),
            }

        return self.run(
            tickers, start_date, end_date,
            spy_ticker=spy_ticker, sector_map=sector_map,
        )
